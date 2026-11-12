from pathlib import Path
from time import monotonic

from caiso.constitution.dicai.preferences import response_view
from caiso.core.serialization import digest, save_json
from caiso.method.config import weight_preference_settings
from caiso.method.preference_collection import collect_preferences

from .base import ConfigurationOptimizer, filter_current_configuration


def generate_dpo_preferences(component, context, *, records=None):
    """Judge full unchanged/changed block trajectories, then project valid DPO pairs."""
    count, attempts, settings = weight_preference_settings(context.config)
    members = context.members or [component.component_id]
    if component.component_id not in members:
        raise ValueError("Weight component must belong to the selected block")
    constitution = context.system_constitution if context.system_constitution is not None else context.constitution
    if constitution is None:
        raise ValueError("Weight preferences require the system constitution")
    cfg = context.state.configurations[component.component_id]
    output = next(iter(component.output_fields))

    def label(a, b, seed):
        left, right = [next((e for e in t.executions if e.component_id == component.component_id), None)
                       for t in (a, b)]
        # DPO compares completions for ONE prompt. An upstream change can give
        # this component different inputs; such a trajectory pair cannot train it.
        if (left is None or right is None
                or digest(left.component_input) != digest(right.component_input)
                or left.component_output[output] == right.component_output[output]):
            return None
        return context.judge.choose(constitution, response_view(a.model_dump()),
                                    response_view(b.model_dump()), seed)

    examples = context.preference_examples if context.preference_examples is not None else context.examples
    with context.runtime.fresh_preference_executions():
        preferences = collect_preferences(
            context.runtime, context.state, examples, count, attempts, context.seed,
            settings=settings, component_ids=members, labeler=label, allow_empty=True,
        )
    if records is not None:
        records.extend(preferences)
    pairs = []
    for pair in preferences:
        a, b = [next(e for e in c["trajectory"]["executions"] if e["component_id"] == component.component_id)
                for c in (pair.candidate_a, pair.candidate_b)]
        chosen, rejected = (a, b) if pair.preferred == "A" else (b, a)
        prompt = context.runtime.training_prompt(
            component.component_id, a["component_input"], cfg
        )
        chosen, rejected = chosen["component_output"][output], rejected["component_output"][output]
        raw = component.settings.get("prompt_format") == "completion"
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen if raw else [{"role": "assistant", "content": chosen}],
                "rejected": rejected if raw else [{"role": "assistant", "content": rejected}],
            }
        )
    return pairs


def train_dpo(model_id, current_adapter, pairs, output_dir, config, budget, seed):
    if not pairs:
        raise ValueError("DPO requires distinct chosen/rejected pairs")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import DPOConfig, DPOTrainer

    set_seed(seed)
    dtype = config.get("dtype", "bfloat16")
    options = {"torch_dtype": getattr(torch, dtype), "trust_remote_code": False}
    if config.get("device_map"):
        options["device_map"] = config["device_map"]
    if config.get("quantization"):
        if config["quantization"] != "4bit":
            raise ValueError("DPO supports optional 4bit quantization")
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, dtype)
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **options)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    peft_config = LoraConfig(
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        task_type="CAUSAL_LM",
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    if current_adapter:
        # TRL 0.29 snapshots the loaded default adapter as a frozen reference.
        model = PeftModel.from_pretrained(model, current_adapter, is_trainable=True)
        peft_config = None

    class AccountingCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            budget.check()
            if budget.usage.training_steps >= config.get("max_total_training_steps", 10000):
                from caiso.core.budget import BudgetExceeded

                raise BudgetExceeded("Training-step budget exhausted")

        def on_step_end(self, args, state, control, **kwargs):
            budget.usage.training_steps += 1

    args = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.get("learning_rate", 5e-6),
        num_train_epochs=config.get("epochs", 1),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        max_steps=config.get("max_steps", -1),
        max_length=config.get("max_length", 2048),
        beta=config.get("beta", 0.1),
        bf16=dtype == "bfloat16",
        fp16=dtype == "float16",
        seed=seed,
        data_seed=seed,
        report_to="none",
        disable_tqdm=config.get("disable_tqdm", True),
        save_strategy="no",
        use_cpu=config.get("use_cpu", False),
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(pairs),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[AccountingCallback()],
    )
    started = monotonic()
    try:
        trainer.train()
    finally:
        budget.usage.training_seconds += monotonic() - started
    trainer.model.save_pretrained(str(output_dir), selected_adapters=["default"])
    saved = Path(output_dir)
    tokenizer.save_pretrained(str(saved))
    return str(saved.resolve())


class WeightOptimizer(ConfigurationOptimizer):
    def propose(self, component, context):
        current = context.state.configurations[component.component_id]
        records = []
        pairs = generate_dpo_preferences(component, context, records=records)
        if not pairs:
            return filter_current_configuration(current, [current.model_copy(deep=True)], context.config)
        cfg = context.config.get("dpo", {})
        provider = context.runtime.providers.get(current.value["model"])
        if not hasattr(provider, "model_id"):
            raise ValueError("DPO requires a local model provider")
        directory = (
            context.output_dir
            / "dpo"
            / component.component_id
            / digest([pairs, current, cfg, context.seed])[:16]
        )
        save_json(directory / "preferences.json", pairs)
        save_json(directory / "trajectory_preferences.json", records)
        constitution = context.system_constitution if context.system_constitution is not None else context.constitution
        save_json(directory / "preference_collection.json", {
            "protocol": "weight-configuration-pair-v1", "component_id": component.component_id,
            "members": context.members or [component.component_id],
            "system_constitution": {"principles": constitution.principles},
            "requested_pairs": cfg.get("preference_pairs", 20), "retained_pairs": len(pairs),
            "attempt_limit": cfg.get("preference_attempts", -1),
            "temperature": cfg.get("preference_temperature", 0.9), "seed": context.seed,
        })
        adapter = train_dpo(
            provider.model_id,
            current.value.get("adapter"),
            pairs,
            directory,
            cfg,
            context.runtime.budget,
            context.seed,
        )
        proposed = current.model_copy(deep=True)
        proposed.value = {**current.value, "adapter": adapter}
        proposed.metadata = {"training_data_hash": digest(pairs), "seed": context.seed, "dpo": cfg}
        return filter_current_configuration(current, [current.model_copy(deep=True), proposed], context.config)
