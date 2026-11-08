from pathlib import Path
from time import monotonic

from caiso.core.content_filter import ContentFilteredError, record_skip
from caiso.core.serialization import digest, save_json

from .base import ConfigurationOptimizer, filter_current_configuration


def generate_dpo_preferences(component, context):
    cfg = context.state.configurations[component.component_id]
    pairs = []
    for i, trajectory in enumerate(context.trajectories):
        execution = next(
            t for t in trajectory.executions if t.component_id == component.component_id
        )
        seed = context.seed + i * 2
        try:
            a, _ = context.runtime.execute_component(
                component.component_id, execution.component_input, cfg, seed
            )
            b, _ = context.runtime.execute_component(
                component.component_id, execution.component_input, cfg, seed + 1
            )
            if a == b:
                continue
            choice = context.judge.choose(
                context.constitution,
                {"input": execution.component_input, "output": a},
                {"input": execution.component_input, "output": b},
                seed,
            )
        except ContentFilteredError as error:
            record_skip(
                context.runtime.providers, error, "dpo_preferences", example_id=trajectory.input_id
            )
            continue
        output = next(iter(component.output_fields))
        chosen, rejected = (a[output], b[output]) if choice == "A" else (b[output], a[output])
        prompt = context.runtime.training_prompt(
            component.component_id, execution.component_input, cfg
        )
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
        pairs = generate_dpo_preferences(component, context)
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
