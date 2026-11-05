"""Exercise real benchmark classes with deterministic HTTP/LLM responses.

These check request/execution parity, not benchmark accuracy or model quality.
"""

from copy import deepcopy
import re
from types import SimpleNamespace

import dspy
import litellm
import pytest

from caiso.core.budget import BudgetTracker
from caiso.integration.runtime import UpstreamRuntime, adapt_examples
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example


@pytest.fixture
def fake_completion(monkeypatch):
    calls = []

    def complete(**kwargs):
        calls.append(deepcopy(kwargs))
        text = kwargs["messages"][-1]["content"]
        field = re.search(r"starting with the field `\[\[ ## (\w+) ## \]\]`", text).group(1)
        values = {
            "rewritten_query": "Who wrote this?",
            "search_keywords": "writer",
            "hints": "The writer is Alice.",
            "answer": "Alice",
        }
        return litellm.ModelResponse(
            model=kwargs["model"],
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": f"[[ ## {field} ## ]]\n{values[field]}\n[[ ## completed ## ]]",
                    }
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    monkeypatch.setattr(litellm, "completion", complete)
    return calls


def assert_parity(system, example, calls):
    """Compare direct upstream execution against CAISO's full-system execution."""
    for component in system.components.values():
        component.num_retry = 1
        component.retry_after = 0
    runtime = UpstreamRuntime(system, ProviderRegistry({}), BudgetTracker(len(system.components)))
    calls.clear()
    direct_score, direct = system.evaluate(example, return_pred=True)
    direct_calls = deepcopy(calls)
    calls.clear()
    adapted = runtime.execute(
        runtime.initial_state, adapt_examples(system, [example], "train")[0], 42
    )
    assert adapted.reward == direct_score
    assert adapted.final_output == {
        field: getattr(direct, field) for field in system.final_output_fields
    }
    assert [
        (entry.component_id, entry.component_input, entry.component_output)
        for entry in adapted.executions
    ] == [(cid, entry["input"], entry["output"]) for cid, entry in direct.traj.items()]
    assert calls == direct_calls
    return runtime


def test_hotpotqa_real_pipeline_preserves_requests_and_f1(monkeypatch, fake_completion):
    from examples.systems.hotpotqa import five_components as hotpot

    monkeypatch.setattr(
        dspy.ColBERTv2,
        "__call__",
        lambda self, query, k=10, **kwargs: [dspy.Example(long_text="Alice wrote the book.")],
    )
    system = hotpot.system_engine()
    runtime = assert_parity(
        system, Example(question="Who wrote this?", gd_answer="Alice"), fake_completion
    )
    assert len(fake_completion) == 4
    assert all(
        call["max_tokens"] == 1024 and call["temperature"] == 0.6 for call in fake_completion
    )
    assert runtime.initial_state.configurations["wikipedia_retriever"].value == {"k": 1}


def test_pubmed_real_pipeline_preserves_whitespace_models_and_metric(monkeypatch):
    from examples.systems.pubmed import three_components_with_model_selection as pubmed

    calls = []

    def complete(**kwargs):
        calls.append(deepcopy(kwargs))
        if "Please summarize" in kwargs["message"]:
            return "  Relevant clinical evidence.\n"
        return "Knowledge suggests uncertainty. Answer: yes"

    monkeypatch.setattr(pubmed, "get_llm_output", complete)
    system = pubmed.system_engine()
    runtime = assert_parity(
        system, Example(question="Does it work?", context="Trial evidence", groundtruth="no"), calls
    )
    assert len(calls) == 2
    assert all(call["max_new_tokens"] == 4096 and call["temperature"] == 0 for call in calls)
    assert all(call["model"] == "gpt-4o-mini" for call in calls)
    assert '"  Relevant clinical evidence.\n"' in calls[-1]["message"]
    # Preserve the upstream first-substring scorer, including its 'Knowledge' case.
    example = adapt_examples(
        system,
        [Example(question="Does it work?", context="Trial evidence", groundtruth="no")],
        "train",
    )[0]
    assert runtime.execute(runtime.initial_state, example, 42).reward == 1
    # Blank-summary interventions must satisfy the original nonempty-input check.
    runtime.execute(
        runtime.initial_state, example, 42, intervention="context_analyst->problem_solver"
    )


def test_amazon_real_pipeline_and_training_prompt_use_original_code(monkeypatch, fake_completion, caplog):
    from examples.systems.amazon import local_models_for_next_item_selection as amazon

    monkeypatch.setattr(amazon, "initialize_adapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(amazon, "load_lora_adapter", lambda *args, **kwargs: None)
    calls = []

    def post(prompt, *args, **kwargs):
        calls.append((prompt, deepcopy(kwargs)))
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"choices": [{"text": " local output "}]}
        )

    monkeypatch.setattr(amazon, "post_http_request", post)
    system = amazon.system_engine()
    runtime = assert_parity(
        system, Example(sequence="Viewed a pen", choices=["Pen", "Cup"], gd_answer="0"), calls
    )
    assert len(calls) == 2
    assert all(item[1]["max_new_tokens"] == 512 for item in calls)
    assert "self.variable is defined but was not accessed" not in caplog.text
    inputs = {"sequence": "Viewed a pen"}
    config = runtime.initial_state.configurations["session_analyzer"]
    before = len(calls)
    prompt = runtime.training_prompt("session_analyzer", inputs, config)
    assert prompt == calls[0][0]
    assert len(calls) == before  # Capture makes no completion request.
    profiler_prompt = runtime.training_prompt(
        "candidate_profiler",
        {"context": "local output", "choices": ["Pen", "Cup"]},
        runtime.initial_state.configurations["candidate_profiler"],
    )
    assert profiler_prompt == calls[1][0]


def test_amazon_dpo_candidate_uses_original_prompt_and_adapter_callbacks(
    monkeypatch, fake_completion, tmp_path
):
    from examples.systems.amazon import local_models_for_next_item_selection as amazon
    from caiso.integration.services import LocalModelReference
    from caiso.optimizers.base import OptimizationContext
    from caiso.optimizers import weight_optimizer

    loaded = []
    monkeypatch.setattr(amazon, "initialize_adapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        amazon, "load_lora_adapter", lambda name, path, *args: loaded.append((name, str(path)))
    )
    counter = iter(range(100))

    def post(prompt, *args, **kwargs):
        text = f"sample {next(counter)}"
        return SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"choices": [{"text": text}]}
        )

    monkeypatch.setattr(amazon, "post_http_request", post)
    system = amazon.system_engine()
    budget = BudgetTracker(3)
    registry = ProviderRegistry({"session_analyzer": LocalModelReference("original-qwen")})
    runtime = UpstreamRuntime(system, registry, budget)
    example = adapt_examples(
        system,
        [Example(sequence="Viewed a pen", choices=["Pen", "Cup"], gd_answer="0")],
        "train",
    )[0]
    reference = runtime.execute(runtime.initial_state, example, 42)
    training = []
    adapter = str(tmp_path / "new-adapter")

    def train(model, current, pairs, directory, config, budget, seed):
        training.append((model, current, pairs))
        return adapter

    monkeypatch.setattr(weight_optimizer, "train_dpo", train)
    context = OptimizationContext(
        runtime,
        runtime.initial_state,
        None,
        SimpleNamespace(choose=lambda *args: "A"),
        [example],
        [reference],
        42,
        {"dpo": {"max_steps": 1}},
        tmp_path,
    )
    component = runtime.library["session_analyzer"]
    proposals = weight_optimizer.WeightOptimizer().propose(component, context)
    assert len(proposals) == 2
    assert training[0][0] == "original-qwen"
    assert training[0][2][0]["prompt"] == runtime.training_prompt(
        "session_analyzer", {"sequence": "Viewed a pen"}, proposals[0]
    )
    assert isinstance(training[0][2][0]["chosen"], str)
    assert training[0][2][0]["chosen"] != training[0][2][0]["rejected"]
    runtime.execute(
        runtime.initial_state.with_configs({"session_analyzer": proposals[1]}), example, 42
    )
    assert ("SessionAnalyzerModule", adapter) in loaded
    assert str(system.components["session_analyzer"].variable) == proposals[0].value["adapter"]
