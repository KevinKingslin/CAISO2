"""Weight training uses constitution-labeled configuration trajectories in its block."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from caiso.arch.base import BaseComponent
from caiso.arch.system import CompoundAISystem
from caiso.constitution.models import Constitution
from caiso.core.budget import BudgetExceeded, BudgetTracker
from caiso.core.content_filter import ContentFilteredError
from caiso.integration.runtime import UpstreamRuntime, adapt_examples
from caiso.method.config import validate_method
from caiso.optimizers import weight_optimizer
from caiso.optimizers.base import OptimizationContext
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example


def constitution(condition):
    principle = f"Select the trajectory where weight {condition}."
    return Constitution(principles=[principle], source_preference_pair_ids=[],
                        settled_principles=[principle], cluster_ids=[0], abstraction_outputs=[principle])


SYSTEM = constitution("preserves the input")
BLOCK = constitution("doubles the input")


class Outside(BaseComponent):
    def __init__(self):
        self.calls = 0
        super().__init__("Read input", ["question"], ["number"], variable={"multiplier": 1},
                         variable_search_space={"multiplier": [1, 3]}, num_retry=1)

    def forward(self, **inputs):
        self.calls += 1
        return {"number": str(int(inputs["question"]) * self.variable["multiplier"])}


class Weight(BaseComponent):
    variable_applied_on_update = True

    def __init__(self):
        self.calls, self.identical = [], False
        super().__init__("Transform input", ["number"], ["result"], variable=Path("old-adapter"),
                         config={"temperature": 0.0}, num_retry=1)

    def forward(self, **inputs):
        self.calls.append((deepcopy(inputs), self.config.temperature))
        multiplier = 1 if self.identical or self.config.temperature == 0 else 2
        return {"result": str(int(inputs["number"]) * multiplier)}


class Answer(BaseComponent):
    def __init__(self):
        self.calls = 0
        super().__init__("Answer", ["result"], ["answer"], num_retry=1)

    def forward(self, **inputs):
        self.calls += 1
        return {"answer": inputs["result"]}


def setup(tmp_path, *, count=1, attempts=1, members=None):
    components = {"outside": Outside(), "weight": Weight(), "answer": Answer()}
    system = CompoundAISystem(components, final_output_fields=["answer"], ground_fields=["gd_answer"],
                              eval_func=lambda answer, gd_answer: float(answer == gd_answer))
    runtime = UpstreamRuntime(system, ProviderRegistry({"weight": SimpleNamespace(model_id="local")}),
                              BudgetTracker(3))
    examples = adapt_examples(system, [Example(question=str(i), gd_answer=str(2 * i)) for i in range(1, 5)], "train")
    runtime.training_prompt = lambda cid, inputs, config: "Original prompt: " + inputs["number"]
    judged = []

    def choose(constitution, a, b, seed):
        assert constitution is SYSTEM
        assert all(key not in json.dumps([a, b]) for key in ('"reward"', '"target"', '"metadata"'))
        assert {e["component_id"] for e in a["executions"]} == set(components)
        judged.append((a, b))
        return "A"  # Deliberately opposes the task score; the constitution owns these labels.

    context = OptimizationContext(
        runtime, runtime.initial_state, BLOCK, SimpleNamespace(choose=choose), examples[:1], [], 42,
        {"dpo": {"preference_pairs": count, "preference_attempts": attempts}}, tmp_path,
        members=members or ["weight"], system_constitution=SYSTEM, preference_examples=examples,
    )
    return runtime, runtime.library["weight"], context, judged


def test_actual_configuration_branches_share_prefix_and_use_system_constitution(tmp_path):
    runtime, component, context, judged = setup(tmp_path)
    before = runtime.initial_state.model_copy(deep=True)
    records = []
    with runtime.reuse_component_executions():
        search_cache = runtime._component_cache
        pairs = weight_optimizer.generate_dpo_preferences(component, context, records=records)
        assert runtime._component_cache is search_cache
        assert search_cache == {}  # Preference draws neither read nor populate search observations.
    assert len(pairs) == len(records) == len(judged) == 1
    assert records[0].metadata["component_id"] == "weight"
    assert records[0].metadata["label_source"] == "constitution"
    assert records[0].metadata["global_scores"] == [0, 1]
    assert records[0].preferred == "A"
    a, b = judged[0]
    assert pairs[0] == {"prompt": "Original prompt: " + a["executions"][1]["component_input"]["number"],
                        "chosen": a["executions"][1]["component_output"]["result"],
                        "rejected": b["executions"][1]["component_output"]["result"]}
    assert runtime.engine.components["outside"].calls == 1
    assert [call[1] for call in runtime.engine.components["weight"].calls] == [0, 0.9]
    assert runtime.engine.components["answer"].calls == 2
    assert runtime.budget.usage.full_system_runs == 2
    assert runtime.initial_state == before
    assert runtime.engine.components["weight"].config.temperature == 0


def test_dataset_target_uses_training_pool_and_skips_different_component_inputs(tmp_path):
    runtime, component, context, judged = setup(tmp_path, count=4, attempts=20, members=["outside", "weight"])
    records = []
    pairs = weight_optimizer.generate_dpo_preferences(component, context, records=records)
    assert len(pairs) == len(judged) == 4
    assert all(record.metadata["component_id"] == "weight" for record in records)
    assert len({record.metadata["example_id"] for record in records}) > len(context.examples)
    # Changing the upstream block member changes the target's prompt, so those
    # comparisons execute but cannot become chosen/rejected completions for it.
    assert runtime.budget.usage.full_system_runs > 2 * len(pairs)
    assert runtime.engine.components["outside"].variable == {"multiplier": 1}


def test_joint_search_passes_block_system_constitution_and_full_preference_pool(monkeypatch, tmp_path):
    runtime, _, context, judged = setup(tmp_path, count=2, attempts=2)
    trained = []

    def train(model, current, pairs, directory, config, budget, seed):
        trained.append(pairs)
        return str(directory / "adapter")

    monkeypatch.setattr(weight_optimizer, "train_dpo", train)
    optimizer = JointBlockOptimizer(runtime, context.judge,
                                    {**context.config, "reuse_component_executions": True}, tmp_path)
    candidates = optimizer.propose(context.state, ["weight"], BLOCK, context.examples, 42,
                                   system_constitution=SYSTEM, preference_examples=context.preference_examples)
    assert len(trained[0]) == len(judged) == 2
    assert len(candidates) == 2
    evidence, = list(tmp_path.glob("dpo/weight/*/trajectory_preferences.json"))
    records = json.loads(evidence.read_text())
    assert len(records) == 2
    report = json.loads(evidence.with_name("preference_collection.json").read_text())
    assert report["system_constitution"]["principles"] == SYSTEM.principles
    assert report["members"] == ["weight"]
    assert report["retained_pairs"] == report["requested_pairs"] == 2
    assert runtime._component_cache is None
    assert str(runtime.engine.components["weight"].variable) == "old-adapter"


def test_identical_completions_do_not_train_or_invent_a_preference(monkeypatch, tmp_path):
    runtime, component, context, judged = setup(tmp_path, attempts=2)
    runtime.engine.components["weight"].identical = True
    monkeypatch.setattr(weight_optimizer, "train_dpo", lambda *args: pytest.fail("No valid DPO data"))
    assert weight_optimizer.WeightOptimizer().propose(component, context) == [context.state.configurations["weight"]]
    assert judged == []


def test_filtered_comparisons_are_not_resampled_for_the_same_input(tmp_path, monkeypatch):
    runtime, component, context, judged = setup(tmp_path, attempts=10)
    context.preference_examples = context.examples
    original = runtime.run_preference_configuration
    calls = []

    def filtered(*args):
        calls.append(args)
        if args[-1] == 0.9:
            raise ContentFilteredError("blocked")
        return original(*args)

    monkeypatch.setattr(runtime, "run_preference_configuration", filtered)
    assert weight_optimizer.generate_dpo_preferences(component, context) == []
    assert len(calls) == 2 and judged == []


def test_failure_restores_search_cache_and_generation_settings(tmp_path):
    runtime, component, context, _ = setup(tmp_path)

    def fail(*args):
        raise BudgetExceeded("stop")

    context.judge.choose = fail
    with runtime.reuse_component_executions():
        cache = runtime._component_cache
        with pytest.raises(BudgetExceeded):
            weight_optimizer.generate_dpo_preferences(component, context)
        assert runtime._component_cache is cache
    assert runtime.engine.components["weight"].config.temperature == 0


@pytest.mark.parametrize("split", ["validation", "test"])
def test_weight_collection_is_training_only(tmp_path, split):
    runtime, component, context, _ = setup(tmp_path)
    context.preference_examples[0].split = split
    with pytest.raises(ValueError, match="training data"):
        weight_optimizer.generate_dpo_preferences(component, context)
    assert runtime.budget.usage.component_executions == 0


@pytest.mark.parametrize("settings", [
    {"preference_pairs": 0}, {"preference_pairs": None}, {"preference_pairs": True},
    {"preference_pairs": 1.5}, {"preference_attempts": 0}, {"preference_attempts": -2},
    {"preference_temperature": float("nan")}, {"preference_temperature": 3},
])
def test_invalid_dataset_controls_fail_before_execution(settings):
    with pytest.raises(ValueError, match="block_optimization.dpo.preference_"):
        validate_method({"block_optimization": {"dpo": settings}})
