"""Task evidence, shared training observations and comparable staged selection."""

from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from caiso.arch.base import BaseComponent
from caiso.arch.system import CompoundAISystem
from caiso.core.budget import BudgetExceeded, BudgetTracker
from caiso.core.content_filter import ContentFilteredError
from caiso.core.recovery import ModelResponseError
from caiso.integration.config import load_experiment, load_method
from caiso.integration.runtime import UpstreamRuntime, adapt_examples
from caiso.method.algorithm import initial_preferences
from caiso.method.block_optimization import validate_candidate
from caiso.method.config import validate_method
from caiso.optim.optimizer import CaisoOptimizer
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
from caiso.optimizers.prompt_optimizer import PromptProposal
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example
from tests.test_candidate_configuration_toggle import Proposals
from tests.test_runtime import make_runtime
from tests.offline import OfflineEmbeddings, OfflineProvider, training_preferences


CONSTITUTION = SimpleNamespace(principles=["Select the trajectory that follows the requested operation."])


def test_bootstrap_abstains_on_identical_outputs_in_both_attempt_modes(tmp_path):
    for attempts in (-1, 2):
        runtime = make_runtime()
        train = adapt_examples(runtime.engine, [Example(question="2", gd_answer="4")], "train")
        path = tmp_path / f"comparisons-{attempts}.jsonl"
        with pytest.raises(ModelResponseError, match="No usable unequal-score"):
            initial_preferences(runtime, runtime.initial_state, train, 1, attempts, 42, path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == (1 if attempts == -1 else 2)
        assert all(row["preferred"] is None and row["label_source"] is None for row in rows)
        assert all(row["abstention_reason"] == "identical_trajectories" for row in rows)


def test_bootstrap_skips_ties_and_small_margins_before_using_task_preference(monkeypatch, tmp_path):
    runtime = make_runtime()
    train = adapt_examples(runtime.engine, [Example(question="2", gd_answer="4")], "train")
    base = runtime.execute(runtime.initial_state, train[0], 42)
    scores = iter([0, 0, 0.6, 0.605, 0.2, 0.8])
    calls = []

    def execute(state, example, seed):
        result = base.model_copy(deep=True)
        result.reward, result.seed = next(scores), seed
        result.final_output = {"answer": f"distinct answer {seed}"}
        calls.append(seed)
        return result

    monkeypatch.setattr(runtime, "execute", execute)
    path = tmp_path / "comparisons.jsonl"
    pairs = initial_preferences(runtime, runtime.initial_state, train, 1, 3, 42, path, min_margin=0.01)
    assert len(calls) == 6
    assert len(pairs) == 1 and pairs[0].preferred == "B"
    assert pairs[0].metadata["label_source"] == "training_task_metric"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["abstention_reason"] for row in rows] == ["task_score_tie", "insufficient_margin", None]


def test_all_tied_bootstrap_stops_before_constitution_induction(tmp_path):
    runtime = make_runtime()
    dicai = SimpleNamespace(induce=lambda *args, **kwargs: pytest.fail("No preference signal to induce"))
    optimizer = CaisoOptimizer(
        runtime.engine, [Example(question="2", gd_answer="4")],
        [Example(question="3", gd_answer="6")],
        {"dataset": "pubmed", "caiso": {"method": {"initial_preference_attempts": 1}}},
        tmp_path, providers=ProviderRegistry({}), dicai=dicai,
    )
    with pytest.raises(RuntimeError, match="initialization failed"):
        optimizer.optimize()
    assert json.loads((tmp_path / "run.json").read_text())["metrics"]["stop_reason"] == "initialization_failed"
    assert not (tmp_path / "checkpoint/checkpoint.json").exists()


def test_diagnostics_reuse_filtered_attribution_observation_without_resampling(monkeypatch, tmp_path):
    provider, embeddings = OfflineProvider(None), OfflineEmbeddings(None)
    registry = ProviderRegistry({"offline": provider}, dict.fromkeys(
        ["system_constitution", "constitution_judge", "block_selection", "prompt_optimizer"], "offline",
    ))
    optimizer = CaisoOptimizer(
        make_runtime().engine,
        [Example(question="1", gd_answer="2"), Example(question="3", gd_answer="6")],
        [Example(question="5", gd_answer="5")],
        {"dataset": "pubmed", "num_iterations": 1, "val_size": 1, "caiso": {
            "edge_baselines": {
                "source->transform": {"number": {"value": "0"}},
                "transform->sink": {"result": {"value": "0"}},
            }, "method": {
            "attribution": {"batch_size": 2}, "block_selection": {"diagnostic_batch_size": 2},
            "dicai": {"clustering": {"min_clusters": 2, "max_clusters": 4}},
            "reporting": {"cli": False, "plots": False},
        }}}, tmp_path, providers=registry, embeddings=embeddings,
    )
    provider.budget = embeddings.budget = optimizer.budget
    preferences = training_preferences(optimizer)
    original, filtered = optimizer.runtime.execute, []

    def execute(state, example, seed, **kwargs):
        if example.example_id == "train-1":
            filtered.append((example.example_id, seed))
            raise ContentFilteredError("policy")
        return original(state, example, seed, **kwargs)

    def stop_before_search(*args, **kwargs):
        raise BudgetExceeded("Stop after scheduling")

    monkeypatch.setattr(optimizer.runtime, "execute", execute)
    monkeypatch.setattr(JointBlockOptimizer, "propose", stop_before_search)
    optimizer.optimize(preferences=preferences)
    assert filtered == [("train-1", 42)]
    votes = json.loads((tmp_path / "iterations/0001/03_selection/block_selection.json").read_text())["votes"]
    assert len(votes) == 1 and votes[0]["trajectory"]["input_id"] == "train-0"


def counted_runtime():
    runtime, calls = make_runtime(), Counter()
    for cid, component in runtime.engine.components.items():
        original = component.forward

        def forward(original=original, cid=cid, **inputs):
            calls[cid] += 1
            return original(**inputs)

        component.forward = forward
    return runtime, calls


def test_training_replay_invalidates_changed_inputs_and_fresh_validation_stays_fresh():
    runtime, calls = counted_runtime()
    state = runtime.initial_state
    candidate = state.model_copy(deep=True)
    candidate.configurations["transform"].value = "double"
    train = adapt_examples(runtime.engine, [Example(question="3", gd_answer="6")], "train")[0]
    with runtime.reuse_component_executions():
        a = runtime.execute(state, train, 42)
        b = runtime.execute(candidate, train, 42)
        again = runtime.execute(state, train, 42)
        assert [a.reward, b.reward, again.reward] == [0, 1, 0]
        assert calls == {"source": 1, "transform": 2, "sink": 2}
        assert b.metadata["reused_components"] == ["source"]
        assert again.metadata["reused_components"] == ["source", "transform", "sink"]
        assert runtime.engine.components["transform"].traj["variable"] is None
        assert runtime.budget.usage.component_executions == 5
        assert runtime.budget.usage.reused_component_executions == 4
        # A different repeat ID denotes fresh component draws, even in a search.
        runtime.execute(candidate, train, 43)
        assert calls == {"source": 2, "transform": 3, "sink": 3}
    validation = adapt_examples(runtime.engine, [Example(question="3", gd_answer="6")], "validation")
    before = runtime.budget.usage.component_executions
    observed = []
    _, gate = validate_candidate(runtime, state, candidate, validation, [42, 43], on_evaluation=lambda *args: observed.extend(args[2]))
    assert gate.accepted
    assert runtime.budget.usage.component_executions - before == 12
    assert all("reused_components" not in trajectory.metadata for trajectory in observed)
    with runtime.reuse_component_executions(), pytest.raises(ValueError, match="restricted to training"):
        runtime.execute(state, validation[0], 42)


def test_replay_is_example_and_generation_specific_and_scope_clears_on_failure(monkeypatch):
    runtime, calls = counted_runtime()
    state = runtime.initial_state
    train = adapt_examples(runtime.engine, [Example(question="3", gd_answer="6")] * 2, "train")
    original_call = runtime.engine._single_call
    with pytest.raises(RuntimeError, match="boom"):
        with runtime.reuse_component_executions():
            runtime.execute(state, train[0], 42)
            runtime.execute(state, train[1], 42)
            assert calls == {"source": 2, "transform": 2, "sink": 2}
            runtime.engine.components["transform"].config.temperature = 0.8
            runtime.execute(state, train[0], 42)
            assert calls == {"source": 2, "transform": 3, "sink": 2}
            monkeypatch.setattr(runtime.engine.components["transform"], "forward", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
            runtime.engine.components["transform"].config.temperature = 0.9
            runtime.execute(state, train[0], 42)
    assert runtime._component_cache is None
    assert runtime.engine._single_call == original_call


def test_replay_does_not_resample_a_cached_policy_rejection(monkeypatch):
    runtime = make_runtime()
    train = adapt_examples(runtime.engine, [Example(question="3", gd_answer="6")], "train")[0]
    calls = []

    def filtered(**inputs):
        calls.append(inputs)
        raise ContentFilteredError("policy")

    monkeypatch.setattr(runtime.engine.components["transform"], "forward", filtered)
    with runtime.reuse_component_executions():
        for _ in range(2):
            with pytest.raises(ContentFilteredError):
                runtime.execute(runtime.initial_state, train, 42)
    assert len(calls) == 1


class PromptNode(BaseComponent):
    def __init__(self, source, target):
        super().__init__("Transform", [source], [target], variable="base", num_retry=1, retry_after=0)

    def forward(self, **inputs):
        return {self.output_fields[0]: next(iter(inputs.values())) + "|" + self.variable}


def pair_runtime():
    def reward(answer, gd_answer):
        left, right = answer.split("|")[-2:]
        return 1.0 if left == right == "synergy" else 0.2 if left == right == "base" else 0.1

    engine = CompoundAISystem(
        {"left": PromptNode("question", "text"), "right": PromptNode("text", "answer")},
        final_output_fields=["answer"], ground_fields=["gd_answer"], eval_func=reward,
    )
    return UpstreamRuntime(engine, ProviderRegistry({}), BudgetTracker(2))


@pytest.mark.parametrize("reuse", [False, True])
def test_staging_screens_complete_cooperations_and_preserves_common_full_batch(tmp_path, reuse):
    runtime = pair_runtime()
    train = adapt_examples(runtime.engine, [Example(question=f"Q{i}", gd_answer="unused") for i in range(20)], "train")
    optimizer = JointBlockOptimizer(runtime, None, {
        "reuse_component_executions": reuse,
        "screening": {"enabled": True, "batch_size": 5, "finalists": 3},
    }, tmp_path, optimizers={runtime.library["left"].config_type: Proposals(["one", "two", "synergy"])})
    scored = optimizer.propose(runtime.initial_state, ["left", "right"], CONSTITUTION, train, 42)
    assert len(optimizer.last_search_report["screened_candidates"]) == 16
    assert len(scored) == 4
    assert scored[0].score == scored[0].task_score == 1
    assert all(configuration.value == "synergy" for configuration in scored[0].configurations.values())
    assert all(item.evaluation["examples"] == 20 and item.evaluation["stage"] == "full_training" for item in scored)
    assert len({item.evaluation["batch_fingerprint"] for item in scored}) == 1
    # Individually harmful changes were retained long enough to discover synergy.
    assert runtime.budget.usage.full_system_runs == 140
    if reuse:
        assert runtime.budget.usage.equivalent_full_system_runs < 140
        assert runtime.budget.usage.reused_component_executions > 0
    else:
        assert runtime.budget.usage.equivalent_full_system_runs == pytest.approx(140)


def test_filtered_screen_observations_stay_zero_and_are_not_retried_when_promoted(monkeypatch, tmp_path):
    runtime = pair_runtime()
    train = adapt_examples(runtime.engine, [Example(question=f"Q{i}", gd_answer="unused") for i in range(20)], "train")
    execute, filtered = runtime.execute, []

    def policy(state, example, seed, **kwargs):
        if all(config.value == "synergy" for config in state.configurations.values()) and example.example_id == "train-0":
            filtered.append(example.example_id)
            raise ContentFilteredError("policy")
        return execute(state, example, seed, **kwargs)

    monkeypatch.setattr(runtime, "execute", policy)
    optimizer = JointBlockOptimizer(runtime, None, {
        "screening": {"enabled": True, "batch_size": 5, "finalists": 3},
    }, tmp_path, optimizers={runtime.library["left"].config_type: Proposals(["one", "two", "synergy"])})
    scored = optimizer.propose(runtime.initial_state, ["left", "right"], CONSTITUTION, train, 42)
    assert scored[0].task_score == pytest.approx(19 / 20)
    assert filtered == ["train-0"]


def test_archived_prompts_are_checkpointable_and_rescored_for_new_training_targets(monkeypatch, tmp_path):
    runtime = make_runtime()
    first = adapt_examples(runtime.engine, [Example(question="2", gd_answer="4")], "train")
    replies, messages = iter(["double", "other", "new-one", "new-two"]), []

    def propose(role, content, **kwargs):
        messages.append(deepcopy(content))
        return PromptProposal(prompt=next(replies))

    monkeypatch.setattr(runtime.providers, "complete", propose)
    optimizer = JointBlockOptimizer(runtime, None, {"prompt_candidates": 2}, tmp_path)
    optimizer.propose(runtime.initial_state, ["transform"], CONSTITUTION, first, 42)
    saved = optimizer.state_dict()
    restored = JointBlockOptimizer(runtime, None, {"prompt_candidates": 2}, tmp_path)
    restored.load_state_dict(saved)
    second = adapt_examples(runtime.engine, [Example(question="2", gd_answer="2")], "train")
    restored.propose(runtime.initial_state, ["transform"], CONSTITUTION, second, 43)
    history = json.JSONDecoder().raw_decode(messages[2][-1]["content"].split("PREVIOUS INSTRUCTIONS AND SCORES:\n", 1)[1])[0]
    assert next(row for row in history if row["prompt"] == "double")["score"] == 0  # Previously 1.
    assert next(row for row in history if row["prompt"] == "identity")["score"] == 1
    feedback = json.JSONDecoder().raw_decode(messages[2][-1]["content"].split("EXAMPLE INPUTS AND OUTPUTS:\n", 1)[1])[0]
    assert feedback[0]["target"] == {"gd_answer": "2"}
    assert feedback[0]["final_output"] == {"answer": "2"}
    assert feedback[0]["task_score"] == 1
    assert feedback[0]["downstream_executions"] == [{"component_id": "sink", "component_output": {"answer": "2"}}]
    assert saved == {"prompt_archive": {"transform": ["double", "other"]}}
    assert restored.state_dict()["prompt_archive"]["transform"] == ["double", "other", "new-one", "new-two"]


def test_reserved_budget_survives_exhausted_search():
    budget = BudgetTracker(1, max_equivalent_full_runs=10)
    with budget.reserve(4):
        for _ in range(6):
            budget.component(1)
        with pytest.raises(BudgetExceeded):
            budget.component(1)
    for _ in range(4):
        budget.component(1)
    assert budget.usage.equivalent_full_system_runs == 10
    with pytest.raises(BudgetExceeded):
        with budget.reserve(1):
            pytest.fail("Reservation should fail before search starts")


@pytest.mark.parametrize("setting,value", [
    ("validation_repeats", 0), ("validation_repeats", None), ("validation_repeats", True),
    ("prompt_history_limit", -1), ("prompt_history_limit", None),
    ("feedback_examples", 0), ("feedback_max_chars", "2000"),
    ("prompt_archive_size", 0), ("reuse_component_executions", "true"),
    ("sample_discrete", None), ("screening", []),
    ("screening", {"enabled": None}), ("screening", {"batch_size": 0}),
    ("screening", {"finalists": -1}), ("screening", {"finalists": None}),
])
def test_search_settings_fail_before_paid_work(setting, value):
    with pytest.raises(ValueError):
        validate_method({"block_optimization": {setting: value}})


def test_active_hotpot_profile_preserves_pool_and_repeats_acceptance():
    method = load_method(load_experiment("configs/experiments/hotpotqa.yaml"))
    assert method["validation_sampling"] == "seeded"
    assert method["validation_batch_size"] == 50
    assert method["diagnostics_split"] == "train"
    assert method["block_optimization"]["validation_repeats"] == 2
    assert method["block_optimization"]["sample_discrete"] is False
