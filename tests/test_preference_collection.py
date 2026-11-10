"""Real upstream execution: fixed producer draws, fresh descendants, objective labels."""

from copy import deepcopy
import json

import pytest

from caiso.arch.base import BaseComponent
from caiso.arch.system import CompoundAISystem
from caiso.constitution.dicai.preferences import candidate_views, task_view
from caiso.constitution.dicai.coverage import observed_components
from caiso.core.budget import BudgetExceeded, BudgetTracker
from caiso.core.content_filter import ContentFilteredError
from caiso.core.recovery import ModelResponseError
from caiso.integration.config import load_method
from caiso.integration.runtime import UpstreamRuntime, adapt_examples
from caiso.method.algorithm import initial_preferences
from caiso.method.config import validate_method
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example


class Producer(BaseComponent):
    def __init__(self, mode="temperature"):
        self.temperatures, self.calls, self.mode = [], 0, mode
        super().__init__(
            "Transform the input", ["number"], ["result"],
            variable={"multiplier": 1} if mode == "hyperparameter" else "unchanged prompt",
            variable_search_space={"multiplier": [1, 2]} if mode == "hyperparameter" else None,
            config={"temperature": 0.0}, num_retry=1, retry_after=0,
        )

    def forward(self, **inputs):
        self.calls += 1
        self.temperatures.append(self.config.temperature)
        multiplier = self.variable["multiplier"] if self.mode == "hyperparameter" else (2 if self.config.temperature > 0.5 else 1)
        return {"result": str(int(inputs["number"]) * multiplier)}


class Fixed(BaseComponent):
    def __init__(self, name, input_field, output_field):
        self.calls = 0
        super().__init__(name, [input_field], [output_field], num_retry=1, retry_after=0)

    def forward(self, **inputs):
        self.calls += 1
        return {self.output_fields[0]: next(iter(inputs.values()))}


class Consumer(BaseComponent):
    def __init__(self, noisy=False):
        self.noisy, self.calls, self.temperatures = noisy, {}, []
        super().__init__("Answer from result", ["result", "unrelated"], ["answer"],
                         config={"temperature": 0.2}, num_retry=1, retry_after=0)

    def forward(self, **inputs):
        result = inputs["result"]
        count = self.calls.get(result, 0)
        self.calls[result] = count + 1
        self.temperatures.append(self.config.temperature)
        if self.noisy:
            outcomes = {"1": [False, True, False, False], "2": [False, True, True]}
            return {"answer": "2" if outcomes[result][count] else "wrong"}
        return {"answer": result}


def fixture(mode="temperature", noisy=False):
    components = {
        "source": Fixed("Input", "question", "number"),
        "producer": Producer(mode),
        # This node follows the producer in execution order but is not its descendant.
        "independent": Fixed("Independent input", "question", "unrelated"),
        "consumer": Consumer(noisy),
    }
    system = CompoundAISystem(components, final_output_fields=["answer"], ground_fields=["gd_answer"],
                              eval_func=lambda answer, gd_answer: float(answer == gd_answer))
    system.execution_order = list(components)  # A valid order with a later non-descendant.
    runtime = UpstreamRuntime(system, ProviderRegistry({}), BudgetTracker(4))
    examples = adapt_examples(system, [Example(question="1", gd_answer="2")], "train")
    return runtime, examples


def collect(runtime, examples, **kwargs):
    return initial_preferences(runtime, runtime.initial_state, examples, 1, 1, 42,
                               collection={"mode": "component", "candidates": 3,
                                           "downstream_rollouts": 3, "temperature": 0.9}, **kwargs)


@pytest.mark.parametrize("mode", ["temperature", "hyperparameter"])
def test_frozen_outputs_shared_non_descendants_normal_downstream_and_real_accounting(mode, tmp_path):
    runtime, examples = fixture(mode)
    state = runtime.initial_state.model_copy(deep=True)
    components = runtime.engine.components
    original_call = runtime.engine._single_call
    path = tmp_path / "comparisons.jsonl"
    pair, = collect(runtime, examples, comparison_path=path)
    assert observed_components(pair) == set(components)
    assert pair.metadata["mean_scores"] == ([0, 1] if pair.preferred == "B" else [1, 0])
    assert components["source"].calls == components["independent"].calls == 1
    assert components["producer"].calls == 3
    assert sum(components["consumer"].calls.values()) == 7  # baseline + two frozen outputs × three runs
    assert components["consumer"].temperatures == [0.2] * 7
    assert components["producer"].temperatures == [0, 0.9, 0.9]
    assert components["producer"].config.temperature == 0
    assert components["producer"].variable == state.configurations["producer"].value
    assert runtime.initial_state == state
    assert runtime.engine._single_call == original_call
    assert runtime.budget.usage.component_executions == 12
    assert runtime.budget.usage.equivalent_full_system_runs == pytest.approx(3)
    for candidate in (pair.candidate_a, pair.candidate_b):
        output = candidate["trajectory"]["executions"][-1]["component_output"]
        assert len(candidate["continuations"]) == 3
        for continuation in candidate["continuations"]:
            assert continuation["executions"][1]["component_output"] == output
            assert continuation["metadata"]["preference_replayed_components"] == ["source", "independent", "producer"]
    view = json.dumps([task_view(pair.context), *candidate_views(pair)])
    assert all(f'"{key}"' not in view for key in ("reward", "gold", "target", "metadata", "mean_scores", "rollout_scores"))
    log = json.loads(path.read_text())
    assert log["status"] == "labeled" and len(log["candidates"]) == 2
    assert all(len(row["scores"]) == 3 for row in log["candidates"])


def test_labels_use_mean_continuation_scores_instead_of_first_observed_answers():
    runtime, examples = fixture(noisy=True)
    pair, = collect(runtime, examples)
    chosen = pair.candidate_a if pair.preferred == "A" else pair.candidate_b
    rejected = pair.candidate_b if pair.preferred == "A" else pair.candidate_a
    assert chosen["continuations"][0]["reward"] == 0
    assert rejected["continuations"][0]["reward"] == 1
    assert sorted(pair.metadata["mean_scores"]) == [1 / 3, 2 / 3]
    assert chosen["trajectory"]["executions"][-1]["component_output"] == {"result": "2"}


def test_original_candidate_can_be_chosen_regardless_of_array_index():
    runtime, examples = fixture()
    examples[0].target = {"gd_answer": "1"}
    pair, = collect(runtime, examples)
    chosen = pair.candidate_a if pair.preferred == "A" else pair.candidate_b
    assert chosen["trajectory"]["executions"][-1]["component_output"] == {"result": "1"}


@pytest.mark.parametrize("temperature,expected_reason", [(None, "identical_component_outputs"), (0.9, "equal_or_insufficient_score_margin")])
def test_identical_outputs_and_equal_objective_scores_abstain(temperature, expected_reason, tmp_path):
    runtime, examples = fixture()
    examples[0].target = {"gd_answer": "neither"}
    path = tmp_path / "pools.jsonl"
    with pytest.raises(ModelResponseError, match="No usable unequal-score"):
        initial_preferences(runtime, runtime.initial_state, examples, 1, 1, 42, comparison_path=path,
                            collection={"mode": "component", "candidates": 2, "temperature": temperature,
                                        "downstream_rollouts": 1})
    assert json.loads(path.read_text())["abstention_reason"] == expected_reason


def test_filtered_sampling_context_is_not_resampled(monkeypatch, tmp_path):
    runtime, examples = fixture()
    calls = []

    def filtered(*args):
        calls.append(args)
        raise ContentFilteredError("blocked sample")

    monkeypatch.setattr(runtime, "sample_preference_candidate", filtered)
    with pytest.raises(ModelResponseError, match="No usable unequal-score"):
        initial_preferences(runtime, runtime.initial_state, examples, 1, 3, 42,
                            collection={"mode": "component"}, comparison_path=tmp_path / "pools.jsonl")
    assert len(calls) == 1


def test_filtered_continuation_is_excluded_without_a_replacement(monkeypatch, tmp_path):
    runtime, examples = fixture()
    calls = []
    original = runtime.continue_preference_candidate

    def filtered(*args):
        output = args[3][-1].component_output["result"]
        calls.append(output)
        if output == "2":
            raise ContentFilteredError("blocked continuation")
        return original(*args)

    monkeypatch.setattr(runtime, "continue_preference_candidate", filtered)
    with pytest.raises(ModelResponseError, match="No usable unequal-score"):
        collect(runtime, examples, comparison_path=tmp_path / "pools.jsonl")
    assert calls.count("1") == 3 and calls.count("2") == 1
    log = json.loads((tmp_path / "pools.jsonl").read_text())
    assert log["excluded_candidates"][0]["status"] == "skipped_content_filter"


@pytest.mark.parametrize("split", ["validation", "test"])
def test_training_only_before_any_sampling_calls(split):
    runtime, examples = fixture()
    reference = runtime.execute(runtime.initial_state, examples[0], 42)
    example = examples[0].model_copy(update={"split": split})
    with pytest.raises(ValueError, match="training"):
        collect(runtime, [example])
    with pytest.raises(ValueError, match="training"):
        runtime.sample_preference_candidate(runtime.initial_state, example, reference, "producer", 43, 0.9)
    with pytest.raises(ValueError, match="training"):
        runtime.continue_preference_candidate(runtime.initial_state, example, reference, reference.executions[:2], "producer", 44)
    assert runtime.engine.components["producer"].calls == 1


@pytest.mark.parametrize("error", [RuntimeError("service unavailable"), BudgetExceeded("budget"), ContentFilteredError("filtered")])
def test_sampling_failure_restores_generation_settings_variables_and_observer(error):
    runtime, examples = fixture("hyperparameter")
    reference = runtime.execute(runtime.initial_state, examples[0], 42)
    producer = runtime.engine.components["producer"]
    original = runtime.engine._single_call
    candidate = runtime.initial_state.model_copy(deep=True)
    candidate.configurations["producer"].value = {"multiplier": 2}

    def fail(**inputs):
        assert producer.variable == {"multiplier": 2}
        assert producer.config.temperature == 0.9
        raise error

    producer.forward = fail
    with pytest.raises(type(error), match=str(error)):
        runtime.sample_preference_candidate(candidate, examples[0], reference, "producer", 43, 0.9)
    assert producer.variable == {"multiplier": 1}
    assert producer.config.temperature == 0
    assert runtime.engine._single_call == original


def test_replay_rejects_mismatched_inputs():
    runtime, examples = fixture()
    reference = runtime.execute(runtime.initial_state, examples[0], 42)
    candidate = deepcopy(reference.executions[:2])
    candidate[-1].component_input = {"number": "different"}
    with pytest.raises(ValueError, match="mismatch"):
        runtime.continue_preference_candidate(runtime.initial_state, examples[0], reference, candidate, "producer", 43)


@pytest.mark.parametrize("settings", [None, {"mode": "unknown"}, {"candidates": 1}, {"candidates": True},
                                     {"downstream_rollouts": 0}, {"pairs_per_context": None},
                                     {"temperature": float("nan")}, {"temperature": 3}, {"temperature": True}])
def test_invalid_sampling_config_fails_before_execution(settings):
    with pytest.raises(ValueError, match="preference_collection"):
        validate_method({"preference_collection": settings})


def test_new_collector_is_the_resolved_default():
    assert load_method({})["preference_collection"] == {
        "mode": "component", "candidates": 4, "temperature": 0.9,
        "downstream_rollouts": 3, "pairs_per_context": 1,
    }


def test_optimizer_collects_component_preferences_before_induction(tmp_path):
    from caiso.optim.optimizer import CaisoOptimizer
    from tests.offline import OfflineEmbeddings, OfflineProvider

    runtime, _ = fixture()
    provider, embeddings = OfflineProvider(None), OfflineEmbeddings(None)
    registry = ProviderRegistry({"offline": provider}, dict.fromkeys(
        ["system_constitution", "constitution_judge", "block_selection", "prompt_optimizer"], "offline",
    ))
    optimizer = CaisoOptimizer(
        runtime.engine, [Example(question="1", gd_answer="2")],
        [Example(question="3", gd_answer="6")],
        {"dataset": "pubmed", "caiso": {"method": {
            "max_iterations": 0, "initial_preference_pairs": 1, "initial_preference_attempts": 1,
            "dicai": {"clustering": {"min_clusters": 2, "max_clusters": 4}},
            "reporting": {"cli": False, "plots": False, "capture_model_output": False},
        }}}, tmp_path, providers=registry, embeddings=embeddings,
    )
    provider.budget = embeddings.budget = optimizer.budget
    optimizer.optimize()
    setup = json.loads((tmp_path / "setup.json").read_text())
    pair, = setup["preferences"]
    assert pair["metadata"]["protocol"] == "component-continuations-v1"
    assert setup["system_constitution"]["source_preference_pair_ids"] == [pair["pair_id"]]
    assert runtime.engine.components["producer"].config.temperature == 0
    assert runtime.engine.components["consumer"].temperatures == [0.2] * len(runtime.engine.components["consumer"].temperatures)
