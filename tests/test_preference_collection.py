"""One unchanged/changed run per sample, random components and random metric ties."""

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


def collect(runtime, examples, temperature=0.9, count=1, attempts=1, seed=42, **kwargs):
    return initial_preferences(runtime, runtime.initial_state, examples, count, attempts, seed,
                               collection={"mode": "component", "temperature": temperature}, **kwargs)


@pytest.mark.parametrize("mode", ["temperature", "hyperparameter"])
def test_exactly_two_configuration_runs_share_only_the_upstream_prefix(mode, tmp_path):
    runtime, examples = fixture(mode)
    state = runtime.initial_state.model_copy(deep=True)
    components = runtime.engine.components
    original_call = runtime.engine._single_call
    path = tmp_path / "comparisons.jsonl"
    pair, = collect(runtime, examples, comparison_path=path)
    assert pair.preferred == "B"
    assert pair.metadata["global_scores"] == [0, 1]
    assert pair.metadata["candidate_roles"] == {"A": "unchanged", "B": "changed"}
    assert components["source"].calls == 1
    assert components["producer"].calls == components["independent"].calls == 2
    assert sum(components["consumer"].calls.values()) == 2
    assert components["consumer"].temperatures == [0.2, 0.2]
    assert components["producer"].temperatures == ([0, 0.9] if mode == "temperature" else [0, 0])
    assert components["producer"].config.temperature == 0
    assert components["producer"].variable == state.configurations["producer"].value
    assert runtime.initial_state == state
    assert runtime.engine._single_call == original_call
    assert runtime.budget.usage.component_executions == 7
    assert runtime.budget.usage.full_system_runs == 2
    assert runtime.budget.usage.equivalent_full_system_runs == pytest.approx(1.75)
    assert observed_components(pair) == set(components)
    for candidate in (pair.candidate_a, pair.candidate_b):
        assert "continuations" not in candidate
        assert candidate["trajectory"]["metadata"]["preference_replayed_components"] == ["source"]
    assert pair.candidate_a["trajectory"]["executions"][0] == pair.candidate_b["trajectory"]["executions"][0]
    view = json.dumps([task_view(pair.context), *candidate_views(pair)])
    assert all(f'"{key}"' not in view for key in ("reward", "gold", "target", "metadata", "global_scores"))
    assert json.loads(path.read_text())["preference"]["preferred"] == "B"


def test_unchanged_configuration_is_preferred_when_its_metric_is_higher():
    runtime, examples = fixture()
    examples[0].target = {"gd_answer": "1"}
    pair, = collect(runtime, examples)
    assert pair.preferred == "A" and pair.metadata["global_scores"] == [1, 0]


@pytest.mark.parametrize("target,score", [("neither", 0), ("1", 1)])
def test_equal_metrics_and_identical_outputs_are_retained_with_random_winners(target, score):
    results = []
    for _ in range(2):
        runtime, examples = fixture()
        examples[0].target = {"gd_answer": target}
        pairs = collect(runtime, examples, temperature=0.1, count=20, attempts=20)
        assert len(pairs) == 20
        assert all(pair.metadata["task_score_tied"] for pair in pairs)
        assert all(pair.metadata["label_source"] == "random_task_tie" for pair in pairs)
        assert all(pair.metadata["global_scores"] == [score, score] for pair in pairs)
        assert all(pair.candidate_a["trajectory"]["final_output"] == pair.candidate_b["trajectory"]["final_output"] for pair in pairs)
        results.append([pair.preferred for pair in pairs])
    assert set(results[0]) == {"A", "B"}
    assert results[0] == results[1]  # Collection RNG, not model generations, is reproducible.


def test_score_margin_does_not_override_a_strictly_higher_metric(monkeypatch):
    runtime, examples = fixture()
    original = runtime.run_preference_configuration

    def score(*args):
        result = original(*args)
        result.reward = 0.505 if args[-1] == 0.9 else 0.5
        return result

    monkeypatch.setattr(runtime, "run_preference_configuration", score)
    pair, = collect(runtime, examples, min_margin=1)
    assert pair.preferred == "B" and not pair.metadata["task_score_tied"]


def test_component_is_randomly_selected_per_sample_instead_of_cycled():
    class OtherProducer(BaseComponent):
        def __init__(self):
            super().__init__("Other LLM", ["number"], ["extra"], variable="fixed prompt",
                             config={"temperature": 0.0}, num_retry=1, retry_after=0)

        def forward(self, **inputs):
            return {"extra": str(self.config.temperature) + self.variable}

    runtime, _ = fixture()
    runtime.engine.components["other"] = OtherProducer()
    runtime.engine.components["consumer"].input_fields.append("extra")
    runtime.engine.execution_order = ["source", "producer", "other", "independent", "consumer"]
    runtime = UpstreamRuntime(runtime.engine, ProviderRegistry({}), BudgetTracker(5))
    examples = adapt_examples(runtime.engine, [Example(question="1", gd_answer="2")], "train")
    pairs = collect(runtime, examples, count=20, attempts=20)
    selected = [pair.metadata["component_id"] for pair in pairs]
    assert set(selected) == {"producer", "other"}
    assert any(a == b for a, b in zip(selected, selected[1:]))


def test_first_component_needs_no_prefix_and_is_executed_twice():
    component = Producer()
    system = CompoundAISystem({"producer": component}, final_output_fields=["result"],
                              ground_fields=["gd_answer"], eval_func=lambda result, gd_answer: float(result == gd_answer))
    runtime = UpstreamRuntime(system, ProviderRegistry({}), BudgetTracker(1))
    examples = adapt_examples(system, [Example(number="1", gd_answer="2")], "train")
    pair, = collect(runtime, examples)
    assert pair.preferred == "B" and component.calls == 2
    assert runtime.budget.usage.component_executions == 2


def test_filtered_changed_branch_is_not_retried(monkeypatch, tmp_path):
    runtime, examples = fixture()
    calls = []
    original = runtime.run_preference_configuration

    def filtered(*args):
        calls.append(args)
        if args[-1] == 0.9:
            raise ContentFilteredError("blocked changed configuration")
        return original(*args)

    monkeypatch.setattr(runtime, "run_preference_configuration", filtered)
    path = tmp_path / "comparisons.jsonl"
    with pytest.raises(ModelResponseError, match="No unfiltered configuration"):
        collect(runtime, examples, attempts=3, comparison_path=path)
    assert len(calls) == 2
    assert json.loads(path.read_text())["status"] == "skipped_content_filter"


@pytest.mark.parametrize("split", ["validation", "test"])
def test_training_only_before_any_sampling_calls(split):
    runtime, examples = fixture()
    prefix = runtime.preference_prefix(runtime.initial_state, examples[0], "producer", 42)
    example = examples[0].model_copy(update={"split": split})
    with pytest.raises(ValueError, match="training"):
        collect(runtime, [example])
    with pytest.raises(ValueError, match="training"):
        runtime.preference_prefix(runtime.initial_state, example, "producer", 43)
    with pytest.raises(ValueError, match="training"):
        runtime.run_preference_configuration(runtime.initial_state, example, prefix, "producer", 44)
    assert runtime.engine.components["producer"].calls == 0


@pytest.mark.parametrize("error", [RuntimeError("service unavailable"), BudgetExceeded("budget"), ContentFilteredError("filtered")])
def test_branch_failure_restores_generation_settings_variables_and_observer(error):
    runtime, examples = fixture("hyperparameter")
    prefix = runtime.preference_prefix(runtime.initial_state, examples[0], "producer", 42)
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
        runtime.run_preference_configuration(candidate, examples[0], prefix, "producer", 43, 0.9)
    assert producer.variable == {"multiplier": 1}
    assert producer.config.temperature == 0
    assert runtime.engine._single_call == original


def test_replay_rejects_mismatched_inputs_and_incomplete_prefixes():
    runtime, examples = fixture()
    prefix = runtime.preference_prefix(runtime.initial_state, examples[0], "producer", 42)
    invalid = deepcopy(prefix)
    invalid[0].component_input = {"question": "different"}
    with pytest.raises(ValueError, match="mismatch"):
        runtime.run_preference_configuration(runtime.initial_state, examples[0], invalid, "producer", 43)
    with pytest.raises(ValueError, match="exactly the shared"):
        runtime.run_preference_configuration(runtime.initial_state, examples[0], [], "producer", 43)


@pytest.mark.parametrize("settings", [None, {"mode": "unknown"}, {"candidates": 4},
                                     {"downstream_rollouts": 3}, {"pairs_per_context": 1},
                                     {"temperature": float("nan")}, {"temperature": 3}, {"temperature": True}])
def test_invalid_sampling_config_fails_before_execution(settings):
    with pytest.raises(ValueError, match="preference_collection"):
        validate_method({"preference_collection": settings})


def test_new_collector_is_the_resolved_default():
    assert load_method({})["preference_collection"] == {"mode": "component", "temperature": 0.9}


def test_optimizer_collects_configuration_preferences_before_induction(tmp_path):
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
    assert pair["metadata"]["protocol"] == "configuration-pair-v2"
    assert setup["system_constitution"]["source_preference_pair_ids"] == [pair["pair_id"]]
    assert runtime.engine.components["producer"].config.temperature == 0
