from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from caiso.arch.base import BaseComponent
from caiso.arch.system import CompoundAISystem
from caiso.core.budget import BudgetTracker
from caiso.core.serialization import canonical
from caiso.integration.runtime import UpstreamRuntime, adapt_examples, zero_like
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example


class Source(BaseComponent):
    def __init__(self):
        super().__init__("Preserve input", ["question"], ["number"], num_retry=1, retry_after=0)

    def forward(self, **inputs):
        return {"number": inputs["question"]}


class Transform(BaseComponent):
    def __init__(self):
        super().__init__(
            "Transform number",
            ["number"],
            ["result"],
            variable="identity",
            num_retry=1,
            retry_after=0,
        )

    def forward(self, **inputs):
        return {"result": str(int(inputs["number"]) * (2 if self.variable == "double" else 1))}


class Sink(BaseComponent):
    def __init__(self):
        super().__init__("Return result", ["result"], ["answer"], num_retry=1, retry_after=0)

    def forward(self, **inputs):
        return {"answer": inputs["result"]}


def make_system():
    return CompoundAISystem(
        {"source": Source(), "transform": Transform(), "sink": Sink()},
        final_output_fields=["answer"],
        ground_fields=["gd_answer"],
        eval_func=lambda answer, gd_answer: float(answer == gd_answer),
    )


def test_unused_variable_warning_still_applies_to_other_components(caplog):
    class Unused(BaseComponent):
        def forward(self, **inputs):
            return {"answer": "test"}

    component = Unused("test", ["question"], ["answer"], variable="unused", num_retry=1, retry_after=0)
    component(question="test")
    assert "self.variable is defined but was not accessed" in caplog.text


def make_runtime(system=None):
    return UpstreamRuntime(
        system or make_system(),
        ProviderRegistry({}),
        BudgetTracker(3),
        {
            "source->transform": {"number": {"value": "0"}},
            "transform->sink": {"result": {"value": "0"}},
        },
    )


def test_direct_and_adapted_execution_agree():
    runtime = make_runtime()
    system = runtime.engine
    example = Example(question="3", gd_answer="6")
    direct_score, direct_prediction = system.evaluate(example, return_pred=True)
    adapted = adapt_examples(system, [example], "train")[0]
    trajectory = runtime.execute(runtime.initial_state, adapted, 42)
    assert trajectory.reward == direct_score == 0
    assert trajectory.final_output == {"answer": direct_prediction.answer}
    assert {item.component_id: item.component_input for item in trajectory.executions} == {
        cid: record["input"] for cid, record in direct_prediction.traj.items()
    }
    assert runtime.budget.usage.equivalent_full_system_runs == pytest.approx(1)


def test_candidate_is_executed_and_state_restored_concurrently():
    runtime = make_runtime()
    example = adapt_examples(runtime.engine, [Example(question="3", gd_answer="6")], "train")[0]
    candidate = runtime.initial_state.model_copy(deep=True)
    candidate.configurations["transform"].value = "double"
    states = [runtime.initial_state, candidate] * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda state: runtime.execute(state, example, 42).reward, states))
    assert outputs == [0, 1] * 4
    assert runtime.engine.components["transform"].variable == "identity"
    runtime.apply_state(candidate)
    assert runtime.engine.evaluate(Example(question="3", gd_answer="6")) == 1


def test_intervention_changes_only_the_target_edge():
    runtime = make_runtime()
    example = adapt_examples(runtime.engine, [Example(question="3", gd_answer="3")], "train")[0]
    trajectory = runtime.execute(
        runtime.initial_state, example, 42, intervention="source->transform"
    )
    assert trajectory.executions[0].component_output == {"number": "3"}
    assert trajectory.executions[1].component_input == {"number": "0"}
    assert trajectory.final_output == {"answer": "0"}
    assert runtime.engine(question="3").answer == "3"


def test_multiple_ground_fields_are_retained():
    system = make_system()
    system.ground_fields = ["gd_answer", "weight"]
    system.eval_func = lambda answer, gd_answer, weight: weight if answer == gd_answer else 0
    example = Example(question="3", gd_answer="3", weight=0.75)
    runtime = make_runtime(system)
    adapted = adapt_examples(system, [example], "train")[0]
    assert adapted.target == {"gd_answer": "3", "weight": 0.75}
    assert runtime.execute(runtime.initial_state, adapted, 42).reward == 0.75


def test_failure_restores_variables_and_call_observer():
    runtime = make_runtime()
    original = runtime.engine._single_call
    candidate = runtime.initial_state.model_copy(deep=True)
    candidate.configurations["transform"].value = "double"
    example = adapt_examples(runtime.engine, [Example(question="bad", gd_answer="3")], "train")[0]
    with pytest.raises(ValueError):
        runtime.execute(candidate, example, 42)
    assert runtime.engine._single_call == original
    assert runtime.engine.components["transform"].variable == "identity"


def test_zero_like_and_array_serialization():
    assert zero_like({"a": [1, 2], "b": "text"}) == {"a": [0, 0], "b": " "}
    assert np.array_equal(zero_like(np.ones(3)), np.zeros(3))
    assert canonical({"a": np.array([1, 2]), "b": np.float64(0.5)}) == '{"a": [1, 2], "b": 0.5}'
