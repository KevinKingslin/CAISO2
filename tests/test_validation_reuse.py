"""Reusing an incumbent score must still execute every proposed candidate."""

import pytest

from caiso.integration.runtime import adapt_examples
from caiso.method.block_optimization import validate_candidate
from tests.test_runtime import make_runtime
from caiso.wrappers.example import Example


def setup():
    runtime = make_runtime()
    initial = runtime.initial_state
    candidate = initial.model_copy(deep=True)
    candidate.configurations["transform"].value = "double"
    validation = adapt_examples(runtime.engine, [Example(question="3", gd_answer="6")], "validation")
    return runtime, initial, candidate, validation


def test_default_still_executes_current_and_candidate():
    runtime, current, candidate, validation = setup()
    observed, reused = [], []
    updated, gate = validate_candidate(
        runtime, current, candidate, validation, [42],
        on_evaluation=lambda label, score, traces: observed.append((label, score, len(traces))),
        on_current_reuse=reused.append,
    )
    assert updated == candidate
    assert gate.accepted is True
    assert observed == [("current", 0.0, 1), ("candidate", 1.0, 1)]
    assert reused == []
    assert runtime.budget.usage.full_system_runs == 2


def test_reuse_accepts_against_zero_and_keeps_the_accepted_score_after_rejection():
    runtime, current, candidate, validation = setup()
    observed, reused = [], []
    options = {
        "reevaluate_current": False,
        "on_evaluation": lambda label, score, traces: observed.append((label, score, len(traces))),
        "on_current_reuse": reused.append,
    }
    updated, accepted = validate_candidate(runtime, current, candidate, validation, [42], current_score=0.0, **options)
    assert accepted.accepted is True
    assert updated == candidate
    assert runtime.budget.usage.full_system_runs == 1
    final, rejected = validate_candidate(
        runtime, updated, current, validation, [42], current_score=accepted.candidate_score, **options
    )
    assert rejected.accepted is False
    assert final == updated
    assert rejected.current_score == accepted.candidate_score == 1.0
    assert runtime.budget.usage.full_system_runs == 2
    assert observed == [("candidate", 1.0, 1), ("candidate", 0.0, 1)]
    assert reused == [0.0, 1.0]


@pytest.mark.parametrize("saved", [None, True, "0.5", float("nan"), float("inf")])
def test_invalid_saved_score_fails_before_executing_a_candidate(saved):
    runtime, current, candidate, validation = setup()
    with pytest.raises(ValueError, match="finite saved current_score"):
        validate_candidate(runtime, current, candidate, validation, [42], reevaluate_current=False, current_score=saved)
    assert runtime.budget.usage.full_system_runs == 0


def test_a_failed_candidate_never_reuses_its_previous_score(monkeypatch):
    runtime, current, candidate, validation = setup()
    def fail(*args, **kwargs):
        raise RuntimeError("candidate execution failed")
    monkeypatch.setattr(runtime, "execute", fail)
    with pytest.raises(RuntimeError, match="candidate execution failed"):
        validate_candidate(runtime, current, candidate, validation, [42], reevaluate_current=False, current_score=0.5)
    assert current.configurations["transform"].value == "identity"
