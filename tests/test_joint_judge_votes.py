"""Joint constitution scoring repeats judgments without repeating system execution."""

from types import SimpleNamespace

import pytest

from caiso.core.budget import BudgetExceeded
from caiso.core.content_filter import ContentFilteredError
from caiso.integration.runtime import adapt_examples
from caiso.method.config import validate_method
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
from caiso.wrappers.example import Example
from tests.test_runtime import make_runtime


def setup(tmp_path, monkeypatch, choose, *, samples=2, votes=None):
    runtime = make_runtime()
    examples = adapt_examples(runtime.engine, [
        Example(question=str(i), gd_answer=str(2 * i)) for i in range(1, samples + 1)
    ], "train")
    references = [runtime.execute(runtime.initial_state, example, 42) for example in examples]
    config = {"task_scoring": {"enabled": False}}
    if votes is not None:
        config["constitution_judge_votes"] = votes
    optimizer = JointBlockOptimizer(runtime, SimpleNamespace(choose=choose), config, tmp_path)
    candidate = runtime.initial_state.configurations["transform"].model_copy(update={"value": "double"})
    executions = []
    original = runtime.execute_block

    def execute(*args):
        executions.append(args)
        return original(*args)

    monkeypatch.setattr(runtime, "execute_block", execute)
    return optimizer, examples, references, candidate, executions


@pytest.mark.parametrize("votes", [None, 5, 7])
def test_panel_votes_are_averaged_per_sample_with_distinct_seeds(tmp_path, monkeypatch, votes):
    size, calls = votes or 5, []

    def choose(constitution, a, b, seed):
        sample, vote = divmod(len(calls), size)
        calls.append((a, b, seed))
        return "A" if vote < (3 if sample == 0 else 1) else "B"

    optimizer, examples, references, candidate, executions = setup(tmp_path, monkeypatch, choose, votes=votes)
    scored = optimizer.score_candidate(optimizer.runtime.initial_state, {"transform": candidate},
                                        None, examples, references, 42)
    assert len(calls) == 2 * size
    assert len({seed for _, _, seed in calls}) == len(calls)
    assert len(executions) == 4  # One baseline and candidate execution per sample.
    for offset in (0, size):
        assert all(a is calls[offset][0] and b is calls[offset][1] for a, b, _ in calls[offset:offset + size])
    assert scored.constitution_score == pytest.approx(2 / size)
    report = scored.evaluation["constitution_judging"]
    assert report["votes_per_sample"] == size
    assert report["scored_samples"] == 2
    assert report["completed_judge_votes"] == 2 * size
    assert [sample["score"] for sample in report["samples"]] == pytest.approx([3 / size, 1 / size])
    assert [sample["seeds"] for sample in report["samples"]] == [
        list(range(42, 42 + size)), list(range(42 + size, 42 + 2 * size)),
    ]


def test_twenty_samples_make_one_hundred_judge_calls(tmp_path, monkeypatch):
    calls = []

    def choose(*args):
        calls.append(args)
        return "A"

    optimizer, examples, references, candidate, executions = setup(tmp_path, monkeypatch, choose, samples=20)
    scored = optimizer.score_candidate(optimizer.runtime.initial_state, {"transform": candidate},
                                        None, examples, references, 42)
    assert len(calls) == 100
    assert len(executions) == 40
    assert scored.constitution_score == 1


def test_identical_outputs_still_receive_five_votes_and_remain_neutral(tmp_path, monkeypatch):
    calls = []

    def choose(*args):
        calls.append(args)
        return "A"

    optimizer, examples, references, _, executions = setup(tmp_path, monkeypatch, choose)
    current = optimizer.runtime.initial_state
    scored = optimizer.score_candidate(current, {"transform": current.configurations["transform"]},
                                        None, examples, references, 42)
    assert len(calls) == 10 and len(executions) == 4
    assert scored.constitution_score == 0.5
    assert all(sample["identical_outputs"] for sample in scored.evaluation["constitution_judging"]["samples"])


def test_policy_rejection_skips_partial_panel_without_resampling(tmp_path, monkeypatch):
    calls = []

    def choose(*args):
        calls.append(args)
        if len(calls) == 3:
            raise ContentFilteredError("policy rejection")
        return "A" if len(calls) < 3 else "B"

    optimizer, examples, references, candidate, _ = setup(tmp_path, monkeypatch, choose)
    scored = optimizer.score_candidate(optimizer.runtime.initial_state, {"transform": candidate},
                                        None, examples, references, 42)
    assert len(calls) == 8  # Three attempts for the filtered sample, five for the next.
    assert scored.constitution_score == 0
    report = scored.evaluation["constitution_judging"]
    assert report["scored_samples"] == 1
    assert report["samples"][0]["choices"] == ["A", "A"]
    assert report["samples"][0]["score"] is None
    assert report["samples"][0]["status"] == "skipped_content_filter"
    assert report["samples"][1]["status"] == "completed"


def test_budget_failure_does_not_return_a_score_from_partial_votes(tmp_path, monkeypatch):
    calls = []

    def choose(*args):
        calls.append(args)
        if len(calls) == 3:
            raise BudgetExceeded("judge budget")
        return "A"

    optimizer, examples, references, candidate, _ = setup(tmp_path, monkeypatch, choose)
    with pytest.raises(BudgetExceeded, match="judge budget"):
        optimizer.score_candidate(optimizer.runtime.initial_state, {"transform": candidate},
                                  None, examples, references, 42)
    assert len(calls) == 3
    assert optimizer.runtime.initial_state.configurations["transform"].value == "identity"


@pytest.mark.parametrize("invalid", [None, True, False, -1, 0, 1, 4, 5.0, 6.5, "5"])
def test_vote_count_cannot_be_less_than_five_or_noninteger(tmp_path, invalid):
    config = {"constitution_judge_votes": invalid}
    with pytest.raises(ValueError, match="constitution_judge_votes.*at least 5"):
        validate_method({"block_optimization": config})
    with pytest.raises(ValueError, match="constitution_judge_votes.*at least 5"):
        JointBlockOptimizer(None, None, config, tmp_path)
