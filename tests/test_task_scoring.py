"""Task feedback and joint selection must use the same training task metric."""

from collections import Counter
import json
from types import SimpleNamespace

import pytest

from caiso.integration.runtime import adapt_examples
from caiso.method.config import validate_method
from caiso.optimizers import joint_block_optimizer as joint_module
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
from caiso.optimizers.prompt_optimizer import PromptProposal
from caiso.wrappers.example import Example
from tests.test_runtime import make_runtime


def unexpected_judge(*args):
    pytest.fail("Task-only candidate scoring must not call the constitution judge")


def training(runtime):
    return adapt_examples(runtime.engine, [Example(question="2", gd_answer="4")], "train")


@pytest.mark.parametrize("use_history", [True, False])
def test_task_only_feedback_ranking_and_joint_scores_skip_local_judging(monkeypatch, tmp_path, use_history):
    runtime = make_runtime()
    examples = training(runtime)
    replies = iter(["double", "other"])
    messages, evaluations = [], Counter()
    evaluate = joint_module.evaluate

    def propose(role, content, **kwargs):
        messages.append(content)
        return PromptProposal(prompt=next(replies))

    def observe(runtime, state, *args, **kwargs):
        evaluations[state.configurations["transform"].value] += 1
        return evaluate(runtime, state, *args, **kwargs)

    monkeypatch.setattr(runtime.providers, "complete", propose)
    monkeypatch.setattr(runtime, "execute_block", unexpected_judge)
    monkeypatch.setattr(joint_module, "evaluate", observe)
    optimizer = JointBlockOptimizer(
        runtime, SimpleNamespace(choose=unexpected_judge),
        {"prompt_candidates": 2, "max_joint_candidates": 16, "use_history": use_history}, tmp_path,
    )
    scored = optimizer.propose(
        runtime.initial_state, ["source", "transform", "sink"],
        SimpleNamespace(principles=["Select the trajectory that preserves its input."]), examples, 42,
    )
    assert len(scored) == 3
    assert scored[0].configurations["transform"].value == "double"
    assert scored[0].score == scored[0].task_score == 1.0
    assert all(candidate.constitution_score is None for candidate in scored)
    assert all(candidate.score == candidate.task_score for candidate in scored)
    assert all(set(candidate.configurations) == {"source", "transform", "sink"} for candidate in scored)
    # Full joint scoring reuses feedback for the same effective system state.
    assert evaluations["double"] == evaluations["other"] == 1
    assert runtime.engine.components["transform"].variable == "identity"
    history = json.loads(messages[1][-1]["content"].split(
        "PREVIOUS INSTRUCTIONS AND SCORES:\n", 1,
    )[1].split("\n\nOPTIMIZATION TIP:", 1)[0])
    assert history == [{"prompt": "double", "score": 1.0}]


@pytest.mark.parametrize("mode,expected,task_score", [
    ({"enabled": False}, 0.0, None),
    ({"enabled": True, "constitution_weight": 1, "task_weight": 3}, 0.75, 1.0),
])
def test_explicit_constitution_and_mixed_modes_remain_available(tmp_path, monkeypatch, mode, expected, task_score):
    runtime = make_runtime()
    examples = training(runtime)
    _, references = joint_module.evaluate(runtime, runtime.initial_state, examples, [42])
    candidate = runtime.initial_state.configurations["transform"].model_copy(update={"value": "double"})
    optimizer = JointBlockOptimizer(
        runtime, SimpleNamespace(choose=lambda *args: "B"), {"task_scoring": mode}, tmp_path,
    )
    if not mode["enabled"]:
        monkeypatch.setattr(joint_module, "evaluate", unexpected_judge)
    scored = optimizer.score_candidate(
        runtime.initial_state, {"transform": candidate}, None, examples, references, 42,
    )
    assert scored.constitution_score == 0.0
    assert scored.task_score == task_score
    assert scored.score == expected


@pytest.mark.parametrize("invalid", [
    None, [], {"enabled": None}, {"enabled": "true"},
    {"constitution_weight": -1}, {"constitution_weight": None},
    {"constitution_weight": True}, {"constitution_weight": "1"},
    {"constitution_weight": float("nan")}, {"task_weight": 0},
    {"task_weight": float("inf")},
])
def test_task_scoring_settings_are_validated_before_running(tmp_path, invalid):
    with pytest.raises(ValueError, match="task_scoring"):
        validate_method({"block_optimization": {"task_scoring": invalid}})
    with pytest.raises(ValueError, match="task_scoring"):
        JointBlockOptimizer(None, None, {"task_scoring": invalid}, tmp_path)
