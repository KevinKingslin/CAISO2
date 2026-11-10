"""The selection-history toggle must preserve feedback everywhere else."""

from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace

import pytest

from caiso.core.configuration import ConfigurationType
from caiso.core.serialization import digest
from caiso.integration.runtime import adapt_examples
from caiso.method.algorithm import Algorithm
from caiso.method.config import resume_config_hashes
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
from caiso.optimizers.prompt_optimizer import PromptOptimizer
from caiso.workflow_search import compatibility_adaptation
from caiso.workflow_search.generator import WorkflowGenerator
from caiso.wrappers.example import Example
from tests.prompt_requests import PRINCIPLES, Recorder
from tests.test_candidate_configuration_toggle import context
from tests.test_runtime import make_runtime


def test_prompt_feedback_is_always_scored_and_preserved(tmp_path):
    component, ctx = context(tmp_path, ConfigurationType.PROMPT, "Original instruction", True, count=3)
    feedback = []
    scores = iter([0.5, 0.3, 0.1, 0.2])

    def score(configs):
        feedback.append(configs[component.component_id].value)
        return next(scores)

    ctx.score_config = score
    candidates = PromptOptimizer().propose(component, ctx)
    histories = [
        json.loads(call["messages"][-1]["content"].split("PREVIOUS INSTRUCTIONS AND SCORES:\n", 1)[1].split("\n\nOPTIMIZATION TIP:", 1)[0])
        for call in ctx.runtime.providers.calls
    ]
    assert len(candidates) == 4
    assert len(feedback) == 4  # Include the current instruction's reference score.
    assert [[item["score"] for item in history] for history in histories] == (
        [[0.5], [0.3, 0.5], [0.1, 0.3, 0.5]]
    )


@pytest.mark.parametrize("use_history", [True, False])
def test_workflow_feedback_is_preserved_for_either_selection_setting(tmp_path, use_history):
    runtime, provider = make_runtime(), Recorder()
    performance = [{"iteration": 1, "score": 0.4}, {"iteration": 2, "score": 0.5}]
    algorithm = Algorithm(
        runtime, None, None,
        {"block_selection": {"use_history": use_history}, "training_batch_size": 1,
         "validation_batch_size": 1, "block_optimization": {}}, tmp_path,
        generator=WorkflowGenerator(provider),
    )
    algorithm.generator.propose(
        runtime.initial_state, SimpleNamespace(principles=PRINCIPLES), runtime.library,
        performance, [], 42,
    )
    content = provider.calls[0]["messages"][-1]["content"]
    supplied = json.loads(content.split("RECENT PERFORMANCE:\n", 1)[1].split("\n\nEXECUTION TRAJECTORIES:", 1)[0])
    assert supplied == performance
    # Current workflow and constitution are still available to generate proposals.
    assert "transform" in content
    assert PRINCIPLES[0] in content


def test_compatibility_prompt_feedback_is_preserved(monkeypatch, tmp_path):
    runtime, provider = make_runtime(), Recorder()
    examples = adapt_examples(runtime.engine, [Example(question="2", gd_answer="4")], "train")
    optimizer = JointBlockOptimizer(runtime, None, {}, tmp_path)
    monkeypatch.setattr(runtime.providers, "complete", provider.complete)
    monkeypatch.setattr(compatibility_adaptation, "affected_components", lambda *args: ["transform"])
    _, report = compatibility_adaptation.compatibility_adapt(
        runtime, runtime.initial_state, runtime.initial_state, {},
        SimpleNamespace(principles=PRINCIPLES), optimizer, examples, [], 42, 20, 1e-6,
        search_config={"prompt_candidates": 2},
    )
    histories = [
        json.loads(call["messages"][-1]["content"].split("PREVIOUS INSTRUCTIONS AND SCORES:\n", 1)[1].split("\n\nOPTIMIZATION TIP:", 1)[0])
        for call in provider.calls
    ]
    assert len(histories) == 2
    assert histories[0] == [{"prompt": "identity", "score": 0.0}]
    assert histories[1][0]["score"] == 0.0
    assert "no_candidates" not in report


@pytest.mark.parametrize("use_history", [True, False])
def test_selection_history_resume_hash_is_strict_except_for_implicit_true(use_history):
    config = {
        "seed": 42,
        "block_selection": {"use_history": use_history},
        "block_optimization": {"include_current_configuration": True, "reevaluate_current": True},
    }
    before = deepcopy(config)
    hashes = resume_config_hashes(config)
    paths = [("block_selection", "use_history"),
             ("block_optimization", "include_current_configuration"), ("block_optimization", "reevaluate_current")]
    for omitted in product([True, False], repeat=len(paths)):
        older = deepcopy(config)
        for (section, key), remove in zip(paths, omitted, strict=True):
            if remove:
                older[section].pop(key)
        equivalent = use_history or not omitted[0]
        assert (digest(older) in hashes) == equivalent
        assert (digest({**older, "use_history": True}) in hashes) == equivalent
        assert digest({**older, "use_history": False}) not in hashes
    changed = deepcopy(config)
    changed["block_selection"]["use_history"] = not use_history
    assert digest(changed) not in hashes
    assert config == before
