"""Optimization memory can be disabled without removing candidate evaluation."""

from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace

import pytest

from caiso.core.configuration import ConfigurationType
from caiso.core.serialization import digest
from caiso.method.config import resume_config_hashes
from caiso.optimizers.prompt_optimizer import PromptOptimizer
from caiso.workflow_search import compatibility_adaptation
from caiso.workflow_search.generator import WorkflowGenerator
from tests.prompt_requests import PRINCIPLES, Recorder
from tests.test_candidate_configuration_toggle import context, joint, joint_runtime
from tests.test_runtime import make_runtime


@pytest.mark.parametrize("use_history", [True, False])
def test_prompt_feedback_is_omitted_and_not_scored_when_disabled(tmp_path, use_history):
    component, ctx = context(tmp_path, ConfigurationType.PROMPT, "Original instruction", True, count=3)
    ctx.config["use_history"] = use_history
    feedback = []
    scores = iter([0.3, 0.1, 0.2])

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
    assert len(feedback) == (3 if use_history else 0)
    assert [[item["score"] for item in history] for history in histories] == (
        [[], [0.3], [0.1, 0.3]] if use_history else [[], [], []]
    )


@pytest.mark.parametrize("use_history", [True, False])
def test_workflow_feedback_respects_the_history_switch(use_history):
    runtime, provider = make_runtime(), Recorder()
    performance = [{"iteration": 1, "score": 0.4}, {"iteration": 2, "score": 0.5}]
    WorkflowGenerator(provider, use_history=use_history).propose(
        runtime.initial_state, SimpleNamespace(principles=PRINCIPLES), runtime.library,
        performance, [], 42,
    )
    content = provider.calls[0]["messages"][-1]["content"]
    supplied = json.loads(content.split("RECENT PERFORMANCE:\n", 1)[1].split("\n\nEXECUTION TRAJECTORIES:", 1)[0])
    assert supplied == (performance if use_history else [])
    # Current workflow and constitution are still available to generate proposals.
    assert "transform" in content
    assert PRINCIPLES[0] in content


def test_compatibility_optimizer_inherits_the_global_history_setting(monkeypatch, tmp_path):
    runtime, examples = joint_runtime()
    optimizer = joint(runtime, tmp_path, True)
    optimizer.config["use_history"] = False
    observed = []
    monkeypatch.setattr(compatibility_adaptation, "affected_components", lambda *args: ["retriever"])

    def propose(self, *args):
        observed.append(self.config["use_history"])
        return []

    monkeypatch.setattr(type(optimizer), "propose", propose)
    _, report = compatibility_adaptation.compatibility_adapt(
        runtime, runtime.initial_state, runtime.initial_state, {}, None, optimizer,
        examples, [], 42, 20, 1e-6, search_config={"use_history": True},
    )
    assert observed == [False]
    assert report["no_candidates"] is True


@pytest.mark.parametrize("use_history", [True, False])
def test_global_history_resume_hash_is_strict_except_for_implicit_true(use_history):
    config = {
        "seed": 42, "use_history": use_history,
        "block_selection": {"use_history": True},
        "block_optimization": {"include_current_configuration": True, "reevaluate_current": True},
    }
    before = deepcopy(config)
    hashes = resume_config_hashes(config)
    paths = [(None, "use_history"), ("block_selection", "use_history"),
             ("block_optimization", "include_current_configuration"), ("block_optimization", "reevaluate_current")]
    for omitted in product([True, False], repeat=len(paths)):
        older = deepcopy(config)
        for (section, key), remove in zip(paths, omitted, strict=True):
            if remove:
                (older if section is None else older[section]).pop(key)
        assert (digest(older) in hashes) == (use_history or not omitted[0])
    changed = deepcopy(config)
    changed["use_history"] = not use_history
    assert digest(changed) not in hashes
    assert config == before
