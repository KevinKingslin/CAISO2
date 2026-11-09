"""Selection-history controls preserve all other feedback and rejection memory."""

from copy import deepcopy
import json

import pytest

from caiso.core.checkpoint import Checkpoint
from caiso.core.serialization import digest
from caiso.method.optimization_history import OptimizationHistory
from caiso.optim.optimizer import CaisoOptimizer
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example
from tests.offline import OfflineEmbeddings, OfflineProvider
from tests.test_runtime import make_system


@pytest.mark.parametrize("selection_history,block_constitution", [
    (True, False), (False, False), (True, True), (False, True),
])
def test_history_controls_and_resume(tmp_path, monkeypatch, selection_history, block_constitution):
    experiment = {
        "dataset": "pubmed", "system": "synthetic-test-only", "num_iterations": 2,
        "val_size": 1, "per_component_train_size": 1, "per_iteration_new_data_size": 1,
        "caiso": {
            "edge_baselines": {
                "source->transform": {"number": {"value": "0"}},
                "transform->sink": {"result": {"value": "0"}},
            },
            "method": {
                "initial_preference_pairs": 1,
                "block_selection": {"use_history": selection_history},
                "block_constitution": {"enabled": block_constitution, "preference_pairs": 1},
                "block_optimization": {"prompt_candidates": 3, "max_joint_candidates": 4},
                "attribution": {"bootstrap_samples": 10},
                "dicai": {
                    "self_consistency_samples": 1, "self_refine_steps": 0,
                    "clustering": {"min_clusters": 2, "max_clusters": 4},
                },
                "reporting": {"cli": False, "capture_model_output": False},
            },
        },
    }
    provider, embeddings = OfflineProvider(None), OfflineEmbeddings(None)
    registry = ProviderRegistry({"offline": provider}, dict.fromkeys(
        ["system_constitution", "block_constitution", "constitution_judge", "block_selection", "prompt_optimizer"], "offline",
    ))
    summaries = []
    summarize = OptimizationHistory.selection_summary

    def observe_summary(self, *args, **kwargs):
        summary = summarize(self, *args, **kwargs)
        summaries.append(summary)
        return summary

    monkeypatch.setattr(OptimizationHistory, "selection_summary", observe_summary)
    filters = []
    filter_candidates = OptimizationHistory.filter_candidates

    def observe_filter(self, *args, **kwargs):
        filters.append(True)
        return filter_candidates(self, *args, **kwargs)

    monkeypatch.setattr(OptimizationHistory, "filter_candidates", observe_filter)

    def optimizer():
        result = CaisoOptimizer(
            make_system(), [Example(question="1", gd_answer="2")],
            [Example(question="2", gd_answer="2")], experiment, tmp_path,
            providers=registry, embeddings=embeddings,
        )
        provider.budget = embeddings.budget = result.budget
        return result

    original = optimizer()
    system, metrics = original.optimize()
    assert metrics["completed_iterations"] == 2
    assert system.components["transform"].variable == "identity"
    calls = [messages for messages, _, schema in provider.calls if schema and schema.__name__ == "BlockVote"]
    assert len(calls) == 2
    histories = [json.loads(messages[-1]["content"].split("OPTIMIZATION HISTORY:\n", 1)[1]) for messages in calls]
    selection_enabled = selection_history
    assert len(summaries) == (2 if selection_enabled else 0)
    if selection_enabled:
        assert histories == summaries
        first = next(value for value in histories[0].values() if "transform" in value["members"])
        second = next(value for value in histories[1].values() if "transform" in value["members"])
        assert first["current_context_attempts"] == 0
        assert second["validated_rejections"] == 1
        assert second["recent_attempts"][0]["accepted"] is False
    else:
        assert histories == [{}, {}]
    for iteration, history in enumerate(histories, 1):
        phase = tmp_path / f"iterations/{iteration:04}/03_selection"
        assert json.loads((phase / "status.json").read_text())["history_enabled"] is selection_enabled
        assert json.loads((phase / "block_selection.json").read_text())["optimization_history"] == history
        optimization = phase.parent / "05_block_optimization/status.json"
        if optimization.exists():
            assert json.loads(optimization.read_text())["history_enabled"] is True
        if block_constitution:
            induction = phase.parent / "04_block_constitution/status.json"
            assert json.loads(induction.read_text())["history_enabled"] is True

    prompt_calls = [messages for messages, _, schema in provider.calls if schema and schema.__name__ == "PromptProposal"]
    prompt_histories = [json.loads(messages[-1]["content"].split("PREVIOUS INSTRUCTIONS AND SCORES:\n", 1)[1].split("\n\nOPTIMIZATION TIP:", 1)[0]) for messages in prompt_calls]
    assert bool(prompt_histories)
    assert any(prompt_histories)
    assert len(filters) == (3 if block_constitution else 2)
    checkpoint = Checkpoint.load(tmp_path / "checkpoint")
    assert [attempt.outcome for attempt in checkpoint.optimization_history.attempts] == ["validated", "no_new_candidate"]
    assert checkpoint.optimization_history.attempts[0].accepted is False
    phase = "04_block_constitution" if block_constitution else "05_block_optimization"
    skipped = json.loads((tmp_path / f"iterations/0002/{phase}/status.json").read_text())
    assert "No new block candidates" in skipped["reason"]
    assert not (tmp_path / "iterations/0002/05_block_optimization/validation_metrics.json").exists()
    # True retains the old implicit behavior; false must remain in the run hash.
    legacy = deepcopy(original.config)
    legacy["use_history"] = True  # Previously shipped redundant top-level default.
    if selection_history:
        legacy["block_selection"].pop("use_history")
    checkpoint.config_hash = digest(legacy)
    checkpoint.save(tmp_path / "checkpoint")
    restored = optimizer()
    _, resumed_metrics = restored.optimize(resume=tmp_path / "checkpoint")
    assert resumed_metrics["completed_iterations"] == 2
    assert len(summaries) == (2 if selection_enabled else 0)
    assert Checkpoint.load(tmp_path / "checkpoint").config_hash == digest(restored.config)
    assert len(Checkpoint.load(tmp_path / "checkpoint").optimization_history.attempts) == 2
