"""History can be withheld from selection without losing rejected-state evidence."""

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


@pytest.mark.parametrize("use_history", [True, False])
def test_selection_history_toggle_preserves_rejection_filtering_and_resume(tmp_path, monkeypatch, use_history):
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
                "block_selection": {"use_history": use_history},
                "block_constitution": {"enabled": False},
                "block_optimization": {"prompt_candidates": 1, "max_joint_candidates": 4},
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
        ["system_constitution", "constitution_judge", "block_selection", "prompt_optimizer"], "offline",
    ))
    summaries = []
    summarize = OptimizationHistory.selection_summary

    def observe_summary(self, *args, **kwargs):
        summary = summarize(self, *args, **kwargs)
        summaries.append(summary)
        return summary

    monkeypatch.setattr(OptimizationHistory, "selection_summary", observe_summary)

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
    assert len(summaries) == (2 if use_history else 0)
    if use_history:
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
        assert json.loads((phase / "status.json").read_text())["history_enabled"] is use_history
        assert json.loads((phase / "block_selection.json").read_text())["optimization_history"] == history

    checkpoint = Checkpoint.load(tmp_path / "checkpoint")
    assert [attempt.outcome for attempt in checkpoint.optimization_history.attempts] == ["validated", "no_new_candidate"]
    assert checkpoint.optimization_history.attempts[0].accepted is False
    skipped = json.loads((tmp_path / "iterations/0002/05_block_optimization/status.json").read_text())
    assert "No new block candidates" in skipped["reason"]
    assert not (tmp_path / "iterations/0002/05_block_optimization/validation_metrics.json").exists()
    # True retains the old implicit behavior; false must remain in the run hash.
    legacy = deepcopy(original.config)
    if use_history:
        legacy["block_selection"].pop("use_history")
    checkpoint.config_hash = digest(legacy)
    checkpoint.save(tmp_path / "checkpoint")
    restored = optimizer()
    _, resumed_metrics = restored.optimize(resume=tmp_path / "checkpoint")
    assert resumed_metrics["completed_iterations"] == 2
    assert len(summaries) == (2 if use_history else 0)
    assert Checkpoint.load(tmp_path / "checkpoint").config_hash == digest(restored.config)
    assert len(Checkpoint.load(tmp_path / "checkpoint").optimization_history.attempts) == 2
