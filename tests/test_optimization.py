import json

import pytest

from caiso.core.checkpoint import Checkpoint
from caiso.core.serialization import digest
from caiso.optim.optimizer import CaisoOptimizer
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example
from tests.offline import OfflineEmbeddings, OfflineProvider, training_preferences
from tests.test_runtime import make_system


@pytest.mark.parametrize("block_constitution_enabled,accept_update", [
    (True, True), (False, True), (False, False), (None, True), (None, False),
])
@pytest.mark.parametrize("reevaluate_current", [True, False])
@pytest.mark.parametrize("include_current_configuration", [True, False])
def test_complete_caiso_cycle_on_original_system_interface(tmp_path, monkeypatch, block_constitution_enabled, accept_update, reevaluate_current, include_current_configuration):
    system = make_system()
    experiment = {
        "dataset": "pubmed",
        "system": "synthetic-test-only",
        "num_iterations": 1,
        "val_size": 1,
        "per_component_train_size": 1,
        "per_iteration_new_data_size": 1,
        "caiso": {
            "edge_baselines": {
                "source->transform": {"number": {"value": "0"}},
                "transform->sink": {"result": {"value": "0"}},
            },
            "method": {
                "initial_preference_pairs": 1,
                "attribution": {"bootstrap_samples": 10},
                "block_constitution": {"enabled": block_constitution_enabled, "preference_pairs": 1},
                "block_optimization": {
                    "prompt_candidates": 1, "max_joint_candidates": 4,
                    "reevaluate_current": reevaluate_current, "include_current_configuration": include_current_configuration,
                    "validation_repeats": 1,
                },
                "dicai": {"clustering": {"min_clusters": 2, "max_clusters": 4}},
                "reporting": {"cli": False, "capture_model_output": False},
            },
        },
    }
    if block_constitution_enabled is None:
        # An omitted override must use the default system constitution too.
        experiment["caiso"]["method"].pop("block_constitution")
    # CaisoOptimizer creates its budget; fixtures adopt it before any call.
    provider = OfflineProvider(None)
    embeddings = OfflineEmbeddings(None)
    proposal_constitutions = []
    propose = JointBlockOptimizer.propose

    def observe_proposal(self, state, members, constitution, examples, seed, **kwargs):
        proposal_constitutions.append(constitution)
        assert kwargs["system_constitution"].metadata["scope"] == "system"
        assert kwargs["preference_examples"] is optimizer.train
        return propose(self, state, members, constitution, examples, seed, **kwargs)

    monkeypatch.setattr(JointBlockOptimizer, "propose", observe_proposal)
    from caiso.method import block_optimization
    validation_calls = []
    evaluate = block_optimization.evaluate

    def observe_validation(*args, **kwargs):
        validation_calls.append(args[1].fingerprint())
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(block_optimization, "evaluate", observe_validation)
    if not block_constitution_enabled:
        def unexpected_block_induction(*args, **kwargs):
            pytest.fail("Disabled block constitution must skip preference collection and induction")

        monkeypatch.setattr("caiso.method.algorithm.build_block_preferences", unexpected_block_induction)
        monkeypatch.setattr("caiso.method.algorithm.induce_block_constitution", unexpected_block_induction)
    registry = ProviderRegistry(
        {"offline": provider},
        dict.fromkeys(
            [
                "system_constitution",
                "block_constitution",
                "constitution_judge",
                "block_selection",
                "prompt_optimizer",
            ] if block_constitution_enabled else [
                "system_constitution", "constitution_judge", "block_selection", "prompt_optimizer",
            ],
            "offline",
        ),
    )
    optimizer = CaisoOptimizer(
        system,
        [Example(question="1", gd_answer="2")],
        [Example(question="2", gd_answer="4" if accept_update else "2")],
        experiment,
        tmp_path,
        providers=registry,
        embeddings=embeddings,
    )
    provider.budget = embeddings.budget = optimizer.budget
    result, metrics = optimizer.optimize(preferences=training_preferences(optimizer))
    assert result is system
    expected_config = "double" if accept_update else "identity"
    assert result.components["transform"].variable == expected_config
    assert metrics["validation_score"] == 1
    assert metrics["completed_iterations"] == 1
    profiles = optimizer.dicai.identities
    assert len(profiles) == 3
    if block_constitution_enabled:
        assert optimizer.block_dicai.identities == profiles
    else:
        assert optimizer.block_dicai is None
    for profile in profiles:
        assert any(profile["prompt_text"] in call[0][0]["content"] for call in provider.calls)
    saved_experiment = json.loads((tmp_path / "experiment.json").read_text())
    assert saved_experiment["caiso"]["personas"] == profiles
    checkpoint = Checkpoint.load(tmp_path / "checkpoint")
    assert checkpoint.best.configurations["transform"].value == expected_config
    if block_constitution_enabled:
        assert checkpoint.block_constitutions
        assert len(proposal_constitutions) == 2
        assert proposal_constitutions[1] == next(iter(checkpoint.block_constitutions.values()))
    else:
        assert checkpoint.block_constitutions == {}
        assert proposal_constitutions == [checkpoint.system_constitution]
        assert not (tmp_path / ".internal/dicai/blocks").exists()
        skipped = json.loads((tmp_path / "iterations/0001/04_block_constitution/status.json").read_text())
        assert skipped["induction_skipped"] is True
        assert "enabled is false" in skipped["reason"]
        for name in ("preliminary_candidates.json", "block_preferences.json", "block_constitution.json"):
            assert not (tmp_path / "iterations/0001/04_block_constitution" / name).exists()
    source = json.loads((tmp_path / "iterations/0001/05_block_optimization/status.json").read_text())
    assert source["constitution_source"] == ("block" if block_constitution_enabled else "system")
    validation = json.loads((tmp_path / "iterations/0001/05_block_optimization/validation_metrics.json").read_text())
    assert validation["current"]["score"] == (0 if accept_update else 1)
    assert validation["candidate"]["score"] == (1 if accept_update else 0)
    assert validation["current"]["reused"] is not reevaluate_current
    assert validation["candidate"]["reused"] is False
    assert validation["current"]["trajectory_count"] == (1 if reevaluate_current else 0)
    if not reevaluate_current:
        assert "coverage" not in validation["current"]
        assert validation["current"]["source"] == "saved_current_score"
    assert len(validation_calls) == (2 if reevaluate_current else 1)
    trajectories = (tmp_path / "iterations/0001/05_block_optimization/validation_trajectories.jsonl").read_text().splitlines()
    assert len(trajectories) == (2 if reevaluate_current else 1)
    assert checkpoint.current_score == 1
    import torch

    snapshot = tmp_path / "system.pth"
    torch.save(result.state_dict(), snapshot)
    reloaded = make_system()
    reloaded.load_state_dict(torch.load(snapshot, weights_only=False))
    assert reloaded.evaluate(Example(question="5", gd_answer="10" if accept_update else "5")) == 1
    assert (tmp_path / ".internal/dicai/system/silhouette.png").is_file()
    assert (tmp_path / "iterations/0001/01_attribution/edge_weights.json").is_file()
    assert (tmp_path / "iterations/0001/02_blocks/weighted_graph.graphml").is_file()
    assert (tmp_path / "iterations/0001/04_block_constitution/silhouette.png").is_file() == bool(block_constitution_enabled)
    configurations = json.loads((tmp_path / "iterations/0001/07_state/component_configurations.json").read_text())
    assert configurations["original"]["transform"]["value"] == "identity"
    assert configurations["current"]["transform"]["value"] == expected_config
    assert configurations["candidate"]["transform"]["value"] == "double"
    assert configurations["candidate_status"] == ("Accepted" if accept_update else "Rejected")
    # Implicit true defaults in older checkpoints are compatible; explicit
    # false protocol settings remain part of the required configuration hash.
    legacy = {**optimizer.config, "block_optimization": {
        key: value for key, value in optimizer.config["block_optimization"].items()
        if key not in {"reevaluate_current", "include_current_configuration"} or value is not True
    }}
    legacy["block_selection"] = {
        key: value for key, value in optimizer.config["block_selection"].items()
        if key != "use_history"
    }
    checkpoint.config_hash = digest(legacy)
    checkpoint.save(tmp_path / "checkpoint")
    # Restore the accepted checkpoint onto a fresh copy of the benchmark system.
    fresh = make_system()
    restored = CaisoOptimizer(
        fresh,
        [Example(question="1", gd_answer="2")],
        [Example(question="2", gd_answer="4" if accept_update else "2")],
        experiment,
        tmp_path,
        providers=registry,
        embeddings=embeddings,
    )
    provider.budget = embeddings.budget = restored.budget
    resumed, resumed_metrics = restored.optimize(resume=tmp_path / "checkpoint")
    assert resumed.evaluate(Example(question="5", gd_answer="10" if accept_update else "5")) == 1
    assert resumed_metrics["completed_iterations"] == 1
    assert len(proposal_constitutions) == (2 if block_constitution_enabled else 1)
    assert len(validation_calls) == (2 if reevaluate_current else 1)
    assert Checkpoint.load(tmp_path / "checkpoint").config_hash == digest(restored.config)
