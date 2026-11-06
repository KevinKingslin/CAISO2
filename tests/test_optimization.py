import json

from caiso.core.checkpoint import Checkpoint
from caiso.optim.optimizer import CaisoOptimizer
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example
from tests.offline import OfflineEmbeddings, OfflineProvider
from tests.test_runtime import make_system


def test_complete_caiso_cycle_on_original_system_interface(tmp_path):
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
                "block_constitution": {"preference_pairs": 1},
                "block_optimization": {"prompt_candidates": 1, "max_joint_candidates": 4},
                "dicai": {"clustering": {"min_clusters": 2, "max_clusters": 4}},
                "reporting": {"cli": False, "capture_model_output": False},
            },
        },
    }
    # CaisoOptimizer creates its budget; fixtures adopt it before any call.
    provider = OfflineProvider(None)
    embeddings = OfflineEmbeddings(None)
    registry = ProviderRegistry(
        {"offline": provider},
        dict.fromkeys(
            [
                "system_constitution",
                "block_constitution",
                "constitution_judge",
                "block_selection",
                "prompt_optimizer",
            ],
            "offline",
        ),
    )
    optimizer = CaisoOptimizer(
        system,
        [Example(question="1", gd_answer="2")],
        [Example(question="2", gd_answer="4")],
        experiment,
        tmp_path,
        providers=registry,
        embeddings=embeddings,
    )
    provider.budget = embeddings.budget = optimizer.budget
    result, metrics = optimizer.optimize()
    assert result is system
    assert result.components["transform"].variable == "double"
    assert metrics["validation_score"] == 1
    assert metrics["completed_iterations"] == 1
    profiles = optimizer.dicai.identities
    assert len(profiles) == 3
    assert optimizer.block_dicai.identities == profiles
    for profile in profiles:
        assert any(profile["prompt_text"] in call[0][0]["content"] for call in provider.calls)
    saved_experiment = json.loads((tmp_path / "experiment.json").read_text())
    assert saved_experiment["caiso"]["personas"] == profiles
    checkpoint = Checkpoint.load(tmp_path / "checkpoint")
    assert checkpoint.best.configurations["transform"].value == "double"
    assert checkpoint.block_constitutions
    import torch

    snapshot = tmp_path / "system.pth"
    torch.save(result.state_dict(), snapshot)
    reloaded = make_system()
    reloaded.load_state_dict(torch.load(snapshot, weights_only=False))
    assert reloaded.evaluate(Example(question="5", gd_answer="10")) == 1
    assert (tmp_path / ".internal/dicai/system/silhouette.png").is_file()
    assert (tmp_path / "iterations/0001/01_attribution/edge_weights.json").is_file()
    assert (tmp_path / "iterations/0001/02_blocks/weighted_graph.graphml").is_file()
    assert (tmp_path / "iterations/0001/04_block_constitution/silhouette.png").is_file()
    # Restore the accepted checkpoint onto a fresh copy of the benchmark system.
    fresh = make_system()
    restored = CaisoOptimizer(
        fresh,
        [Example(question="1", gd_answer="2")],
        [Example(question="2", gd_answer="4")],
        experiment,
        tmp_path,
        providers=registry,
        embeddings=embeddings,
    )
    provider.budget = embeddings.budget = restored.budget
    resumed, resumed_metrics = restored.optimize(resume=tmp_path / "checkpoint")
    assert resumed.evaluate(Example(question="5", gd_answer="10")) == 1
    assert resumed_metrics["completed_iterations"] == 1
