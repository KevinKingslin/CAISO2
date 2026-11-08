"""An evaluator must hold one accepted snapshot while the optimizer advances."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from caiso.constitution.models import Constitution
from caiso.core.budget import BudgetUsage
from caiso.core.checkpoint import Checkpoint
from caiso.integration.evaluation_snapshot import EvaluationSnapshot, apply_snapshot, freeze_checkpoint, save_state_dict, snapshot_models
from caiso.integration.runtime import describe_system
from caiso.method.joint_workflow_block import WorkflowTrigger
from scripts import caiso_cli, eval_system
from scripts.results_layout import write_evaluation
from tests.test_runtime import make_system
from caiso.wrappers.example import Example


def make_run(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "experiment.json").write_text(json.dumps({
        "dataset": "pubmed", "system": "synthetic_system", "evaluation": {
            "num_repeat": 1, "split": "test", "max_sample_workers": 1, "max_eval_workers": 1,
        },
    }))
    system = make_system()
    original, _ = describe_system(system)
    current = original.model_copy(deep=True)
    current.configurations["transform"].value = "double"
    constitution = Constitution(principles=["Select the trajectory where transform computes correctly."],
                                source_preference_pair_ids=[], settled_principles=[], cluster_ids=[], abstraction_outputs=[])
    checkpoint = Checkpoint(
        iteration=3, current=current, best=original, current_score=0.5, best_score=0.6,
        system_constitution=constitution, block_constitutions={}, preferences=[],
        budget=BudgetUsage(), trigger=WorkflowTrigger(), config_hash="test", data_hash="test",
        provider_hash="test", library_hash="test",
    )
    checkpoint.save(path / "checkpoint")
    save_state_dict(system.state_dict(), path / "initial_system.pth")
    return checkpoint


@pytest.mark.parametrize("selection", ["current", "best"])
def test_snapshot_freezes_the_selected_state_and_original_settings(tmp_path, selection):
    run = tmp_path / "run"
    checkpoint = make_run(run)
    original_export = (run / "initial_system.pth").read_bytes()
    snapshot = freeze_checkpoint(run / "checkpoint", selection, tmp_path / "evaluation")
    selected = getattr(checkpoint, selection)
    fingerprint = selected.fingerprint()
    assert snapshot.state == selected
    assert snapshot.state_fingerprint == selected.fingerprint()
    assert snapshot.iteration == 3
    assert Path(snapshot.initial_system_path).read_bytes() == original_export
    checkpoint.current.configurations["transform"].value = "new prompt"
    checkpoint.iteration = 4
    checkpoint.save(run / "checkpoint")
    (run / "initial_system.pth").write_bytes(b"a later export")
    assert snapshot.state.fingerprint() == fingerprint
    assert Path(snapshot.initial_system_path).read_bytes() == original_export
    saved = EvaluationSnapshot.model_validate_json((tmp_path / "evaluation/state_snapshot.json").read_text())
    system = make_system()
    apply_snapshot(system, saved)
    assert system.components["transform"].variable == ("double" if selection == "current" else "identity")


def test_current_evaluation_uses_one_snapshot_even_when_optimizer_advances(monkeypatch, tmp_path):
    run = tmp_path / "run"
    checkpoint = make_run(run)
    fingerprint = checkpoint.current.fingerprint()
    original_export = (run / "initial_system.pth").read_bytes()
    (run / "evaluation.json").write_text("final evaluation sentinel")
    (run / "run.json").write_text("optimizer status sentinel")

    def factory(**kwargs):
        checkpoint.current = checkpoint.best.model_copy(deep=True)
        checkpoint.iteration = 4
        checkpoint.save(run / "checkpoint")
        return make_system()

    monkeypatch.setitem(eval_system.registered_systems, "synthetic_system", factory)
    monkeypatch.setitem(eval_system.registered_datasets, "pubmed", lambda **kwargs: ([], [], [Example(question="2", gd_answer="4")]))
    metrics = caiso_cli.main(["evaluate", str(run), "--state", "current"])
    assert metrics["mean"] == 1.0  # Frozen double; the optimizer's new state is identity.
    report = json.loads((run / "evaluation_current.json").read_text())
    assert report["system_variant"] == "current"
    assert report["state_snapshot"]["iteration"] == 3
    assert report["state_snapshot"]["state_fingerprint"] == fingerprint
    raw = Path(report["state_source"]).parent
    assert raw.parent == run / ".internal/evaluation/current/test"
    assert (raw / "test_metrics.json").is_file()
    assert (run / "evaluation.json").read_text() == "final evaluation sentinel"
    assert (run / "run.json").read_text() == "optimizer status sentinel"
    assert Checkpoint.load(run / "checkpoint").iteration == 4
    assert (run / "initial_system.pth").read_bytes() == original_export


def test_automatic_fallback_and_explicit_selection_with_a_stale_final_export(tmp_path):
    make_run(tmp_path)
    args = caiso_cli.parser().parse_args(["evaluate", str(tmp_path)])
    _, config, _ = caiso_cli.resolve(args)
    assert config["caiso_checkpoint_state"] == "current"
    (tmp_path / "final_optimized_system.pth").touch()
    _, config, _ = caiso_cli.resolve(args)
    assert config["system_state_dict_path"].endswith("final_optimized_system.pth")
    for selection in ("current", "best"):
        args.state = selection
        _, config, _ = caiso_cli.resolve(args)
        assert config["system_state_dict_path"] is None
        assert config["caiso_checkpoint_state"] == selection


@pytest.mark.parametrize("selection", ["current", "best", "final"])
def test_state_selector_requires_a_run(selection):
    with pytest.raises(ValueError, match="requires an optimization run"):
        caiso_cli.resolve(caiso_cli.parser().parse_args(["evaluate", "amazon", "--state", selection]))


def test_corrupt_checkpoint_fails_before_constructing_a_system(monkeypatch, tmp_path):
    make_run(tmp_path)
    (tmp_path / "checkpoint/checkpoint.json").write_text("{}")
    def unexpected(**kwargs):
        pytest.fail("A corrupt checkpoint must not cause unoptimized evaluation")
    monkeypatch.setitem(eval_system.registered_systems, "synthetic_system", unexpected)
    with pytest.raises(SystemExit):
        caiso_cli.main(["evaluate", str(tmp_path), "--state", "current"])


def test_failed_export_keeps_the_last_complete_state(monkeypatch, tmp_path):
    destination = tmp_path / "system.pth"
    save_state_dict({"version": 1}, destination)
    def fail(state, path):
        Path(path).write_bytes(b"partial")
        raise OSError("interrupted export")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save_state_dict({"version": 2}, destination)
    assert torch.load(destination, weights_only=False) == {"version": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_current_best_and_final_reports_have_separate_files(tmp_path):
    for variant, score in (("optimized", 0.5), ("current", 0.6), ("best", 0.7)):
        write_evaluation(tmp_path, {"mean": score, "std": 0}, variant=variant, split="test")
    assert json.loads((tmp_path / "evaluation.json").read_text())["score"] == 0.5
    assert json.loads((tmp_path / "evaluation_current.json").read_text())["score"] == 0.6
    assert json.loads((tmp_path / "evaluation_best.json").read_text())["score"] == 0.7


@pytest.mark.parametrize("interrupt", [False, True])
def test_amazon_snapshot_does_not_replace_training_aliases_or_initialize_adapters(monkeypatch, tmp_path, interrupt):
    from examples.systems.amazon import local_models_for_next_item_selection as amazon
    from caiso.utils import lora

    monkeypatch.setattr(amazon, "BASE_MODEL_PATH", str(tmp_path / "base"))
    initialized, requests = [], []
    server = {}
    def initialize(base, path, *args, **kwargs):
        initialized.append(str(path))
        Path(path).mkdir(parents=True, exist_ok=True)
    def load(name, path, *args, **kwargs):
        server[name] = str(path)
    def unload(name, *args, **kwargs):
        del server[name]
    def post(prompt, *args, adapter_id=None, **kwargs):
        requests.append((adapter_id, server[adapter_id]))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"choices": [{"text": "evidence"}]})
    monkeypatch.setattr(amazon, "initialize_adapter", initialize)
    monkeypatch.setattr(amazon, "load_lora_adapter", load)
    monkeypatch.setattr(amazon, "post_http_request", post)
    monkeypatch.setattr(lora, "unload_lora_adapter", unload)
    optimizer_system = amazon.system_engine()
    state, _ = describe_system(optimizer_system)
    paths = {}
    for cid in ("session_analyzer", "candidate_profiler"):
        adapter = tmp_path / (cid + "-accepted")
        adapter.mkdir()
        paths[cid] = str(adapter)
        state.configurations[cid].value["adapter"] = str(adapter)
    snapshot = EvaluationSnapshot(source="checkpoint.json", selection="current", iteration=3,
                                  validation_score=0.5, state_fingerprint=state.fingerprint(), state=state)
    optimizer_aliases = dict(server)
    with pytest.raises(RuntimeError, match="interrupted evaluation") if interrupt else nullcontext():
        with snapshot_models(snapshot, "amazon_system"):
            evaluation_system = amazon.system_engine()
            apply_snapshot(evaluation_system, snapshot)
            assert len(initialized) == 2  # Only the optimizer initialized base adapters.
            assert all(server[name] == value for name, value in optimizer_aliases.items())
            server["SessionAnalyzerModule"] = "optimizer-next-candidate"
            evaluation_system.components["session_analyzer"](sequence="Viewed a pen")
            evaluation_system.components["candidate_profiler"](context="Pen", choices=["Pen", "Cup"])
            assert all("-evaluation-" in name for name, path in requests)
            assert requests[0][1] == paths["session_analyzer"]
            assert requests[1][1] == paths["candidate_profiler"]
            if interrupt:
                raise RuntimeError("interrupted evaluation")
    assert server == {**optimizer_aliases, "SessionAnalyzerModule": "optimizer-next-candidate"}
    assert amazon.load_lora_adapter is load
    assert amazon.post_http_request is post
