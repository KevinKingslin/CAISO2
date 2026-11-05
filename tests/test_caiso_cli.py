"""The convenience CLI must dispatch to the existing benchmark launchers."""

import json
from types import SimpleNamespace

import pytest

from scripts import caiso_cli


@pytest.mark.parametrize("dataset", ["amazon", "hotpotqa", "pubmedqa"])
def test_dataset_alias_uses_existing_profiles(dataset):
    args = caiso_cli.parser().parse_args(["evaluate", dataset, "--unoptimized"])
    path, config, run = caiso_cli.resolve(args)
    canonical = caiso_cli.DATASETS[dataset]
    assert path == caiso_cli.ROOT / f"scripts/configs/eval/{canonical}_vanilla.yaml"
    assert config["dataset"] == canonical
    assert config["system_state_dict_path"] is None
    assert config["num_repeat"] == 3
    assert run is None


@pytest.mark.parametrize("command", ["optimize", "run"])
def test_optimization_dispatch_uses_original_config(monkeypatch, tmp_path, command):
    from scripts import optimize_system

    calls = []
    monkeypatch.setattr(optimize_system, "main", lambda args: calls.append(args))
    caiso_cli.main([
        command, "amazon", "--output", str(tmp_path),
        "--checkpoint", "checkpoint-dir", "--no-test",
    ])
    assert calls == [[
        str(caiso_cli.ROOT / "scripts/configs/optimize/amazon.yaml"),
        "--output", str(tmp_path), "--resume", "checkpoint-dir", "--no-test",
    ]]


def make_run(path):
    (path / "experiment.json").write_text(json.dumps({"dataset": "amazon", "system": "amazon_system"}))


def test_run_evaluation_never_falls_back_to_unoptimized(tmp_path):
    make_run(tmp_path)
    args = caiso_cli.parser().parse_args(["evaluate", str(tmp_path)])
    with pytest.raises(ValueError, match="Missing optimized checkpoint"):
        caiso_cli.resolve(args)
    checkpoint = tmp_path / "final_optimized_system.pth"
    checkpoint.touch()
    _, config, run = caiso_cli.resolve(args)
    assert config["system_state_dict_path"] == str(checkpoint)
    assert run == tmp_path
    args.unoptimized = True
    _, config, _ = caiso_cli.resolve(args)
    assert config["system_state_dict_path"] is None


def test_evaluation_dispatch_loads_run_and_separates_outputs(monkeypatch, tmp_path):
    from scripts import eval_system

    make_run(tmp_path)
    checkpoint = tmp_path / "final_optimized_system.pth"
    checkpoint.touch()
    calls = []
    monkeypatch.setattr(eval_system, "main", lambda argv, **kw: calls.append((argv, kw)))
    caiso_cli.main(["evaluate", str(tmp_path), "--repeats", "1"])
    argv, options = calls[-1]
    assert argv[argv.index("--system_state_dict_path") + 1] == str(checkpoint)
    assert argv[argv.index("--num_repeat") + 1] == "1"
    assert options["output_dir"] == tmp_path / "eval_optimized"
    caiso_cli.main(["evaluate", str(tmp_path), "--unoptimized"])
    argv, options = calls[-1]
    assert "--system_state_dict_path" not in argv
    assert options["output_dir"] == tmp_path / "eval_unoptimized"


def test_unoptimized_cli_executes_existing_evaluator(monkeypatch, tmp_path):
    from scripts import eval_system

    examples = [object(), object()]
    calls = []
    system = SimpleNamespace(state_dict=lambda: {"original": True})
    monkeypatch.setitem(eval_system.registered_systems, "amazon_system", lambda **kw: system)
    monkeypatch.setitem(eval_system.registered_datasets, "amazon", lambda **kw: ([], [], examples))

    def evaluate(system, testset, num_repeat):
        calls.append((system, testset, num_repeat))
        return {"mean": 0.5, "std": 0.0}, []

    monkeypatch.setattr(eval_system, "eval_system", evaluate)
    caiso_cli.main(["evaluate", "amazon", "--unoptimized", "--repeats", "1", "--output", str(tmp_path)])
    assert calls == [(system, examples, 1)]
    assert json.loads((tmp_path / "test_metrics.json").read_text())["mean"] == 0.5
    assert (tmp_path / "system_state_dict.pth.bak").exists()


def test_legacy_system_flag_and_yaml_configs():
    args = caiso_cli.parser().parse_args(["optimize", "--system", "pubmedqa"])
    path, config, _ = caiso_cli.resolve(args)
    assert path.name == "pubmed.yaml"
    assert config["system"] == "pubmed_system"
    args = caiso_cli.parser().parse_args(["optimize", "--config", str(path)])
    assert caiso_cli.resolve(args)[1] == config


@pytest.mark.parametrize("argv", [
    ["evaluate", "amazon", "--system", "amazon"],
    ["evaluate", "does-not-exist"],
])
def test_bad_targets_fail_before_loading_models(argv):
    with pytest.raises(ValueError):
        caiso_cli.resolve(caiso_cli.parser().parse_args(argv))
