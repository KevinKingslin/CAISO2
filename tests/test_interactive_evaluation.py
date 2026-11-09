"""Terminal overrides must be the configuration actually evaluated and saved."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
import torch

from caiso.arch.base import BaseComponent
from caiso.integration.interactive_evaluation import configure_interactively, parse_value
from caiso.wrappers.example import Example
from scripts import caiso_cli, eval_system
from tests.test_runtime import make_system


def answer_with(monkeypatch, values):
    replies = iter(values)
    monkeypatch.setattr("builtins.input", lambda label: next(replies))


class Multiplier(BaseComponent):
    def __init__(self):
        super().__init__(
            "Multiply the result", ["result"], ["answer"], variable={"factor": 1},
            variable_search_space={"factor": [1, 2]},
            config={"temperature": 0.6, "max_tokens": 10}, num_retry=1, retry_after=0,
        )

    def forward(self, **inputs):
        # A changed generation config must reach the same workers as the variable.
        assert self.config.temperature == 0.2
        assert self.config.max_tokens == 20
        return {"answer": str(int(inputs["result"]) * self.variable["factor"])}


def configurable_system(**kwargs):
    system = make_system()
    system.components["sink"] = Multiplier()
    return system


def test_multiline_input_keeps_prompt_text_and_enter_keeps_defaults():
    system = make_system()
    replies = iter(["/multiline", "First line", "", "  indented line", "/end"])
    records = configure_interactively(system, read=lambda _: next(replies), write=lambda _: None)
    assert records["source"]["kind"] == "NONE"
    assert system.components["transform"].variable == "First line\n\n  indented line"
    configure_interactively(system, read=lambda _: "", write=lambda _: None)
    assert system.components["transform"].variable == "First line\n\n  indented line"


def test_typed_overrides_retry_and_reach_workers_outside_optimizer_grid():
    system = configurable_system()
    replies = iter(["double", "bad", "true", "3", "0.2", "20"])
    messages = []
    records = configure_interactively(system, read=lambda _: next(replies), write=messages.append)
    assert sum("Invalid value" in message for message in messages) == 2
    assert records["sink"]["variable"] == {"factor": 3}
    with ThreadPoolExecutor(max_workers=2) as pool:
        scores = list(pool.map(lambda _: system.evaluate(Example(question="2", gd_answer="12")), range(2)))
    assert scores == [1.0, 1.0]


def test_cancelling_input_does_not_apply_partial_edits():
    system = configurable_system()
    replies = iter(["double"])
    with pytest.raises(StopIteration):
        configure_interactively(system, read=lambda _: next(replies), write=lambda _: None)
    assert system.components["transform"].variable == "identity"


@pytest.mark.parametrize("text,current", [("1.5", 1), ("true", 1), ("NaN", 0.6), ("Infinity", 0.6)])
def test_invalid_numeric_values_are_rejected(text, current):
    with pytest.raises(ValueError):
        parse_value(text, current)


def test_interactive_cli_runs_test_set_and_saves_reloadable_configuration(monkeypatch, tmp_path):
    monkeypatch.setitem(eval_system.registered_systems, "hotpotqa_system", configurable_system)
    examples = [Example(question="2", gd_answer="12"), Example(question="3", gd_answer="18")]
    monkeypatch.setitem(eval_system.registered_datasets, "hotpotqa", lambda **_: ([], [], examples))
    answer_with(monkeypatch, ["double", "3", "0.2", "20"])
    output = tmp_path / "manual"
    metrics = caiso_cli.main([
        "evaluate", "hotpotqa", "--interactive", "--repeats", "1", "--workers", "2",
        "--output", str(output),
    ])
    assert metrics["mean"] == 1.0
    document = json.loads((output / "component_config.json").read_text())
    assert document["evaluation"]["split"] == "test"
    assert document["components"]["transform"]["variable"] == "double"
    assert document["components"]["sink"]["variable"] == {"factor": 3}
    report = json.loads((output / "evaluation.json").read_text())
    assert report["system_variant"] == "manual"
    assert report["state_source"] == str(output / "manual_system.pth")
    saved = torch.load(output / "manual_system.pth", weights_only=False)
    assert saved["sink"]["config"].temperature == 0.2
    assert (output / ".internal/evaluation/manual/test/test_metrics.json").is_file()

    # Re-evaluate the exported system without any further terminal input.
    replay = eval_system.main([
        "--dataset", "hotpotqa", "--system_name", "hotpotqa_system", "--num_repeat", "1",
        "--system_state_dict_path", str(output / "manual_system.pth"),
        "--max_eval_workers", "2",
    ], output_dir=tmp_path / "replay")
    assert replay["mean"] == metrics["mean"]
    with pytest.raises(ValueError, match="already contains a run"):
        caiso_cli.main(["evaluate", "hotpotqa", "--interactive", "--output", str(output)])


def test_each_default_interactive_run_gets_a_new_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(eval_system.registered_systems, "hotpotqa_system", lambda **_: make_system())
    monkeypatch.setitem(eval_system.registered_datasets, "hotpotqa", lambda **_: (
        [], [], [Example(question="2", gd_answer="4")],
    ))
    answer_with(monkeypatch, ["double", "double"])
    for _ in range(2):
        caiso_cli.main(["evaluate", "hotpotqa", "--interactive", "--repeats", "1"])
    folders = list((tmp_path / "results").glob("hotpotqa-manual-*"))
    assert len(folders) == 2
    assert all((path / "evaluation.json").is_file() for path in folders)


def test_nonempty_output_is_rejected_before_constructing_a_system(monkeypatch, tmp_path):
    sentinel = tmp_path / "evaluation_unoptimized.json"
    sentinel.write_text("existing baseline")

    def unexpected(**kwargs):
        pytest.fail("A nonempty output directory must be rejected before system construction")

    monkeypatch.setitem(eval_system.registered_systems, "hotpotqa_system", unexpected)
    with pytest.raises(ValueError, match="already contains a run"):
        caiso_cli.main(["evaluate", "hotpotqa", "--interactive", "--output", str(tmp_path)])
    assert sentinel.read_text() == "existing baseline"
    assert list(tmp_path.iterdir()) == [sentinel]


def test_interactive_yaml_setting_and_conflicting_saved_state(tmp_path):
    path = tmp_path / "evaluation.yaml"
    path.write_text("dataset: hotpotqa\nsystem_name: hotpotqa_system\ninteractive: true\n")
    args = caiso_cli.parser().parse_args(["evaluate", str(path)])
    assert caiso_cli.resolve(args)[1]["interactive"] is True
    path.write_text(path.read_text() + "system_state_dict_path: saved.pth\n")
    with pytest.raises(ValueError, match="without a saved system state"):
        caiso_cli.resolve(args)


def test_interactive_rejects_optimization_run_directory(tmp_path):
    (tmp_path / "experiment.json").write_text(json.dumps({"dataset": "amazon", "system": "amazon_system"}))
    args = caiso_cli.parser().parse_args(["evaluate", str(tmp_path), "--interactive"])
    with pytest.raises(ValueError, match="requires a dataset"):
        caiso_cli.resolve(args)


def test_model_selection_and_adapter_paths(tmp_path):
    system = make_system()
    selector = system.components["transform"]
    selector.update({"model_selection": "original"})
    selector.variable_search_space = {"model_selection": ["original"]}
    replies = iter(["custom-model"])
    records = configure_interactively(system, read=lambda _: next(replies), write=lambda _: None)
    assert records["transform"]["kind"] == "MODEL_SELECTION"
    assert selector.variable == {"model_selection": "custom-model"}
    assert parse_value(str(tmp_path), Path("adapter")) == tmp_path
    with pytest.raises(ValueError, match="existing adapter directory"):
        parse_value(str(tmp_path / "missing"), Path("adapter"))
