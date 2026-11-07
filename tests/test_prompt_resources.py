"""Missing optional prompt data must not break runs; required data fails early."""

from pathlib import Path
from types import SimpleNamespace

import pytest

import caiso.prompts as prompts
from caiso.core.component import Component
from caiso.core.configuration import Configuration, ConfigurationType
from caiso.optim.optimizer import CaisoOptimizer
from caiso.optimizers.prompt_optimizer import PromptOptimizer
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example
from scripts.verify_prompts import main as verify_prompts
from tests.offline import OfflineEmbeddings, OfflineProvider
from tests.prompt_requests import Recorder
from tests.test_optimization import test_complete_caiso_cycle_on_original_system_interface as run_cycle
from tests.test_runtime import make_system

ROOT = Path(__file__).resolve().parents[1]


def mask_resources(monkeypatch, missing):
    root = ROOT / "caiso/prompts"

    class Resources:
        def joinpath(self, name):
            if name in missing:
                return root / "missing-in-this-installation" / name
            return root / name

        def __str__(self):
            return str(root)

    monkeypatch.setattr(prompts, "files", lambda package: Resources())


def test_optimization_works_without_unused_defaults_or_disabled_workflow_prompt(monkeypatch, tmp_path):
    missing = {f"method/default_personas/{i:02}.txt" for i in range(1, 4)}
    missing |= {f"method/prompt_optimizer/default_tips/{i:02}.txt" for i in range(1, 5)}
    missing |= {"method/generator/prompt.txt", "method/services/schema_message.txt"}
    mask_resources(monkeypatch, missing)
    # Reproduce a real full run, checkpoint reload, both DICAI scopes and prompt
    # optimization with configured personas/tips. No model/network calls.
    run_cycle(tmp_path)


def test_all_missing_required_files_are_reported_before_model_calls(monkeypatch, tmp_path):
    missing = {
        "dicai/reasoning/cot.txt", "dicai/abstraction.txt",
        "method/block_selection/prompt.txt", "systems/pubmedqa/problem_solver/prompt.txt",
    }
    mask_resources(monkeypatch, missing)
    provider = OfflineProvider(None)
    with pytest.raises(prompts.PromptResourceError) as error:
        CaisoOptimizer(
            make_system(), [Example(question="1", gd_answer="2")],
            [Example(question="2", gd_answer="4")], {"dataset": "pubmed"}, tmp_path,
            providers=ProviderRegistry({"offline": provider}), embeddings=OfflineEmbeddings(None),
        )
    assert {name + ".txt" for name in error.value.missing} == missing
    assert provider.calls == []
    assert "scripts.verify_prompts" in str(error.value)
    assert str(ROOT / "caiso/prompts") in str(error.value)


def test_default_tips_still_required_when_no_override_exists(monkeypatch):
    mask_resources(monkeypatch, {"method/prompt_optimizer/default_tips/01.txt"})
    current = Configuration(kind=ConfigurationType.PROMPT, value="Original instruction")
    component = Component(
        component_id="transform", implementation="synthetic",
        input_fields={"x": "str"}, output_fields={"answer": "str"}, config=current,
    )
    recorder = Recorder()
    context = SimpleNamespace(
        state=SimpleNamespace(configurations={"transform": current}), config={},
        runtime=SimpleNamespace(providers=recorder),
    )
    with pytest.raises(prompts.PromptResourceError, match="default_tips/01.txt"):
        PromptOptimizer().propose(component, context)
    assert recorder.calls == []


def test_disabled_prompt_search_does_not_load_tips_or_call_models(monkeypatch):
    mask_resources(monkeypatch, {"method/prompt_optimizer/default_tips/01.txt"})
    current = Configuration(kind=ConfigurationType.PROMPT, value="Original instruction")
    component = Component(
        component_id="transform", implementation="synthetic",
        input_fields={"x": "str"}, output_fields={"answer": "str"}, config=current,
    )
    context = SimpleNamespace(
        state=SimpleNamespace(configurations={"transform": current}),
        config={"prompt_candidates": 0},
    )
    assert PromptOptimizer().propose(component, context) == [current]


def test_resource_checker_reports_optional_and_required_missing_files(monkeypatch, capsys):
    missing = {"method/default_personas/01.txt", "dicai/debate/chairperson.txt"}
    mask_resources(monkeypatch, missing)
    with pytest.raises(SystemExit) as error:
        verify_prompts([])
    assert error.value.code == 1
    message = capsys.readouterr().err
    assert all(name in message for name in missing)
    assert "pip install" in message
