"""Dataset persona profiles must reach both DICAI scopes and saved runs intact."""

from copy import deepcopy
from pathlib import Path

import pytest

from caiso.constitution.dicai.personas import Persona, assign_personas
from caiso.integration.config import load_experiment, load_personas
from caiso.optim.optimizer import CaisoOptimizer
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example
from tests.offline import OfflineEmbeddings, OfflineProvider
from tests.test_runtime import make_system

ROOT = Path(__file__).resolve().parents[1]
IDENTITIES = {
    "amazon": [
        "E-commerce behavior analyst",
        "Product matching specialist",
        "Recommendation evaluator",
    ],
    "hotpotqa": [
        "Multi-hop question answering researcher",
        "Information retrieval specialist",
        "Fact verification specialist",
    ],
    "pubmedqa": [
        "Biomedical evidence analyst",
        "Clinical research methodologist",
        "Scientific question answering evaluator",
    ],
}


@pytest.mark.parametrize("dataset", IDENTITIES)
def test_three_dataset_personas_reach_both_constitution_scopes(dataset, tmp_path):
    experiment = load_experiment(ROOT / f"configs/experiments/{dataset}.yaml")
    profiles = experiment["caiso"]["personas"]
    assert [profile["identity"] for profile in profiles] == IDENTITIES[dataset]
    personas = assign_personas(profiles, seed=42)
    assert [persona.strategy for persona in personas] == [
        "reasoning_cot",
        "reasoning_self_consistency",
        "reasoning_self_refine",
    ]
    for persona, profile in zip(personas, profiles, strict=True):
        assert persona.prompt() == profile["prompt_text"]
        assert persona.prompt().startswith(persona.identity + " Expert Persona\nAge: ")
        assert Persona.model_validate(persona.model_dump()).prompt() == profile["prompt_text"]

    registry = ProviderRegistry({"offline": OfflineProvider(None)})
    optimizer = CaisoOptimizer(
        make_system(),
        [Example(question="1", gd_answer="2")],
        [Example(question="2", gd_answer="4")],
        experiment,
        tmp_path,
        providers=registry,
        embeddings=OfflineEmbeddings(None),
    )
    assert optimizer.dicai.identities == profiles
    assert optimizer.block_dicai.identities == profiles
    assert optimizer.experiment["caiso"]["personas"] == profiles


def test_explicit_personas_override_catalog_and_are_copied():
    personas = ["First custom expert", "Second custom expert", "Third custom expert"]
    experiment = {
        "dataset": "amazon",
        "caiso": {
            "persona_catalog": "personas/amazon",
            "personas": personas,
        },
    }
    resolved = load_personas(experiment)
    assert resolved == personas
    resolved[0] = "Changed copy"
    assert experiment["caiso"]["personas"] == personas


def test_resolved_personas_do_not_reload_modified_source_files(monkeypatch):
    experiment = load_experiment(ROOT / "configs/experiments/pubmedqa.yaml")
    expected = deepcopy(experiment["caiso"]["personas"])

    def unexpected_reload(*args):
        raise AssertionError("Resolved runs must use their saved persona text")

    monkeypatch.setattr("caiso.prompts.catalog_entries", unexpected_reload)
    assert load_personas(experiment) == expected


@pytest.mark.parametrize("personas", [[], ["One", "Two"], ["One"] * 4, "abc"])
def test_invalid_persona_counts_fail_before_model_calls(personas):
    with pytest.raises(ValueError, match="exactly three"):
        load_personas({"caiso": {"personas": personas}})


def test_blank_persona_text_is_rejected():
    with pytest.raises(ValueError, match="prompt_text must be a nonempty string"):
        load_personas(
            {
                "caiso": {
                    "personas": [
                        {"identity": "Expert", "prompt_text": " "},
                        "Second",
                        "Third",
                    ]
                }
            }
        )
