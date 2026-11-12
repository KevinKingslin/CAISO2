"""Component-subject format and source grounding cannot be bypassed by stages."""

import json

import pytest

from caiso.constitution.dicai import pipeline
from caiso.constitution.dicai.abstraction import abstract_clusters
from caiso.constitution.dicai.atomicity import ensure_atomic
from caiso.constitution.dicai.parsing import normalize_principle, parse_principles
from caiso.constitution.dicai.personas import assign_personas
from caiso.constitution.dicai.reasoning_assembly import assemble
from caiso.constitution.models import PreferencePair
from caiso.core.budget import BudgetTracker
from caiso.core.recovery import ModelResponseError
from caiso.providers.base import ProviderRegistry
from tests.offline import OfflineEmbeddings, OfflineProvider, recorded_candidate
from tests.test_model_response_recovery import SequenceProvider, registry

IDS = ["answer_generator", "hint_generator", "encoder.layer", "search-v2"]
ONE = "Select the trajectory where answer_generator uses evidence from hint_generator."
TWO = "Select the trajectory where hint_generator preserves evidence."


def pair():
    return PreferencePair(pair_id="format", preferred="A", context={},
                          candidate_a=recorded_candidate("A", IDS), candidate_b=recorded_candidate("B", IDS))


@pytest.mark.parametrize("text", [
    "Select the trajectory that answer_generator uses evidence.",
    "Select the trajectory which answer_generator uses evidence.",
    "Select the trajectory with answer_generator evidence.",
    "Select the trajectory where the final answer is grounded.",
    "Select the trajectory where missing_component uses evidence.",
    "Select the trajectory where <component_id> uses evidence.",
    "Select the trajectory where answer_generator <condition>.",
    "Select the trajectory where answer_generator <criterion>.",
    "Select the trajectory where answer_generator",
    "Select the trajectory where answer_generator ...",
    "Select the trajectory where answer_generator …",
    "Select the trajectory where answer_generator .",
    "Select the trajectory where answer_generator, hint_generator agree.",
    "Select the trajectory where Answer_Generator uses evidence.",
])
def test_invalid_subject_format_is_rejected_in_text_and_json(text):
    with pytest.raises(ModelResponseError):
        normalize_principle(text, IDS)
    # A valid first item must not hide an invalid second item.
    for reply in (f"FINAL LIST:\n- Q1: {ONE}\n- Q2: {text}", json.dumps({"principles": [ONE, text]})):
        with pytest.raises(ModelResponseError):
            parse_principles(reply, IDS)


@pytest.mark.parametrize("subject", IDS)
def test_exact_component_ids_and_interaction_conditions_are_preserved(subject):
    text = f"Select the trajectory where {subject} uses evidence from hint_generator."
    assert normalize_principle(f"- **Q2:** “{text}”", IDS) == text
    assert normalize_principle(text.upper().replace(subject.upper(), subject).replace("HINT_GENERATOR", "hint_generator"), IDS).startswith(f"Select the trajectory where {subject} ")


@pytest.mark.parametrize("bad", [
    {"dimensions": ["evidence support"]},
    {"dimensions": ["evidence support"], "questions": []},
    {"questions": ["Select the trajectory where missing_component preserves evidence."]},
])
def test_reasoning_retries_missing_questions_and_unknown_subjects(bad):
    good = json.dumps({"reasoning": "Evidence matters.", "questions": [ONE], "dimensions": ["evidence support"]})
    source = SequenceProvider([json.dumps({"reasoning": "Evidence matters.", **bad}), *([good] * 4)])
    catalog = assemble(pair(), assign_personas(["One", "Two", "Three"], 42), registry(source), "test", 42, {
        "reasoning_workers": 1, "self_consistency_samples": 1, "self_refine_steps": 0,
    })
    assert [row["principles"] for row in catalog] == [[ONE]] * 3
    assert len(source.calls) == 5
    assert "ALLOWED COMPONENT IDS" in source.calls[0][0][-1]["content"]
    assert "Previous unusable" in source.calls[1][0][-1]["content"] or len(source.calls[1][0]) > 2


def test_abstraction_retries_a_subject_outside_its_source_cluster():
    source = SequenceProvider([TWO, ONE])
    assert abstract_clusters([ONE], [0], registry(source), "test", 42) == [ONE]
    assert len(source.calls) == 2
    # Without an external inventory, only source subjects establish allowed IDs.
    assert '"answer_generator"' in source.calls[0][0][-1]["content"]


def test_cluster_repair_cannot_switch_to_an_unrelated_component():
    review_bad = {"assessments": [{"index": 1, "atomic": False, "qualities": ["support", "clarity"], "reason": "Independent qualities."}]}
    review_good = {"assessments": [{"index": 1, "atomic": True, "qualities": ["support"], "reason": "One quality."}]}
    def replace(text):
        return {"replacements": [{"index": 1, "principles": [text]}]}
    unrelated = "Select the trajectory where search-v2 retrieves evidence."
    source = SequenceProvider([json.dumps(v) for v in (review_bad, replace(unrelated), replace(ONE), review_good)])
    values, report = ensure_atomic([ONE], registry(source), "test", 42, kind="abstraction",
                                  component_ids=IDS, context=[{"index": 1, "principles": [ONE]}])
    assert values == [ONE]
    assert report["passed"]
    assert len(source.calls) == 4


def test_final_constitution_rejects_invalid_coverage_restoration(monkeypatch):
    budget = BudgetTracker(1)
    provider = OfflineProvider(budget)
    dicai = pipeline.DICAI(ProviderRegistry({"test": provider}), OfflineEmbeddings(budget), ["One", "Two", "Three"], {
        "max_workers": 1, "reasoning_workers": 1, "debate_rounds": 1,
        "self_consistency_samples": 1, "self_refine_steps": 0,
        "atomicity": {"enabled": False}, "clustering": {"fixed_clusters": 2},
    }, role="test")
    comparison = PreferencePair(pair_id="final", preferred="A", context={},
                                candidate_a=recorded_candidate("A", ["transform"]), candidate_b=recorded_candidate("B", ["transform"]))
    monkeypatch.setattr(pipeline, "preserve_component_coverage", lambda *a, **kw: (
        ["Select the trajectory where invented_component computes correctly."], {}
    ))
    with pytest.raises(ModelResponseError, match="invented_component"):
        dicai.induce([comparison], 42)
    count = len(provider.calls)
    comparison.candidate_a = {"text": "A"}
    comparison.candidate_b = {"text": "B"}
    with pytest.raises(ValueError, match="recorded trajectory component IDs"):
        dicai.induce([comparison], 42)
    assert len(provider.calls) == count
