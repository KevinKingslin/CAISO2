"""Training score evidence stays separate from blind, regret-aware selection."""

from copy import deepcopy
import json

import pytest

from caiso.constitution.dicai import reconstruction
from caiso.constitution.dicai.debate import debate
from caiso.constitution.dicai.personas import assign_personas
from caiso.constitution.dicai.pipeline import DICAI
from caiso.constitution.dicai.preferences import candidate_views, score_feedback, task_view
from caiso.constitution.dicai.reasoning_assembly import assemble
from caiso.constitution.models import PreferencePair
from caiso.core.budget import BudgetTracker
from caiso.core.content_filter import ContentFilteredError
from caiso.integration.config import load_method
from caiso.method.config import validate_method
from caiso.providers.base import ProviderRegistry
from tests.offline import OfflineEmbeddings, OfflineProvider
from tests.prompt_requests import PRINCIPLES, Recorder


def pair(scores=(0.2, 0.8), preferred="B", pair_id="scores"):
    return PreferencePair(
        pair_id=pair_id, preferred=preferred,
        context={"question": "Q", "gold": "REFERENCE_ONLY", "score_feedback": "HIDDEN"},
        candidate_a={"trajectory": {"final_output": {"answer": "A"}, "reward": scores[0]}},
        candidate_b={"trajectory": {"final_output": {"answer": "B"}, "reward": scores[1]}},
        metadata={"global_scores": list(scores), "target": "REFERENCE_ONLY"},
    )


@pytest.mark.parametrize("scores,preferred,relation", [
    ((0.2, 0.8), "B", "B_better"),
    ((0.8, 0.2), "B", "A_better"),  # A/B order is independent of the label.
    ((0.0, 0.0), "A", "tie"),
    ((1.0, 1.0), "B", "tie"),
])
def test_feedback_preserves_score_order_labels_and_raw_records(scores, preferred, relation):
    comparison = pair(scores, preferred)
    before = deepcopy(comparison)
    feedback = score_feedback(comparison, "answer_f1")
    assert feedback == {
        "metric": "answer_f1", "score_a": scores[0], "score_b": scores[1],
        "delta_a_minus_b": scores[0] - scores[1], "relation": relation,
    }
    assert comparison == before
    assert "REFERENCE_ONLY" not in json.dumps(feedback)
    blind = json.dumps([task_view(comparison.context), *candidate_views(comparison)])
    assert all(value not in blind for value in ("REFERENCE_ONLY", "HIDDEN", '"reward"', '"global_scores"'))
    comparison.metadata.clear()
    assert score_feedback(comparison) is None


@pytest.mark.parametrize("scores", [None, [], [0.2], [0, 1, 2], "0,1", [True, 1], ["0", 1], [float("nan"), 1], [0, float("inf")]])
def test_invalid_scores_are_rejected(scores):
    comparison = pair()
    comparison.metadata["global_scores"] = scores
    with pytest.raises(ValueError, match="two finite A/B scores"):
        score_feedback(comparison)


def test_metric_name_is_resolved_from_task_configuration_and_can_be_overridden():
    assert load_method({"dataset": "hotpotqa"})["dicai"]["score_metric"] == "answer_f1"
    assert load_method({"dataset": "amazon"})["dicai"]["score_metric"] == "task_metric"
    config = {"dataset": "hotpotqa", "caiso": {"method": {"dicai": {"score_metric": "joint_f1"}}}}
    assert load_method(config)["dicai"]["score_metric"] == "joint_f1"


@pytest.mark.parametrize("metric", [None, "", " ", 1, True, []])
def test_metric_names_must_be_nonempty_strings(metric):
    with pytest.raises(ValueError, match="dicai.score_metric"):
        validate_method({"dicai": {"score_metric": metric}})


@pytest.mark.parametrize("preferred,minimum,variant", [
    ("A", 3, ""), ("B", 3, "_reconstruction_failed"), ("A", 4, "_too_few"),
])
def test_all_reasoning_and_debate_variants_receive_feedback(preferred, minimum, variant):
    comparison = pair((0.5, 0.5), preferred)
    before = deepcopy(comparison)
    recorder = Recorder()
    config = {
        "score_metric": "answer_f1", "reasoning_workers": 1,
        "self_consistency_samples": 2, "self_refine_steps": 1,
        "min_principles": minimum, "atomicity": {"enabled": False},
    }
    catalog = assemble(comparison, assign_personas(["One", "Two", "Three"], 42), recorder, "audit", 42, config)
    reasoning_calls = list(recorder.calls)
    assert len(reasoning_calls) == 7  # CoT, two samples + consensus, draft + critique + revision.
    recorder.calls.clear()
    _, history = debate(comparison, catalog, recorder, "audit", 42, rounds=2, config=config)
    induction = reasoning_calls + [c for c in recorder.calls if "You are a preference judge." not in c["messages"][0]["content"]]
    for call in induction:
        messages = call["messages"]
        feedback = json.loads(messages[-1]["content"].split("TRAINING TASK FEEDBACK (A/B order, training only):\n")[1])
        assert feedback == score_feedback(comparison, "answer_f1")
        assert "no measured task-metric advantage" in messages[0]["content"]
        assert "without reference answers, scores or preference labels" in messages[0]["content"]
        assert "REFERENCE_ONLY" not in json.dumps(messages)
    if variant:
        assert len([h for h in history if h["speaker"] == "chairperson"]) == 2
        assert "DEBATE_OVER REJECTED" in json.dumps(recorder.calls)
    assert history[-1]["reconstructed"] == (preferred == "A")  # Random-tie label gate is unchanged.
    assert comparison == before


@pytest.mark.parametrize("seed,prediction,selected,regret", [(0, "A", 0.2, 0.6), (1, "B", 0.8, 0.0)])
def test_regret_uses_mapped_choices_after_blind_shuffled_inference(seed, prediction, selected, regret):
    comparison = pair()
    recorder = Recorder()
    choice, _ = reconstruction.apply_constitution(recorder, "audit", comparison, PRINCIPLES, seed=seed)
    assert choice == prediction
    result = reconstruction._score_selection(comparison, choice, "answer_f1")
    assert result["selected_score"] == selected
    assert result["regret"] == pytest.approx(regret)
    assert result["oracle_score"] == 0.8
    messages = json.dumps(recorder.calls[0]["messages"])
    assert all(key not in messages for key in ("score_feedback", "score_a", "score_b", "answer_f1", "global_scores", "REFERENCE_ONLY", '"reward"'))


def test_validation_distinguishes_score_loss_ties_and_unscored_pairs(monkeypatch):
    pairs = [pair((0.8, 0.79), "A", "small"), pair((1, 0), "A", "large"), pair((0.5, 0.5), "B", "tie"), pair(pair_id="unscored")]
    pairs[-1].metadata.clear()
    monkeypatch.setattr(reconstruction, "apply_constitution", lambda *args: ("A", {"predicted": "A"}))
    report = reconstruction.validate_constitution(None, "audit", pairs, PRINCIPLES, score_metric="answer_f1")
    assert report["n"] == 4
    assert report["accuracy"] == 0.5  # Original labels, including ties, remain in this metric.
    assert report["non_tied_accuracy"] == 1.0
    assert report["non_tied_n"] == 2
    assert report["tied_pairs"] == 1
    assert report["scored_pairs"] == report["scored_selections"] == 3
    assert report["mean_selected_score"] == pytest.approx(2.3 / 3)
    assert report["mean_regret"] == 0
    assert report["results"][2]["regret"] == 0
    monkeypatch.setattr(reconstruction, "apply_constitution", lambda *args: ("B", {"predicted": "B"}))
    worse = reconstruction.validate_constitution(None, "audit", pairs, PRINCIPLES)
    assert worse["accuracy"] == report["accuracy"]
    assert worse["mean_regret"] == pytest.approx(1.01 / 3)


def test_no_selection_is_not_reported_as_zero_regret():
    report = reconstruction.validate_constitution(None, "audit", [pair()], [])
    assert report["scored_pairs"] == 1
    assert report["scored_selections"] == 0
    assert report["mean_regret"] is None
    assert report["mean_selected_score"] is None
    assert report["results"][0]["regret"] is None
    empty = reconstruction.validate_constitution(None, "audit", [], [])
    assert empty["non_tied_accuracy"] is None
    assert empty["mean_regret"] is None


@pytest.mark.parametrize("scored,strict,expected", [
    (True, False, ["large_rule"]), (True, True, ["large_rule", "small_rule"]),
    (False, False, ["small_rule"]),
])
def test_minimization_preserves_regret_instead_of_equal_accuracy(monkeypatch, scored, strict, expected):
    pairs = [pair((0.8, 0.79), "A", "small"), pair((1, 0), "A", "large")]
    if not scored:
        for comparison in pairs:
            comparison.metadata.clear()

    def choose(providers, role, comparison, principles, seed, temperature):
        protect_large = "large_rule" in principles
        selected = "A" if (comparison.pair_id == "large") == protect_large else "B"
        return selected, {"predicted": selected}

    monkeypatch.setattr(reconstruction, "apply_constitution", choose)
    kept, history = reconstruction.minimize_constitution(None, "audit", pairs, ["large_rule", "small_rule"], strict=strict)
    assert kept == expected
    assert history[0]["accuracy"] == 0.5
    assert history[0]["objective"] == ("mean_regret" if scored else "accuracy")
    assert history[0]["dropped"] == (not scored)
    if scored:
        assert history[0]["mean_regret"] == 0.5


@pytest.mark.parametrize("missing", ["empty", "content_filter"])
def test_minimization_does_not_improve_regret_by_losing_a_selection(monkeypatch, missing):
    pairs = [pair((0.8, 0.79), "A", "small"), pair((1, 0), "A", "large")]

    def choose(providers, role, comparison, principles, seed, temperature):
        if missing == "content_filter" and comparison.pair_id == "large" and "guard" not in principles:
            raise ContentFilteredError("Blocked trial")
        selected = "A" if comparison.pair_id == "small" else "B" if "guard" in principles else ""
        return selected, {"predicted": selected}

    monkeypatch.setattr(reconstruction, "apply_constitution", choose)
    kept, history = reconstruction.minimize_constitution(None, "audit", pairs, ["guard", "other"])
    assert kept == ["guard"]
    assert history[0]["mean_regret"] == 0
    assert history[0]["scored_selections"] == 1
    assert history[0]["dropped"] is False


def test_real_induction_retains_ties_and_saves_score_diagnostics(tmp_path):
    budget = BudgetTracker(1)
    provider = OfflineProvider(budget)
    dicai = DICAI(ProviderRegistry({"test": provider}), OfflineEmbeddings(budget), ["One", "Two", "Three"], {
        "score_metric": "answer_f1", "max_workers": 1, "reasoning_workers": 1,
        "debate_rounds": 1, "self_consistency_samples": 1, "self_refine_steps": 0,
        "atomicity": {"enabled": False}, "clustering": {"fixed_clusters": 2},
        "minimization": {"enabled": True},
    }, role="test")
    comparisons = [pair(pair_id="unequal"), pair((0.5, 0.5), "B", "tie")]
    before = deepcopy(comparisons)
    constitution = dicai.induce(comparisons, 42, output_dir=tmp_path)
    assert constitution.source_preference_pair_ids == ["unequal", "tie"]
    report = constitution.metadata["reconstruction"]
    assert report["score_metric"] == "answer_f1"
    assert report["scored_pairs"] == report["scored_selections"] == 2
    assert report["tied_pairs"] == 1
    assert report["mean_regret"] == pytest.approx(0.3)
    assert all(row["objective"] == "mean_regret" for row in constitution.metadata["minimization"])
    assert comparisons == before
    path = tmp_path / constitution.metadata["provenance"]
    stored = json.loads((path / "constitution.json").read_text())
    assert stored["metadata"]["reconstruction"] == report
    assert all('"score_feedback"' not in json.dumps(messages) for messages, _, _ in provider.calls if "You are a preference judge." in messages[0]["content"])
    count = len(provider.calls)
    resumed = dicai.induce(comparisons, 42, output_dir=tmp_path)
    assert resumed.principles == constitution.principles
    assert resumed.metadata["reconstruction"] == report
    assert len(provider.calls) == count
    comparisons[0].metadata["global_scores"] = [float("nan"), 1]
    with pytest.raises(ValueError, match="two finite A/B scores"):
        dicai.induce(comparisons, 42, output_dir=tmp_path)
    assert len(provider.calls) == count
