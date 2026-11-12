"""Semantic verdicts are scripted; real gates, repairs, caches and budgets run."""

import json
from collections import deque

import pytest

from caiso.constitution.dicai.atomicity import AtomicityError, ensure_atomic
from caiso.constitution.dicai.debate import debate
from caiso.constitution.dicai.personas import assign_personas
from caiso.constitution.dicai.pipeline import DICAI
from caiso.constitution.models import PreferencePair
from caiso.core.budget import BudgetExceeded, BudgetTracker
from caiso.core.content_filter import ContentFilteredError
from caiso.core.recovery import ModelResponseError, TransportExhausted, TransportRequestError
from caiso.constitution.dicai.request_cache import RequestCache
from caiso.method.config import validate_method
from caiso.providers.base import ProviderRegistry
from tests.offline import OfflineEmbeddings, OfflineProvider, atomicity_review_fixture, recorded_candidate
from tests.prompt_requests import PRINCIPLES, Recorder

BUNDLE = "Select the trajectory where hint_generator produces detailed hints and answer_generator gives a concise answer."
PARTS = [PRINCIPLES[2], PRINCIPLES[1]]


def assessment(index, atomic):
    return {"index": index, "atomic": atomic,
            "qualities": ["hint detail"] if atomic else ["hint detail", "answer concision"],
            "reason": "One independently testable property." if atomic else "Detailed hints can accompany a verbose answer."}


def review(*atomic):
    return {"assessments": [assessment(i, value) for i, value in enumerate(atomic, 1)]}


def replacement(index, principles):
    return {"replacements": [{"index": index, "principles": principles}]}


class ScriptedProvider:
    def __init__(self, replies):
        self.replies, self.calls = deque(replies), []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        result = self.replies.popleft()
        if isinstance(result, Exception):
            raise result
        return json.dumps(result)


def registry(replies):
    source = ScriptedProvider(replies)
    providers = ProviderRegistry({"test": source})
    providers.recovery.config = {"response_retries": 0, "backoff_seconds": 0}
    return providers, source


def test_compound_is_split_and_reviewed_while_interaction_principle_is_preserved():
    interaction = "Select the trajectory where hint_generator and answer_generator remain consistent."
    providers, source = registry([review(False, True), replacement(1, PARTS), review(True, True, True)])
    values, report = ensure_atomic([BUNDLE, interaction], providers, "test", 42)
    assert values == [*PARTS, interaction]
    assert report["passed"] is True
    assert report["revisions"] == 1
    assert report["calls_before_retries"] == len(source.calls) == 3
    assert report["last_checked_principles"] == values
    assert "hint detail" in source.calls[1][0][-1]["content"]
    assert "meaning, not punctuation" in source.calls[0][0][0]["content"]


@pytest.mark.parametrize("verdict", [
    review(True),
    {"assessments": [assessment(1, True), assessment(1, True)]},
    {"assessments": [assessment(1, True), assessment(3, True)]},
    {"assessments": [{**assessment(1, True), "atomic": "true"}, assessment(2, True)]},
    {"assessments": [{**assessment(1, True), "index": True}, assessment(2, True)]},
    {"assessments": [{**assessment(1, True), "qualities": ["detail", "concision"]}, assessment(2, True)]},
])
def test_incomplete_or_inconsistent_reviews_cannot_approve_a_list(verdict):
    providers, source = registry([verdict])
    with pytest.raises(AtomicityError):
        ensure_atomic(PARTS, providers, "test", 42)
    assert len(source.calls) == 1


@pytest.mark.parametrize("bad", [replacement(1, [PARTS[0]]), replacement(2, PARTS), replacement(1, [PARTS[0], PARTS[0]])])
def test_repairs_cannot_drop_a_compound_quality_or_edit_an_unflagged_item(bad):
    providers, source = registry([review(False, True), bad])
    with pytest.raises(AtomicityError):
        ensure_atomic([BUNDLE, PRINCIPLES[0]], providers, "test", 42)
    assert len(source.calls) == 2


@pytest.mark.parametrize("limit", [0, 2])
def test_semantic_revisions_are_bounded_and_unverified_results_are_not_returned(limit):
    replies = [review(False)]
    for _ in range(limit):
        replies.extend([replacement(1, [BUNDLE]), review(False)])
    providers, source = registry(replies)
    with pytest.raises(AtomicityError) as error:
        ensure_atomic([BUNDLE], providers, "test", 42, kind="abstraction", config={"atomicity": {"max_revisions": limit}})
    assert len(source.calls) == 1 + 2 * limit
    assert error.value.report["passed"] is False
    assert error.value.report["last_checked_principles"] == [BUNDLE]


@pytest.mark.parametrize("error", [BudgetExceeded("cap"), TransportRequestError("auth"), TransportExhausted("quota"), ContentFilteredError("policy")])
@pytest.mark.parametrize("during_repair", [False, True])
def test_atomicity_never_masks_budget_or_service_errors(error, during_repair):
    providers, _ = registry([review(False), error] if during_repair else [error])
    with pytest.raises(type(error)):
        ensure_atomic([BUNDLE], providers, "test", 42)


def test_disabled_atomicity_makes_no_requests():
    providers, source = registry([])
    values, report = ensure_atomic([BUNDLE], providers, "test", 42, config={"atomicity": {"enabled": False}})
    assert values == [BUNDLE]
    assert report["enabled"] is False
    assert source.calls == []


def test_large_lists_are_checked_in_bounded_batches_with_global_indices():
    principles = [f"Select the trajectory where component_{i} preserves its input." for i in range(21)]
    providers, source = registry([
        {"assessments": [assessment(i, True) for i in range(1, 21)]},
        {"assessments": [assessment(21, True)]},
    ])
    values, report = ensure_atomic(principles, providers, "test", 42)
    assert values == principles
    assert len(source.calls) == report["calls_before_retries"] == 2
    assert len(report["last_review"]["assessments"]) == 21
    assert '"index": 21' in source.calls[1][0][-1]["content"]


def test_revision_batches_cannot_shift_indices_or_overwrite_other_repairs():
    providers, _ = registry([
        {"assessments": [assessment(1, False)]},
        {"assessments": [assessment(2, False)]},
        replacement(1, [PARTS[0]]), replacement(2, [PARTS[1]]),
        {"assessments": [assessment(1, True)]},
        {"assessments": [assessment(2, True)]},
    ])
    values, report = ensure_atomic([BUNDLE, BUNDLE], providers, "test", 42, kind="abstraction", config={"atomicity": {"batch_size": 1}})
    assert values == PARTS
    assert report["passed"] is True
    assert report["calls_before_retries"] == 6


def test_malformed_review_retries_with_feedback_and_complete_coverage():
    providers, source = registry([review(True), review(True, True)])
    providers.recovery.config["response_retries"] = 1
    values, report = ensure_atomic(PARTS, providers, "test", 42)
    assert values == PARTS
    assert report["calls_before_retries"] == 1
    assert len(source.calls) == 2
    assert "exactly once" in source.calls[1][0][-1]["content"]


def test_semantic_check_and_repair_resume_without_repeating_a_successful_review(tmp_path):
    providers, source = registry([
        review(False), BudgetExceeded("interrupted before revision"),
        replacement(1, PARTS), review(True, True),
    ])
    cache = RequestCache(providers, tmp_path)
    with pytest.raises(BudgetExceeded):
        ensure_atomic([BUNDLE], cache, "test", 42)
    resumed = RequestCache(providers, tmp_path)
    values, report = ensure_atomic([BUNDLE], resumed, "test", 42)
    assert values == PARTS
    assert report["passed"] is True
    assert resumed.hits == 1
    assert len(source.calls) == 4


class DebateRecorder(Recorder):
    def __init__(self, chairs, reviews, revisions=()):
        super().__init__()
        self.chairs, self.reviews, self.revisions = map(deque, (chairs, reviews, revisions))

    def complete(self, role, messages, *, seed, schema=None, validate=None, **kwargs):
        if schema and schema.__name__ in {"AtomicityReview", "PrincipleRevisions"}:
            output = (self.reviews if schema.__name__ == "AtomicityReview" else self.revisions).popleft()
            result = schema.model_validate(output)
        elif "You are the Moderator/Judge." in messages[0]["content"]:
            result = self.chairs.popleft()
        else:
            return super().complete(role, messages, seed=seed, schema=schema, validate=validate, **kwargs)
        return validate(result) if validate else result


def pair_and_catalog():
    components = ["answer_generator", "hint_generator", "transform"]
    pair = PreferencePair(pair_id="atomicity-pair", preferred="A", context={},
                          candidate_a=recorded_candidate("A", components), candidate_b=recorded_candidate("B", components))
    catalog = [{"persona": p.model_dump(), "reasoning": "Evidence", "principles": PRINCIPLES} for p in assign_personas(["One", "Two", "Three"], 42)]
    return pair, catalog


def test_debate_reconstructs_the_repaired_list_before_finalizing():
    provider = DebateRecorder([
        "FINAL LIST\n- Q1: " + BUNDLE + "\n- Q2: " + PRINCIPLES[0] + "\nDEBATE_OVER"
    ], [review(False, True), review(True, True, True)], [replacement(1, PARTS)])
    pair, catalog = pair_and_catalog()
    values, history = debate(pair, catalog, provider, "test", 42, rounds=1)
    assert values == [*PARTS, PRINCIPLES[0]]
    assert history[-1]["finalized"] is True
    audited = next(row for row in history if row["speaker"] == "chairperson_atomicity")
    assert audited["accepted_principles"] == values
    final_judge = [call for call in provider.calls if "You are a preference judge." in call["messages"][0]["content"]][-1]
    assert BUNDLE not in final_judge["messages"][-1]["content"]
    assert all(p in final_judge["messages"][-1]["content"] for p in values)


def test_failed_later_semantic_check_retains_only_the_previous_verified_list():
    provider = DebateRecorder([
        "FINAL LIST\n" + "\n".join("- " + p for p in PRINCIPLES) + "\nDEBATE_OVER",
        "FINAL LIST\n- " + BUNDLE + "\nDEBATE_OVER",
    ], [review(True, True, True), review(False)])
    pair, catalog = pair_and_catalog()
    values, history = debate(pair, catalog, provider, "test", 42, rounds=2, config={"min_debate_rounds": 2, "atomicity": {"max_revisions": 0}})
    assert values == PRINCIPLES
    assert history[-1]["finalized"] is False
    rejected = next(row for row in history if row["speaker"] == "chairperson_atomicity_error")
    assert rejected["retained_verified_list"] is True


def test_no_unverified_seed_or_compound_list_is_reported_as_settled():
    provider = DebateRecorder(["- " + BUNDLE], [review(False)])
    pair, catalog = pair_and_catalog()
    with pytest.raises(ModelResponseError, match="No constitution was fabricated"):
        debate(pair, catalog, provider, "test", 42, rounds=1, config={"atomicity": {"max_revisions": 0}})


@pytest.mark.parametrize("repair", [True, False])
@pytest.mark.parametrize("scope", ["system", "block:transform"])
def test_induction_checks_abstracted_outputs_and_uses_verified_medoids_on_failure(tmp_path, repair, scope):
    compound = "Select the trajectory where transform computes correctly and sink explains concisely."

    class Provider(OfflineProvider):
        def complete(self, messages, *, seed, schema=None, **kwargs):
            if schema and schema.__name__ == "AtomicityReview":
                output = atomicity_review_fixture(messages)
                entries = json.loads(messages[-1]["content"].split("PRINCIPLES:\n", 1)[1])
                for item in entries:
                    if item["principle"] == compound:
                        output["assessments"][item["index"] - 1] = assessment(item["index"], False)
            elif schema and schema.__name__ == "PrincipleRevisions":
                issues = json.JSONDecoder().raw_decode(messages[-1]["content"].split("COMPOUND ITEMS AND INDEPENDENT QUALITIES:\n", 1)[1])[0]
                output = {"replacements": [{"index": issue["index"], "principles": [
                    "Select the trajectory where transform computes correctly." if repair else compound
                ]} for issue in issues]}
            elif "You abstract a cluster of preference principles" in messages[0]["content"]:
                output = compound
            else:
                return super().complete(messages, seed=seed, schema=schema, **kwargs)
            self.budget.call()
            self.calls.append((messages, seed, schema))
            return json.dumps(output) if not isinstance(output, str) else output

    budget = BudgetTracker(3)
    source = Provider(budget)
    providers = ProviderRegistry({"test": source})
    induction = DICAI(providers, OfflineEmbeddings(budget), ["One", "Two", "Three"], {
        "max_workers": 1, "reasoning_workers": 1, "debate_rounds": 1,
        "atomicity": {"max_revisions": 1}, "clustering": {"min_clusters": 2, "max_clusters": 2},
    }, role="test")
    pair, _ = pair_and_catalog()
    constitution = induction.induce([pair], 42, scope=scope, output_dir=tmp_path)
    assert compound not in constitution.principles
    audit = constitution.metadata["abstraction_atomicity"]
    assert audit["output_verified"] is True
    assert audit["calls_before_retries"] == 3
    if repair:
        assert audit["passed"] is True
    else:
        assert audit["fallback"] == "verified_cluster_medoids"
        assert audit["fallback_cluster_indices"]
        assert all(p in constitution.settled_principles for p in constitution.principles)
    assert len(constitution.cluster_ids) == len(constitution.settled_principles)
    count = len(source.calls)
    resumed = induction.induce([pair], 42, scope=scope, output_dir=tmp_path)
    assert resumed.principles == constitution.principles
    assert len(source.calls) == count


@pytest.mark.parametrize("settings", [{"enabled": "true"}, {"enabled": None}, {"max_revisions": -1}, {"max_revisions": True}, {"max_revisions": 1.5}, {"batch_size": 0}, {"batch_size": 1.5}])
def test_invalid_atomicity_settings_fail_before_requests(settings):
    with pytest.raises(ValueError, match="atomicity"):
        validate_method({"dicai": {"atomicity": settings}})
