"""Realistic model formatting, failed turns and partial induction restarts."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from caiso.constitution.dicai.abstraction import abstract_clusters
from caiso.constitution.dicai.debate import debate, parse_principles
from caiso.constitution.dicai.personas import assign_personas
from caiso.constitution.dicai.parsing import normalize_principle
from caiso.constitution.dicai.pipeline import DICAI
from caiso.constitution.dicai.reconstruction import parse_choice
from caiso.constitution.dicai.request_cache import RequestCache
from caiso.constitution.dicai.reasoning_assembly import Reasoning, assemble
from caiso.constitution.models import PreferencePair
from caiso.core.budget import BudgetExceeded
from caiso.core.recovery import ModelResponseError, RetryExhausted, TransportExhausted, TransportRequestError
from caiso.integration.services import MethodProvider
from caiso.optimizers.prompt_optimizer import PromptProposal
from caiso.providers.base import ProviderRegistry
from caiso.providers.responses import json_object
from tests.offline import OfflineEmbeddings, recorded_candidate

ONE = "Select the trajectory where info_extractor preserves evidence."
TWO = "Select the trajectory where answer_generator answers the question."


@pytest.mark.parametrize("answer", [
    f"FINAL LIST:\n- Q1: {ONE}\n- Q2: {TWO}\nDEBATE_OVER",
    f"**FINAL LIST:**\n- **Q1:** {ONE}\n- **Q2:** {TWO}\nDEBATE_OVER",
    f"### FINAL LIST\n1. **{ONE}**\n2. **{TWO}**",
    f"WORKING LIST:\n- Q1: Select the trajectory where the earlier criterion is superseded.\n\n**FINAL LIST (Cycle 3):**\n- Q1: {ONE}\n- Q2: {TWO}",
    f"UPDATED WORKING LIST:\n• {ONE}\n• {TWO}",
    f"FINAL LIST:\nQ1 — {ONE}\nQ2 — {TWO}",
    f"FINAL LIST:\nQ1:\n{ONE}\nQ2:\n{TWO}",
    f"```markdown\nFINAL LIST:\n- Q1: {ONE}\n- Q2: {TWO}\n```",
    f"FINAL LIST:\n- Q1: {ONE}\n- Q2: {TWO}\n\nThe final list is stable.\nDEBATE_OVER",
    json.dumps({"principles": [ONE, TWO]}),
    json.dumps({"final_list": {"Q1": ONE, "Q2": TWO}}),
    json.dumps([{"select_prompt": ONE}, {"select_prompt": TWO}]),
    f"| Question | Principle |\n|---|---|\n| Q1 | {ONE} |\n| Q2 | {TWO} |",
])
def test_chairperson_format_variants_preserve_criteria(answer):
    assert parse_principles(answer) == [ONE, TWO]


def test_last_published_list_and_multiline_principles():
    answer = (
        f"WORKING LIST:\n- Q1: {TWO}\n\nFINAL LIST:\n"
        "- **Q1:** Select the trajectory where info_extractor preserves\n"
        "  the API_key and user_id values.\nDEBATE_OVER"
    )
    assert parse_principles(answer) == ["Select the trajectory where info_extractor preserves the API_key and user_id values."]


@pytest.mark.parametrize("answer", [
    "DEBATE_OVER", "There are no principles.", "FINAL LIST:\n- Q1: Prefer good answers.",
    'FINAL LIST:\n- Q1: Select the trajectory where <condition>.',
    '{"principles": []}', '{"principles": [42]}', '{"principles": ["Select the trajectory where',
    "FINAL LIST:\n- Q1: Select the trajectory where",
])
def test_invalid_principles_are_not_fabricated(answer):
    with pytest.raises(ModelResponseError):
        parse_principles(answer)


@pytest.mark.parametrize("principle", [
    "Select the response that answer_generator is concise.",
    "Select the trajectory",
    "Select the trajectory where",
    "Select the trajectory that",
    "Select the trajectory which",
    "Select the trajectory with",
    "Select the trajectory whose",
    "Select the trajectory in which",
    "Select the trajectory A",
    "Select the trajectory B.",
    "Select the trajectory where ...",
    "Select the trajectory that <condition>.",
    "Select the trajectory which <actual criterion>.",
    "Select the trajectory where answer_generator is concise. Select the trajectory where hint_generator is detailed.",
])
def test_trajectory_prefix_requires_one_actual_criterion(principle):
    with pytest.raises(ModelResponseError):
        normalize_principle(principle)


def test_trajectory_normalization_preserves_component_identifiers():
    assert normalize_principle(
        "- **Q2:** SELECT  THE trajectory WHERE hint_generator passes detailed hints to answer_generator."
    ) == "Select the trajectory where hint_generator passes detailed hints to answer_generator."


@pytest.mark.parametrize("original", [
    "Select the trajectory where answer_generator uses evidence from info_extractor.",
    "Select the trajectory that uses info_extractor's evidence in answer_generator's answer.",
    "Select the trajectory which uses info_extractor's evidence in answer_generator's answer.",
    "Select the trajectory with an answer_generator output supported by info_extractor's evidence.",
    "Select the trajectory whose answer_generator uses evidence from info_extractor.",
    "Select the trajectory in which answer_generator uses evidence from info_extractor.",
])
def test_only_component_subject_format_is_accepted(original):
    if original.startswith("Select the trajectory where answer_generator "):
        assert normalize_principle(original) == original
    else:
        with pytest.raises(ModelResponseError):
            normalize_principle(original)
        with pytest.raises(ModelResponseError):
            parse_principles(f"| Question | Principle |\n|---|---|\n| Q1 | {original} |")


def test_reported_seed_questions_retry_invalid_format():
    questions = [
        "Select the trajectory where the retrieved evidence explicitly confirms the actress's involvement.",
        "Select the trajectory that ensures the final answer is supported by accurate and verifiable information.",
    ]
    valid = [ONE, TWO]
    def response(values):
        return json.dumps({"reasoning": "The preferred trajectory uses supporting evidence.", "dimensions": ["evidence support"], "questions": values})
    provider = SequenceProvider([response(questions), response(valid)])
    models = registry(provider)
    events = []
    models.recovery.on_error = events.append
    result = models.complete("test", [{"role": "user", "content": "Draft candidate questions."}], seed=7143, schema=Reasoning)
    assert [q.select_prompt for q in result.questions] == valid
    assert len(provider.calls) == 2
    assert events
    with pytest.raises(ModelResponseError):
        parse_principles("FINAL LIST:\n" + "\n".join(f"- Q{i}: {p}" for i, p in enumerate(questions, 1)))


@pytest.mark.parametrize("answer", [
    '{"prompt":"Keep API_key intact."}',
    '```json\n{"prompt":"Keep API_key intact."}\n```',
    'Here is the result:\n{"prompt":"Keep API_key intact."}\nThank you.',
])
def test_json_wrappers_preserve_values(answer):
    assert json_object(answer) == {"prompt": "Keep API_key intact."}


@pytest.mark.parametrize("answer", [
    '{"prompt":"truncated', '[]', 'null', '{"prompt":"a"} {"prompt":"b"}',
    '{"outer": {"prompt":"do not salvage this nested object"}',
    '{"prompt":"a", "prompt":"b"}', '{"score": NaN}', '{"score": Infinity}',
])
def test_ambiguous_or_truncated_json_is_rejected(answer):
    with pytest.raises(ModelResponseError):
        json_object(answer)


class SequenceProvider:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def registry(provider, retries=2):
    result = ProviderRegistry({"test": provider})
    result.recovery.config = {"response_retries": retries, "backoff_seconds": 0}
    return result


def test_repair_includes_bad_answer_and_exact_schema_error_in_logs():
    provider = SequenceProvider(['{"wrong_field":"bad"}', '{"prompt":"Correct"}'])
    models = registry(provider)
    events = []
    models.recovery.on_error = events.append
    messages = [{"role": "user", "content": "Return the instruction."}]
    result = models.complete("test", messages, seed=42, schema=PromptProposal, response_context={"pair_id": "p", "stage": "chairperson"})
    assert result.prompt == "Correct"
    retry = provider.calls[1][0]
    assert retry[-2] == {"role": "assistant", "content": '{"wrong_field":"bad"}'}
    assert "prompt" in retry[-1]["content"]
    assert events[0]["raw_response"] == '{"wrong_field":"bad"}'
    assert events[0]["response_context"]["pair_id"] == "p"
    assert provider.calls[0][1]["seed"] != provider.calls[1][1]["seed"]


def test_method_transport_preserves_wire_json_mode_and_accepts_fences():
    replies = ['```json\n{"prompt":"Valid fenced response"}\n```', '{"prompt":"fallback"}']
    # Exercise the real helper; method parsing stays inside the retry boundary.
    def completion(**kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies.pop(0)))])
    models = registry(MethodProvider("synthetic", None))
    with patch("caiso.utils.api.litellm.completion", completion):
        result = models.complete("test", [{"role": "user", "content": "Return JSON."}], seed=42, schema=PromptProposal)
    assert result.prompt == "Valid fenced response"
    assert len(replies) == 1


def test_invalid_api_json_repairs_instead_of_escaping_retry_loop():
    replies = ['{"prompt":', '{"prompt":"Repaired"}']
    seen = []
    def completion(**kwargs):
        seen.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies.pop(0)))])
    models = registry(MethodProvider("synthetic", None))
    with patch("caiso.utils.api.litellm.completion", completion):
        result = models.complete("test", [{"role": "user", "content": "Return JSON."}], seed=42, schema=PromptProposal)
    assert result.prompt == "Repaired"
    assert any(m == {"role": "assistant", "content": '{"prompt":'} for m in seen[1])


@pytest.mark.parametrize("error", [BudgetExceeded("cap"), TransportRequestError("deployment"), TransportExhausted("quota"), TypeError("programming bug")])
def test_non_response_errors_are_not_hidden_or_retried(error):
    provider = SequenceProvider([error, '{"prompt":"must not be used"}'])
    with pytest.raises(type(error)):
        registry(provider).complete("test", [{"role": "user", "content": "Request"}], seed=42, schema=PromptProposal)
    assert len(provider.calls) == 1


def pair():
    components = ["info_extractor", "answer_generator", "hint_generator", "transform"]
    return PreferencePair(pair_id="pair-1", preferred="A", context={"question": "Q"},
                          candidate_a=recorded_candidate("A", components), candidate_b=recorded_candidate("B", components))


def catalog():
    return [{"persona": p.model_dump(), "reasoning": "Rationale", "principles": []} for p in assign_personas(["One", "Two", "Three"], 42)]


class DebateProvider:
    def __init__(self, chairs):
        self.chairs = iter(chairs)
    def complete(self, role, messages, *, validate=None, **kwargs):
        if kwargs.get("schema") is not None and kwargs["schema"].__name__ == "AtomicityReview":
            from tests.offline import atomicity_review_fixture
            result = kwargs["schema"].model_validate(atomicity_review_fixture(messages))
            return validate(result) if validate else result
        if "Moderator/Judge" in messages[0]["content"]:
            reply = next(self.chairs)
            if isinstance(reply, Exception):
                raise reply
            return validate(reply) if validate else reply
        if "You are a preference judge." in messages[0]["content"]:
            text = '{"choice": "1", "reason": "The first trajectory satisfies the criteria."}'
            return validate(text) if validate else text
        return "5. ADD:\n- " + ONE


def test_unchanged_finalization_retains_actual_validated_working_list():
    principles, transcript = debate(pair(), catalog(), DebateProvider([f"UPDATED WORKING LIST:\n- **Q1:** {ONE}", "DEBATE_OVER"]), "test", 42, rounds=2, config={"min_principles": 1})
    assert principles == [ONE]
    assert transcript[-1]["finalized"] is True


def test_bad_chair_turn_retains_valid_list_and_continues_later_rounds():
    principles, transcript = debate(pair(), catalog(), DebateProvider([f"- Q1: {ONE}", ModelResponseError("invalid"), f"FINAL LIST:\n- Q1: {TWO}\nDEBATE_OVER"]), "test", 42, rounds=3)
    assert principles == [TWO]
    assert any(row["speaker"] == "chairperson_format_error" and row["retained_principles"] == [ONE] for row in transcript)


def test_bad_first_round_can_recover_without_seed_questions():
    principles, _ = debate(pair(), catalog(), DebateProvider([ModelResponseError("invalid"), f"FINAL LIST:\n- Q1: {ONE}"]), "test", 42, rounds=2)
    assert principles == [ONE]


def test_all_bad_rounds_never_create_fake_or_unsettled_principles():
    with pytest.raises(ModelResponseError, match="No constitution was fabricated"):
        debate(pair(), catalog(), DebateProvider([ModelResponseError("bad")] * 3), "test", 42, rounds=3)


@pytest.mark.parametrize("answer", [
    "UPDATED WORKING LIST:\n- Q1: Select the trajectory where <actual criterion>\n\n"
    "(none, as Q1 has been unanimously deemed redundant and fully addressed in the response)",
    "FINAL LIST:\n(none)\nDEBATE_OVER",
])
def test_reported_empty_or_placeholder_reply_retains_published_seed_criterion(answer):
    from tests.prompt_requests import Recorder

    drafts = assemble(pair(), assign_personas(["One", "Two", "Three"], 42), Recorder(), "test", 42, {"reasoning_workers": 1})
    actual_seed = drafts[0]["principles"][0]

    class Provider(DebateProvider):
        def __init__(self):
            self.chairs = []
        def complete(self, messages, **kwargs):
            if kwargs.get("schema") is not None and kwargs["schema"].__name__ == "AtomicityReview":
                from tests.offline import atomicity_review_fixture
                return json.dumps(atomicity_review_fixture(messages))
            if "Moderator/Judge" in messages[0]["content"]:
                assert actual_seed in messages[1]["content"]
                self.chairs.append(messages)
                if len(self.chairs) == 1:
                    return "UPDATED WORKING LIST:\n- Q1: " + actual_seed
                return answer
            if "You are a preference judge." in messages[0]["content"]:
                return '{"choice": "1", "reason": "Preserves the criteria."}'
            assert "YOUR SEED QUESTIONS:\n- " + actual_seed in messages[0]["content"]
            return "1. DEFEND:\n- Q1: This criterion captures the supplied rationale."

    source = Provider()
    models = registry(source, retries=1)
    values, transcript = debate(pair(), drafts, models, "test", 42, rounds=3)
    assert values == [actual_seed]
    assert len(source.chairs) == 5
    assert len([e for e in transcript if e["speaker"] == "chairperson_format_error"]) == 2
    assert transcript[-1]["finalized"] is False


def test_proposed_seeds_are_never_reported_as_settled_without_a_valid_chair_turn():
    drafts = catalog()
    for row in drafts:
        row["principles"] = [ONE]
    with pytest.raises(ModelResponseError, match="No constitution was fabricated"):
        debate(pair(), drafts, DebateProvider([ModelResponseError("bad")] * 3), "test", 42, rounds=3)


def test_empty_list_retry_feedback_never_offers_a_copyable_placeholder():
    source = SequenceProvider(["FINAL LIST:\n(none)\nDEBATE_OVER", f"- Q1: {ONE}"])
    result = registry(source).complete("test", [{"role": "user", "content": "Publish the list."}], seed=42, validate=parse_principles)
    assert result == [ONE]
    repair = source.calls[1][0][-1]["content"]
    assert "<actual criterion>" not in repair
    assert "Q1: Select the trajectory where" not in repair
    assert "rationales and candidate questions" in repair


@pytest.mark.parametrize("questions", [[], [""], ["Select the trajectory where <actual criterion>"], ["Select the trajectory where"], ["Prefer response A"]])
def test_seed_drafting_rejects_missing_or_fake_criteria(questions):
    from pydantic import ValidationError

    with pytest.raises((ModelResponseError, ValidationError)):
        Reasoning(reasoning="Rationale", questions=questions)


@pytest.mark.parametrize("error", [BudgetExceeded("cap"), TransportExhausted("quota"), TransportRequestError("auth")])
def test_chairperson_fallback_never_hides_service_or_budget_errors(error):
    with pytest.raises(type(error)):
        debate(pair(), catalog(), DebateProvider([f"- Q1: {ONE}", error]), "test", 42, rounds=2)


@pytest.mark.parametrize("answer", ['{"choice": "1"}', '```json\n{"choice": "2"}\n```', '{"choice": 1}'])
def test_judge_accepts_only_presentation_variations(answer):
    assert parse_choice(answer) in {"1", "2"}


@pytest.mark.parametrize("principle", [
    "Select the trajectory where info_extractor preserves API_key values.",
])
def test_abstraction_accepts_markdown_and_typographic_quotes(principle):
    class Provider:
        def complete(self, role, messages, *, validate, **kwargs):
            return validate(f'“**{principle}**”')
    assert abstract_clusters([ONE, TWO], [0, 0], Provider(), "test", 42) == [principle]


def test_successful_requests_resume_after_a_later_failure(tmp_path):
    source = SequenceProvider(['{"prompt":"Saved"}', ModelResponseError("bad"), '{"prompt":"Recovered"}'])
    models = registry(source, retries=0)
    cache = RequestCache(models, tmp_path)
    request = [{"role": "user", "content": "First"}]
    assert cache.complete("test", request, seed=42, schema=PromptProposal).prompt == "Saved"
    with pytest.raises(RetryExhausted):
        cache.complete("test", [{"role": "user", "content": "Second"}], seed=43, schema=PromptProposal)
    resumed = RequestCache(models, tmp_path)
    assert resumed.complete("test", request, seed=42, schema=PromptProposal).prompt == "Saved"
    assert resumed.complete("test", [{"role": "user", "content": "Second"}], seed=43, schema=PromptProposal).prompt == "Recovered"
    assert resumed.hits == 1
    assert len(source.calls) == 3


def test_request_cache_hits_still_respect_budget_limits(tmp_path):
    from caiso.core.budget import BudgetTracker

    source = SequenceProvider(['{"prompt":"Saved"}'])
    models = registry(source)
    models.recovery.budget = BudgetTracker(3)
    cached = RequestCache(models, tmp_path)
    messages = [{"role": "user", "content": "Request"}]
    cached.complete("test", messages, seed=42, schema=PromptProposal)
    cached.complete("test", messages, seed=42, schema=PromptProposal)
    assert models.recovery.budget.snapshot().cache_hits == 1
    models.recovery.budget.max_seconds = 0
    with pytest.raises(BudgetExceeded):
        cached.complete("test", messages, seed=42, schema=PromptProposal)
    assert len(source.calls) == 1


def test_real_induction_resumes_successful_calls_after_interruption(tmp_path):
    from caiso.core.budget import BudgetTracker
    from tests.offline import OfflineProvider

    class Interrupted(OfflineProvider):
        failed = False
        def complete(self, messages, **kwargs):
            if not self.failed and len(self.calls) == 4:
                self.failed = True
                raise BudgetExceeded("Simulated interruption after successful requests")
            return super().complete(messages, **kwargs)

    budget = BudgetTracker(3)
    source = Interrupted(budget)
    models = ProviderRegistry({"test": source})
    induction = DICAI(
        models, OfflineEmbeddings(budget), ["One", "Two", "Three"],
        {"max_workers": 1, "reasoning_workers": 1, "debate_rounds": 2,
         "clustering": {"min_clusters": 2, "max_clusters": 2}},
        role="test",
    )
    with pytest.raises(BudgetExceeded):
        induction.induce([pair()], 42, output_dir=tmp_path)
    earlier_requests = list(source.calls)
    assert len(earlier_requests) == 4
    constitution = induction.induce([pair()], 42, output_dir=tmp_path)
    assert constitution.principles
    assert constitution.metadata["request_cache_hits"] >= 4
    assert not any(request in earlier_requests for request in source.calls[4:])
    count = len(source.calls)
    cached = induction.induce([pair()], 42, output_dir=tmp_path)
    assert cached.principles == constitution.principles
    assert cached.metadata["reused_pair_records"] == 1
    assert len(source.calls) == count


@pytest.mark.parametrize("bad", [None, "", "I cannot provide JSON.", '{"prompt":42}', '{"prompt":"a","prompt":"b"}', '{"prompt":NaN}'])
def test_actual_method_transport_repairs_unusable_content_without_changing_schema(bad):
    replies = [bad, '{"prompt":"Repaired instruction"}']
    calls = []
    def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=replies.pop(0)))])
    models = registry(MethodProvider("synthetic", None), retries=1)
    with patch("caiso.utils.api.litellm.completion", completion):
        result = models.complete("test", [{"role": "user", "content": "Return JSON."}], seed=42, schema=PromptProposal)
    assert result.prompt == "Repaired instruction"
    assert len(calls) == 2
    assert all(call["response_format"] == {"type": "json_object"} for call in calls)
