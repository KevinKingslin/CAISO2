"""Regressions for component criteria lost during constitution abstraction."""

from copy import deepcopy
import json

import pytest

from caiso.constitution.dicai import pipeline
from caiso.constitution.dicai.coverage import coverage_targets, observed_components, preserve_component_coverage
from caiso.constitution.models import PreferencePair
from caiso.core.budget import BudgetTracker
from caiso.evaluation.iteration_report import IterationReport
from caiso.method.block_constitution import induce_block_constitution
from caiso.method.system_constitution import induce_system_constitution
from caiso.providers.base import ProviderRegistry
from tests.offline import OfflineEmbeddings, OfflineProvider

RULES = {
    "question_rewriter": "Select the trajectory where question_rewriter preserves the question's requested relation.",
    "info_extractor": "Select the trajectory where info_extractor generates keywords for the requested relation.",
    "wikipedia_retriever": "Select the trajectory where wikipedia_retriever retrieves evidence for the requested relation.",
    "hint_generator": "Select the trajectory where hint_generator grounds its hints in the retrieved evidence.",
    "answer_generator": "Select the trajectory where answer_generator produces a concise answer.",
}


def pair(pair_id="p1", components=None):
    executions = [{"component_id": cid, "component_input": {"query": "Q"}, "component_output": {"text": "A"}} for cid in (RULES if components is None else components)]
    return PreferencePair(
        pair_id=pair_id, preferred="A", context={"question": "Q"},
        candidate_a={"trajectory": {"executions": executions, "final_output": {"answer": "A"}}},
        candidate_b={"trajectory": {"executions": deepcopy(executions), "final_output": {"answer": "B"}}},
    )


def preserve(outputs, settled, origins, pairs, expected):
    return preserve_component_coverage(outputs, settled, origins, pairs, sorted(expected), scope="system", target_source="provided_components")


def test_restores_only_missing_components_using_settled_rules_and_source_pairs():
    pairs = [pair("p1", ["question_rewriter"]), pair("p2", ["answer_generator"])]
    settled = [RULES["question_rewriter"], RULES["answer_generator"]]
    before = deepcopy(pairs)
    values, report = preserve([RULES["answer_generator"]], settled, ["p1", "p2"], pairs, RULES)
    assert values == [RULES["answer_generator"], RULES["question_rewriter"]]
    assert report["covered_before"] == ["answer_generator"]
    assert report["covered_after"] == ["answer_generator", "question_rewriter"]
    assert report["restored"][0]["source_pair_ids"] == ["p1"]
    assert report["restored"][0]["settled_principle_indices"] == [1]
    assert report["principle_indices_by_component"]["question_rewriter"] == [2]
    assert report["status"] == "insufficient_evidence"
    assert set(report["missing_components"]) == set(RULES) - {"answer_generator", "question_rewriter"}
    assert pairs == before


def test_prefers_distinct_pair_support_over_repetition_within_one_pair():
    common = RULES["hint_generator"]
    rare = "Select the trajectory where hint_generator is concise."
    settled = [rare, rare, rare, common, common]
    values, report = preserve([], settled, ["p1", "p1", "p1", "p1", "p2"], [pair("p1"), pair("p2")], ["hint_generator"])
    assert values == [common]
    assert report["restored"][0]["source_pair_ids"] == ["p1", "p2"]
    assert report["status"] == "complete"
    again, repeated = preserve(values, settled, ["p1", "p1", "p1", "p1", "p2"], [pair("p1"), pair("p2")], ["hint_generator"])
    assert again == values
    assert repeated["restored"] == []


def test_generic_aliases_and_partial_ids_do_not_count_as_component_coverage():
    components = ["retriever", "wikipedia_retriever", "encoder.layer", "encoder", "search-v2", "search"]
    text = "Select the trajectory where wikipedia_retriever uses encoder.layer and search-v2."
    generic = "Select the trajectory where the final answer uses the retrieved content."
    values, report = preserve([text, generic], [], [], [pair(components=components)], components)
    assert values == [text, generic]
    assert report["covered_after"] == ["encoder.layer", "search-v2", "wikipedia_retriever"]
    assert report["missing_components"] == ["encoder", "retriever", "search"]
    assert report["unattributed_principle_indices"] == [2]


def test_unsupported_or_unobserved_component_mentions_are_not_restored():
    first, second = pair("p1", ["answer_generator"]), pair("p2", ["hint_generator"])
    hallucinated = "Select the trajectory where imaginary_component supplies evidence."
    settled = [RULES["hint_generator"], hallucinated, "Select the trajectory where the final answer is concise."]
    values, report = preserve([hallucinated], settled, ["p1"] * 3, [first, second], ["hint_generator", "imaginary_component"])
    assert values == [hallucinated]
    assert report["restored"] == []
    assert report["covered_after"] == []
    assert report["insufficient_evidence_components"] == ["hint_generator", "imaginary_component"]
    assert report["unobserved_components"] == ["imaginary_component"]


def test_one_supported_interaction_can_cover_both_components():
    rule = "Select the trajectory where answer_generator uses evidence from hint_generator."
    values, report = preserve([], [rule], ["p1"], [pair()], ["answer_generator", "hint_generator"])
    assert values == [rule]
    assert report["restored"][0]["restored_for"] == ["answer_generator", "hint_generator"]
    assert report["missing_components"] == []


def test_interaction_restoration_requires_a_source_pair_observing_both_components():
    rule = "Select the trajectory where answer_generator uses evidence from hint_generator."
    pairs = [pair("p1", ["answer_generator"]), pair("p2", ["hint_generator"])]
    values, report = preserve([], [rule], ["p1"], pairs, ["answer_generator", "hint_generator"])
    assert values == []
    assert report["covered_after"] == []
    assert report["missing_components"] == ["answer_generator", "hint_generator"]


def test_targets_use_execution_records_or_the_block_scope_without_inventing_ids():
    comparison = pair(components=["answer_generator", "hint_generator"])
    comparison.context["component_id"] = "task_metadata"
    comparison.candidate_a["trajectory"]["executions"][0]["component_output"]["component_id"] = "task_output"
    assert observed_components(comparison) == {"answer_generator", "hint_generator"}
    assert coverage_targets([comparison], "system") == (["answer_generator", "hint_generator"], "observed_trajectories")
    assert coverage_targets([comparison], "block:hint_generator") == (["hint_generator"], "block_scope")
    simple = PreferencePair(pair_id="text", preferred="A", context={}, candidate_a={"text": "A"}, candidate_b={"text": "B"})
    targets, source = coverage_targets([simple], "system")
    values, report = preserve_component_coverage([RULES["answer_generator"]], [], [], [simple], targets, scope="system", target_source=source)
    assert values == [RULES["answer_generator"]]
    assert report["status"] == "not_applicable"


@pytest.mark.parametrize("invalid", ["answer_generator", [None], [" "]])
def test_invalid_target_ids_fail_before_induction(invalid):
    with pytest.raises(ValueError, match="Coverage component IDs"):
        coverage_targets([pair()], "system", invalid)


class CollapsingProvider(OfflineProvider):
    """Keep the actual reasoning/debate pipeline, collapse cluster abstractions."""

    def complete(self, messages, *, seed, schema=None, **kwargs):
        if "You are the Moderator/Judge." in messages[0]["content"] or "You abstract a cluster of preference principles" in messages[0]["content"]:
            self.budget.call()
            self.calls.append((messages, seed, schema))
            if "Moderator/Judge" in messages[0]["content"]:
                return "FINAL LIST\n" + "\n".join(f"- Q{i}: {text}" for i, text in enumerate(RULES.values(), 1)) + "\nDEBATE_OVER"
            allowed = json.JSONDecoder().raw_decode(messages[-1]["content"].split("ALLOWED COMPONENT IDS:\n", 1)[1])[0]
            return RULES["answer_generator" if "answer_generator" in allowed else allowed[0]]
        result = super().complete(messages, seed=seed, schema=schema, **kwargs)
        if schema and schema.__name__ == "Reasoning":
            values = json.loads(result)
            values["questions"] = [{"select_prompt": text, "reason": "Observed component difference."} for text in RULES.values()]
            return json.dumps(values)
        return result


@pytest.mark.parametrize("scope", ["system", "block"])
@pytest.mark.parametrize("minimize", [False, True])
def test_real_induction_restores_coverage_after_abstraction_and_minimization(tmp_path, monkeypatch, scope, minimize):
    budget = BudgetTracker(3)
    source = CollapsingProvider(budget)
    providers = ProviderRegistry({"test": source})
    dicai = pipeline.DICAI(providers, OfflineEmbeddings(budget), ["One", "Two", "Three"], {
        "max_workers": 1, "reasoning_workers": 1, "debate_rounds": 1,
        "self_consistency_samples": 1, "self_refine_steps": 0,
        "clustering": {"min_clusters": 2, "max_clusters": 2},
        "minimization": {"enabled": minimize},
    }, role="test")

    def remove_other_components(providers, role, pairs, values, *args, **kwargs):
        assert RULES["answer_generator"] in values
        return [RULES["answer_generator"]], [{"principle": p, "dropped": True} for p in values if p != RULES["answer_generator"]]

    monkeypatch.setattr(pipeline, "minimize_constitution", remove_other_components)
    targets = list(RULES) if scope == "system" else ["hint_generator"]

    def induce(components=targets):
        if scope == "system":
            return induce_system_constitution(dicai, [pair()], 42, component_ids=components, cache_dir=tmp_path / "cache")
        return induce_block_constitution(dicai, [pair()], components, 42, cache_dir=tmp_path / "cache")

    constitution = induce()
    coverage = constitution.metadata["component_coverage"]
    assert coverage["expected_components"] == sorted(targets)
    assert coverage["covered_after"] == sorted(targets)
    assert coverage["status"] == "complete"
    assert bool(coverage["restored"]) == bool(set(targets) - set(coverage["covered_before"]))
    assert all(entry["principle"] in constitution.settled_principles for entry in coverage["restored"])
    assert all(entry["source_pair_ids"] == ["p1"] for entry in coverage["restored"])
    assert constitution.metadata["clustering"]["selected_k"] == 2
    last_judge = [messages for messages, _, _ in source.calls if "You are a preference judge." in messages[0]["content"]][-1]
    assert all(text in last_judge[-1]["content"] for text in constitution.principles)
    path = tmp_path / "cache" / constitution.metadata["provenance"]
    assert json.loads((path / "component_coverage.json").read_text()) == coverage
    if scope == "system":
        assert json.loads((tmp_path / "cache/component_coverage.json").read_text()) == coverage
    count = len(source.calls)
    resumed = induce()
    assert resumed.metadata["component_coverage"] == coverage
    assert resumed.principles == constitution.principles
    assert len(source.calls) == count

    report = IterationReport(tmp_path / "results", {"reporting": {"cli": False}}, budget)
    report.begin(0, 0.3, 0.3)
    if scope == "system":
        report.save_system_induction(constitution, [])
        phase = "00_system_constitution"
    else:
        report.save_phase("04_block_constitution", constitution=constitution)
        phase = "04_block_constitution"
    assert json.loads((tmp_path / f"results/iterations/0001/{phase}/component_coverage.json").read_text()) == coverage

    if scope == "system" and not minimize:
        changed = induce([*targets, "unobserved_component"])
        assert changed.metadata["provenance"] != constitution.metadata["provenance"]
        assert changed.metadata["component_coverage"]["missing_components"] == ["unobserved_component"]
        assert changed.principles == constitution.principles
