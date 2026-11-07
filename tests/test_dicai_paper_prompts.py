"""Published Appendix G instructions and their runtime formats must stay aligned."""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from caiso.constitution.dicai.debate import debate
from caiso.constitution.dicai.personas import assign_personas
from caiso.constitution.dicai.reasoning_assembly import assemble
from caiso.constitution.dicai.reconstruction import apply_constitution, parse_choice
from caiso.constitution.models import PreferencePair
from caiso.core.recovery import ModelResponseError
from caiso.integration.services import MethodProvider
from caiso.prompts import render_messages
from tests.prompt_requests import Recorder, PRINCIPLES

ROOT = Path(__file__).resolve().parents[1]
PAPER = json.loads((ROOT / "tests/fixtures/dicai_paper_prompts.json").read_text())


def paper_words(value):
    """Ignore TeX layout, typography and list numbering, retaining every word."""
    value = re.sub(r"\\vspace\{[^}]*\}", "", value)
    value = re.sub(r"\\\\\[[^]]*\]", "", value)
    value = re.sub(r"\\(?:begin|end)\{[^}]*\}(?:\[[^]]*\])?", "", value)
    value = re.sub(r"\\item(?:\[[^]]*\])?", "", value)
    value = re.sub(r"\\(?:noindent|texttt|textbf)", "", value)
    value = re.sub(r"\\textless(?:\{\})?", "<", value)
    value = re.sub(r"\\textgreater(?:\{\})?", ">", value)
    value = value.replace(r"\_", "_").replace("’", "'")
    value = re.sub(r"^\s*\d+\.\s*", "", value, flags=re.M)
    return re.findall(r"[\w]+(?:'[\w]+)?", value)


def test_transcriptions_match_published_latex_words():
    source = (ROOT / "third_party/democratic_icai/paper/appendix_g.tex").read_text()
    blocks = re.findall(r"\\begin\{tcolorbox\}\[.*?\]\n(.*?)\\end\{tcolorbox\}", source, re.S)
    keys = [
        "reasoning.cot", "reasoning.self_consistency", "reasoning.self_refine",
        "debate.chairperson", "debate.debater", "abstraction", "reconstruction",
    ]
    for key, block in zip(keys, blocks, strict=False):
        expected = PAPER["instructions"][key]
        if isinstance(expected, dict):
            expected = "System: " + expected["system"] + " User: " + expected["user"]
        assert paper_words(block) == paper_words(expected), key
    assert len(blocks) >= len(keys)


@pytest.mark.parametrize("key", [
    "reasoning.cot", "reasoning.self_consistency", "reasoning.self_refine",
    "debate.chairperson", "debate.debater", "abstraction",
])
def test_active_instructions_match_paper_transcription(key):
    identity = "Persona {braces}\n[USER]\nEvidence"
    messages = render_messages(
        "dicai", key, identity=identity, agent_name="Expert", context={},
        response_a={}, response_b={}, preferred="A", pair="Pair", trace="Rationale",
        seeds="(none)", working_list="(none)", debate_history="History",
        cycle=1, total_cycles=3, principles="Principles",
    )
    expected = PAPER["instructions"][key]
    if key.startswith("reasoning."):
        expected = identity + "\n\n" + expected
    if key == "debate.debater":
        expected = expected.format(agent_name="Expert")
    assert messages[0] == {"role": "system", "content": expected}
    assert [m["role"] for m in messages] == ["system", "user"]
    assert not any("PRINCIPLE STYLE" in m["content"] for m in messages)


def pair(preferred="A"):
    return PreferencePair(
        pair_id="paper-test", preferred=preferred, context={"question": "Q", "gold": "HIDDEN"},
        candidate_a={"answer": "A", "reward": 1}, candidate_b={"answer": "B", "reward": 0},
    )


def test_paper_reasoning_is_unchanged_and_separate_drafting_seeds_each_persona():
    recorder = Recorder()
    personas = assign_personas(["First", "Second", "Third"], 42)
    catalog = assemble(
        pair(), personas, recorder, "audit", 42,
        {"reasoning_workers": 1, "self_consistency_samples": 50, "self_refine_steps": 50},
    )
    assert len(recorder.calls) == 6
    for persona, call, draft, row in zip(personas, recorder.calls[::2], recorder.calls[1::2], catalog, strict=True):
        assert call["schema"] == "Reasoning"
        assert persona.prompt() in call["messages"][0]["content"]
        assert '"final_reasoning"' in call["messages"][0]["content"]
        assert set(row["trace"][0]["output"]) == {"final_reasoning"}
        assert draft["schema"] == "SeedQuestions"
        assert row["reasoning"] in draft["messages"][1]["content"]
        assert persona.prompt() in draft["messages"][1]["content"]
        assert row["trace"][1]["stage"] == "seed_questions"
        assert row["calls_before_retries"] == 2
        assert row["principles"] == [PRINCIPLES[0]]


def test_debate_runs_three_rounds_without_reconstruction_gates():
    recorder = Recorder()
    personas = assign_personas(["First", "Second", "Third"], 42)
    catalog = assemble(pair(), personas, recorder, "audit", 42, {"reasoning_workers": 1})
    recorder.calls.clear()
    principles, history = debate(
        pair(), catalog, recorder, "audit", 42, rounds=3,
        config={"min_principles": 100, "min_debate_rounds": 100},
    )
    assert principles == PRINCIPLES
    assert len(recorder.calls) == 12
    assert history[0]["content"] == "- Q1: " + PRINCIPLES[0]
    for call in recorder.calls:
        user = call["messages"][1]["content"]
        assert "CURRENT WORKING LIST:\n(none)" not in user
        assert "YOUR OWN PROPOSED QUESTION LIST:\n(none)" not in user
    assert [row["speaker"] for row in history[1:-1]] == [
        "debater_1", "debater_2", "debater_3", "chairperson",
    ] * 3
    assert history[-1] == {"speaker": "debate_outcome", "cycles": 3, "finalized": True}
    assert not any("RECONSTRUCTION" in str(c["messages"]) for c in recorder.calls)


@pytest.mark.parametrize("seed,expected", [(0, "A"), (1, "B")])
def test_paper_inference_is_blind_and_maps_shuffled_outputs(seed, expected):
    recorder = Recorder()
    choice, detail = apply_constitution(recorder, "audit", pair(), PRINCIPLES, seed=seed)
    call = recorder.calls[0]
    assert call["schema"] is None
    assert choice == expected
    assert detail["response"] == "Output(a)"
    assert "HIDDEN" not in str(call["messages"])
    assert "reward" not in str(call["messages"])
    source = PAPER["instructions"]["reconstruction"]
    assert call["messages"][0]["content"] == source["system"]
    assert call["messages"][1]["content"] == source["user"].format(
        constitution="\n".join(PRINCIPLES),
        output_1=json.dumps({"answer": "A" if expected == "A" else "B"}),
        output_2=json.dumps({"answer": "B" if expected == "A" else "A"}),
    )


@pytest.mark.parametrize("text", ['{"choice": "1"}', "Output(c)", "Output(a) because..."])
def test_non_paper_inference_format_is_rejected(text):
    with pytest.raises(ModelResponseError):
        parse_choice(text)


def test_method_transport_does_not_append_schema_instructions_to_paper_prompt():
    from caiso.constitution.dicai.reasoning_assembly import Reasoning

    request = render_messages(
        "dicai", "reasoning.cot", identity="Persona", context={},
        response_a={}, response_b={}, preferred="A",
    )
    with patch("caiso.integration.services.api.get_llm_output", return_value={"final_reasoning": "Final"}) as api:
        MethodProvider("synthetic", None).complete(
            request, seed=42, schema=Reasoning, include_schema_instruction=False,
        )
    assert api.call_args.args[0] == request
    assert api.call_args.kwargs["json_object"] is False
    assert api.call_args.kwargs["response_format"] == {"type": "json_object"}
