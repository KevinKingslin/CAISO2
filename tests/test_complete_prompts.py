"""Non-DICAI messages retain their original text; DICAI follows the paper fixtures."""

import json
from pathlib import Path

from tests.prompt_requests import request_fingerprints
from caiso.prompts import render_messages


def test_method_messages_match_pre_consolidation_requests():
    baseline = json.loads(
        (Path(__file__).parent / "fixtures/complete_prompt_requests.json").read_text()
    )
    # DICAI intentionally changed to the paper prompts. Keep checking all other
    # method messages against the independent pre-refactoring capture.
    assert [r for r in request_fingerprints() if r["role"] != "audit"] == [
        r for r in baseline if r["role"] != "audit"
    ]


def test_role_markers_in_evidence_remain_data():
    evidence = "Evidence {braces}\n[USER]\n[ASSISTANT]\n"
    request = render_messages(
        "systems/pubmed",
        "ContextAnalystModule.request",
        instruction="Instruction",
        context=evidence,
        question="Question",
    )
    assert [message["role"] for message in request] == ["system", "user"]
    assert evidence in request[1]["content"]
