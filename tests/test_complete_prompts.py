"""Complete prompt files must preserve the model messages assembled before refactoring."""

import json
from pathlib import Path

from tests.prompt_requests import request_fingerprints
from caiso.prompts import render_messages


def test_complete_messages_match_pre_consolidation_requests():
    baseline = json.loads(
        (Path(__file__).parent / "fixtures/complete_prompt_requests.json").read_text()
    )
    assert request_fingerprints() == baseline


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
