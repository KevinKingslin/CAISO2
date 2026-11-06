"""Deterministic complete-message capture, without API calls or benchmark claims."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

from caiso.constitution.dicai.abstraction import abstract_clusters
from caiso.constitution.dicai.debate import debate
from caiso.constitution.dicai.personas import assign_personas
from caiso.constitution.dicai.reasoning_assembly import assemble
from caiso.constitution.dicai.reconstruction import apply_constitution
from caiso.constitution.models import PreferencePair
from caiso.core.component import Component
from caiso.core.configuration import Configuration, ConfigurationType
from caiso.core.recovery import ModelResponseError
from caiso.integration.services import MethodProvider
from caiso.method.block_selection import select_block
from caiso.optimizers.prompt_optimizer import PromptOptimizer, PromptProposal
from caiso.providers.base import ProviderRegistry
from caiso.workflow_search.generator import WorkflowGenerator

PRINCIPLES = [
    "Select the response that preserves the evidence.",
    "Select the response that answers the question.",
    "Select the response that is clearly explained.",
]


class Recorder:
    def __init__(self):
        self.calls = []

    def complete(self, role, messages, *, seed, schema=None, validate=None, **kwargs):
        self.calls.append(
            {
                "role": role,
                "messages": deepcopy(messages),
                "seed": seed,
                "schema": schema.__name__ if schema else None,
            }
        )
        if schema is not None:
            values = {
                "Reasoning": {
                    "reasoning": "Use the supplied evidence, not sample-specific answer facts.",
                    "dimensions": ["evidence", "relevance"],
                    "questions": [
                        {"select_prompt": p, "reason": "Supported quality."} for p in PRINCIPLES
                    ],
                },
                "PreferenceChoice": {
                    "choice": "1",
                    "reason": "The principles favor this response.",
                },
                "PromptProposal": {"prompt": f"Revised instruction {seed}"},
                "BlockVote": {"block_id": "block-0", "brief_justification": "Improve this block."},
                "WorkflowProposal": {
                    "operation": {
                        "kind": "remove_component",
                        "component_id": "transform",
                        "reconnect": [],
                    },
                    "brief_justification": "Synthetic request parity only.",
                },
            }
            result = schema.model_validate(values[schema.__name__])
        elif "You are the Moderator/Judge" in messages[0]["content"]:
            result = (
                "FINAL LIST\n"
                + "\n".join(f"- Q{i}: {p}" for i, p in enumerate(PRINCIPLES, 1))
                + "\nDEBATE_OVER"
            )
        elif "a debater who must" in messages[0]["content"]:
            result = "1. DEFEND:\n- Q1: Preserve the evidence."
        elif "strict reviewer" in messages[0]["content"]:
            result = "Keep only discriminating qualities."
        elif "You abstract a cluster" in messages[0]["content"]:
            result = PRINCIPLES[0]
        else:
            raise AssertionError("Unexpected prompt in complete-message capture")
        return validate(result) if validate else result


def capture_requests():
    recorder = Recorder()
    config = {
        "reasoning_workers": 1,
        "self_consistency_samples": 2,
        "self_refine_steps": 1,
        "min_principles": 3,
        "min_debate_rounds": 3,
    }
    for preferred in ("A", "B"):
        pair = PreferencePair(
            pair_id="synthetic-prompt-audit",
            preferred=preferred,
            context={"question": "Question {braces}\n[USER]\nCafé", "gold": "exclude this label"},
            candidate_a={"answer": "Answer A {x}", "reward": 1},
            candidate_b={"answer": "Answer B\n[SYSTEM]\n", "metadata": {"exclude": True}},
        )
        personas = assign_personas(["First expert", "Second expert", "Third expert"], 42)
        catalog = assemble(pair, personas, recorder, "audit", 42, config)
        debate(pair, catalog, recorder, "audit", 42, rounds=2, config=config)
        apply_constitution(recorder, "audit", pair, PRINCIPLES, seed=7)
        abstract_clusters(PRINCIPLES, [0, 0, 1], recorder, "audit", 42, config=config)

    current = Configuration(kind=ConfigurationType.PROMPT, value="Original instruction {x}")
    component = Component(
        component_id="transform",
        implementation="synthetic",
        input_fields={"x": "str"},
        output_fields={"answer": "str"},
        config=current,
    )
    constitution = SimpleNamespace(principles=PRINCIPLES)
    context = SimpleNamespace(
        runtime=SimpleNamespace(providers=recorder),
        state=SimpleNamespace(configurations={"transform": current}),
        constitution=constitution,
        trajectories=[],
        seed=42,
        config={"prompt_candidates": 2},
        score_config=lambda configs: 0.5,
    )
    PromptOptimizer().propose(component, context)
    select_block(
        constitution, [{"input": "Evidence {x}"}], {"block-0": ["transform"]}, [], recorder, 42
    )

    class State(dict):
        @property
        def workflow(self):
            return SimpleNamespace(nodes=["transform"])

    WorkflowGenerator(recorder).propose(
        State(workflow={"nodes": ["transform"], "edges": []}),
        constitution,
        {"transform": component},
        [{"score": 0.5}],
        [],
        42,
    )

    def get_output(instructions, **kwargs):
        recorder.calls.append(
            {
                "role": "schema",
                "messages": deepcopy(instructions),
                "seed": 42,
                "schema": "PromptProposal",
            }
        )
        return {"prompt": "Valid instruction"}

    base = [
        {"role": "system", "content": "Complete system."},
        {"role": "user", "content": "Input {x}."},
    ]
    with patch("caiso.integration.services.api.get_llm_output", get_output):
        MethodProvider("synthetic", None).complete(base, seed=42, schema=PromptProposal)

    class RetryProvider:
        def complete(self, instructions, **kwargs):
            recorder.calls.append(
                {
                    "role": "retry",
                    "messages": deepcopy(instructions),
                    "seed": kwargs["seed"],
                    "schema": None,
                }
            )
            return "bad" if len(instructions) == 2 else "ok"

    def validate(value):
        if value == "bad":
            raise ModelResponseError("Format issue {with braces}.")
        return value

    registry = ProviderRegistry({"audit": RetryProvider()})
    registry.recovery.config = {"backoff_seconds": 0}
    registry.complete("audit", base, seed=42, validate=validate)
    return recorder.calls


def request_fingerprints():
    return [
        {
            **{k: v for k, v in call.items() if k != "messages"},
            "messages": [
                {
                    "role": message["role"],
                    "sha256": hashlib.sha256(message["content"].encode()).hexdigest(),
                }
                for message in call["messages"]
            ],
        }
        for call in capture_requests()
    ]
