"""Deterministic complete-message capture, without API calls or benchmark claims."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

from caiso.constitution.dicai.abstraction import abstract_clusters
from caiso.constitution.dicai.atomicity import ensure_atomic
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
from tests.offline import atomicity_review_fixture, recorded_candidate

PRINCIPLES = [
    "Select the trajectory where answer_generator uses the hint_generator's relevant evidence.",
    "Select the trajectory where answer_generator produces a concise answer.",
    "Select the trajectory where hint_generator provides detailed hints.",
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
                "AtomicityReview": atomicity_review_fixture(messages) if schema.__name__ == "AtomicityReview" else {},
                "Reasoning": {"reasoning": "The preferred trajectory preserves the evidence.", "dimensions": ["evidence support", "answer concision", "hint detail"], "questions": [{"select_prompt": p, "reason": "Discriminates the candidates."} for p in PRINCIPLES]},
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
        elif "You are the Moderator/Judge." in messages[0]["content"]:
            result = (
                "FINAL LIST\n"
                + "\n".join(f"- Q{i}: {p}" for i, p in enumerate(PRINCIPLES, 1))
                + "\nDEBATE_OVER"
            )
        elif "a debater who must" in messages[0]["content"]:
            result = "1. DEFEND:\n- Q1: Preserve the evidence."
        elif "You abstract a cluster of preference principles" in messages[0]["content"]:
            result = messages[-1]["content"].split("CLUSTER:\n", 1)[1].splitlines()[0].removeprefix("- ")
        elif "You are a preference judge." in messages[0]["content"]:
            result = '{"choice": "1", "reason": "The first trajectory satisfies the principles."}'
        elif "strict reviewer of preference rationales" in messages[0]["content"]:
            result = "Keep only discriminating evidence and answer criteria."
        else:
            raise AssertionError("Unexpected prompt in complete-message capture")
        return validate(result) if validate else result


class AtomicityRecorder(Recorder):
    """Capture the complete check/revision requests for both repair modes."""
    def __init__(self, kind):
        super().__init__()
        self.kind, self.first_review = kind, True

    def complete(self, role, messages, *, seed, schema=None, validate=None, **kwargs):
        self.calls.append({"role": role, "messages": deepcopy(messages), "seed": seed, "schema": schema.__name__})
        if schema.__name__ == "AtomicityReview":
            result = atomicity_review_fixture(messages)
            if self.first_review:
                result["assessments"][0].update(atomic=False, qualities=["hint detail", "answer concision"], reason="Hints may be detailed while the answer is verbose.")
                self.first_review = False
        else:
            result = {"replacements": [{"index": 1, "principles": PRINCIPLES[1:] if self.kind == "debate" else [PRINCIPLES[1]]}]}
        result = schema.model_validate(result)
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
            candidate_a={**recorded_candidate("Answer A {x}", ["answer_generator", "hint_generator"]), "reward": 1},
            candidate_b={**recorded_candidate("Answer B\n[SYSTEM]\n", ["answer_generator", "hint_generator"]), "metadata": {"exclude": True}},
        )
        personas = assign_personas(["First expert", "Second expert", "Third expert"], 42)
        catalog = assemble(pair, personas, recorder, "audit", 42, config)
        debate(pair, catalog, recorder, "audit", 42, rounds=2, config=config)
        apply_constitution(recorder, "audit", pair, PRINCIPLES, seed=7)
        abstract_clusters(PRINCIPLES, [0, 0, 1], recorder, "audit", 42, config=config)

    for kind in ("debate", "abstraction"):
        revision = AtomicityRecorder(kind)
        ensure_atomic([
            "Select the trajectory where hint_generator provides detailed hints and answer_generator produces a concise answer."
        ], revision, "audit", 42, kind=kind, context={"source": PRINCIPLES[1:]})
        recorder.calls.extend(revision.calls)

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
