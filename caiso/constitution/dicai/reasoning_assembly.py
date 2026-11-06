"""Adapted from Democratic ICAI reasoning.py; MIT, third_party/democratic_icai.

The registry supplies transport, validation, retries and budget accounting.
"""

from collections import Counter
from threading import BoundedSemaphore

from pydantic import Field, field_validator

from caiso.core.parallel import ordered_map
from caiso.core.serialization import Record
from caiso.prompts import render_messages

from .preferences import candidate_views, task_view


def valid_principle(value):
    value = value.strip()
    if not value.startswith("Select the response") or value == "Select the response":
        raise ValueError('Invalid DICAI seed principle: begin with "Select the response"')
    return value


class Question(Record):
    select_prompt: str
    reason: str

    _valid = field_validator("select_prompt")(valid_principle)


class Reasoning(Record):
    reasoning: str = Field(min_length=1)
    dimensions: list[str] = Field(min_length=1)
    questions: list[Question] = Field(min_length=1)

    @field_validator("dimensions")
    @classmethod
    def dimensions_clean(cls, values):
        values = list(dict.fromkeys(d.strip().lower() for d in values if d.strip()))
        if not values:
            raise ValueError("Reasoning must identify discriminating dimensions")
        return values


class Principles(Record):
    """Compatibility schema for callers supplying explicit seed principles."""

    principles: list[str] = Field(min_length=1)

    @field_validator("principles")
    @classmethod
    def valid_seed_principles(cls, values):
        return [valid_principle(p) for p in values]


def assemble(pair, personas, providers, role, seed, config=None):
    config = config or {}
    samples = config.get("self_consistency_samples", 5)
    steps = config.get("self_refine_steps", 2)
    temperatures = config.get("temperatures", {})
    a, b = candidate_views(pair)
    evidence = {
        "pref": pair.preferred, "preferred": pair.preferred,
        "context": task_view(pair.context), "response_a": a, "response_b": b,
    }
    workers = config.get("reasoning_workers", 4)
    # Branches and independent samples overlap, with one shared request cap per
    # pair. No worker holds a slot while waiting for another branch/sample.
    gate = BoundedSemaphore(workers)

    def reason(item):
        index, persona = item
        count = {"reasoning_cot": 1, "reasoning_self_consistency": samples + 1}.get(
            persona.strategy, 1 + steps * 2
        )
        trace, call_index = [None] * count, 0

        # Upstream reasoning is persona-neutral; identities enter during debate.
        def request(stage, temperature, schema=Reasoning, number=None, **values):
            nonlocal call_index
            if number is None:
                number = call_index
                call_index += 1
            with gate:
                result = providers.complete(
                    role,
                    render_messages("dicai", "reasoning." + stage, **evidence, **values),
                    seed=seed + index * 100 + number,
                    schema=schema,
                    temperature=temperatures.get(stage, temperature),
                )
            # Each request owns a slot; completion order never changes traces.
            trace[number] = {"stage": stage, "output": result.model_dump() if schema else result}
            return result

        if persona.strategy == "reasoning_cot":
            rationale = request(
                "cot",
                0.4,
            )
        elif persona.strategy == "reasoning_self_consistency":
            call_index = samples
            chains = ordered_map(
                lambda number: request(
                    "self_consistency_sample",
                    0.9,
                    number=number,
                ),
                range(samples),
                workers,
            )
            counter = Counter()
            for chain in chains:
                # Stable ties across Python processes; upstream counts set membership.
                counter.update(sorted(set(chain.dimensions)))
            threshold = max(1, samples // 2)  # Literal upstream rule: 2 of 5 by default.
            stable = [d for d, count in counter.most_common() if count >= threshold]
            stable = stable or [d for d, _ in counter.most_common(3)]
            joined = "\n".join("- " + c.reasoning for c in chains)
            rationale = request(
                "self_consistency_consensus",
                0.2,
                samples=samples, joined=joined, stable=stable,
            )
            if stable:
                rationale.dimensions = stable
        else:
            rationale = request(
                "self_refine_draft", 0.6
            )
            for _ in range(steps):
                critique = request(
                    "self_refine_critique",
                    0.3,
                    schema=None,
                    rationale=rationale.reasoning, dimensions=rationale.dimensions,
                )
                rationale = request(
                    "self_refine_revision",
                    0.4,
                    rationale=rationale.reasoning, critique=critique,
                )
        return {
            "persona": persona.model_dump(),
            "reasoning": rationale.reasoning,
            "dimensions": rationale.dimensions,
            "questions": [q.model_dump() for q in rationale.questions],
            "principles": [q.select_prompt for q in rationale.questions],
            "trace": trace,
            "calls_before_retries": call_index,
        }

    return ordered_map(reason, enumerate(personas), workers)
