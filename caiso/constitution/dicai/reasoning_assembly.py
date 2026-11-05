"""Adapted from Democratic ICAI reasoning.py; MIT, third_party/democratic_icai.

The registry supplies transport, validation, retries and budget accounting.
"""

from collections import Counter
from importlib.resources import files
from threading import BoundedSemaphore

from pydantic import Field, field_validator

from caiso.core.parallel import ordered_map
from caiso.core.serialization import Record
from caiso.providers.base import messages

from .preferences import pair_block


def prompt(name):
    return files("caiso.constitution.dicai").joinpath("prompts", name + ".txt").read_text()


def principle_instructions(config=None):
    """General, concise wording guidance without a numeric word-count limit.

    The argument remains compatible with existing callers; legacy max_words
    settings no longer add a hidden cap to induction prompts.
    """
    return prompt("principle_style")


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
    base = (
        prompt("reasoning_base").format(pref=pair.preferred) + "\n" + principle_instructions(config)
    )
    instruction = prompt("reasoning_json")
    block = pair_block(pair, labeled=True)
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
        def request(user, stage, temperature, schema=Reasoning, system=base, number=None):
            nonlocal call_index
            if number is None:
                number = call_index
                call_index += 1
            with gate:
                result = providers.complete(
                    role,
                    messages(system, user),
                    seed=seed + index * 100 + number,
                    schema=schema,
                    temperature=temperatures.get(stage, temperature),
                )
            # Each request owns a slot; completion order never changes traces.
            trace[number] = {"stage": stage, "output": result.model_dump() if schema else result}
            return result

        if persona.strategy == "reasoning_cot":
            rationale = request(
                block
                + "\nThink step by step through the differences, then conclude. "
                + instruction,
                "cot",
                0.4,
            )
        elif persona.strategy == "reasoning_self_consistency":
            call_index = samples
            chains = ordered_map(
                lambda number: request(
                    block + "\nReason step by step, then conclude. " + instruction,
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
                f"{samples} independent analyses were produced for the same preference judgement."
                f"\nANALYSES:\n{joined}\nDIMENSIONS THAT RECUR ACROSS A MAJORITY: {stable}\n"
                "Write one consensus rationale that relies only on the recurring dimensions "
                "and discards idiosyncratic points. " + instruction,
                "self_consistency_consensus",
                0.2,
            )
            if stable:
                rationale.dimensions = stable
        else:
            rationale = request(
                block + "\nWrite a first-draft rationale. " + instruction, "self_refine_draft", 0.6
            )
            for _ in range(steps):
                critique = request(
                    f"{block}\nDRAFT RATIONALE:\n{rationale.reasoning}\n"
                    f"DRAFT DIMENSIONS: {rationale.dimensions}\n"
                    "List concrete weaknesses: vague claims, points that are not discriminating "
                    "(true of both responses), and missing high-tension differences. Be brief.",
                    "self_refine_critique",
                    0.3,
                    schema=None,
                    system="You are a strict reviewer of preference rationales.",
                )
                rationale = request(
                    f"{block}\nPREVIOUS RATIONALE:\n{rationale.reasoning}\n"
                    f"REVIEWER CRITIQUE:\n{critique}\n"
                    "Revise the rationale to fix every weakness. Keep only discriminating, "
                    "specific points. " + instruction,
                    "self_refine_revision",
                    0.4,
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
