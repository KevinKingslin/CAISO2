"""Original DICAI CoT, sampled consensus and iterative self-refinement.

The user-supplied source is preserved under third_party/democratic_icai/original.
Each structured rationale supplies its own dimensions and candidate questions.
"""

from collections import Counter

from pydantic import Field, field_validator, model_validator

from caiso.core.parallel import ordered_map
from caiso.core.serialization import Record
from caiso.prompts import render_messages

from .preferences import candidate_views, task_view
from .parsing import dedupe
from .parsing import normalize_principle as valid_principle

__all__ = ["Reasoning", "assemble", "valid_principle"]


class Question(Record):
    select_prompt: str
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def allow_text_question(cls, value):
        return {"select_prompt": value} if isinstance(value, str) else value

    @field_validator("select_prompt")
    @classmethod
    def actual_criterion(cls, value):
        return valid_principle(value)


class Reasoning(Record):
    reasoning: str = Field(min_length=1)
    dimensions: list[str] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)

    @field_validator("reasoning")
    @classmethod
    def nonempty_reasoning(cls, value):
        if not value.strip():
            raise ValueError("reasoning must contain a nonempty rationale")
        return value

    @field_validator("dimensions")
    @classmethod
    def normalize_dimensions(cls, values):
        return [value.strip().lower() for value in values if value.strip()]

    @model_validator(mode="after")
    def usable_criteria(self):
        if not self.questions and not self.dimensions:
            raise ValueError("Supply actual candidate questions or discriminating dimensions")
        return self


def assemble(pair, personas, providers, role, seed, config=None):
    config = config or {}
    temperatures = config.get("temperatures", {})
    samples, steps = config.get("self_consistency_samples", 5), config.get("self_refine_steps", 2)
    if samples < 1 or steps < 0:
        raise ValueError("Self-consistency needs at least one sample; refinement steps cannot be negative")
    a, b = candidate_views(pair)
    evidence = {"preferred": pair.preferred, "context": task_view(pair.context), "response_a": a, "response_b": b}

    def reason(item):
        index, persona = item
        stage = persona.strategy.removeprefix("reasoning_")
        trace = []
        base_seed = seed + index * 100

        def request(key, temperature_key, default, extra=None, structured=True):
            result = providers.complete(
                role, render_messages("dicai", "reasoning." + key, **evidence, **(extra or {})),
                seed=base_seed + len(trace), schema=Reasoning if structured else None,
                include_schema_instruction=False,
                response_context={"pair_id": pair.pair_id, "stage": "reasoning." + key, "strategy": stage},
                temperature=temperatures.get(temperature_key, default),
            )
            trace.append({"stage": key, "output": result.model_dump() if structured else result})
            return result

        if stage == "cot":
            result = request("cot", "cot", 0.4)
        elif stage == "self_consistency":
            chains = [request("self_consistency", "self_consistency_sample", 0.9) for _ in range(samples)]
            # Count each axis once per chain, as upstream, with a reproducible
            # first-seen order for tied counts instead of process-random set order.
            counter = Counter(d for chain in chains for d in dict.fromkeys(chain.dimensions))
            stable = [d for d, count in counter.most_common() if count >= max(1, samples // 2)]
            if not stable:
                stable = [d for d, _ in counter.most_common(3)]
            result = request(
                "self_consistency_consensus", "self_consistency_consensus", 0.2,
                {"samples": samples, "analyses": "\n".join("- " + chain.reasoning for chain in chains), "stable_dimensions": stable},
            )
            if stable:
                result.dimensions = stable
        elif stage == "self_refine":
            result = request("self_refine", "self_refine_draft", 0.6)
            for _ in range(steps):
                critique = request(
                    "self_refine_critique", "self_refine_critique", 0.3,
                    {"reasoning": result.reasoning, "dimensions": result.dimensions}, structured=False,
                )
                result = request(
                    "self_refine_revision", "self_refine_revision", 0.4,
                    {"reasoning": result.reasoning, "critique": critique},
                )
        else:
            raise ValueError(f"Unknown DICAI reasoning strategy: {stage}")
        # Original seed_questions() uses the rationale's questions, or its named
        # dimensions when the model supplied no questions. No extra model call.
        principles = dedupe([q.select_prompt for q in result.questions] or [
            f"Select the trajectory that demonstrates higher {d}." for d in result.dimensions
        ])
        return {
            "persona": persona.model_dump(), "reasoning": result.reasoning,
            "dimensions": result.dimensions, "questions": [q.model_dump() for q in result.questions],
            "principles": principles, "trace": trace, "calls_before_retries": len(trace),
        }

    return ordered_map(reason, enumerate(personas), config.get("reasoning_workers", 4))
