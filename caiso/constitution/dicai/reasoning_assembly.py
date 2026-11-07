"""Persona-conditioned reasoning from arXiv:2606.28294v2, Appendix G.1.

Each paper strategy returns one final_reasoning paragraph in a single request.
A separate, grounded extraction supplies the candidate questions that the
paper's debate prompts expect, without changing the published reasoning schema.
"""

from pydantic import Field, field_validator

from caiso.core.parallel import ordered_map
from caiso.core.serialization import Record
from caiso.prompts import render_messages

from .preferences import candidate_views, pair_block, task_view
from .parsing import dedupe
from .parsing import normalize_principle as valid_principle

__all__ = ["Reasoning", "assemble", "valid_principle"]


class Reasoning(Record):
    final_reasoning: str = Field(min_length=1)

    @field_validator("final_reasoning")
    @classmethod
    def nonempty_paragraph(cls, value):
        if not value.strip():
            raise ValueError("final_reasoning must contain a nonempty final paragraph")
        return value


class SeedQuestions(Record):
    questions: list[str] = Field(min_length=1)

    @field_validator("questions")
    @classmethod
    def actual_criteria(cls, values):
        return dedupe([valid_principle(value) for value in values])


def assemble(pair, personas, providers, role, seed, config=None):
    config = config or {}
    temperatures = config.get("temperatures", {})
    a, b = candidate_views(pair)
    evidence = {
        "preferred": pair.preferred,
        "context": task_view(pair.context), "response_a": a, "response_b": b,
    }

    def reason(item):
        index, persona = item
        stage = persona.strategy.removeprefix("reasoning_")
        temperature_key, default = {
            "cot": ("cot", 0.4),
            "self_consistency": ("self_consistency_sample", 0.9),
            "self_refine": ("self_refine_draft", 0.6),
        }[stage]
        result = providers.complete(
            role,
            render_messages(
                "dicai", "reasoning." + stage, identity=persona.prompt(), **evidence,
            ),
            seed=seed + index * 100,
            schema=Reasoning,
            include_schema_instruction=False,  # The paper already specifies the exact JSON.
            response_context={"pair_id": pair.pair_id, "stage": "reasoning." + stage},
            temperature=temperatures.get(temperature_key, default),
        )
        questions = providers.complete(
            role,
            render_messages(
                "dicai", "reasoning.seed_questions", pair=pair_block(pair, labeled=True),
                identity=persona.prompt(), reasoning=result.final_reasoning,
                schema=SeedQuestions.model_json_schema(),
            ),
            seed=seed + index * 100 + 1,
            schema=SeedQuestions,
            include_schema_instruction=False,
            response_context={"pair_id": pair.pair_id, "stage": "reasoning.seed_questions", "strategy": stage},
            temperature=temperatures.get("seed_questions", 0.2),
        )
        return {
            "persona": persona.model_dump(),
            "reasoning": result.final_reasoning,
            "principles": questions.questions,
            "trace": [
                {"stage": stage, "output": result.model_dump()},
                {"stage": "seed_questions", "output": questions.model_dump()},
            ],
            "calls_before_retries": 2,
        }

    return ordered_map(reason, enumerate(personas), config.get("reasoning_workers", 4))
