"""Persona-conditioned reasoning from arXiv:2606.28294v2, Appendix G.1.

Each paper strategy returns one final_reasoning paragraph in a single request.
"""

from pydantic import Field

from caiso.core.parallel import ordered_map
from caiso.core.serialization import Record
from caiso.prompts import render_messages

from .preferences import candidate_views, task_view


def valid_principle(value):
    value = value.strip()
    if not value.startswith("Select the response") or value == "Select the response":
        raise ValueError('Invalid DICAI principle: begin with "Select the response"')
    return value


class Reasoning(Record):
    final_reasoning: str = Field(min_length=1)


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
            temperature=temperatures.get(temperature_key, default),
        )
        return {
            "persona": persona.model_dump(),
            "reasoning": result.final_reasoning,
            "principles": [],  # Questions are proposed during debate, not added to the paper schema.
            "trace": [{"stage": stage, "output": result.model_dump()}],
            "calls_before_retries": 1,
        }

    return ordered_map(reason, enumerate(personas), config.get("reasoning_workers", 4))
