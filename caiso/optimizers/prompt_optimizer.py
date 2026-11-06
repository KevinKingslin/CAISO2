"""Score-history prompt generation adapted from OPTIMAS OPRO, Apache-2.0."""

import random

from pydantic import Field

from caiso.core.serialization import Record
from caiso.prompts import entry as prompt_entry
from caiso.prompts import render_messages

from .base import ConfigurationOptimizer


class PromptProposal(Record):
    prompt: str = Field(min_length=1)


class PromptOptimizer(ConfigurationOptimizer):
    def propose(self, component, context):
        current = context.state.configurations[component.component_id]
        candidates, history = [current.model_copy(deep=True)], []
        tips = context.config.get(
            "prompt_tips",
            prompt_entry('method', 'prompt_optimizer.default_tips'),
        )
        if not tips:
            raise ValueError("prompt_tips must contain at least one instruction")
        rng = random.Random(context.seed)
        count = context.config.get("prompt_candidates", 3)
        for i in range(count):
            result = context.runtime.providers.complete(
                "prompt_optimizer",
                render_messages(
                    "method", "prompt_optimizer.request", tip=rng.choice(tips),
                    component=component, current_prompt=current.value,
                    constitution=context.constitution.principles,
                    history=sorted(history, key=lambda x: x["score"]),
                    examples=[
                        {
                            "inputs": t.inputs,
                            "executions": [
                                {"component_input": e.component_input, "component_output": e.component_output}
                                for e in t.executions
                                if e.component_id == component.component_id
                            ],
                        }
                        for t in context.trajectories
                    ],
                ),
                seed=context.seed + i,
                schema=PromptProposal,
                include_schema_instruction=False,
            )
            c = current.model_copy(deep=True)
            c.value = result.prompt
            if any(x.value == c.value for x in candidates):
                continue
            candidates.append(c)
            if context.score_config:
                score = context.score_config({component.component_id: c})
                history.append({"prompt": c.value, "score": score})
        return candidates
