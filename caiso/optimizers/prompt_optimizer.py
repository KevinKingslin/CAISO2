"""Score-history prompt generation adapted from OPTIMAS OPRO, Apache-2.0."""

import random

from pydantic import Field

from caiso.core.serialization import Record, canonical
from caiso.providers.base import messages

from .base import ConfigurationOptimizer


class PromptProposal(Record):
    prompt: str = Field(min_length=1)


class PromptOptimizer(ConfigurationOptimizer):
    def propose(self, component, context):
        current = context.state.configurations[component.component_id]
        candidates, history = [current.model_copy(deep=True)], []
        tips = context.config.get(
            "prompt_tips",
            [
                "Keep the instruction clear and concise.",
                "Make the instruction informative and descriptive.",
                "Include specific conditions when useful.",
                "Consider a different formulation of the instruction.",
            ],
        )
        if not tips:
            raise ValueError("prompt_tips must contain at least one instruction")
        rng = random.Random(context.seed)
        count = context.config.get("prompt_candidates", 3)
        for i in range(count):
            result = context.runtime.providers.complete(
                "prompt_optimizer",
                messages(
                    "You are an AI specialized in improving prompts for a certain component. "
                    "Use the complete block constitution. Previous prompts and scores are ordered "
                    "from lowest to highest. Propose an improved instruction. "
                    + rng.choice(tips)
                    + " Return JSON with a prompt field.",
                    canonical(
                        {
                            "component": component,
                            "current_prompt": current.value,
                            "constitution": context.constitution.principles,
                            "history": sorted(history, key=lambda x: x["score"]),
                            "examples": [
                                {
                                    "inputs": t.inputs,
                                    "executions": [
                                        {
                                            "component_input": e.component_input,
                                            "component_output": e.component_output,
                                        }
                                        for e in t.executions
                                        if e.component_id == component.component_id
                                    ],
                                }
                                for t in context.trajectories
                            ],
                        }
                    ),
                ),
                seed=context.seed + i,
                schema=PromptProposal,
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
