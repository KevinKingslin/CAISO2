"""Score-history prompt generation adapted from OPTIMAS OPRO, Apache-2.0."""

import random

from pydantic import Field

from caiso.core.serialization import Record
from caiso.prompts import entry as prompt_entry
from caiso.prompts import render_messages

from .base import ConfigurationOptimizer, filter_current_configuration


def training_feedback(component, context):
    """Bounded, failure-first optimizer evidence; never changes inference inputs."""
    targets = {example.example_id: example.target for example in getattr(context, "examples", [])}
    limit = context.config.get("feedback_examples", 8)
    max_chars = context.config.get("feedback_max_chars", 2000)

    def bounded(value):
        if isinstance(value, str):
            return value if len(value) <= max_chars else value[:max_chars] + " [truncated]"
        if isinstance(value, dict):
            return {key: bounded(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [bounded(item) for item in value[:20]]
        return value

    ordered = sorted(context.trajectories, key=lambda trajectory: (trajectory.reward, trajectory.input_id))
    selected = ordered[:limit]
    if len(ordered) > limit and limit > 1 and ordered[-1].reward > selected[-1].reward:
        selected[-1] = ordered[-1]  # Include a successful contrast when available.
    return [bounded({
        "inputs": trajectory.inputs,
        "target": targets.get(trajectory.input_id),
        "final_output": trajectory.final_output,
        "task_score": trajectory.reward,
        "executions": [
            {"component_input": execution.component_input, "component_output": execution.component_output}
            for execution in trajectory.executions if execution.component_id == component.component_id
        ],
        "downstream_executions": [
            {"component_id": execution.component_id, "component_output": execution.component_output}
            for execution in trajectory.executions[
                next((index + 1 for index, execution in enumerate(trajectory.executions)
                      if execution.component_id == component.component_id), len(trajectory.executions)):
            ]
        ],
    }) for trajectory in selected]


class PromptProposal(Record):
    prompt: str = Field(min_length=1)


class PromptOptimizer(ConfigurationOptimizer):
    def propose(self, component, context):
        current = context.state.configurations[component.component_id]
        candidates = [current.model_copy(deep=True)]
        history = list(getattr(context, "prompt_history", []))
        count = context.config.get("prompt_candidates", 3)
        if count == 0:
            return filter_current_configuration(current, candidates, context.config)
        tips = (
            context.config["prompt_tips"]
            if "prompt_tips" in context.config
            else prompt_entry("method", "prompt_optimizer.default_tips")
        )
        if not tips:
            raise ValueError("prompt_tips must contain at least one instruction")
        if context.score_config and not any(item["prompt"] == current.value for item in history):
            history.append({"prompt": current.value, "score": context.score_config({component.component_id: current})})
        feedback = training_feedback(component, context)
        rng = random.Random(context.seed)
        for i in range(count):
            result = context.runtime.providers.complete(
                "prompt_optimizer",
                render_messages(
                    "method", "prompt_optimizer.request", tip=rng.choice(tips),
                    component=component, current_prompt=current.value,
                    constitution=context.constitution.principles,
                    history=sorted(history, key=lambda x: x["score"]),
                    examples=feedback,
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
            if getattr(context, "record_prompt", None):
                context.record_prompt(c.value)
            if context.score_config:
                score = context.score_config({component.component_id: c})
                history.append({"prompt": c.value, "score": score})
        return filter_current_configuration(current, candidates, context.config)
