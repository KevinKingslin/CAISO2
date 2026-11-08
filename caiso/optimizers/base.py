from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from caiso.core.configuration import Configuration
from caiso.core.serialization import Record, digest


def filter_current_configuration(current, proposals, config):
    """Exclude unchanged execution values, even if proposal metadata differs."""
    include = config.get("include_current_configuration", True)
    if not isinstance(include, bool):
        raise ValueError("include_current_configuration must be true or false")
    if include:
        return proposals
    current_key = digest([current.kind, current.value])
    return [p for p in proposals if digest([p.kind, p.value]) != current_key]


class CandidateScore(Record):
    configurations: dict[str, Configuration]
    constitution_score: float
    task_score: float | None = None
    score: float


@dataclass
class OptimizationContext:
    runtime: Any
    state: Any
    constitution: Any
    judge: Any
    examples: list
    trajectories: list
    seed: int
    config: dict
    output_dir: Path
    score_config: Callable | None = None


class ConfigurationOptimizer(ABC):
    @abstractmethod
    def propose(self, component, context) -> list[Configuration]: ...

    def score(self, component, candidates, context):
        if context.score_config is None:
            raise ValueError("A whole-constitution scoring callback is required")
        return [context.score_config({component.component_id: c}) for c in candidates]

    def select(self, candidates, scores, seed=0, final=False):
        if not candidates or len(candidates) != len(scores):
            raise ValueError("Candidates and scores must be nonempty and aligned")
        return candidates[max(range(len(candidates)), key=lambda i: scores[i])]

    def apply(self, state, component_id, configuration):
        # Returns a candidate state only. Global acceptance is owned by the method layer.
        return state.with_configs({component_id: configuration})
