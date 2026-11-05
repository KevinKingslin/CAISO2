"""Section 4.7, equation 18: joint proposal and global validation acceptance."""

import math

from caiso.core.serialization import Record
from caiso.evaluation.evaluator import evaluate


class Acceptance(Record):
    accepted: bool
    current_score: float
    candidate_score: float
    improvement: float


def acceptance_gate(current_score, candidate_score, epsilon=1e-6):
    if epsilon < 0 or not all(map(math.isfinite, [current_score, candidate_score])):
        raise ValueError("Acceptance needs finite scores and nonnegative epsilon")
    delta = candidate_score - current_score
    return Acceptance(
        accepted=delta > 0 and delta >= epsilon,
        current_score=current_score,
        candidate_score=candidate_score,
        improvement=delta,
    )


def validate_candidate(
    runtime,
    current,
    candidate,
    validation,
    seeds,
    epsilon=1e-6,
    trajectory_path=None,
    *,
    on_evaluation=None,
    progress=None,
):
    a, current_trajectories = evaluate(
        runtime,
        current,
        validation,
        seeds,
        trajectory_path,
        progress=(lambda done, total: progress("Current system", done, total))
        if progress
        else None,
    )
    if on_evaluation:
        on_evaluation("current", a, current_trajectories)
    b, candidate_trajectories = evaluate(
        runtime,
        candidate,
        validation,
        seeds,
        trajectory_path,
        progress=(lambda done, total: progress("Candidate system", done, total))
        if progress
        else None,
    )
    if on_evaluation:
        on_evaluation("candidate", b, candidate_trajectories)
    gate = acceptance_gate(a, b, epsilon)
    return (
        candidate.model_copy(deep=True) if gate.accepted else current.model_copy(deep=True)
    ), gate
