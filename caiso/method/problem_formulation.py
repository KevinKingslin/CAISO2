"""Section 4.1, equations 1-3: S=(G,v), maximize expected task reward."""

from statistics import mean


def objective(trajectories):
    if not trajectories:
        raise ValueError("Objective requires at least one execution")
    return mean(t.reward for t in trajectories)
