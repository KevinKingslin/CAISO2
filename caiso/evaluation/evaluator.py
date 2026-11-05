from caiso.core.content_filter import execute_example
from caiso.core.parallel import ordered_map


def evaluate(runtime, state, examples, seeds, trajectory_path=None, *, progress=None):
    if not examples or not seeds:
        raise ValueError("Evaluation requires inputs and seeds")
    trajectories = ordered_map(
        lambda job: execute_example(runtime, state, job[0], job[1]),
        [(example, seed) for example in examples for seed in seeds],
        max_workers=getattr(runtime, "max_workers", 1),
        progress=progress,
    )
    # Keep the requested denominator: filtering must not inflate task performance.
    requested = len(trajectories)
    trajectories = [trajectory for trajectory in trajectories if trajectory is not None]
    for trajectory in trajectories:
        trajectory.metadata["evaluation_coverage"] = {
            "requested": requested,
            "completed": len(trajectories),
            "skipped_content_filter": requested - len(trajectories),
        }
    # Only the caller writes trajectories, in input order, after workers finish.
    if trajectory_path:
        for trajectory in trajectories:
            trajectory.save(trajectory_path)
    return sum(trajectory.reward for trajectory in trajectories) / requested, trajectories
