"""Early validation for user-editable method parameters; defaults live in YAML."""

import math


def validate_method(config):
    if config.get("validation_sampling", "seeded") not in {"seeded", "first"}:
        raise ValueError("validation_sampling must be seeded or first")
    recovery = config.get("error_handling", {})
    if not isinstance(recovery, dict):
        raise ValueError("error_handling must be a configuration mapping")
    if any(value is None for value in recovery.values()):
        raise ValueError(
            "error_handling settings cannot be null; omit a setting to use its default"
        )
    positive = (
        "validation_batch_size",
        "execution.max_workers",
        "training_batch_size",
        "initial_preference_pairs",
        "dicai.debate_rounds",
        "dicai.max_workers",
        "dicai.reasoning_workers",
        "dicai.self_consistency_samples",
        "dicai.min_debate_rounds",
        "dicai.min_principles",
        "dicai.reconstruction_votes",
        "dicai.history_turns",
        "dicai.clustering.min_clusters",
        "dicai.clustering.max_clusters",
        "dicai.clustering.cluster_step",
        "attribution.batch_size",
        "attribution.stochastic_repeats",
        "block_selection.diagnostic_batch_size",
        "block_constitution.preference_pairs",
        "block_optimization.max_joint_candidates",
        "workflow.candidates",
        "workflow.trigger_every_k_accepted_blocks",
        "workflow.plateau_patience",
        "workflow.compatibility_search.training_batch_size",
        "workflow.compatibility_search.max_joint_candidates",
        "reporting.refresh_per_second",
        "reporting.plot_dpi",
        "budget.max_equivalent_full_runs",
        "budget.max_wallclock_hours",
        "budget.max_llm_calls",
        "error_handling.max_consecutive_errors",
    )
    nonnegative = (
        "max_iterations",
        "dicai.self_refine_steps",
        "dicai.minimization.sample_size",
        "attribution.bootstrap_samples",
        "leiden.resolution_gamma",
        "block_constitution.score_tie_epsilon",
        "block_optimization.acceptance_epsilon",
        "block_optimization.prompt_candidates",
        "block_optimization.softmax_temperature",
        "workflow.compatibility_budget",
        "workflow.compatibility_search.prompt_candidates",
        "budget.overhead_call_cost",
        "error_handling.response_retries",
        "error_handling.operation_retries",
        "error_handling.backoff_seconds",
        "error_handling.max_backoff_seconds",
    )

    def value(path):
        node = config
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None  # Backward-compatible optional fields use module defaults.
            node = node[part]
        return node

    attempts = value("initial_preference_attempts")
    if attempts is not None and (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or (attempts != -1 and attempts < 1)
    ):
        raise ValueError("initial_preference_attempts must be -1 or a positive integer")

    for path in (*positive, *nonnegative):
        number = value(path)
        if path == "validation_batch_size" and number == "all":
            continue
        if number is not None and (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or number < 0
            or not math.isfinite(number)
            or (path in positive and number == 0)
        ):
            raise ValueError(f"{path} must be {'positive' if path in positive else 'nonnegative'}")
    integers = [
        p
        for p in positive
        if p
        not in {
            "budget.max_equivalent_full_runs",
            "budget.max_wallclock_hours",
            "reporting.refresh_per_second",
        }
    ]
    integers += [
        "max_iterations",
        "dicai.self_refine_steps",
        "dicai.minimization.sample_size",
        "dicai.clustering.fixed_clusters",
        "attribution.bootstrap_samples",
        "block_optimization.prompt_candidates",
        "workflow.compatibility_search.prompt_candidates",
        "leiden.n_iterations",
        "error_handling.response_retries",
        "error_handling.operation_retries",
    ]
    for path in integers:
        number = value(path)
        if path == "validation_batch_size" and number == "all":
            continue
        if number is not None and not isinstance(number, int):
            raise ValueError(f"{path} must be an integer")
    if (
        value("leiden.n_iterations") is not None
        and value("leiden.n_iterations") not in (-1,)
        and value("leiden.n_iterations") < 1
    ):
        raise ValueError("leiden.n_iterations must be -1 or positive")
    if value("dicai.personas") not in (None, 3):
        raise ValueError("The DICAI appendix protocol requires exactly three personas")
    dicai = config.get("dicai", {})
    if not isinstance(dicai, dict):
        raise ValueError("dicai must be a configuration mapping")
    for key, setting in dicai.items():
        if setting is None:
            raise ValueError(f"dicai.{key} cannot be null; omit it to use its default")
    for subsection in ("clustering", "minimization", "temperatures"):
        if subsection in dicai and not isinstance(dicai[subsection], dict):
            raise ValueError(f"dicai.{subsection} must be a configuration mapping")
        for key, setting in dicai.get(subsection, {}).items():
            if setting is None and (subsection, key) not in {
                ("clustering", "fixed_clusters"),
                ("clustering", "max_clusters"),
            }:
                raise ValueError(f"dicai.{subsection}.{key} cannot be null")
    if "n_init" in config.get("dicai", {}).get("clustering", {}):
        raise ValueError(
            "dicai.clustering.n_init belongs to the old KMeans implementation; "
            "use the current method YAML and a fresh run for official DICAI"
        )
    rounds = value("dicai.debate_rounds") or 4
    minimum = value("dicai.min_debate_rounds") or 1
    if minimum > rounds:
        raise ValueError("dicai.min_debate_rounds cannot exceed dicai.debate_rounds")
    if (
        value("dicai.clustering.fixed_clusters") is not None
        and value("dicai.clustering.fixed_clusters") < 2
    ):
        raise ValueError("dicai.clustering.fixed_clusters must be at least 2 or null")
    for key, number in config.get("dicai", {}).get("temperatures", {}).items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 2:
            raise ValueError(f"dicai.temperatures.{key} must be between zero and two")
    for path in ("dicai.minimization.enabled", "dicai.minimization.strict"):
        if value(path) is not None and not isinstance(value(path), bool):
            raise ValueError(f"{path} must be true or false")
    if value("attribution.baseline_policy") not in (None, "per_edge"):
        raise ValueError(
            "Set attribution.baseline_policy to per_edge; edit baselines in system YAML"
        )
    if value("dicai.clustering.selection") not in (None, "silhouette"):
        raise ValueError("dicai.clustering.selection must be silhouette")
    lower, upper = value("dicai.clustering.min_clusters"), value("dicai.clustering.max_clusters")
    if (lower is not None and lower < 2) or (
        lower is not None and upper is not None and lower > upper
    ):
        raise ValueError("dicai.clustering requires 2 <= min_clusters <= max_clusters")
    level = value("attribution.confidence_level")
    if level is not None and not 0 < level < 1:
        raise ValueError("attribution.confidence_level must be between zero and one")
    if (
        value("error_handling.max_backoff_seconds") is not None
        and value("error_handling.max_backoff_seconds") > 60
    ):
        raise ValueError("error_handling.max_backoff_seconds must be at most 60")
    if value("error_handling.continue_after_errors") is not None and not isinstance(
        value("error_handling.continue_after_errors"), bool
    ):
        raise ValueError("error_handling.continue_after_errors must be true or false")
    if value("block_constitution.reuse_trajectories") is not None and not isinstance(
        value("block_constitution.reuse_trajectories"), bool
    ):
        raise ValueError("block_constitution.reuse_trajectories must be true or false")
    if "execution" in config and not isinstance(config["execution"], dict):
        raise ValueError("execution must be a configuration mapping")
    for path in (
        "dicai.max_workers",
        "execution.max_workers",
        "block_constitution.reuse_trajectories",
    ):
        section, key = path.split(".")
        if key in config.get(section, {}) and value(path) is None:
            raise ValueError(f"{path} cannot be null; omit it to use its default")
