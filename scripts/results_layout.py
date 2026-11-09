"""Historical CAISO result locations with scores from the current evaluator."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4


def dataset_label(config):
    dataset = config["dataset"]
    return "pubmedqa" if dataset == "pubmed" else dataset


def optimization_output(config, method):
    payload = json.dumps([config, method], sort_keys=True, separators=(",", ":"))
    identifier = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return Path("results") / f"{dataset_label(config)}-{identifier}"


def manual_output(config):
    return Path("results") / f"{dataset_label(config)}-manual-{uuid4().hex[:12]}"


def write_evaluation(directory, metrics, *, variant, split, state_source=None, state_snapshot=None):
    """Keep historical filenames without changing scoring or inventing seeds."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "evaluation" if split == "test" else "validation"
    suffix = "_" + variant if variant in {"unoptimized", "current", "best"} else ""
    path = directory / f"{stem}{suffix}.json"
    report = {
        **metrics,
        "system_variant": variant,
        "state_source": str(state_source) if state_source else "initial_system_factory",
        "split": split,
        "score": metrics["mean"],
        "score_mean": metrics["mean"],
        "score_std": metrics["std"],
        "per_repeat": metrics.get("per_repeat") or [
            {"repeat": int(key.removeprefix("trial_")), "score": metrics[key]}
            for key in sorted(
                (key for key in metrics if key.startswith("trial_")),
                key=lambda key: int(key.removeprefix("trial_")),
            )
        ],
    }
    if state_snapshot is not None:
        report["state_snapshot"] = state_snapshot
    from caiso.core.serialization import save_json
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        save_json(temporary, report)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Evaluation results: {path}")
    return path
