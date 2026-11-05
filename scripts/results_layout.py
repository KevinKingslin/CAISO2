"""Historical CAISO result locations with scores from the current evaluator."""

import hashlib
import json
from pathlib import Path


def dataset_label(config):
    dataset = config["dataset"]
    return "pubmedqa" if dataset == "pubmed" else dataset


def optimization_output(config, method):
    payload = json.dumps([config, method], sort_keys=True, separators=(",", ":"))
    identifier = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return Path("results") / f"{dataset_label(config)}-{identifier}"


def write_evaluation(directory, metrics, *, variant, split, state_source=None):
    """Keep historical filenames without changing scoring or inventing seeds."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "evaluation" if split == "test" else "validation"
    suffix = "_unoptimized" if variant == "unoptimized" else ""
    path = directory / f"{stem}{suffix}.json"
    report = {
        **metrics,
        "system_variant": variant,
        "state_source": str(state_source) if state_source else "initial_system_factory",
        "split": split,
        "score": metrics["mean"],
        "score_mean": metrics["mean"],
        "score_std": metrics["std"],
        "per_repeat": [
            {"repeat": int(key.removeprefix("trial_")), "score": metrics[key]}
            for key in sorted(
                (key for key in metrics if key.startswith("trial_")),
                key=lambda key: int(key.removeprefix("trial_")),
            )
        ],
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Evaluation results: {path}")
    return path
