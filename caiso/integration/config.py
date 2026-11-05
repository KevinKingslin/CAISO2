"""Resolve method settings without changing any benchmark defaults."""

from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import yaml

from caiso.method.config import validate_method


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def load_method(experiment):
    options = experiment.get("caiso", {})
    path = options.get("method_file")
    text = (
        Path(path).read_text()
        if path
        else files("caiso").joinpath("config/default.yaml").read_text()
    )
    config = yaml.safe_load(text)
    # These sizes come from the SAME released experiment files, not CAISO copies.
    config = merge(
        config,
        {
            "max_iterations": experiment.get("num_iterations", 10),
            "training_batch_size": experiment.get("per_component_train_size", 20),
            "validation_batch_size": experiment.get("val_size", 10),
            "validation_sampling": "first",
            "attribution": {"batch_size": experiment.get("per_iteration_new_data_size", 20)},
            "block_selection": {
                "diagnostic_batch_size": experiment.get("per_component_train_size", 20)
            },
            "block_optimization": {"prompt_candidates": experiment.get("num_prompt_candidates", 3)},
        },
    )
    config = merge(config, options.get("method", {}))
    if config.get("workflow", {}).get("enabled"):
        raise ValueError("workflow.enabled must be false to keep the original benchmark systems")
    validate_method(config)
    return config
