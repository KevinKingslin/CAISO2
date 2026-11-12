"""Resolve method settings without changing any benchmark defaults."""

from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import yaml

from caiso.method.config import validate_method

DATASET_PERSONAS = {
    "amazon": "personas/amazon",
    "hotpotqa": "personas/hotpotqa",
    "pubmed": "personas/pubmedqa",
    "pubmedqa": "personas/pubmedqa",
}


def load_personas(experiment):
    """Resolve exactly three dataset personas; explicit inline overrides win."""
    from caiso.constitution.dicai.personas import assign_personas
    from caiso.prompts import catalog_entries, entry, validate_resources

    options = experiment.get("caiso", {})
    if "personas" in options:
        personas = deepcopy(options["personas"])
    else:
        catalog = options.get("persona_catalog", DATASET_PERSONAS.get(experiment.get("dataset")))
        if catalog is None:
            validate_resources({"method": ["default_personas"]})
            personas = entry("method", "default_personas")
        else:
            if not isinstance(catalog, str):
                raise ValueError("caiso.persona_catalog must name a prompt catalog")
            try:
                validate_resources({catalog: None})
                profiles = catalog_entries(catalog)
            except KeyError as error:
                raise ValueError(f"Unknown persona catalog: {catalog}") from error
            personas = [
                {"identity": identity, "prompt_text": profile}
                for identity, profile in profiles.items()
            ]
    if not isinstance(personas, list) or len(personas) != 3:
        raise ValueError("caiso.personas must contain exactly three personas")
    if any(isinstance(persona, str) and not persona.strip() for persona in personas):
        raise ValueError("Persona identity must be a nonempty string")
    try:
        assign_personas(personas, seed=0)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid CAISO personas: {error}") from error
    return personas


def load_experiment(path, mode="optimize"):
    """Read either a CAISO experiment or a preserved OPTIMAS launcher profile."""
    path = Path(path)
    experiment = yaml.safe_load(path.read_text())
    if not isinstance(experiment, dict):
        raise ValueError(f"Expected a configuration mapping in {path}")
    if "method_config" not in experiment:
        if mode == "optimize":
            experiment.setdefault("caiso", {})["personas"] = load_personas(experiment)
        return experiment
    common = {"dataset": experiment["dataset"], "dotenv_path": experiment.get("dotenv_path", ".env")}
    if mode == "evaluate":
        return {**common, "system_name": experiment["system"], **experiment["evaluation"]}
    method_file = Path(experiment["method_config"])
    if not method_file.is_absolute():
        method_file = (path.parent / method_file).resolve()
    options = merge(experiment.get("caiso", {}), {
        "method_file": str(method_file),
        "method": merge(experiment.get("method", {}), experiment.get("caiso", {}).get("method", {})),
    })
    resolved = {
        **common, "system": experiment["system"],
        **experiment["optimization"], "evaluation": experiment["evaluation"], "caiso": options,
    }
    resolved["caiso"]["personas"] = load_personas(resolved)
    return resolved


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else deepcopy(value)
        )
    return result


def validate_optimization_prompts(method, dataset=None):
    """Check resources needed by this run before models, data or paid calls."""
    from caiso.prompts import CATALOGS, validate_resources

    keys = ["block_selection.request", "preferences.labeled_pair", "providers.format_retry"]
    from caiso.constitution.dicai.atomicity import PROMPT_KEYS, enabled
    if enabled(method.get("dicai", {})):
        keys.extend(PROMPT_KEYS)
    optimization = method.get("block_optimization", {})
    if optimization.get("prompt_candidates", 3) > 0:
        keys.append("prompt_optimizer.request")
        if "prompt_tips" not in optimization:
            keys.append("prompt_optimizer.default_tips")
    workflow = method.get("workflow", {})
    if workflow.get("enabled", False):
        keys.append("generator.request")
        search = workflow.get("compatibility_search", {})
        if search.get("prompt_candidates", 3) > 0:
            keys.append("prompt_optimizer.request")
            if "prompt_tips" not in search:
                keys.append("prompt_optimizer.default_tips")
    selection = {"dicai": None, "method": keys}
    catalog = "systems/" + ("pubmed" if dataset == "pubmedqa" else str(dataset))
    if catalog in CATALOGS:
        selection[catalog] = None
    return validate_resources(selection)


def load_method(experiment):
    options = experiment.get("caiso", {})
    path = options.get("method_file")
    text = (
        Path(path).read_text()
        if path
        else files("caiso").joinpath("config/default.yaml").read_text()
    )
    config = yaml.safe_load(text)
    # Only names the stored benchmark scores; never changes the task evaluator.
    config["dicai"].setdefault(
        "score_metric", "answer_f1" if experiment.get("dataset") == "hotpotqa" else "task_metric"
    )
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
