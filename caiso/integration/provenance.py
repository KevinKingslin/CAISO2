"""Record benchmark identity and software versions without recording credentials."""

from importlib.metadata import PackageNotFoundError, version

from caiso.core.serialization import digest

UPSTREAM_COMMIT = "71871a2dd98081231d733debd41178ba47e13528"


def benchmark_manifest(system, splits, method):
    packages = {}
    for package in (
        "caiso2",
        "dspy",
        "litellm",
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "pandas",
        "numpy",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "upstream_repository": "https://github.com/snap-stanford/optimas",
        "upstream_commit": UPSTREAM_COMMIT,
        "datasets": {
            name: {
                "size": len(examples),
                "sha256": digest([example.to_dict() for example in examples]),
            }
            for name, examples in splits.items()
        },
        "validation_prefix_size": method["validation_batch_size"],
        "validation_sampling": method.get("validation_sampling", "first"),
        "validation_repeats": method.get("block_optimization", {}).get("validation_repeats", 1),
        "diagnostics_split": method.get("diagnostics_split", "validation"),
        "optimization_protocol": method.get("optimization_protocol", "legacy"),
        "components": {
            cid: {
                "class": f"{type(comp).__module__}.{type(comp).__qualname__}",
                "inputs": comp.input_fields,
                "outputs": comp.output_fields,
                "model": getattr(comp.config, "model", None),
                "base_model_path": getattr(comp.config, "base_model_path", None),
                "temperature": getattr(comp.config, "temperature", None),
                "max_tokens": getattr(comp.config, "max_tokens", None),
                "max_new_tokens": getattr(comp.config, "max_new_tokens", None),
            }
            for cid, comp in system.components.items()
        },
        "packages": packages,
        "task_generation_seeded": False,
        "budget_notes": "Equivalent runs count component executions across all CAISO phases. LLM/token counters cover method services, not HTTP calls inside unchanged benchmark components.",
    }
