"""Run CAISO on the original OPTIMAS experiment factories and datasets.

The launcher replaces reward-model initialization/training with CAISO. Released
experiment files and their model/data defaults are otherwise used as supplied.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
import yaml

from caiso.core.serialization import save_json
from caiso.integration.config import load_method
from caiso.integration.provenance import benchmark_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Released experiment YAML, optionally with a caiso section")
    parser.add_argument("--output", type=Path, help="Exact output directory (otherwise use the experiment paths)")
    parser.add_argument("--resume", help="CAISO checkpoint directory")
    parser.add_argument(
        "--no-test", action="store_true", help="Optimize without evaluating the test split"
    )
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    method = load_method(config)  # Fail invalid settings before loading models/data.
    load_dotenv(config.get("dotenv_path", ".env"))

    import torch
    from examples.systems import registered_systems
    from examples.datasets import registered_datasets
    from caiso.eval import eval_system
    from caiso.optim.optimizer import CaisoOptimizer

    output = (
        Path(config.get("output_dir", "outputs/optim"))
        / config["dataset"]
        / config["system"]
        / config.get("wandb_run_name", "caiso")
    )
    if args.output is not None:
        output = args.output
    output.mkdir(parents=True, exist_ok=True)
    system_factory = registered_systems[config["system"]]
    dataset_factory = registered_datasets[config["dataset"]]
    if system_factory is None or dataset_factory is None:
        raise RuntimeError(
            "The selected benchmark's optional dependencies are missing; see the import diagnostics above"
        )
    system = system_factory(
        log_dir=str(output), max_sample_workers=config.get("max_sample_workers", 4)
    )
    trainset, valset, testset = dataset_factory()
    save_json(
        output / "benchmark_manifest.json",
        benchmark_manifest(
            system,
            {"train": trainset, "validation": valset, "test": testset or []},
            method,
        ),
    )
    # The optimizer applies the released valset[:val_size] prefix to every validation
    # phase; an explicit method override can request 'all'. Test data stays separate.
    if config.get("system_state_dict_path"):
        system.load_state_dict(torch.load(config["system_state_dict_path"], weights_only=False))
    torch.save(system.state_dict(), output / "initial_system.pth")
    optimizer = CaisoOptimizer(system, trainset, valset, config, output)
    optimized, metrics = optimizer.optimize(resume=args.resume)
    save_json(output / "optimization_history.json", metrics)
    torch.save(optimized.state_dict(), output / "final_optimized_system.pth")
    save_json(output / "final_adapter_state.json", optimizer._get_current_adapter_state())
    if not args.no_test and testset:
        evaluation, _ = eval_system(optimized, testset, num_repeat=config.get("num_repeat", 1))
        save_json(output / "eval_metrics.json", evaluation)
        print(json.dumps(evaluation, indent=2))
    print(f"CAISO results: {output}")


if __name__ == "__main__":
    main()
