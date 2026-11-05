"""Convenience commands for the current benchmark launchers and configuration files."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from scripts.results_layout import dataset_label, optimization_output, write_evaluation


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"hotpotqa": "hotpotqa", "amazon": "amazon", "pubmed": "pubmed", "pubmedqa": "pubmed"}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "optimize"):
        command = commands.add_parser(name, aliases=["run"] if name == "optimize" else [])
        command.set_defaults(command=name)
        command.add_argument("target", nargs="?", help="Dataset, YAML config, or (evaluate) run directory")
        command.add_argument("--system", choices=tuple(DATASETS), help="Alias for the dataset target")
        command.add_argument("--config", type=Path, help="Existing launcher YAML configuration")
        command.add_argument("--output", type=Path, help="Exact output directory")
        if name == "evaluate":
            command.add_argument("--unoptimized", action="store_true", help="Evaluate the initial factory system")
            command.add_argument("--repeats", type=positive_int, help="Override the evaluation profile's repetition count")
            command.add_argument("--split", choices=("test", "val"), help="Default: the evaluation profile's split")
        else:
            command.add_argument("--resume", "--checkpoint", type=Path, help="CAISO checkpoint directory")
            command.add_argument("--no-test", action="store_true", help="Skip post-optimization test evaluation")
    return result


def resolve(args):
    """Resolve existing files before importing the heavyweight benchmark modules."""
    if args.target and args.system:
        raise ValueError("Use a target or --system, not both")
    target = args.target or args.system
    run = None
    if args.config:
        if args.target:
            raise ValueError("Use a target or --config, not both")
        path = args.config
    elif target in DATASETS:
        dataset = DATASETS[target]
        label = "pubmedqa" if dataset == "pubmed" else dataset
        path = ROOT / "configs/experiments" / f"{label}.yaml"
    elif target and Path(target).is_file():
        path = Path(target)
    elif target and Path(target).is_dir() and args.command == "evaluate":
        run = Path(target)
        experiment_path = run / "experiment.json"
        if not experiment_path.is_file():
            raise ValueError(f"No saved experiment.json in {run}; pass an actual optimization run directory")
        experiment = json.loads(experiment_path.read_text())
        dataset = experiment["dataset"]
        path = ROOT / "scripts/configs/eval" / f"{dataset}_vanilla.yaml"
    else:
        raise ValueError("Specify hotpotqa, amazon, pubmedqa, a YAML config, or an evaluation run directory")
    if run and "evaluation" in experiment:
        config = {
            **experiment["evaluation"], "dataset": experiment["dataset"],
            "system_name": experiment["system"], "dotenv_path": experiment.get("dotenv_path", ".env"),
        }
    else:
        from caiso.integration.config import load_experiment

        config = load_experiment(path, mode=args.command)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a configuration mapping in {path}")
    if args.system and config.get("dataset") != DATASETS[args.system]:
        raise ValueError("--system does not match the configuration's dataset")
    if run:
        config["system_name"] = experiment["system"]
        config["dotenv_path"] = experiment.get("dotenv_path", ".env")
        if args.unoptimized:
            config["system_state_dict_path"] = None
        else:
            checkpoint = run / "final_optimized_system.pth"
            if not checkpoint.is_file():
                raise ValueError(f"Missing optimized checkpoint: {checkpoint}")
            config["system_state_dict_path"] = str(checkpoint)
        config["run_name"] = "unoptimized" if args.unoptimized else "optimized"
    if args.command == "evaluate":
        if args.unoptimized:
            config["system_state_dict_path"] = None
            config["run_name"] = "unoptimized"
        if args.repeats is not None:
            config["num_repeat"] = args.repeats
        if args.split is not None:
            config["split"] = args.split
    return path, config, run


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        path, config, run = resolve(args)
    except (ValueError, KeyError, OSError) as error:
        command_parser.error(str(error))
    # Load before importing factories: Amazon reads BASE_MODEL_PATH at import time.
    load_dotenv(config.get("dotenv_path", ".env"))
    if args.command == "optimize":
        from scripts.optimize_system import main as optimize
        from caiso.integration.config import load_method

        arguments = [str(path)]
        output = args.output
        if output is None and args.resume:
            checkpoint = args.resume
            if checkpoint.name == "checkpoint.json":
                checkpoint = checkpoint.parent
            if not (checkpoint / "checkpoint.json").is_file():
                command_parser.error(f"No checkpoint.json in {checkpoint}")
            output = checkpoint.parent
        if output is None:
            output = optimization_output(config, load_method(config))
        arguments.extend(("--output", str(output)))
        if args.resume:
            arguments.extend(("--resume", str(args.resume)))
        if args.no_test:
            arguments.append("--no-test")
        return optimize(arguments)

    from scripts.eval_system import main as evaluate

    arguments = []
    for key, value in config.items():
        if value is not None:
            arguments.extend((f"--{key}", str(value)))
    output = args.output
    variant = "optimized" if config.get("system_state_dict_path") else "unoptimized"
    if output is None:
        output = run if run else Path("results") / f"{dataset_label(config)}-{variant}"
    split = config.get("split", "test")
    raw_output = output / ".internal" / "evaluation" / variant / split
    metrics = evaluate(arguments, output_dir=raw_output)
    write_evaluation(
        output, metrics, variant=variant, split=split,
        state_source=config.get("system_state_dict_path"),
    )
    return metrics


if __name__ == "__main__":
    main()
