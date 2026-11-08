"""Convenience commands for the current benchmark launchers and configuration files."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

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
            state = command.add_mutually_exclusive_group()
            state.add_argument("--unoptimized", action="store_true", help="Evaluate the initial factory system")
            state.add_argument("--state", choices=("current", "best", "final"),
                               help="Run state to evaluate; default: final export if present, otherwise current checkpoint")
            command.add_argument("--repeats", type=positive_int, help="Override the evaluation profile's repetition count")
            command.add_argument("--split", choices=("test", "val"), help="Default: the evaluation profile's split")
            command.add_argument(
                "--workers", type=positive_int,
                help="Parallel evaluation examples; default: the evaluation profile's max_eval_workers",
            )
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
            selection = args.state or ("final" if checkpoint.is_file() else "current")
            if selection == "final":
                if not checkpoint.is_file():
                    raise ValueError(f"Missing optimized checkpoint: {checkpoint}")
                config["system_state_dict_path"] = str(checkpoint)
            else:
                checkpoint = run / "checkpoint/checkpoint.json"
                if not checkpoint.is_file():
                    raise ValueError(
                        f"Missing optimized checkpoint: {checkpoint}. No saved {selection} state is available yet; "
                        "wait for initialization to finish or use --unoptimized explicitly."
                    )
                config["system_state_dict_path"] = None
                config["caiso_checkpoint_path"] = str(checkpoint)
                config["caiso_checkpoint_state"] = selection
        config["run_name"] = "unoptimized" if args.unoptimized else config.get("caiso_checkpoint_state", "optimized")
    if args.command == "evaluate":
        if args.state and not run:
            raise ValueError("--state requires an optimization run directory")
        if args.unoptimized:
            config["system_state_dict_path"] = None
            config["run_name"] = "unoptimized"
        if args.repeats is not None:
            config["num_repeat"] = args.repeats
        if args.split is not None:
            config["split"] = args.split
        if args.workers is not None:
            config["max_eval_workers"] = args.workers
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

    snapshot = None
    checkpoint_path = config.pop("caiso_checkpoint_path", None)
    checkpoint_state = config.pop("caiso_checkpoint_state", None)
    variant = checkpoint_state or ("optimized" if config.get("system_state_dict_path") else "unoptimized")
    output = args.output
    if output is None:
        output = run if run else Path("results") / f"{dataset_label(config)}-{variant}"
    split = config.get("split", "test")
    raw_output = output / ".internal" / "evaluation" / variant / split
    if checkpoint_path:
        from caiso.integration.evaluation_snapshot import freeze_checkpoint

        raw_output /= uuid4().hex
        try:
            snapshot = freeze_checkpoint(checkpoint_path, checkpoint_state, raw_output)
        except (ValueError, OSError) as error:
            command_parser.error(f"Cannot load saved {checkpoint_state} system: {error}")
        print(f"Evaluating {snapshot.selection} system from checkpoint iteration {snapshot.iteration} "
              f"(validation score={snapshot.validation_score:.6f}, state={snapshot.state_fingerprint[:12]})")
    arguments = []
    for key, value in config.items():
        if value is not None:
            arguments.extend((f"--{key}", str(value)))
    options = {"output_dir": raw_output, "results_dir": output}
    if snapshot is not None:
        options.update(state_snapshot=snapshot, results_dir=raw_output)
    metrics = evaluate(arguments, **options)
    write_evaluation(
        output, metrics, variant=variant, split=split,
        state_source=raw_output / "state_snapshot.json" if snapshot else config.get("system_state_dict_path"),
        state_snapshot=snapshot.provenance() if snapshot else None,
    )
    return metrics


if __name__ == "__main__":
    main()
