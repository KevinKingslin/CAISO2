import os
import os.path as osp
import sys
import json
from dataclasses import dataclass
from typing import Optional

import torch
from dotenv import load_dotenv
from transformers import HfArgumentParser

from examples.systems import registered_systems
from examples.datasets import registered_datasets
from caiso.utils.logger import setup_logger
from caiso.eval import eval_system
from caiso.integration.config import load_experiment
from caiso.providers.transport import configure_transport
from pathlib import Path
from caiso.integration.evaluation_snapshot import apply_snapshot, snapshot_models
from uuid import uuid4
from types import SimpleNamespace


@dataclass
class ScriptArgs:
    """
    Configuration class for system evaluation and optimization script.
    """
    # Model and Pipeline Configuration
    run_name: str = "default"
    dataset: str = "hotpotqa"
    system_name: str = "hotpotqa_system"
    num_repeat: int = 3
    split: str = "test"  # Options: "val", "test"
    interactive: bool = False  # Read each component's configuration from the terminal.

    # Paths and Environment
    system_state_dict_path: Optional[str] = None
    output_dir: str = "outputs/optim"
    dotenv_path: str = ".env"

    # Worker Config (injected dynamically)
    max_sample_workers: int = 4
    max_eval_workers: int = 2

    def __post_init__(self):
        """Convert string 'None' values to None objects."""
        for field_name, field_value in vars(self).items():
            if field_value == "None":
                setattr(self, field_name, None)


def main(argv=None, *, output_dir=None, results_dir=None, state_snapshot=None):
    parser = HfArgumentParser(ScriptArgs)
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) == 1 and argv[0].endswith(".yaml"):
        yaml_config = load_experiment(argv[0], mode="evaluate")
        args = parser.parse_dict(yaml_config)[0]
    else:
        args = parser.parse_args_into_dataclasses(args=argv)[0]
    if state_snapshot is not None and args.system_state_dict_path:
        raise ValueError("Choose a CAISO snapshot or a .pth system state, not both")
    if args.interactive and (state_snapshot is not None or args.system_state_dict_path):
        raise ValueError("Interactive evaluation starts from the factory system; omit saved system states")

    # Load environment variables
    load_dotenv(args.dotenv_path)

    # Setup output directory
    args.output_dir = (
        str(output_dir) if output_dir is not None
        else osp.join(args.output_dir, args.dataset, args.system_name, args.run_name)
    )
    if args.interactive and output_dir is None:
        args.output_dir = osp.join(args.output_dir, "manual-" + uuid4().hex[:12])
    results_dir = Path(results_dir or args.output_dir)
    if args.interactive and results_dir.exists() and any(results_dir.iterdir()):
        raise ValueError("This output directory already contains a run; choose a new --output directory")
    os.makedirs(args.output_dir, exist_ok=True)
    limits = configure_transport(results_dir)

    logger = setup_logger(__name__, log_file=osp.join(args.output_dir, "output.log"))
    logger.info("Starting evaluation script.")
    logger.info(f"Azure transport: {limits['requests_per_minute']} requests/minute, {limits['concurrency']} concurrent per deployment")
    logger.info(f"Parsed arguments:\n{json.dumps(vars(args), indent=2)}")

    with snapshot_models(state_snapshot, args.system_name):
        # Construct system
        logger.info(f"Instantiating system: {args.system_name}")
        system_kwargs = {
            "log_dir": args.output_dir,
            "max_sample_workers": args.max_sample_workers,
            "max_eval_workers": args.max_eval_workers,
        }
        system = registered_systems[args.system_name](**system_kwargs)
        system._caiso_skip_log = results_dir / ".internal/logs/skipped_examples.jsonl"
        system._caiso_eval_split = args.split
        system._caiso_eval_variant = "optimized" if args.system_state_dict_path else "unoptimized"

        if state_snapshot is not None:
            apply_snapshot(system, state_snapshot)
            system._caiso_eval_variant = state_snapshot.selection

        if args.interactive:
            from caiso.core.serialization import save_json
            from caiso.integration.evaluation_snapshot import save_state_dict
            from caiso.integration.interactive_evaluation import configure_interactively

            components = configure_interactively(system)
            system._caiso_eval_variant = "manual"
            save_state_dict(system.state_dict(), results_dir / "manual_system.pth")
            save_json(results_dir / "component_config.json", {
                "system_name": args.system_name, "dataset": args.dataset,
                "evaluation": vars(args), "components": components,
            })
            print(f"Manual configuration saved: {results_dir / 'component_config.json'}")

        cur_system_state_dict = system.state_dict()
        logger.info("Saving current system state.")
        torch.save(cur_system_state_dict, osp.join(args.output_dir, "system_state_dict.pth.bak"))

        # Load evaluation datasets
        trainset, valset, testset = registered_datasets[args.dataset](**vars(args))

        # Restore state if specified
        if args.system_state_dict_path:
            logger.info(f"Loading system state dict from {args.system_state_dict_path}")
            state_dict = torch.load(args.system_state_dict_path, weights_only=False)
            system.load_state_dict(state_dict)
            # Saved manual generation settings must also reach new worker threads.
            for cid in state_dict:
                if cid in system.components:
                    component = system.components[cid]
                    component.default_config = SimpleNamespace(**vars(component.config))
            logger.info(f"Loaded state dict keys: {list(state_dict.keys())}")

        metrics_path = osp.join(args.output_dir, f"{args.split}_metrics.json")
        dataset = valset if args.split == "val" else testset
        metrics, preds = eval_system(system=system, testset=dataset, num_repeat=args.num_repeat)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        logger.info(f"Metrics written to {metrics_path}")
        logger.info(f"Final metrics:\n{json.dumps(metrics, indent=2)}")
        if args.interactive:
            from scripts.results_layout import write_evaluation

            write_evaluation(
                results_dir, metrics, variant="manual", split=args.split,
                state_source=results_dir / "manual_system.pth",
            )
        return metrics


if __name__ == "__main__":
    main()
