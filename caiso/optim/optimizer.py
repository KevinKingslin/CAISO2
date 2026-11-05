"""CAISO orchestration on the upstream system interface; no OPTIMAS reward model."""

from pathlib import Path

from caiso.constitution.dicai.pipeline import DICAI
from caiso.core.budget import BudgetTracker
from caiso.core.serialization import read_json, save_json
from caiso.integration.config import load_method
from caiso.integration.runtime import UpstreamRuntime, adapt_examples
from caiso.integration.services import build_services
from caiso.method.algorithm import Algorithm

DEFAULT_PERSONAS = [
    "A domain expert who checks whether the system solves its stated task accurately.",
    "A critical reviewer who examines evidence, errors, and unsupported conclusions.",
    "An end user who values relevant, clear, and usable final outputs.",
]


class CaisoOptimizer:
    def __init__(
        self,
        system,
        train_dataset,
        val_dataset,
        config,
        output_dir,
        providers=None,
        embeddings=None,
        dicai=None,
        block_dicai=None,
    ):
        self.system = system
        self.experiment = config
        self.config = load_method(config)
        self.output = Path(output_dir)
        self.budget = BudgetTracker(len(system.components), **self.config["budget"])
        options = config.get("caiso", {})
        if providers is None:
            providers, embeddings = build_services(system, options.get("services", {}), self.budget)
        self.runtime = UpstreamRuntime(
            system, providers, self.budget, options.get("edge_baselines")
        )
        self.runtime.initial_state.system = config.get("system", "upstream")
        self.train = adapt_examples(system, train_dataset, "train")
        size = self.config["validation_batch_size"]
        validation = val_dataset if size == "all" else val_dataset[:size]
        self.validation = adapt_examples(system, validation, "validation")
        personas = options.get("personas", DEFAULT_PERSONAS)
        self.dicai = dicai or DICAI(providers, embeddings, personas, self.config["dicai"])
        self.block_dicai = block_dicai or DICAI(
            providers, embeddings, personas, self.config["dicai"], role="block_constitution"
        )

    def optimize(self, resume=None):
        self.output.mkdir(parents=True, exist_ok=True)
        save_json(self.output / "experiment.json", self.experiment)
        save_json(self.output / "method.json", self.config)
        initial = self.runtime.initial_state
        algorithm = Algorithm(self.runtime, self.dicai, self.block_dicai, self.config, self.output)
        best = algorithm.run(initial, self.train, self.validation, resume=resume)
        self.runtime.apply_state(best)
        run = read_json(self.output / "run.json")
        if run["metrics"].get("stop_reason") in {
            "initialization_failed",
            "initialization_budget_exhausted",
        }:
            raise RuntimeError(
                "CAISO initialization failed; see setup.json and .internal/logs/phase_errors.jsonl"
            )
        self.best_state = best
        save_json(self.output / "best_caiso_state.json", best)
        return self.system, run["metrics"]

    def _get_current_adapter_state(self):
        return {
            cid: str(comp.variable)
            for cid, comp in self.system.components.items()
            if isinstance(comp.variable, Path)
        }
