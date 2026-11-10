"""Offline probes for the optimization audit; these do not measure benchmark F1.

Run from the repository root with ``python -m scripts.audit_optimization``.
All benchmark forward calls and method requests are replaced with deterministic
stubs. A socket guard rejects network connections, including accidental calls.
The real configuration loader, HotpotQA factory, runtime, attribution estimator,
candidate search, acceptance evaluator, and preference bootstrap are exercised.
Pass ``--run RUN_DIRECTORY`` instead to summarize existing run artifacts without
loading models or running probes. That mode uses only the Python standard library.
"""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import MethodType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def inspect_run(directory):
    directory = Path(directory)
    if not (directory / "method.json").is_file():
        raise ValueError(f"No method.json in {directory}; pass a CAISO optimization run")

    def read(relative, default=None):
        path = directory / relative
        return json.loads(path.read_text()) if path.is_file() else default

    method, setup = read("method.json"), read("setup.json", {})
    run, checkpoint = read("run.json", {}), read("checkpoint/checkpoint.json", {})
    comparisons = directory / "setup/bootstrap_comparisons.jsonl"
    bootstrap = [json.loads(line) for line in comparisons.read_text().splitlines() if line.strip()] if comparisons.is_file() else []
    labeled = [row for row in bootstrap if row.get("preferred") in {"A", "B"}]
    iterations, memberships = [], Counter()
    for path in sorted(directory.glob("iterations/*/summary.json")):
        row = json.loads(path.read_text())
        phase = path.parent.relative_to(directory) / "05_block_optimization"
        candidates = read(phase / "candidates.json", [])
        candidate_state = read(phase / "block_candidate_state.json", {})
        chosen_configs = candidate_state.get("configurations", {})
        chosen = next((item for item in candidates if item["configurations"] and all(
            chosen_configs.get(cid) == value for cid, value in item["configurations"].items()
        )), {})
        gate = read(phase / "block_acceptance.json", {})
        blocks = row.get("blocks", {})
        members = blocks.get(row.get("selected_block"), [])
        memberships[",".join(sorted(members)) or "unknown"] += 1
        task_scores = [item["task_score"] for item in candidates if item.get("task_score") is not None]
        search = read(phase / "candidate_search.json", {})
        iterations.append({
            "directory": path.parent.name,
            "status": row.get("status"),
            "selected_members": members,
            "accepted": gate.get("accepted", row.get("accepted", False)),
            "current_validation_score": gate.get("current_score"),
            "candidate_validation_score": gate.get("candidate_score"),
            "best_training_task_score_in_saved_candidates": max(task_scores) if task_scores else None,
            "selected_training_task_score": chosen.get("task_score"),
            "equivalent_runs": row.get("budget_delta", {}).get("equivalent_full_system_runs"),
            "screened_candidates": len(search.get("screened_candidates", [])),
            "finalist_states": len(search.get("finalist_states", [])),
            "error": row.get("error"),
        })
    evaluations = {}
    for name in ("evaluation.json", "evaluation_unoptimized.json", "evaluation_best.json", "evaluation_current.json"):
        value = read(name)
        if value:
            evaluations[name] = {key: value.get(key) for key in (
                "mean", "std", "per_repeat", "coverage", "state_snapshot",
            )}
    constitution = setup.get("system_constitution", {})
    budget = run.get("metrics", {}).get("budget") or checkpoint.get("budget")
    return {
        "scope": "Read-only summary of saved run artifacts; no new benchmark measurements",
        "run_name": directory.name,
        "method": {key: method.get(key) for key in (
            "seed", "max_iterations", "training_batch_size", "validation_batch_size",
            "validation_sampling", "initial_preference_attempts", "budget",
        )},
        "candidate_selection": {key: method.get("block_optimization", {}).get(key) for key in (
            "sample_discrete", "softmax_temperature", "prompt_candidates", "task_scoring", "validation_repeats", "screening",
        )},
        "stop_reason": run.get("metrics", {}).get("stop_reason"),
        "budget": budget,
        "baseline_validation_score": setup.get("baseline", {}).get("score"),
        "best_validation_score": run.get("metrics", {}).get("validation_score", checkpoint.get("best_score")),
        "bootstrap": {
            "comparisons_logged": len(bootstrap), "labeled": len(labeled),
            "label_sources": dict(Counter(row.get("label_source", "unknown") for row in labeled)),
            "abstention_reasons": dict(Counter(row.get("abstention_reason", "unknown") for row in bootstrap if row.get("status") == "abstained")),
            "random_tie_fraction": sum(row.get("label_source") == "random_task_tie" for row in labeled) / len(labeled) if labeled else None,
        },
        "constitution_training_reconstruction": constitution.get("metadata", {}).get("reconstruction", {}).get("accuracy"),
        "selected_block_memberships": dict(memberships),
        "iterations": iterations,
        "evaluations": evaluations,
    }


def run_probes():
    from caiso.attribution.estimator import estimate_all_edges
    from caiso.core.budget import BudgetTracker
    from caiso.core.recovery import ModelResponseError
    from caiso.core.serialization import digest
    from caiso.integration.config import load_experiment, load_method
    from caiso.integration.runtime import UpstreamRuntime, adapt_examples
    from caiso.integration.services import MethodProvider
    from caiso.method.algorithm import initial_preferences, validation_batch
    from caiso.method.block_optimization import validate_candidate
    from caiso.optim.optimizer import CaisoOptimizer
    from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
    from caiso.optimizers.model_selector import ModelSelectorOptimizer
    from caiso.providers.base import ProviderRegistry
    from caiso.wrappers.example import Example
    from examples.systems.hotpotqa.five_components import system_engine

    experiment = load_experiment(ROOT / "configs/experiments/hotpotqa.yaml")
    method = load_method(experiment)

    class OfflineMethod:
        def __init__(self, budget):
            self.budget, self.requests = budget, []

        def fingerprint(self):
            return "audit-deterministic-method"

        def complete(self, messages, *, seed, schema=None, **kwargs):
            assert schema.__name__ == "PromptProposal"
            self.budget.call()
            self.requests.append(messages)
            return json.dumps({"prompt": f"Audit behavioral change {len(self.requests)}"})

    def runtime_factory():
        engine = system_engine()
        for component in engine.components.values():
            def forward(self, **inputs):
                # Keep the real input/output interfaces, but make no external calls.
                _ = self.variable, self.config
                return {
                    field: "audit answer" if field == "answer" else f"audit {field} {digest([self.variable, inputs])[:12]}"
                    for field in self.output_fields
                }

            component.forward = MethodType(forward, component)
            component.num_retry, component.retry_after = 1, 0
        budget = BudgetTracker(len(engine.components), **method["budget"])
        provider = OfflineMethod(budget)
        registry = ProviderRegistry({"audit": provider}, {"prompt_optimizer": "audit"})
        runtime = UpstreamRuntime(engine, registry, budget)
        # The real runtime is serialized; use one worker to keep probes inexpensive.
        runtime.max_workers = 1
        return runtime, provider

    def examples(runtime, count, split):
        return adapt_examples(runtime.engine, [
            Example(question=f"{split} question {i}", gd_answer="audit answer")
            for i in range(count)
        ], split)

    runtime, _ = runtime_factory()
    train, validation = examples(runtime, 50, "train"), examples(runtime, 250, "validation")
    try:
        initial_preferences(runtime, runtime.initial_state, train, 50, -1, 42)
    except ModelResponseError:
        rejected_tied_bootstrap = True
    else:
        raise AssertionError("An all-tied bootstrap must not induce preferences")
    bootstrap_runs = runtime.budget.usage.full_system_runs
    tie_probe = {
        "pairs": 0,
        "all_tied_bootstrap_rejected": rejected_tied_bootstrap,
        "full_system_runs": bootstrap_runs,
    }

    with tempfile.TemporaryDirectory(prefix="caiso-audit-") as directory:
        optimizer = CaisoOptimizer(
            runtime.engine,
            [Example(question=f"train {i}", gd_answer="audit answer") for i in range(50)],
            [Example(question=f"validation {i}", gd_answer="audit answer") for i in range(250)],
            experiment, directory, providers=runtime.providers, dicai=SimpleNamespace(),
        )
        pool = optimizer.validation
        subset = validation_batch(pool, optimizer.config, optimizer.config["seed"])
        validation_probe = {
            "loader_validation_size": len(validation),
            "optimizer_validation_pool_size": len(pool),
            "acceptance_size": len(subset),
            "acceptance_repeats": method["block_optimization"]["validation_repeats"],
            "diagnostics_split": method["diagnostics_split"],
            "acceptance_ids": [item.example_id for item in subset],
            "effective_attribution_batch_size": min(len(pool), method["attribution"]["batch_size"]),
            "effective_diagnostic_batch_size": min(len(pool), method["block_selection"]["diagnostic_batch_size"]),
        }

    costs = {}
    for label, members in {
        "single_prompt": ["answer_generator"],
        "retriever_only": ["wikipedia_retriever"],
        "two_prompts": ["hint_generator", "answer_generator"],
        "prompt_and_retriever": ["info_extractor", "wikipedia_retriever"],
    }.items():
        runtime, provider = runtime_factory()
        state = runtime.initial_state
        train, val = examples(runtime, 20, "train"), examples(runtime, 50, "validation")
        before = 0
        phases = {}

        def record(phase):
            nonlocal before
            now = runtime.budget.usage.equivalent_full_system_runs
            phases[phase], before = now - before, now

        normal = {}
        estimate_all_edges(runtime, state, train[:10], [42], bootstrap_samples=200, normal=normal)
        record("attribution")
        assert len([trajectory for trajectory in normal.values() if trajectory is not None]) == 10
        # The scheduler reads these same normal trajectories instead of rerunning them.
        record("diagnostics")
        search = JointBlockOptimizer(runtime, None, method["block_optimization"], Path("unused"))
        constitution = SimpleNamespace(principles=["Select the trajectory with a concise correct answer."])
        scored = search.propose(state, members, constitution, train, 10042)
        record("proposals_and_candidate_scoring")
        changed = [c for c in scored if state.with_configs(c.configurations) != state]
        chosen = search.select(changed, state, members, 42)
        validate_candidate(runtime, state, state.with_configs(chosen.configurations), val, [42, 104771])
        record("acceptance")
        total = sum(phases.values())
        costs[label] = {
            "members": members,
            "joint_states_screened": len(search.last_search_report["screened_candidates"]),
            "finalist_states": len(scored),
            "phases_equivalent_full_runs": {key: round(value, 8) for key, value in phases.items()},
            "logical_completed_system_runs": runtime.budget.usage.full_system_runs,
            "reused_component_executions": runtime.budget.usage.reused_component_executions,
            "equivalent_full_system_runs": round(runtime.budget.usage.equivalent_full_system_runs, 8),
            "complete_iterations_after_160_setup_runs": math.floor((2000 - 160) / total),
        }
        if label == "single_prompt":
            # A second proposal call rescoring its archive starts with prior evidence.
            search.propose(state, members, constitution, train, 20042)
            users = [request[-1]["content"] for request in provider.requests]

            def history(user):
                return json.JSONDecoder().raw_decode(user.split("PREVIOUS INSTRUCTIONS AND SCORES:\n", 1)[1])[0]

            prompt_probe = {
                "history_sizes_over_two_proposal_calls": [len(history(user)) for user in users],
                "examples_payload_keys": list(json.JSONDecoder().raw_decode(
                    users[0].split("EXAMPLE INPUTS AND OUTPUTS:\n", 1)[1]
                )[0][0]),
            }

    wire_requests = []
    with patch("caiso.integration.services.api.get_llm_output", side_effect=lambda *args, **kwargs: wire_requests.append(kwargs) or "ok"):
        MethodProvider("audit-model", None).complete(
            [{"role": "user", "content": "audit"}], seed=123,
            response_format={"type": "json_object"},
        )
    wire_probe = {
        "seed_forwarded": "seed" in wire_requests[0],
        "schema_free_response_format_forwarded": "response_format" in wire_requests[0],
    }
    scores = [0.42, 0.45, 0.48, 0.50]
    probability_probe = {
        "scores": scores,
        "configured_sample_discrete": method["block_optimization"]["sample_discrete"],
        "selected_index": ModelSelectorOptimizer().select(list(range(4)), scores),
    }
    # Isolate candidate-search savings under the same 20 training questions.
    search_costs = {}
    for label, settings in {
        "baseline_reuse_only": {"reuse_component_executions": False, "screening": {"enabled": False}},
        "staged_no_prefix_reuse": {"reuse_component_executions": False, "screening": {"enabled": True, "batch_size": 5, "finalists": 3}},
        "staged_with_prefix_reuse": method["block_optimization"],
    }.items():
        runtime, _ = runtime_factory()
        search = JointBlockOptimizer(runtime, None, settings, Path("unused"))
        search.propose(runtime.initial_state, ["hint_generator", "answer_generator"], constitution, examples(runtime, 20, "train"), 10042)
        search_costs[label] = round(runtime.budget.usage.equivalent_full_system_runs, 8)
    assert validation_probe["optimizer_validation_pool_size"] == 250
    assert search_costs["baseline_reuse_only"] == 320
    assert search_costs["staged_no_prefix_reuse"] == 140
    assert search_costs["staged_with_prefix_reuse"] < 140
    assert prompt_probe["history_sizes_over_two_proposal_calls"] == [1, 2, 3, 4, 5, 6]
    assert wire_probe["seed_forwarded"] and wire_probe["schema_free_response_format_forwarded"]
    source_files = [
        "caiso/optimizers/joint_block_optimizer.py", "caiso/optimizers/prompt_optimizer.py",
        "caiso/integration/runtime.py", "caiso/method/algorithm.py", "configs/method/default.yaml",
        "configs/experiments/hotpotqa.yaml", "scripts/audit_optimization.py",
    ]
    return {
        "scope": "Offline execution/control-flow probes, not measured HotpotQA performance",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_fingerprint": digest({name: (ROOT / name).read_text() for name in source_files}),
        "source_files": source_files,
        "hotpotqa_edges": [edge.edge_id for edge in runtime.initial_state.workflow.edges],
        "validation": validation_probe,
        "bootstrap_ties": tie_probe,
        "per_iteration_cost": costs,
        "two_prompt_training_search_equivalent_runs": search_costs,
        "setup_assumption": "60 successful bootstrap runs plus 100 initial validation runs; real bootstrap may need more attempts",
        "prompt_feedback": prompt_probe,
        "method_request_forwarding": wire_probe,
        "discrete_selection": probability_probe,
        "test_repetitions": {
            "automatic_post_optimization": experiment["num_repeat"],
            "separate_evaluate_command_default": experiment["evaluation"]["num_repeat"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON destination")
    parser.add_argument("--run", type=Path, help="Read existing artifacts instead of executing offline probes")
    args = parser.parse_args()
    if args.run:
        result = inspect_run(args.run)
    else:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        os.environ["ENABLE_OPTIMAS_LOGGING"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        logging.disable(logging.CRITICAL)
        with tempfile.TemporaryDirectory(prefix="caiso-audit-cache-") as cache:
            os.environ["MPLCONFIGDIR"] = cache
            os.environ["XDG_CACHE_HOME"] = cache
            with patch("socket.socket.connect", side_effect=AssertionError("Audit probes must stay offline")), redirect_stdout(io.StringIO()):
                result = run_probes()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
