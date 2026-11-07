"""Algorithm 1: transactional accepted/candidate/best state orchestration."""

import random
from itertools import count
from pathlib import Path
from time import monotonic

from caiso.constitution.dicai.judge import ConstitutionJudge
from caiso.constitution.models import PreferencePair
from caiso.core.budget import BudgetExceeded
from caiso.core.checkpoint import Checkpoint
from caiso.core.content_filter import execute_example
from caiso.core.recovery import ModelResponseError, Recovery, recoverable
from caiso.core.serialization import append_jsonl, digest, save_json, save_jsonl
from caiso.evaluation.evaluator import evaluate
from caiso.evaluation.iteration_report import IterationReport
from caiso.evaluation.results import update_result, write_overview
from caiso.evaluation.run_accounting import RunAccounting
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
from caiso.workflow_search.generator import WorkflowGenerator

from .block_constitution import build_block_preferences, induce_block_constitution
from .block_discovery import discover_blocks
from .block_optimization import validate_candidate
from .block_selection import select_block
from .causal_edge_attribution import estimate_all_edges
from .config import validate_method
from .joint_workflow_block import WorkflowTrigger, optimize_workflow
from .optimization_history import OptimizationHistory
from .system_constitution import induce_system_constitution


def batch(examples, size, seed):
    if size == "all":
        if not examples:
            raise ValueError("Nonempty data required")
        return list(examples)
    if not examples or size < 1:
        raise ValueError("Nonempty data and positive batch size required")
    return random.Random(seed).sample(examples, min(size, len(examples)))


def validation_batch(examples, config, seed):
    size = config["validation_batch_size"]
    if config.get("validation_sampling", "seeded") == "first" and size != "all":
        if not examples or size < 1:
            raise ValueError("Nonempty data and positive validation batch size required")
        return list(examples[:size])
    return batch(examples, size, seed)


def initial_preferences(runtime, state, train, count, attempts, seed, comparison_path=None):
    """Compare training trajectories; -1 uses one pass with seeded random tie labels."""
    if not train or count < 1:
        raise ValueError("Initial preferences require training samples and a positive sample count")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or (attempts != -1 and attempts < 1)
    ):
        raise ValueError("initial_preference_attempts must be -1 or a positive integer")
    single_pass = attempts == -1
    selected = []
    if single_pass:
        seen = set()
        for example in train:
            key = digest(example.inputs)
            if key not in seen:
                selected.append(example)
                seen.add(key)
            if len(selected) == count:
                break

    def comparisons():
        if single_pass:
            yield from selected
        else:
            for i in range(attempts):
                yield train[i % len(train)]

    if comparison_path:
        save_jsonl(comparison_path, [])
    result = []
    tie_rng = random.Random(seed)
    for i, example in enumerate(comparisons()):
        a = execute_example(runtime, state, example, seed + i * 2)
        b = execute_example(runtime, state, example, seed + i * 2 + 1) if a is not None else None
        if a is None or b is None:
            if comparison_path:
                append_jsonl(
                    comparison_path,
                    {
                        "example_id": example.example_id,
                        "status": "skipped_content_filter",
                        "preferred": None,
                        "candidate_a": a,
                        "candidate_b": b,
                        "seeds": [seed + i * 2, seed + i * 2 + 1],
                    },
                )
            continue
        tied = a.reward == b.reward
        if tied:
            preferred = tie_rng.choice(["A", "B"]) if single_pass else None
        else:
            preferred = "A" if a.reward > b.reward else "B"
        label_source = "random_task_tie" if tied else "training_task_metric"
        if comparison_path:
            append_jsonl(
                comparison_path,
                {
                    "example_id": example.example_id,
                    "seeds": [seed + i * 2, seed + i * 2 + 1],
                    "candidate_a": a,
                    "candidate_b": b,
                    "preferred": preferred,
                    "task_score_tied": tied,
                    "label_source": label_source if preferred is not None else None,
                },
            )
        if preferred is None:
            continue
        result.append(
            PreferencePair(
                pair_id=f"initial-{i}-{example.example_id}",
                context=example.inputs,
                candidate_a={"trajectory": a.model_dump()},
                candidate_b={"trajectory": b.model_dump()},
                preferred=preferred,
                metadata={"label_source": label_source, "task_score_tied": tied},
            )
        )
        if len(result) >= count:
            break
    if not result:
        raise ModelResponseError(
            "No unequal-score training pairs were observed, or all examples were content-filtered. Supply preferences_path "
            "with labeled PreferencePair JSON data; tied scores cannot induce preferences."
        )
    return result


class Algorithm:
    def __init__(
        self,
        runtime,
        dicai,
        block_dicai,
        config,
        output_dir,
        allowed_connections=None,
        optimizer=None,
        generator=None,
        wandb_run=None,
    ):
        validate_method(config)
        if any(key not in config for key in ("training_batch_size", "validation_batch_size")):
            raise ValueError(
                "Question batch sizes belong to the experiment. Pass the resolved method "
                "from load_experiment(...), not the shared method YAML alone."
            )
        self.runtime, self.dicai, self.block_dicai = runtime, dicai, block_dicai
        runtime.max_workers = config.get("execution", {}).get(
            "max_workers", getattr(runtime, "max_workers", 1)
        )
        self.config, self.output = config, Path(output_dir)
        runtime.providers.content_filter_log = self.output / ".internal/logs/skipped_examples.jsonl"
        self.accounting = RunAccounting(self.output, wandb_run=wandb_run)
        self.judge = ConstitutionJudge(
            runtime.providers,
            temperature=config.get("dicai", {}).get("temperatures", {}).get("reconstruction", 0.0),
        )
        self.optimizer = optimizer or JointBlockOptimizer(
            runtime,
            self.judge,
            config["block_optimization"],
            self.output / ".internal/optimization",
        )
        self.generator = generator or WorkflowGenerator(runtime.providers)
        self.allowed_connections = allowed_connections

    def run(self, initial, train, validation, preferences=None, resume=None):
        self.report = IterationReport(self.output, self.config, self.runtime.budget)
        self.report.configuration_state(original=initial, current=initial)
        previous_candidate_observer = None
        if isinstance(self.optimizer, JointBlockOptimizer):
            previous_candidate_observer = self.optimizer.on_candidate

            def show_candidate(candidate):
                self.report.configuration_state(candidate=candidate, status="Evaluating local proposal")
                if previous_candidate_observer:
                    previous_candidate_observer(candidate)

            self.optimizer.on_candidate = show_candidate
        previous_recovery = self.runtime.providers.recovery
        self.recovery = Recovery(
            self.config.get("error_handling"), self.report.error_event, self.runtime.budget
        )
        self.runtime.providers.recovery = self.recovery
        self.report.start()
        transports = [*self.runtime.providers.providers.values()]
        for dicai in (self.dicai, self.block_dicai):
            transports.append(getattr(getattr(dicai, "embeddings", None), "provider", None))
        try:
            with self.recovery.observe_transports(transports), self.report.capture_output():
                return self._run(initial, train, validation, preferences, resume)
        except Exception as exc:
            self.recovery.error(exc, self.report.data["phase"], "stop_run")
            self.report.update(f"Stopped: {type(exc).__name__}: {exc}", status="failed")
            if recoverable(exc) or isinstance(exc, BudgetExceeded):
                # Initialization has no constitution/validated score to fall back to.
                # Save a clear failed-run record instead of inventing either of them.
                update_result(
                    self.output / "setup.json",
                    status="failed",
                    failure={
                        "phase": self.report.data["phase"],
                        "error": str(exc),
                        "state": initial,
                        "budget": self.runtime.budget.snapshot(),
                    },
                )
                save_json(
                    self.output / "run.json",
                    {
                        "metrics": {
                            "validation_score": None,
                            "completed_iterations": 0,
                            "stop_reason": "initialization_budget_exhausted"
                            if isinstance(exc, BudgetExceeded)
                            else "initialization_failed",
                            "error": str(exc),
                            "budget": self.runtime.budget.snapshot(),
                        },
                        "best_system": initial,
                    },
                )
                write_overview(self.output)
                return initial.model_copy(deep=True)
            raise
        finally:
            if isinstance(self.optimizer, JointBlockOptimizer):
                self.optimizer.on_candidate = previous_candidate_observer
            self.runtime.providers.recovery = previous_recovery
            self.report.close()

    def _run(self, initial, train, validation, preferences=None, resume=None):
        cfg = self.config
        seed = cfg["seed"]
        if any(x.split != "train" for x in train) or any(
            x.split != "validation" for x in validation
        ):
            raise ValueError("Optimization accepts only train and validation splits")
        if {digest(x.inputs) for x in train} & {digest(x.inputs) for x in validation}:
            raise ValueError("Training and validation input overlap detected")
        config_hash = digest(cfg)
        data_hash = digest([train, validation])
        provider_hash = self.runtime.providers.fingerprint()
        library_hash = digest(self.runtime.library)
        if resume:
            self.report.update("Loading saved checkpoint")
            cp = Checkpoint.load(resume)
            if (cp.config_hash, cp.data_hash, cp.provider_hash, cp.library_hash) != (
                config_hash,
                data_hash,
                provider_hash,
                library_hash,
            ):
                raise ValueError("Resume config/data/provider/library fingerprint mismatch")
            current, best = cp.current, cp.best
            current.validate_state(self.runtime.library)
            best.validate_state(self.runtime.library)
            qsys, qblocks, preferences = (
                cp.system_constitution,
                cp.block_constitutions,
                cp.preferences,
            )
            trigger, start = cp.trigger, cp.iteration
            score, best_score = cp.current_score, cp.best_score
            blocks, effects = cp.blocks, cp.attribution
            pending_refresh = cp.pending_refresh
            optimization_history = cp.optimization_history
            self.runtime.budget.usage = cp.budget.model_copy(deep=True)
            self.runtime.budget.previous_seconds = cp.budget.elapsed_seconds
        else:
            current = initial.model_copy(deep=True)
            best = current.model_copy(deep=True)
            self.report.update("Preparing initial training preferences")
            preferences = preferences or initial_preferences(
                self.runtime,
                current,
                train,
                cfg["initial_preference_pairs"],
                cfg["initial_preference_attempts"],
                seed,
                comparison_path=self.output / "setup/bootstrap_comparisons.jsonl",
            )
            update_result(
                self.output / "setup.json",
                preferences=preferences,
                method_config=cfg,
                initial_state=current,
            )
            self.report.update("Inducing initial system constitution")
            qsys = induce_system_constitution(
                self.dicai,
                preferences,
                seed,
                cache_dir=self.output / ".internal/dicai/system",
                activity=self.report.induction_activity,
                progress=lambda stage, done, total: self.report.update(
                    progress=f"{stage.replace('_', ' ')}: {done}/{total}"
                ),
            )
            update_result(self.output / "setup.json", system_constitution=qsys)
            qblocks = {}
            trigger = WorkflowTrigger()
            start = 0
            blocks = {}
            effects = []
            pending_refresh = False
            optimization_history = OptimizationHistory()
            self.report.update("Evaluating initial validation baseline")
            score, baseline_trajectories = evaluate(
                self.runtime,
                current,
                validation_batch(validation, cfg, seed),
                [seed],
                progress=lambda done, total: self.report.update(
                    progress=f"Baseline validation: {done}/{total} system runs"
                ),
            )
            best_score = score
            update_result(
                self.output / "setup.json",
                baseline={"score": score, "seed": seed, "trajectories": baseline_trajectories},
                status="completed",
            )

        self.report.configuration_state(current=current)
        # A manual resume gets another bounded recovery window; all spent budgets persist.
        consecutive_errors = 0

        def checkpoint(next_iteration):
            Checkpoint(
                iteration=next_iteration,
                current=current,
                best=best,
                current_score=score,
                best_score=best_score,
                system_constitution=qsys,
                block_constitutions=qblocks,
                preferences=preferences,
                budget=self.runtime.budget.snapshot(),
                trigger=trigger,
                config_hash=config_hash,
                data_hash=data_hash,
                provider_hash=provider_hash,
                library_hash=library_hash,
                blocks=blocks,
                attribution=effects,
                pending_refresh=pending_refresh,
                consecutive_errors=consecutive_errors,
                optimization_history=optimization_history,
            ).save(self.output / "checkpoint")

        checkpoint(start)
        completed = start
        stop_reason = "iteration_limit"
        try:
            for iteration in count(start):
                if cfg.get("max_iterations") is not None and iteration >= cfg["max_iterations"]:
                    break
                self.runtime.budget.check()
                self.report.begin(iteration, score, best_score, state=current)
                budget_before = self.runtime.budget.snapshot()
                accepted, selection, gate, workflow_report = False, None, None, None
                discovered_blocks = {}
                try:
                    self.report.save_phase("01_attribution", state_before=current)
                    step_seed = seed + iteration * 1000000
                    if pending_refresh:
                        self.report.update(
                            "Refreshing system constitution",
                            phase_dir="00_system_constitution",
                        )
                        qsys = induce_system_constitution(
                            self.dicai,
                            preferences,
                            step_seed,
                            cache_dir=self.output / ".internal/dicai/system",
                            activity=self.report.induction_activity,
                            progress=lambda stage, done, total: self.report.update(
                                progress=f"{stage.replace('_', ' ')}: {done}/{total}"
                            ),
                        )
                        self.report.save_system_induction(qsys, preferences)
                        pending_refresh = False
                    self.report.update("Causal edge attribution", phase_dir="01_attribution")
                    attr = batch(validation, cfg["attribution"]["batch_size"], step_seed)
                    attr_seeds = [
                        step_seed + r for r in range(cfg["attribution"]["stochastic_repeats"])
                    ]
                    attribution = estimate_all_edges(
                        self.runtime,
                        current,
                        attr,
                        attr_seeds,
                        None,
                        cfg["attribution"]["bootstrap_samples"],
                        confidence_level=cfg["attribution"].get("confidence_level", 0.95),
                    )
                    effects = [e.model_dump() for e in attribution]
                    self.report.save_phase(
                        "01_attribution",
                        edge_effects=effects,
                        settings=cfg["attribution"],
                        seeds=attr_seeds,
                    )
                    self.report.update("Discovering blocks with Leiden", phase_dir="02_blocks")
                    blocks = discover_blocks(
                        current,
                        attribution,
                        seed=step_seed,
                        output=None,
                        **cfg["leiden"],
                    )
                    self.report.save_phase(
                        "02_blocks",
                        blocks=blocks,
                        settings=cfg["leiden"],
                        seed=step_seed,
                        weighted_graph={"nodes": current.workflow.nodes, "edges": effects},
                    )
                    discovered_blocks = {name: list(nodes) for name, nodes in blocks.items()}
                    self.report.update(
                        "Selecting a block from diagnostic trajectories",
                        phase_dir="03_selection",
                        blocks=blocks,
                    )
                    _, diagnostics = evaluate(
                        self.runtime,
                        current,
                        batch(
                            validation, cfg["block_selection"]["diagnostic_batch_size"], step_seed
                        ),
                        [step_seed],
                        None,
                    )
                    self.report.save_phase(
                        "03_selection",
                        diagnostic_trajectories=diagnostics,
                        system_constitution=qsys,
                        blocks=blocks,
                    )
                    selection = select_block(
                        qsys,
                        diagnostics,
                        blocks,
                        attribution,
                        self.runtime.providers,
                        step_seed,
                        role=cfg["block_selection"]["model_role"],
                        output=None,
                        optimization_history=optimization_history.selection_summary(
                            current, blocks
                        ),
                    )
                    self.report.save_phase("03_selection", selection=selection)
                    members = blocks[selection.selected]
                    self.report.update(
                        "Generating block preference candidates",
                        phase_dir="04_block_constitution",
                        selected_block=selection.selected,
                        vote_fractions=selection.scores,
                    )
                    local_train = batch(train, cfg["training_batch_size"], step_seed)
                    val = validation_batch(validation, cfg, seed)  # fixed acceptance set
                    # Seed proposals use Q_sys; optimization proposals later use the induced Q_B.
                    preliminary = self.optimizer.propose(
                        current, members, qsys, local_train, step_seed
                    )
                    self.report.configuration_state(status="Evaluated for preference generation")
                    preliminary = optimization_history.filter_candidates(current, preliminary)
                    self.report.save_phase(
                        "04_block_constitution",
                        preliminary_candidates=preliminary,
                        members=members,
                    )
                    changed = [
                        c.configurations
                        for c in preliminary
                        if current.with_configs(c.configurations) != current
                    ]
                    pair_configs = [
                        ({n: current.configurations[n] for n in members}, c)
                        for c in changed[: cfg["block_constitution"]["preference_pairs"]]
                    ]
                    gate = None
                    if pair_configs:
                        self.report.update("Building block preferences")
                        preference_started = monotonic()
                        preference_budget = self.runtime.budget.snapshot()
                        preference_stats = {}
                        pairs = build_block_preferences(
                            self.runtime,
                            current,
                            members,
                            pair_configs,
                            local_train,
                            [step_seed],
                            qsys,
                            self.judge,
                            cfg["block_constitution"]["score_tie_epsilon"],
                            reuse_trajectories=cfg["block_constitution"].get(
                                "reuse_trajectories", False
                            ),
                            progress=lambda done, total: self.report.update(
                                progress=f"Preference records: {done}/{total}"
                            ),
                            statistics=preference_stats,
                        )
                        preference_stats.update(
                            seconds=monotonic() - preference_started,
                            llm_calls=self.runtime.budget.snapshot().llm_calls
                            - preference_budget.llm_calls,
                        )
                        self.report.save_phase(
                            "04_block_constitution",
                            preference_metrics=preference_stats,
                            preferences=pairs,
                        )
                        member_key = digest(sorted(members))[:16]
                        self.report.update("Inducing block constitution")
                        induction_budget = self.runtime.budget.snapshot()
                        qb = induce_block_constitution(
                            self.block_dicai,
                            pairs,
                            members,
                            step_seed,
                            cache_dir=self.output / ".internal/dicai/blocks",
                            activity=self.report.induction_activity,
                            progress=lambda stage, done, total: self.report.update(
                                progress=f"{stage.replace('_', ' ')}: {done}/{total}"
                            ),
                        )
                        induction_after = self.runtime.budget.snapshot()
                        self.report.save_phase(
                            "04_block_constitution",
                            constitution=qb,
                            induction_metrics={
                                "preference_records": len(pairs),
                                "llm_calls": induction_after.llm_calls - induction_budget.llm_calls,
                                "embedding_calls": induction_after.embedding_calls
                                - induction_budget.embedding_calls,
                                "tokens": induction_after.tokens - induction_budget.tokens,
                                "timings": qb.metadata.get("timings", {}),
                                "max_workers": cfg["dicai"].get("max_workers", 4),
                                "reasoning_workers": qb.metadata.get("reasoning_workers"),
                                "concurrency": qb.metadata.get("concurrency", {}),
                                "protocol": qb.metadata.get("protocol"),
                                "reconstruction": qb.metadata.get("reconstruction", {}),
                                "reused_pair_records": qb.metadata.get("reused_pair_records", 0),
                            },
                        )
                        qblocks[member_key] = qb

                        accuracy = qb.metadata.get("reconstruction", {}).get("accuracy")
                        self.report.update(
                            "Optimizing selected block configurations",
                            phase_dir="05_block_optimization",
                            constitution_reconstruction_accuracy=accuracy,
                        )
                        scored = self.optimizer.propose(
                            current, members, qb, local_train, step_seed + 10000
                        )
                        self.report.configuration_state(status="Local candidates evaluated")
                        scored = [
                            c
                            for c in optimization_history.filter_candidates(current, scored)
                            if current.with_configs(c.configurations) != current
                        ]
                        if not scored:
                            self.report.configuration_state(clear_candidate=True, status="No new candidate")
                            optimization_history.record_no_candidate(iteration, current, members)
                            self.report.save_phase(
                                "05_block_optimization",
                                status="validation_skipped",
                                reason="No new block candidates after history filtering",
                            )
                        else:
                            chosen = self.optimizer.select(scored, current, members, step_seed)
                            candidate = current.with_configs(chosen.configurations)
                            self.report.configuration_state(candidate=candidate, status="Selected for validation")
                            self.report.save_phase(
                                "05_block_optimization",
                                candidate_state=candidate,
                                candidates=scored,
                            )
                            self.report.update(
                                "Validating block candidate on fixed validation batch"
                            )
                            updated, gate = validate_candidate(
                                self.runtime,
                                current,
                                candidate,
                                val,
                                [seed],
                                cfg["block_optimization"]["acceptance_epsilon"],
                                progress=lambda label, done, total: self.report.update(
                                    progress=f"{label} validation: {done}/{total} system runs"
                                ),
                                on_evaluation=lambda label, value, trajectories, validation=val: (
                                    self.report.save_phase(
                                        "05_block_optimization",
                                        **{
                                            f"{label}_validation": {
                                                "score": value,
                                                "seeds": [seed],
                                                "trajectories": trajectories,
                                                "coverage": {
                                                    "requested": len(validation),
                                                    "completed": len(trajectories),
                                                    "skipped_content_filter": len(validation)
                                                    - len(trajectories),
                                                },
                                            }
                                        },
                                    )
                                ),
                            )
                            self.report.save_phase("05_block_optimization", acceptance=gate)
                            optimization_history.record_validation(
                                iteration, current, candidate, members, gate
                            )
                            current = updated
                            self.report.configuration_state(current=current, candidate=candidate, status="Accepted" if gate.accepted else "Rejected")
                            score = gate.candidate_score if gate.accepted else gate.current_score
                            accepted = gate.accepted
                            self.accounting.event(
                                "block_acceptance", iteration=iteration, **gate.model_dump()
                            )
                    else:
                        self.report.configuration_state(clear_candidate=True, status="No new candidate")
                        optimization_history.record_no_candidate(iteration, current, members)
                        self.report.save_phase(
                            "04_block_constitution",
                            induction_skipped=True,
                            reason="No new block candidates after history filtering",
                        )
                        accepted = False
                        self.accounting.event("no_distinct_block_candidates", iteration=iteration)
                    if score > best_score:
                        best, best_score = current.model_copy(deep=True), score
                    self.report.update(
                        "Checking workflow search trigger",
                        phase_dir="06_workflow",
                        score=score,
                        best_score=best_score,
                        accepted=accepted,
                        candidate_score=gate.candidate_score if gate else None,
                    )
                    trigger.observe(accepted)
                    # Save accepted block changes before potentially expensive workflow work.
                    completed = iteration + 1
                    if accepted:
                        effects = []  # Previous attribution refers to the pre-update configuration.
                    checkpoint(completed)
                    workflow_cfg = {
                        **cfg["workflow"],
                        "acceptance_epsilon": cfg["block_optimization"]["acceptance_epsilon"],
                    }
                    workflow_report = None
                    if workflow_cfg.get("enabled", True) and trigger.fires(
                        workflow_cfg["trigger_every_k_accepted_blocks"],
                        workflow_cfg["plateau_patience"],
                    ):
                        self.report.update(
                            "Proposing workflow change and adapting affected components"
                        )
                        _, representative = evaluate(
                            self.runtime, current, local_train, [step_seed]
                        )
                        updated, workflow_report = optimize_workflow(
                            self.runtime,
                            current,
                            qsys,
                            blocks,
                            self.generator,
                            self.optimizer,
                            local_train,
                            val,
                            representative,
                            score,
                            seed,
                            workflow_cfg,
                            self.allowed_connections,
                        )
                        self.report.save_phase("06_workflow", report=workflow_report)
                        if workflow_report.get("candidate_state"):
                            self.report.configuration_state(
                                candidate=workflow_report["candidate_state"],
                                status="Workflow accepted" if workflow_report["accepted"] else "Workflow rejected",
                            )
                        self.report.update(
                            "Applying workflow acceptance decision", workflow=workflow_report
                        )
                        if workflow_report["accepted"]:
                            current = updated
                            self.report.configuration_state(current=current)
                            score = workflow_report["candidate_score"]
                            if score > best_score:
                                best, best_score = current.model_copy(deep=True), score
                            blocks = {}
                            effects = []
                            pending_refresh = workflow_cfg["reinduce_system_constitution_on_accept"]
                        trigger.reset()
                        checkpoint(completed)
                        if workflow_report["accepted"]:
                            if pending_refresh:
                                self.report.update(
                                    "Refreshing system constitution after workflow update",
                                    phase_dir="00_system_constitution",
                                )
                                qsys = induce_system_constitution(
                                    self.dicai,
                                    preferences,
                                    step_seed + 50000,
                                    cache_dir=self.output / ".internal/dicai/system",
                                    activity=self.report.induction_activity,
                                    progress=lambda stage, done, total: self.report.update(
                                        progress=f"{stage.replace('_', ' ')}: {done}/{total}"
                                    ),
                                )
                                self.report.save_system_induction(qsys, preferences)
                                pending_refresh = False
                            self.report.update(
                                "Recomputing causal effects after workflow update",
                                phase_dir="06_workflow",
                                score=score,
                                best_score=best_score,
                            )
                            refreshed = estimate_all_edges(
                                self.runtime,
                                current,
                                attr,
                                attr_seeds,
                                None,
                                cfg["attribution"]["bootstrap_samples"],
                                confidence_level=cfg["attribution"].get("confidence_level", 0.95),
                            )
                            effects = [e.model_dump() for e in refreshed]
                            self.report.update("Repartitioning updated workflow")
                            blocks = discover_blocks(
                                current,
                                refreshed,
                                seed=step_seed,
                                output=None,
                                **cfg["leiden"],
                            )
                            self.report.save_phase(
                                "06_workflow",
                                refreshed_edge_effects=effects,
                                refreshed_blocks=blocks,
                                refreshed_graph={"nodes": current.workflow.nodes, "edges": effects},
                                partition_settings=cfg["leiden"],
                            )
                        self.accounting.event("workflow", iteration=iteration, **workflow_report)
                    self.report.save_phase(
                        "06_workflow",
                        trigger=trigger,
                        triggered=workflow_report is not None,
                    )
                    record = {
                        "iteration": iteration,
                        "score": score,
                        "accepted": accepted,
                        "selected_block": selection.selected,
                        "vote_fractions": selection.scores,
                        "blocks": discovered_blocks,
                        "blocks_after_workflow": blocks,
                        "best_score": best_score,
                        "candidate_score": gate.candidate_score if gate else None,
                        "block_acceptance": gate.model_dump() if gate else None,
                        "workflow": workflow_report,
                    }
                    self.accounting.iteration(record, self.runtime.budget.snapshot())
                    self.report.finish(record, budget_before, current, qsys)
                    consecutive_errors = 0
                    checkpoint(completed)
                except Exception as exc:
                    if not recoverable(exc):
                        raise
                    consecutive_errors += 1
                    self.recovery.error(
                        exc,
                        self.report.data["phase"],
                        "skip_iteration",
                        consecutive_errors=consecutive_errors,
                    )
                    completed = iteration + 1
                    record = {
                        "iteration": iteration,
                        "status": "skipped",
                        "score": score,
                        "best_score": best_score,
                        "accepted": accepted,
                        "selected_block": selection.selected if selection else None,
                        "blocks": discovered_blocks or self.report.data.get("blocks", {}),
                        "candidate_score": gate.candidate_score if gate else None,
                        "workflow": workflow_report,
                        "error": str(exc),
                        "consecutive_errors": consecutive_errors,
                    }
                    checkpoint(completed)
                    self.accounting.iteration(record, self.runtime.budget.snapshot())
                    self.report.finish(record, budget_before, current, qsys, status="skipped")
                    if not cfg.get("error_handling", {}).get("continue_after_errors", True):
                        stop_reason = "recoverable_error"
                        break
                    if consecutive_errors >= cfg.get("error_handling", {}).get(
                        "max_consecutive_errors", 3
                    ):
                        stop_reason = "consecutive_error_limit"
                        self.report.update("Stopped cleanly after consecutive failed iterations")
                        break
        except BudgetExceeded as exc:
            stop_reason = str(exc)
            self.accounting.event("budget_exhausted", reason=str(exc), next_iteration=completed)
            if self.report.folder and self.report.data.get("status") == "running":
                self.report.finish(
                    {
                        "iteration": iteration,
                        "score": score,
                        "best_score": best_score,
                        "stop_reason": str(exc),
                    },
                    budget_before,
                    current,
                    qsys,
                    status="budget exhausted",
                )
            else:
                self.report.update(str(exc))
            checkpoint(completed)
        if self.report.folder is None:
            self.report.update(
                "No further iterations under saved limits",
                display_iteration=completed,
                max_iterations=cfg.get("max_iterations"),
                score=score,
                best_score=best_score,
            )
        save_json(
            self.output / "run.json",
            {
                "best_system": best,
                "system_constitution": qsys,
                "block_constitutions": qblocks,
                "metrics": {
                    "validation_score": best_score,
                    "completed_iterations": completed,
                    "successful_iterations": sum(
                        r["status"] == "completed" for r in self.report.history
                    ),
                    "skipped_iterations": sum(
                        r["status"] == "skipped" for r in self.report.history
                    ),
                    "stop_reason": stop_reason,
                    "protocol": "proposal-local-constitutional-v2",
                    "budget": self.runtime.budget.snapshot(),
                    "state_hash": best.fingerprint(),
                    "pending_refresh": pending_refresh,
                },
            },
        )
        write_overview(self.output)
        return best
