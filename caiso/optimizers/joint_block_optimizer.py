import math
import random
from contextlib import nullcontext

from caiso.core.budget import BudgetExceeded
from caiso.core.configuration import ConfigurationType
from caiso.core.content_filter import ContentFilteredError, record_skip
from caiso.core.recovery import ModelResponseError
from caiso.core.serialization import digest, plain
from caiso.evaluation.evaluator import evaluate
from caiso.method.config import constitution_judge_votes, task_scoring_settings

from .base import CandidateScore, OptimizationContext, filter_current_configuration
from .hyperparameter_optimizer import HyperparameterOptimizer
from .model_selector import ModelSelectorOptimizer
from .prompt_optimizer import PromptOptimizer
from .weight_optimizer import WeightOptimizer


def default_optimizers(config):
    kwargs = {
        "temperature": config.get("softmax_temperature", 1),
        "sample": config.get("sample_discrete", False),
    }
    return {
        ConfigurationType.PROMPT: PromptOptimizer(),
        ConfigurationType.MODEL_WEIGHTS: WeightOptimizer(),
        ConfigurationType.MODEL_SELECTION: ModelSelectorOptimizer(**kwargs),
        ConfigurationType.HYPERPARAMETER: HyperparameterOptimizer(**kwargs),
    }


class JointBlockOptimizer:
    def __init__(self, runtime, judge, config, output_dir, optimizers=None, on_candidate=None):
        legacy = {
            "task_score_weight",
            "joint_beam_size",
            "candidates_per_component",
            "random_joint_samples",
        } & config.keys()
        if legacy:
            raise ValueError(
                f"Obsolete optimization settings: {sorted(legacy)}. Use the current method YAML "
                "with max_joint_candidates and start a fresh run; old checkpoints use a "
                "different search protocol."
            )
        self.runtime, self.judge, self.config, self.output_dir = runtime, judge, config, output_dir
        self.include_current = config.get("include_current_configuration", True)
        if not isinstance(self.include_current, bool):
            raise ValueError("include_current_configuration must be true or false")
        self.optimizers = optimizers or default_optimizers(config)
        self.on_candidate = on_candidate
        self.task_scoring, self.constitution_weight, self.task_weight = task_scoring_settings(config)
        self.constitution_scoring = not self.task_scoring or self.constitution_weight > 0
        self.judge_votes = constitution_judge_votes(config)
        self.prompt_archive = {}
        self.last_search_report = {}

    def state_dict(self):
        return {"prompt_archive": plain(self.prompt_archive)}

    def load_state_dict(self, state):
        archive = state.get("prompt_archive", {})
        if not isinstance(archive, dict) or any(
            cid not in self.runtime.library or not isinstance(values, list)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            for cid, values in archive.items()
        ):
            raise ValueError("Invalid optimizer prompt archive")
        self.prompt_archive = {cid: list(values) for cid, values in archive.items()}

    def remember_prompt(self, cid, prompt):
        values = self.prompt_archive.setdefault(cid, [])
        if prompt in values:
            values.remove(prompt)
        values.append(prompt)
        del values[:-self.config.get("prompt_archive_size", 30)]

    def add_task_score(self, scored, state, examples, seed):
        if not self.task_scoring or scored.task_score is not None:
            return scored
        task_score, _ = evaluate(
            self.runtime, state.with_configs(scored.configurations), examples, [seed]
        )
        selection_score = task_score
        if self.constitution_scoring:
            total = self.constitution_weight + self.task_weight
            selection_score = (
                self.constitution_weight / total * scored.constitution_score
                + self.task_weight / total * task_score
            )
        return scored.model_copy(
            update={
                "task_score": task_score,
                "score": selection_score,
            }
        )

    def score_candidate(
        self,
        state,
        updates,
        constitution,
        examples,
        references,
        seed,
        members=None,
        baseline_outputs=None,
        include_task_score=True,
    ):
        candidate = state.with_configs(updates)
        if self.on_candidate:
            self.on_candidate(candidate.model_copy(deep=True))
        if not self.constitution_scoring:
            # Proposal feedback must use the same task metric as joint selection.
            # Other components keep their accepted configurations during this run.
            return self.add_task_score(
                CandidateScore(configurations=updates, score=0.0), state, examples, seed
            )
        members = list(members or updates)

        def run_block(current, reference):
            try:
                return self.runtime.execute_block(current, members, reference, seed)
            except ContentFilteredError as error:
                record_skip(
                    self.runtime.providers,
                    error,
                    "block_candidate",
                    example_id=reference.input_id,
                    seed=seed,
                )
                return None

        if baseline_outputs is None:
            baseline_outputs = [run_block(state, ref) for ref in references]
        outputs = [run_block(candidate, ref) for ref in references]
        votes, sample_records = [], []
        for i, (output, baseline) in enumerate(zip(outputs, baseline_outputs, strict=True)):
            record = {"input_id": references[i].input_id, "choices": [], "seeds": [], "score": None}
            sample_records.append(record)
            if output is None or baseline is None:
                record["status"] = "skipped_execution"
                continue
            try:
                # Repeat only the judgment, using the same recorded executions.
                # Each sample contributes one mean, irrespective of vote count.
                for vote in range(self.judge_votes):
                    vote_seed = seed + i * self.judge_votes + vote
                    record["seeds"].append(vote_seed)
                    choice = self.judge.choose(constitution, output, baseline, vote_seed)
                    if choice not in {"A", "B"}:
                        raise ModelResponseError("Constitution judge must choose A or B")
                    record["choices"].append(choice)
                record["identical_outputs"] = output == baseline
                record["score"] = (0.5 if output == baseline else
                                   record["choices"].count("A") / self.judge_votes)
                record["status"] = "completed"
                votes.append(record["score"])
            except ContentFilteredError as error:
                # Do not resample policy rejections or score an incomplete panel.
                record["status"] = "skipped_content_filter"
                record_skip(
                    self.runtime.providers,
                    error,
                    "candidate_constitution_judge",
                    example_id=references[i].input_id,
                    seed=vote_seed,
                )
        constitution_score = sum(votes) / len(votes) if votes else 0.0
        scored = CandidateScore(
            configurations=updates,
            constitution_score=constitution_score,
            score=constitution_score,
            evaluation={"constitution_judging": {
                "votes_per_sample": self.judge_votes,
                "requested_samples": len(references), "scored_samples": len(votes),
                "completed_judge_votes": sum(len(record["choices"]) for record in sample_records),
                "samples": sample_records,
            }},
        )
        return self.add_task_score(scored, state, examples, seed) if include_task_score else scored

    def propose(self, state, members, constitution, examples, seed, *, system_constitution=None,
                preference_examples=None):
        reuse = getattr(self.runtime, "reuse_component_executions", None)
        scope = reuse() if reuse and self.config.get("reuse_component_executions", False) else nullcontext()
        with scope:
            return self._propose(state, members, constitution, examples, seed,
                                 system_constitution=system_constitution,
                                 preference_examples=preference_examples)

    def _propose(self, state, members, constitution, examples, seed, *, system_constitution=None,
                 preference_examples=None):
        if not examples:
            raise ValueError("Local optimization needs training examples")
        if len({example.example_id for example in examples}) != len(examples):
            raise ValueError("Training example IDs must be unique within a proposal")
        if not self.include_current and all(self.runtime.library[c].config_type == ConfigurationType.NONE for c in members):
            return []
        screening = self.config.get("screening", {})
        staged = screening.get("enabled", False) and not self.constitution_scoring
        screen_examples = examples[:screening.get("batch_size", 5)] if staged else examples
        baseline_score, references = evaluate(self.runtime, state, screen_examples, [seed])
        if not references:
            raise ModelResponseError(
                "No unfiltered training trajectories remain for local optimization"
            )
        baseline_outputs = []
        for ref in references if self.constitution_scoring else []:
            try:
                baseline_outputs.append(self.runtime.execute_block(state, members, ref, seed))
            except ContentFilteredError as error:
                record_skip(
                    self.runtime.providers, error, "block_baseline", example_id=ref.input_id
                )
                baseline_outputs.append(None)
        memo = {}
        observations = {}
        baseline_key = state.fingerprint()
        if not self.constitution_scoring:
            observations[baseline_key] = {example.example_id: None for example in screen_examples}
            observations[baseline_key].update({trajectory.input_id: trajectory for trajectory in references})
            memo[baseline_key] = CandidateScore(
                configurations={}, score=baseline_score, task_score=baseline_score,
            )

        def task_score(updates, requested):
            candidate = state.with_configs(updates)
            key = candidate.fingerprint()
            observed = observations.setdefault(key, {})
            missing = [example for example in requested if example.example_id not in observed]
            if missing:
                if self.on_candidate:
                    self.on_candidate(candidate.model_copy(deep=True))
                _, trajectories = evaluate(self.runtime, candidate, missing, [seed])
                observed.update({example.example_id: None for example in missing})
                observed.update({trajectory.input_id: trajectory for trajectory in trajectories})
            value = sum(
                observed[example.example_id].reward if observed[example.example_id] is not None else 0
                for example in requested
            ) / len(requested)
            return CandidateScore(configurations=updates, score=value, task_score=value, evaluation={
                "examples": len(requested), "seed": seed,
                "batch_fingerprint": digest(requested),
                "stage": "full_training" if len(requested) == len(examples) else "screening",
                "shared_component_observations": bool(
                    getattr(self.runtime, "_component_cache", None) is not None
                ),
            })

        def candidate_key(updates):
            return (
                state.with_configs(updates).fingerprint()
                if not self.constitution_scoring else digest(updates)
            )

        def score(updates):
            key = candidate_key(updates)
            if not self.constitution_scoring:
                memo[key] = task_score(updates, screen_examples)
                return memo[key].score
            if key not in memo:
                memo[key] = self.score_candidate(
                    state,
                    updates,
                    constitution,
                    examples,
                    references,
                    seed,
                    members,
                    baseline_outputs,
                    include_task_score=False,
                )
            return memo[key].score

        options = {}
        for i, cid in enumerate(sorted(members)):
            comp = self.runtime.library[cid]
            if comp.config_type == ConfigurationType.NONE:
                options[cid] = [state.configurations[cid]]
                continue
            context = OptimizationContext(
                self.runtime,
                state,
                constitution,
                self.judge,
                screen_examples,
                references,
                seed + i * 100,
                self.config,
                self.output_dir,
                score,
                members=list(members),
                system_constitution=system_constitution if system_constitution is not None else constitution,
                preference_examples=preference_examples if preference_examples is not None else examples,
            )
            if comp.config_type == ConfigurationType.PROMPT:
                history_limit = self.config.get("prompt_history_limit", 3)
                previous = [value for value in self.prompt_archive.get(cid, [])
                            if value != state.configurations[cid].value]
                for value in previous[-history_limit:] if history_limit else []:
                    configuration = state.configurations[cid].model_copy(deep=True)
                    configuration.value = value
                    context.prompt_history.append({"prompt": value, "score": score({cid: configuration})})
                context.record_prompt = lambda value, cid=cid: self.remember_prompt(cid, value)
            try:
                proposals = self.optimizers[comp.config_type].propose(comp, context)
            except ContentFilteredError as error:
                record_skip(self.runtime.providers, error, "candidate_proposals", component_id=cid)
                proposals = [state.configurations[cid]]
            if not proposals:
                if not self.include_current:
                    return []
                raise ValueError(f"No candidates for {cid}")
            # Retain individually weak proposals: they may cooperate with other changes.
            proposals = [state.configurations[cid], *proposals]
            proposals = filter_current_configuration(state.configurations[cid], proposals, self.config)
            if not proposals:
                return []  # Never reinsert an unchanged value to fill a strict menu.
            options[cid] = list({
                digest(p) if self.include_current else digest([p.kind, p.value]): p
                for p in proposals
            }.values())

        # Score complete collaborations, never prune on partial-assignment scores.
        # Mixed-radix indices avoid materializing a potentially huge Cartesian product.
        total = math.prod(len(v) for v in options.values())
        limit = int(self.config.get("max_joint_candidates", 16))
        if limit < 1:
            raise ValueError("max_joint_candidates must be positive")
        if self.include_current:
            indices = [0]  # Every option list begins with the unchanged configuration.
            if total > 1:
                indices += random.Random(seed).sample(range(1, total), min(limit - 1, total - 1))
        else:
            indices = random.Random(seed).sample(range(total), min(limit, total))
        scored = []
        for index in indices:
            updates = {}
            for cid, proposals in options.items():
                index, digit = divmod(index, len(proposals))
                updates[cid] = proposals[digit]
            key = candidate_key(updates)
            try:
                score(updates)
                # Reuse task-only feedback scores when this exact configuration
                # was already evaluated; mixed/local feedback remains block-local.
                completed = self.add_task_score(
                    memo[key].model_copy(update={"configurations": updates}), state, examples, seed
                )
            except BudgetExceeded:
                if not scored:
                    raise
                # Keep completed comparisons when a bounded search runs out of budget.
                break
            scored.append(completed)
        scored = sorted(scored, key=lambda c: (-c.score, digest(c.configurations)))
        self.last_search_report = {
            "screening_enabled": staged and len(screen_examples) < len(examples),
            "screening_examples": len(screen_examples), "training_examples": len(examples),
            "screened_candidates": plain(scored),
            "history_rescored_on": digest(screen_examples),
        }
        if staged and len(screen_examples) < len(examples):
            changed = [item for item in scored if state.with_configs(item.configurations) != state]
            finalists = changed[:screening.get("finalists", 3)]
            unchanged = [item for item in scored if state.with_configs(item.configurations) == state]
            # Every returned score uses the same full batch. Screened-out states
            # cannot compete against finalists with incomparable small-batch scores.
            scored = [task_score(item.configurations, examples) for item in [*unchanged, *finalists]]
            self.last_search_report["finalist_states"] = [
                state.with_configs(item.configurations).fingerprint() for item in scored
            ]
        return sorted(scored, key=lambda c: (-c.score, digest(c.configurations)))

    def select(self, scored, state, members, seed, final=False):
        if not scored:
            raise ValueError("No joint candidates")
        discrete = any(
            self.runtime.library[c].config_type
            in {ConfigurationType.MODEL_SELECTION, ConfigurationType.HYPERPARAMETER}
            for c in members
        )
        if discrete:
            return ModelSelectorOptimizer(
                self.config.get("softmax_temperature", 1), self.config.get("sample_discrete", False)
            ).select(scored, [s.score for s in scored], seed, final)
        return scored[0]
