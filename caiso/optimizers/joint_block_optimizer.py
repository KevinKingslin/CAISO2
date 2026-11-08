import math
import random

from caiso.core.budget import BudgetExceeded
from caiso.core.configuration import ConfigurationType
from caiso.core.content_filter import ContentFilteredError, record_skip
from caiso.core.recovery import ModelResponseError
from caiso.core.serialization import digest
from caiso.evaluation.evaluator import evaluate

from .base import CandidateScore, OptimizationContext, filter_current_configuration
from .hyperparameter_optimizer import HyperparameterOptimizer
from .model_selector import ModelSelectorOptimizer
from .prompt_optimizer import PromptOptimizer
from .weight_optimizer import WeightOptimizer


def default_optimizers(config):
    kwargs = {
        "temperature": config.get("softmax_temperature", 1),
        "sample": config.get("sample_discrete", True),
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
        task = config.get("task_scoring", {})
        if not isinstance(task, dict):
            raise ValueError("task_scoring must be a configuration mapping")
        self.task_scoring = task.get("enabled", False)
        if not isinstance(self.task_scoring, bool):
            raise ValueError("task_scoring.enabled must be true or false")
        self.constitution_weight = float(task.get("constitution_weight", 1))
        self.task_weight = float(task.get("task_weight", 1))
        if self.task_scoring and not all(
            math.isfinite(w) and w > 0 for w in (self.constitution_weight, self.task_weight)
        ):
            raise ValueError(
                "Enabled task_scoring requires finite positive weights for both scores"
            )

    def add_task_score(self, scored, state, examples, seed):
        if not self.task_scoring:
            return scored
        task_score, _ = evaluate(
            self.runtime, state.with_configs(scored.configurations), examples, [seed]
        )
        return scored.model_copy(
            update={
                "task_score": task_score,
                "score": (
                    self.constitution_weight * scored.constitution_score
                    + self.task_weight * task_score
                )
                / (self.constitution_weight + self.task_weight),
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
        votes = []
        for i, (output, baseline) in enumerate(zip(outputs, baseline_outputs, strict=True)):
            if output is None or baseline is None:
                continue
            try:
                votes.append(
                    0.5
                    if output == baseline
                    else float(self.judge.choose(constitution, output, baseline, seed + i) == "A")
                )
            except ContentFilteredError as error:
                record_skip(
                    self.runtime.providers,
                    error,
                    "candidate_constitution_judge",
                    example_id=references[i].input_id,
                )
        constitution_score = sum(votes) / len(votes) if votes else 0.0
        scored = CandidateScore(
            configurations=updates,
            constitution_score=constitution_score,
            score=constitution_score,
        )
        return self.add_task_score(scored, state, examples, seed) if include_task_score else scored

    def propose(self, state, members, constitution, examples, seed):
        if not examples:
            raise ValueError("Local optimization needs training examples")
        if not self.include_current and all(self.runtime.library[c].config_type == ConfigurationType.NONE for c in members):
            return []
        _, references = evaluate(self.runtime, state, examples, [seed])
        if not references:
            raise ModelResponseError(
                "No unfiltered training trajectories remain for local optimization"
            )
        baseline_outputs = []
        for ref in references:
            try:
                baseline_outputs.append(self.runtime.execute_block(state, members, ref, seed))
            except ContentFilteredError as error:
                record_skip(
                    self.runtime.providers, error, "block_baseline", example_id=ref.input_id
                )
                baseline_outputs.append(None)
        memo = {}

        def score(updates):
            key = digest(updates)
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
                examples,
                references,
                seed + i * 100,
                self.config,
                self.output_dir,
                score,
            )
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
            key = digest(updates)
            try:
                score(updates)
                # Only complete joint candidates receive end-to-end task evaluation.
                # Prompt proposal/history callbacks above remain block-local.
                completed = self.add_task_score(memo[key], state, examples, seed)
            except BudgetExceeded:
                if not scored:
                    raise
                # Keep completed comparisons when a bounded search runs out of budget.
                break
            scored.append(completed)
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
                self.config.get("softmax_temperature", 1), self.config.get("sample_discrete", True)
            ).select(scored, [s.score for s in scored], seed, final)
        return scored[0]
