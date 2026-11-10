"""Training preferences from frozen component outputs and fresh downstream runs."""

from itertools import combinations
from math import isfinite
import random

from caiso.constitution.models import PreferencePair
from caiso.core.configuration import ConfigurationType
from caiso.core.content_filter import ContentFilteredError, execute_example, record_skip
from caiso.core.recovery import ModelResponseError
from caiso.core.serialization import append_jsonl, digest, save_jsonl
from .config import validate_method

PROTOCOL = "component-continuations-v1"


def collect_preferences(runtime, state, train, count, attempts, seed, *, settings,
                        comparison_path=None, min_margin=0, progress=None):
    """A comparison attempt is one question/component candidate pool.

    Mean task rewards label candidate *outputs*, not arbitrary individual rollouts.
    Only strict unequal-score pairs are eligible; every continuation stays in the
    evidence and audit log. No constitution or gold-output injection is involved.
    """
    validate_method({"preference_collection": settings})
    if not train or any(example.split != "train" for example in train):
        raise ValueError("Preference collection requires nonempty training data")
    if not hasattr(runtime, "sample_preference_candidate"):
        raise ValueError("Component preference collection requires a sampling runtime; supply labeled preferences or use preference_collection.mode: rerun")
    if count < 1 or (attempts != -1 and attempts < 1) or not isfinite(min_margin) or min_margin < 0:
        raise ValueError("Invalid preference count, attempts or score margin")
    samples = settings.get("candidates", 4)
    rollouts = settings.get("downstream_rollouts", 3)
    temperature = settings.get("temperature", 0.9)
    per_context = settings.get("pairs_per_context", 1)
    if samples < 2 or rollouts < 1 or per_context < 1:
        raise ValueError("Preference collection needs at least two candidates and positive rollout/pair counts")
    if temperature is not None and not 0 <= temperature <= 2:
        raise ValueError("Preference sampling temperature must be between zero and two or null")
    components = [cid for cid in state.workflow.topological_order()
                  if runtime.library[cid].config_type != ConfigurationType.NONE]
    if not components:
        raise ModelResponseError("No optimizable components for preference sampling")
    # Shuffle once, then visit every component on a question before advancing.
    # This also works when the training-set size is a multiple of component count.
    questions = list({digest(example.inputs): example for example in train}.values())
    rng = random.Random(seed)
    rng.shuffle(questions)
    rng.shuffle(components)
    limit = min(count, len(questions)) * len(components) if attempts == -1 else attempts
    if comparison_path:
        save_jsonl(comparison_path, [])
    pairs, reference, reference_key = [], None, None
    filtered_questions, filtered_contexts, filtered_outputs = set(), set(), set()

    def log(record):
        if comparison_path:
            append_jsonl(comparison_path, {"protocol": PROTOCOL, **record})
        if progress:
            progress(attempt + 1, limit, len(pairs), count)

    for attempt in range(limit):
        example = questions[(attempt // len(components)) % len(questions)]
        cid = components[attempt % len(components)]
        base_seed = seed + attempt * 100000
        key = digest(example.inputs)
        if key in filtered_questions or (key, cid) in filtered_contexts:
            log({"example_id": example.example_id, "component_id": cid, "status": "skipped_content_filter"})
            continue
        if key != reference_key or reference is None:
            reference = execute_example(runtime, state, example, base_seed)
            reference_key = key
        if reference is None:
            filtered_questions.add(key)
            log({"example_id": example.example_id, "component_id": cid, "status": "skipped_content_filter"})
            continue
        index = state.workflow.topological_order().index(cid)
        initial = reference.executions[:index + 1]
        candidates = [(state, initial, {"source": "original", "configuration": state.configurations[cid].model_dump()})]
        seen = {digest(initial[-1].component_output)}
        component = runtime.library[cid]
        alternatives = [value for value in component.config_search_space
                        if value != state.configurations[cid].value]
        rng.shuffle(alternatives)
        for sample in range(samples - 1):
            config = state.configurations[cid].model_copy(deep=True)
            if alternatives:
                config.value = alternatives[sample % len(alternatives)]
            candidate_state = state.with_configs({cid: config})
            try:
                executions = runtime.sample_preference_candidate(
                    candidate_state, example, reference, cid, base_seed + sample + 1, temperature,
                )
            except ContentFilteredError as error:
                record_skip(runtime.providers, error, "preference_sampling", example_id=example.example_id, component_id=cid)
                filtered_contexts.add((key, cid))
                break
            output_key = digest(executions[-1].component_output)
            if output_key not in seen:
                seen.add(output_key)
                candidates.append((candidate_state, executions, {
                    "source": "sample", "requested_temperature": temperature,
                    "configuration": config.model_dump(),
                }))
        if (key, cid) in filtered_contexts:
            log({"example_id": example.example_id, "component_id": cid, "status": "skipped_content_filter"})
            continue
        if len(candidates) < 2:
            log({"example_id": example.example_id, "component_id": cid,
                 "status": "abstained", "abstention_reason": "identical_component_outputs"})
            continue
        scored, excluded = [], []
        for candidate_state, executions, sampling in candidates:
            output_key = (key, cid, digest(executions[-1].component_output))
            if output_key in filtered_outputs:
                continue
            continuations = []
            for repeat in range(rollouts):
                try:
                    continuations.append(runtime.continue_preference_candidate(
                        candidate_state, example, reference, executions, cid,
                        base_seed + 10000 + repeat,
                    ))
                except ContentFilteredError as error:
                    record_skip(runtime.providers, error, "preference_continuation", example_id=example.example_id, component_id=cid)
                    filtered_outputs.add(output_key)
                    break  # Never replace a policy rejection with another draw.
            if len(continuations) != rollouts:
                excluded.append({
                    "sampling": sampling, "status": "skipped_content_filter",
                    "executions": [execution.model_dump() for execution in executions],
                    "continuations": [trajectory.model_dump() for trajectory in continuations],
                    "scores": [trajectory.reward for trajectory in continuations],
                })
                continue  # Do not compare estimates with policy-selected denominators.
            scores = [trajectory.reward for trajectory in continuations]
            if not all(isfinite(score) for score in scores):
                raise ValueError("Non-finite preference continuation score")
            evidence = {
                "trajectory": {"inputs": example.inputs, "executions": [e.model_dump() for e in executions]},
                "continuations": [trajectory.model_dump() for trajectory in continuations],
            }
            scored.append({"candidate": evidence, "sampling": sampling, "scores": scores,
                           "mean_score": sum(scores) / len(scores)})
        eligible = [(i, j) for i, j in combinations(range(len(scored)), 2)
                    if abs(scored[i]["mean_score"] - scored[j]["mean_score"]) > min_margin]
        # Larger gaps first; all unordered pairs are considered regardless of index order.
        eligible.sort(key=lambda ij: -abs(scored[ij[0]]["mean_score"] - scored[ij[1]]["mean_score"]))
        retained = []
        for i, j in eligible[:min(per_context, count - len(pairs))]:
            if rng.random() < 0.5:
                i, j = j, i
            a, b = scored[i], scored[j]
            pair = PreferencePair(
                pair_id=f"component-{digest([state, seed, attempt, cid, i, j])[:24]}",
                context={**example.inputs, "preference_comparison": {
                    "component_id": cid,
                    "objective": "Prefer the frozen component output with higher expected task performance over the recorded downstream continuations.",
                }},
                candidate_a=a["candidate"], candidate_b=b["candidate"],
                preferred="A" if a["mean_score"] > b["mean_score"] else "B",
                metadata={"protocol": PROTOCOL, "label_source": "mean_downstream_task_metric",
                          "example_id": example.example_id, "component_id": cid,
                          "mean_scores": [a["mean_score"], b["mean_score"]],
                          "rollout_scores": [a["scores"], b["scores"]],
                          "sampling": [a["sampling"], b["sampling"]],
                          "task_generation_seeded": False},
            )
            pairs.append(pair)
            retained.append(pair.pair_id)
        log({"example_id": example.example_id, "component_id": cid,
             "status": "labeled" if retained else "abstained",
             "abstention_reason": None if retained else "equal_or_insufficient_score_margin",
             "candidates": scored, "excluded_candidates": excluded, "pair_ids": retained})
        if len(pairs) >= count:
            break
    if not pairs:
        raise ModelResponseError("No usable unequal-score training pairs were observed with component sampling. Supply labeled caiso.preferences_path data or change the sampling settings.")
    return pairs
