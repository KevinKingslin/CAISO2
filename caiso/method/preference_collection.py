"""One unchanged/changed configuration pair per sample, with random task ties."""

import random

from caiso.constitution.models import PreferencePair
from caiso.core.configuration import ConfigurationType
from caiso.core.content_filter import ContentFilteredError, record_skip
from caiso.core.recovery import ModelResponseError
from caiso.core.serialization import append_jsonl, digest, save_jsonl
from .config import validate_method

PROTOCOL = "configuration-pair-v2"


def component_changes(runtime, state, temperature, component_ids=None):
    """Only components with an actual alternate parameter value are eligible."""
    if component_ids is not None and (not component_ids or not set(component_ids) <= set(state.workflow.nodes)):
        raise ValueError("Preference changes must name active components")
    changes = {}
    for cid in state.workflow.topological_order():
        if component_ids is not None and cid not in component_ids:
            continue
        component = runtime.library[cid]
        if component.config_type == ConfigurationType.NONE:
            continue
        alternatives = [value for value in component.config_search_space
                        if value != state.configurations[cid].value]
        if alternatives:
            changes[cid] = [{"parameter": "variable", "value": value} for value in alternatives]
        else:
            current = getattr(runtime.engine.components[cid].config, "temperature", None)
            if temperature is not None and current is not None and temperature != current:
                changes[cid] = [{"parameter": "temperature", "value": temperature, "original": current}]
    return changes


def collect_preferences(runtime, state, train, count, attempts, seed, *, settings,
                        comparison_path=None, min_margin=0, progress=None,
                        component_ids=None, labeler=None, allow_empty=False):
    """Select a random component, run each configuration once, and retain ties.

    min_margin belongs to the legacy rerun collector. Here every strictly higher
    task score wins, and exact ties (including identical outputs) use a coin flip.
    Weight induction restricts eligible components and supplies a constitution
    labeler, which may abstain on pairs unsuitable for a common DPO prompt.
    """
    validate_method({"preference_collection": settings})
    if not train or any(example.split != "train" for example in train):
        raise ValueError("Preference collection requires nonempty training data")
    if not hasattr(runtime, "run_preference_configuration"):
        raise ValueError("Configuration preferences require a sampling runtime; supply labeled preferences or use preference_collection.mode: rerun")
    validate_method({"initial_preference_pairs": count, "initial_preference_attempts": attempts})
    changes = component_changes(runtime, state, settings.get("temperature", 0.9), component_ids)
    if not changes:
        if allow_empty:
            return []
        raise ModelResponseError("No components have an alternate temperature or declared hyperparameter/model value")
    questions = list({digest(example.inputs): example for example in train}.values())
    rng, tie_rng = random.Random(seed), random.Random(f"{seed}:task-ties")
    rng.shuffle(questions)
    limit = min(count, len(questions)) if attempts == -1 else attempts
    if comparison_path:
        save_jsonl(comparison_path, [])
    pairs, filtered_questions, filtered_contexts = [], set(), set()
    for attempt in range(limit):
        example = questions[attempt % len(questions)]
        key = digest(example.inputs)
        eligible = [cid for cid in changes if (key, cid) not in filtered_contexts]
        if key in filtered_questions or not eligible:
            continue
        cid = rng.choice(eligible)
        change = rng.choice(changes[cid])
        candidate_state = state
        temperature = None
        if change["parameter"] == "variable":
            config = state.configurations[cid].model_copy(deep=True, update={"value": change["value"]})
            candidate_state = state.with_configs({cid: config})
        else:
            temperature = change["value"]
        base_seed = seed + attempt * 3
        prefix = None
        try:
            prefix = runtime.preference_prefix(state, example, cid, base_seed)
            unchanged = runtime.run_preference_configuration(state, example, prefix, cid, base_seed + 1)
            changed = runtime.run_preference_configuration(candidate_state, example, prefix, cid, base_seed + 2, temperature)
            preferred = labeler(unchanged, changed, base_seed) if labeler else None
        except ContentFilteredError as error:
            record_skip(runtime.providers, error, "configuration_preferences", example_id=example.example_id, component_id=cid)
            if prefix is None:
                filtered_questions.add(key)
            else:
                filtered_contexts.add((key, cid))
            if comparison_path:
                append_jsonl(comparison_path, {"protocol": PROTOCOL, "example_id": example.example_id,
                                               "component_id": cid, "status": "skipped_content_filter"})
            continue
        tied = unchanged.reward == changed.reward
        if labeler is None:
            preferred = tie_rng.choice(("A", "B")) if tied else ("A" if unchanged.reward > changed.reward else "B")
        elif preferred is None:
            if comparison_path:
                append_jsonl(comparison_path, {"protocol": PROTOCOL, "example_id": example.example_id,
                                               "component_id": cid, "status": "abstained"})
            continue
        if preferred not in ("A", "B"):
            raise ModelResponseError("Preference labeler must return A, B, or None")
        pair = PreferencePair(
            pair_id=f"configuration-{digest([seed, attempt, example.example_id, cid, change])[:24]}",
            context=example.inputs,
            candidate_a={"trajectory": unchanged.model_dump()},
            candidate_b={"trajectory": changed.model_dump()},
            preferred=preferred,
            metadata={"protocol": PROTOCOL, "example_id": example.example_id, "component_id": cid,
                      "candidate_roles": {"A": "unchanged", "B": "changed"},
                      "change": change, "global_scores": [unchanged.reward, changed.reward],
                      "task_score_tied": tied,
                      "label_source": "constitution" if labeler else "random_task_tie" if tied else "training_task_metric",
                      "task_generation_seeded": False},
        )
        pairs.append(pair)
        if comparison_path:
            append_jsonl(comparison_path, {"protocol": PROTOCOL, "status": "labeled", "preference": pair})
        if progress:
            progress(attempt + 1, limit, len(pairs), count)
        if len(pairs) >= count:
            break
    if not pairs and not allow_empty:
        raise ModelResponseError("No unfiltered configuration preference pairs were collected")
    return pairs
