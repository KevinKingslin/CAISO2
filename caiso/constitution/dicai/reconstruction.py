"""Official common.apply_constitution/reconstructs, with validated responses.

MIT attribution: third_party/democratic_icai. Gold labels are used only after
inference, never in the blind judge's messages.
"""

import random
from collections import Counter
from typing import Literal

from pydantic import Field

from caiso.core.content_filter import ContentFilteredError, record_skip
from caiso.core.parallel import ordered_map
from caiso.core.serialization import Record, canonical
from caiso.providers.base import messages

from .preferences import candidate_views, task_view
from .reasoning_assembly import prompt


class PreferenceChoice(Record):
    choice: Literal["1", "2"]
    reason: str = Field(min_length=1)


def apply_constitution(providers, role, pair, principles, seed=0, temperature=0.0):
    a, b = candidate_views(pair)
    swapped = random.Random(seed).random() < 0.5
    first, second = (b, a) if swapped else (a, b)
    user = (
        canonical(
            {
                "prompt": task_view(pair.context),
                "principles": principles,
                "response_1": first,
                "response_2": second,
            }
        )
        + '\nReturn JSON: {"choice": "1" or "2", "reason": "brief justification"}.'
    )
    response = providers.complete(
        role,
        messages(prompt("reconstruction"), user),
        seed=seed,
        schema=PreferenceChoice,
        temperature=temperature,
    )
    choice = ("B" if swapped else "A") if response.choice == "1" else ("A" if swapped else "B")
    return choice, {"predicted": choice, "reason": response.reason, "swapped": swapped}


def reconstructs(providers, role, pair, principles, votes=1, temperature=0.0):
    if not principles:
        return {
            "id": pair.pair_id,
            "gold": pair.preferred,
            "predicted": "",
            "reconstructed": False,
            "votes": [],
        }
    try:
        observations = [
            apply_constitution(providers, role, pair, principles, i, temperature)[1]
            for i in range(votes)
        ]
    except ContentFilteredError as error:
        record_skip(providers, error, "constitution_reconstruction", pair_id=pair.pair_id)
        raise
    # Upstream Counter.most_common resolves ties by first prediction.
    prediction = Counter(v["predicted"] for v in observations).most_common(1)[0][0]
    return {
        "id": pair.pair_id,
        "gold": pair.preferred,
        "predicted": prediction,
        "reconstructed": prediction == pair.preferred,
        "votes": observations,
    }


def validate_constitution(
    providers, role, pairs, principles, votes=1, temperature=0.0, progress=None, max_workers=1
):
    def reconstruct(pair):
        try:
            return reconstructs(providers, role, pair, principles, votes, temperature)
        except ContentFilteredError:
            return {
                "id": pair.pair_id,
                "gold": pair.preferred,
                "predicted": "",
                "reconstructed": False,
                "votes": [],
                "skipped_content_filter": True,
            }

    results = ordered_map(
        reconstruct,
        pairs,
        max_workers,
        progress,
    )
    return {
        "accuracy": sum(r["reconstructed"] for r in results) / len(results) if results else 0,
        "n": len(results),
        "skipped_content_filter": sum(r.get("skipped_content_filter", False) for r in results),
        "results": results,
    }


def minimize_constitution(
    providers, role, pairs, principles, votes=1, strict=False, temperature=0.0, max_workers=1
):
    kept = list(principles)
    base = validate_constitution(
        providers, role, pairs, kept, votes, temperature, max_workers=max_workers
    )["accuracy"]
    index, history = 0, []
    while index < len(kept):
        trial = kept[:index] + kept[index + 1 :]
        accuracy = (
            validate_constitution(
                providers, role, pairs, trial, votes, temperature, max_workers=max_workers
            )["accuracy"]
            if trial
            else -1.0
        )
        drop = accuracy > base if strict else accuracy >= base
        history.append({"principle": kept[index], "accuracy": accuracy, "dropped": drop})
        if drop:
            kept, base = trial, accuracy
        else:
            index += 1
    return kept, history
