"""Constitution-guided judging from arXiv:2606.28294v2, Figure 16.

MIT attribution: third_party/democratic_icai. Gold labels are used only after
inference, never in the blind judge's messages.
"""

import random
import re
from collections import Counter

from caiso.core.content_filter import ContentFilteredError, record_skip
from caiso.core.parallel import ordered_map
from caiso.core.recovery import ModelResponseError
from caiso.prompts import render_messages

from .preferences import candidate_views


def parse_choice(text):
    match = re.fullmatch(r"Output\(([ab])\)", text.strip())
    if not match:
        raise ModelResponseError("Constitution judge must return Output(a) or Output(b)")
    return match.group(1)


def apply_constitution(providers, role, pair, principles, seed=0, temperature=0.0):
    a, b = candidate_views(pair)
    swapped = random.Random(seed).random() < 0.5
    first, second = (b, a) if swapped else (a, b)
    response = providers.complete(
        role,
        render_messages(
            "dicai", "reconstruction", constitution="\n".join(principles),
            response_1=first, response_2=second,
        ),
        seed=seed,
        validate=lambda text: (parse_choice(text), text)[1],
        temperature=temperature,
    )
    choice = ("B" if swapped else "A") if parse_choice(response) == "a" else ("A" if swapped else "B")
    return choice, {"predicted": choice, "response": response, "swapped": swapped}


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
