"""Read CAISO pair JSON or the supplied repository's CSV formats."""

import csv
import random
import re
from pathlib import Path

from caiso.constitution.models import PreferencePair
from caiso.core.serialization import digest, read_json


def load_principles(path):
    """Load CAISO/official constitution JSON or numbered principle text for inference."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = read_json(path)
        items = value.get("principles", value) if isinstance(value, dict) else value
        return [(p["principle"] if isinstance(p, dict) else p).strip() for p in items]
    return [
        re.sub(r"^\d+[.)]\s*", "", line.strip())
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def load_pairs(path, seed=0, inference=False):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return [PreferencePair.model_validate(p) for p in read_json(path)]
    pairs, seen = [], set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if inference:
                context, a, b = row["input"], row["text_a"], row["text_b"]
                preferred = row.get("preferred_text", "")
                if preferred and preferred not in {a, b}:
                    raise ValueError("CSV preferred_text must exactly match text_a or text_b")
                labeled = bool(preferred)
            else:
                context, preferred, rejected = row["task_prompt"], row["chosen"], row["rejected"]
                a, b, labeled = preferred, rejected, True
            if not context.strip() or not a.strip() or not b.strip() or a == b:
                raise ValueError(
                    "Preference rows require a prompt and two different nonempty responses"
                )
            identity = digest([context, a, b])
            if identity in seen:
                continue
            seen.add(identity)
            # Upstream induction randomizes A/B per unique pair; inference keeps columns.
            if not inference and random.Random(f"{seed}:{identity}").random() < 0.5:
                a, b = b, a
            pairs.append(
                PreferencePair(
                    pair_id=identity,
                    context={"prompt": context},
                    candidate_a={"text": a},
                    candidate_b={"text": b},
                    preferred="A" if not labeled or preferred == a else "B",
                    metadata={"label_source": "provided" if labeled else "unlabeled"},
                )
            )
    if not pairs:
        raise ValueError("No preference pairs were loaded")
    return pairs
