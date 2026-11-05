"""Hotpot F1 adapted from OPTIMAS, Apache-2.0; see docs/SOURCE_MAP.md."""

import re
import string
from collections import Counter


def normalize_answer(text):
    text = "".join(c for c in str(text).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def f1_score(answer, target):
    a, b = normalize_answer(answer), normalize_answer(target)
    if (a in ("yes", "no", "noanswer") or b in ("yes", "no", "noanswer")) and a != b:
        return 0.0
    aa, bb = a.split(), b.split()
    same = sum((Counter(aa) & Counter(bb)).values())
    if not same:
        return 0.0
    return 2 * same / (len(aa) + len(bb))


def accuracy(answer, target):
    return float(str(answer) == str(target))


def pubmed_label(answer):
    text = str(answer).strip().lower()
    if text in {"yes", "no", "maybe"}:
        return text
    matches = re.findall(r"^\s*answer\s*:\s*(yes|no|maybe)\s*[.!]?\s*$", text, re.M)
    return matches[-1] if matches else "invalid"


def pubmed_accuracy(answer, target):
    return float(pubmed_label(answer) == str(target).lower().strip())


def pubmed_optimas_accuracy(answer, target):
    """Released OPTIMAS scorer, including its first-substring matching behavior.

    Kept under a distinct name for reference comparisons; pubmed_accuracy is the
    corrected final Answer: label metric. Never silently interchange the two.
    """
    text = str(answer)
    match = re.search(r"(?i)\s*(yes|no|maybe|Yes|No|Maybe)", text)
    predicted = match.group(1).lower() if match else text
    return float(predicted.lower() == str(target).lower().strip())


METRICS = {
    "accuracy": accuracy,
    "pubmed_accuracy": pubmed_accuracy,
    "pubmed_optimas_accuracy": pubmed_optimas_accuracy,
    "f1": f1_score,
}
