"""
clustering.py  (Stage 3)

The debate produces a principle set per datapoint. This stage pools every
principle, groups near-duplicates, and abstracts each group into one
representative principle, giving a non-overlapping final constitution.

Steps
    1. Embed  every principle with Azure OpenAI, L2-normalise, and build the
       pairwise cosine distance matrix.
    2. Cluster with average-linkage agglomerative clustering on that matrix.
       The number of clusters is NOT fixed and NOT set by a distance threshold:
       k is swept over a range and the k with the highest silhouette score on
       the precomputed cosine distances is selected.
    3. Abstract each cluster into a single principle that represents all its
       members, at the lowest level of abstraction that still covers them.

Reconstruction remains the test of the result: validate_constitution applies the
final constitution back to the datapoints, and minimize_constitution trims it to
the smallest set that still recovers the original preferences.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from common import (
    PreferencePair,
    Principle,
    chat,
    embed,
    reconstructs,
)

# k is searched over [K_MIN, min(K_MAX, n - 1)]
K_MIN = 2
K_MAX = 12


# ------------------------------ geometry ---------------------------------- #

def _normalize(vecs: List[List[float]]) -> np.ndarray:
    m = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def _cosine_distance_matrix(unit: np.ndarray) -> np.ndarray:
    sim = np.clip(unit @ unit.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def _fit(dist: np.ndarray, k: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average")
        return model.fit_predict(dist)
    except TypeError:  # older sklearn uses affinity=
        model = AgglomerativeClustering(
            n_clusters=k, affinity="precomputed", linkage="average")
        return model.fit_predict(dist)


def select_k(dist: np.ndarray,
             k_min: int = K_MIN,
             k_max: int = K_MAX,
             verbose: bool = True) -> Tuple[int, np.ndarray, Dict[int, float]]:
    """Sweep k and keep the partition with the highest silhouette score,
    computed on the precomputed cosine distances."""
    n = dist.shape[0]
    hi = min(k_max, n - 1)
    scores: Dict[int, float] = {}
    best_k, best_labels, best_score = 1, np.zeros(n, dtype=int), -1.0

    for k in range(k_min, hi + 1):
        labels = _fit(dist, k)
        if len(set(labels)) < 2:
            continue
        s = float(silhouette_score(dist, labels, metric="precomputed"))
        scores[k] = s
        if s > best_score:
            best_k, best_labels, best_score = k, labels, s

    if verbose and scores:
        table = "  ".join(f"k={k}:{s:.3f}" for k, s in sorted(scores.items()))
        print(f"[silhouette] {table}")
        print(f"[silhouette] selected k={best_k} (score {best_score:.3f})")
    return best_k, best_labels, scores


def _medoid(dist: np.ndarray, idxs: List[int]) -> int:
    """Index of the cluster member with the smallest total distance to the
    rest of its cluster. Used as an auditable stand-in for the cluster."""
    sub = dist[np.ix_(idxs, idxs)]
    return idxs[int(np.argmin(sub.sum(axis=1)))]


# ------------------------------ abstraction -------------------------------- #

_ABSTRACT_SYS = "You abstract a cluster of preference principles into one principle."

_ABSTRACT_USER = """You are given a cluster of preference principles that all express the same
underlying rule. Each has the form "Select the response that <condition>".

Write ONE principle that captures what they share. Make it general enough to
cover every member, but specific enough that two responses could actually
differ on it. Use the exact format: "Select the response that <condition>."

CLUSTER:
{principles}

Reply with only the principle."""


def abstract_cluster(members: List[str]) -> str:
    """Turn one cluster into a single representative principle."""
    if len(members) == 1:
        return members[0].strip()
    bullets = "\n".join(f"- {m}" for m in members)
    raw = chat(
        [{"role": "system", "content": _ABSTRACT_SYS},
         {"role": "user", "content": _ABSTRACT_USER.format(principles=bullets)}],
        temperature=0.2,
        max_tokens=80,
    )
    return raw.strip().strip('"').strip()


# ------------------------------ pipeline ---------------------------------- #

def cluster_principles(principles: List[Principle],
                       k: Optional[int] = None,
                       k_min: int = K_MIN,
                       k_max: int = K_MAX,
                       verbose: bool = True) -> List[Dict]:
    """Cluster all principles (k chosen by silhouette unless k is given), then
    abstract each cluster into one representative principle.

    Returns [{"principle", "members", "support", "medoid", "silhouette"}]."""
    texts = [p.text.strip() for p in principles if p.text.strip()]
    if not texts:
        return []
    if len(texts) <= 2:
        return [{"principle": t, "members": [t], "support": 1, "medoid": t,
                 "silhouette": None} for t in texts]

    unit = _normalize(embed(texts))
    dist = _cosine_distance_matrix(unit)

    if k is None:
        k, labels, scores = select_k(dist, k_min, k_max, verbose=verbose)
        score = scores.get(k)
    else:
        labels = _fit(dist, k)
        score = float(silhouette_score(dist, labels, metric="precomputed"))
        if verbose:
            print(f"[silhouette] k={k} fixed by caller (score {score:.3f})")

    groups: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    if verbose:
        print(f"[cluster] {len(texts)} principles -> {len(groups)} clusters")

    out: List[Dict] = []
    for idxs in groups.values():
        members = [texts[i] for i in idxs]
        med = texts[_medoid(dist, idxs)]
        abstracted = abstract_cluster(members)
        if verbose:
            print(f"[abstract] {len(members)} -> {abstracted}")
        out.append({"principle": abstracted, "members": members,
                    "support": len(members), "medoid": med,
                    "silhouette": score})

    # two clusters can abstract to the same sentence; collapse those
    merged: Dict[str, Dict] = {}
    for c in out:
        key = re.sub(r"[^a-z0-9 ]", "", c["principle"].lower()).strip()
        if key in merged:
            m = merged[key]
            m["members"].extend(c["members"])
            m["support"] += c["support"]
        else:
            merged[key] = c
    if verbose and len(merged) < len(out):
        print(f"[abstract] collapsed {len(out)} -> {len(merged)} identical abstractions")
    return list(merged.values())


# --------------------------- reconstruction check ------------------------- #

def validate_constitution(pairs: List[PreferencePair],
                          principle_texts: List[str],
                          votes: int = 1) -> Dict:
    """Apply the final constitution back to every datapoint and measure how
    often it reconstructs the original preference."""
    results = []
    correct = 0
    for p in pairs:
        ok, pred = reconstructs(p, principle_texts, votes=votes)
        correct += int(ok)
        results.append({"id": p.id, "gold": p.preferred.upper(),
                        "predicted": pred, "reconstructed": ok})
    acc = correct / len(pairs) if pairs else 0.0
    return {"accuracy": acc, "n": len(pairs), "results": results}


def minimize_constitution(pairs: List[PreferencePair],
                          principle_texts: List[str],
                          votes: int = 1,
                          strict: bool = False,
                          verbose: bool = True) -> List[str]:
    """Greedily drop principles whose removal does not lower reconstruction
    accuracy. With strict=True a principle is dropped only if removing it
    strictly improves accuracy."""
    kept = list(principle_texts)
    base = validate_constitution(pairs, kept, votes)["accuracy"]
    i = 0
    while i < len(kept):
        trial = kept[:i] + kept[i + 1:]
        acc = validate_constitution(pairs, trial, votes)["accuracy"] if trial else -1.0
        drop = acc > base if strict else acc >= base
        if drop:
            if verbose:
                print(f"[minimize] dropped '{kept[i][:60]}' (acc {base:.2f} -> {acc:.2f})")
            kept = trial
            base = acc
        else:
            i += 1
    return kept


if __name__ == "__main__":
    sample = [
        Principle("Select the response that demonstrates higher tension and conflict."),
        Principle("Select the response that sustains dramatic friction between characters."),
        Principle("Select the response that demonstrates higher specificity of imagery."),
        Principle("Select the response that grounds description in concrete detail."),
        Principle("Select the response that shows greater conceptual originality."),
        Principle("Select the response that avoids familiar or generic tropes."),
    ]
    for c in cluster_principles(sample):
        print(f"[{c['support']}] {c['principle']}")
        for m in c["members"]:
            print("     <-", m)
