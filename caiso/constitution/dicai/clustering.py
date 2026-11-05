"""Official normalized cosine/average-linkage silhouette search.

Adapted from Democratic ICAI clustering.py (MIT; third_party/democratic_icai).
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def cluster_principles(
    embeddings, min_clusters=2, max_clusters=12, seed=0, fixed_clusters=None, cluster_step=1
):
    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Expected nonempty finite embedding matrix")
    if min_clusters < 2 or (max_clusters is not None and max_clusters < min_clusters):
        raise ValueError("Silhouette search requires 2 <= min_clusters <= max_clusters")
    if isinstance(cluster_step, bool) or not isinstance(cluster_step, int) or cluster_step < 1:
        raise ValueError("cluster_step must be a positive integer")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.where(norms == 0, 1.0, norms)
    distance = np.clip(1 - np.clip(unit @ unit.T, -1, 1), 0, 2)
    np.fill_diagonal(distance, 0)
    n, scores = len(vectors), {}
    if n <= 2:
        # Upstream emits each tiny-set item directly, without an embedding call.
        best = np.arange(n)
    else:
        if fixed_clusters is not None and not 2 <= fixed_clusters < n:
            raise ValueError("fixed_clusters must satisfy 2 <= k < number of principles")
        upper = min(max_clusters, n - 1) if max_clusters is not None else n - 1
        if fixed_clusters is not None:
            candidates = [fixed_clusters]
        elif upper < min_clusters:
            raise ValueError("No valid k: min_clusters exceeds number of principles minus one")
        else:
            # Include the configured minimum, then larger step-aligned values.
            start = (min_clusters // cluster_step + 1) * cluster_step
            candidates = [min_clusters, *range(start, upper + 1, cluster_step)]
        best, best_score = None, -np.inf
        for k in candidates:
            labels = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            ).fit_predict(distance)
            score = float(silhouette_score(distance, labels, metric="precomputed"))
            if not np.isfinite(score):
                raise ValueError(f"Non-finite silhouette score for k={k}")
            scores[k] = score
            if score > best_score:
                best, best_score = labels, score
    mapping = {}
    labels = [mapping.setdefault(int(v), len(mapping)) for v in best]
    medoids = []
    for cluster in range(len(mapping)):
        indices = [i for i, v in enumerate(labels) if v == cluster]
        sums = distance[np.ix_(indices, indices)].sum(axis=1)
        medoids.append(indices[int(np.argmin(sums))])
    return labels, {
        "algorithm": "agglomerative",
        "metric": "cosine",
        "linkage": "average",
        "silhouette_scores": {str(k): v for k, v in scores.items()},
        "selected_k": len(mapping),
        "selected_score": scores.get(len(mapping)),
        "principle_count": n,
        "min_clusters": min_clusters,
        "max_clusters": max_clusters,
        "cluster_step": cluster_step,
        "selection": "fixed" if fixed_clusters is not None else "highest_silhouette",
        "tie_break": "smallest_k",
        "medoid_indices": medoids,
    }
