"""Silhouette selection artifacts; no model calls or changes to cluster selection."""

import csv
from pathlib import Path

from caiso.core.serialization import save_json


def save_silhouette_artifacts(directory, clustering):
    # A private Agg canvas avoids GUI windows and changing pyplot's global backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "silhouette_scores.json", clustering)
    scores = sorted((int(k), score) for k, score in clustering["silhouette_scores"].items())
    selected = clustering["selected_k"]
    csv_path = directory / "silhouette_scores.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["k", "silhouette_score", "selected"])
        writer.writeheader()
        writer.writerows(
            {"k": k, "silhouette_score": score, "selected": k == selected} for k, score in scores
        )
    temporary.replace(csv_path)
    figure = Figure(figsize=(8, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    if scores:
        ax.plot(*zip(*scores, strict=True), "o-", label="Mean silhouette score")
        score = dict(scores)[selected]
        label = "Selected" if clustering.get("selection") == "fixed" else "Best"
        ax.scatter(
            [selected],
            [score],
            color="tab:orange",
            marker="*",
            s=150,
            zorder=3,
            label=f"{label}: k={selected}, score={score:.4f}",
        )
        ax.axvline(selected, color="tab:orange", linestyle="--", alpha=0.6)
        ax.legend()
        if len(scores) <= 12:
            ax.set_xticks([k for k, _ in scores])
    else:
        ax.text(
            0.5,
            0.5,
            "Silhouette requires at least 3 principles.\n"
            f"Retained {selected} singleton cluster(s); no score is defined.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set(
        title="DICAI silhouette vs k", xlabel="Number of clusters (k)", ylabel="Silhouette score"
    )
    ax.grid(alpha=0.25)
    png = directory / "silhouette.png"
    temporary = png.with_suffix(".png.tmp")
    figure.savefig(temporary, format="png", dpi=150)
    temporary.replace(png)
    figure.clear()
