"""Detailed phase artifacts, retaining filenames from before commit 500d21c.

The mapping is deliberately explicit: adding a saved result means choosing its
phase and filename here. This module does no model calls or optimization.
"""

import csv
import json
from pathlib import Path

from caiso.core.serialization import plain, read_json, save_json, save_jsonl

PHASE_FILES = {
    "00_system_constitution": {
        "preferences": "preferences.json",
        "constitution": "system_constitution.json",
    },
    "01_attribution": {
        "state_before": "state_before.json",
        "edge_effects": "edge_weights.json",
        "settings": "settings.json",
        "seeds": "seeds.json",
    },
    "02_blocks": {
        "blocks": "blocks.json",
        "settings": "settings.json",
        "seed": "seed.json",
        "weighted_graph": "weighted_graph.json",
    },
    "03_selection": {
        "diagnostic_trajectories": "diagnostic_trajectories.jsonl",
        "system_constitution": "system_constitution.json",
        "blocks": "blocks.json",
        "selection": "block_selection.json",
    },
    "04_block_constitution": {
        "members": "members.json",
        "preliminary_candidates": "preliminary_candidates.json",
        "preferences": "block_preferences.json",
        "preference_metrics": "block_preference_metrics.json",
        "constitution": "block_constitution.json",
        "induction_metrics": "block_induction_metrics.json",
    },
    "05_block_optimization": {
        "candidates": "candidates.json",
        "candidate_state": "block_candidate_state.json",
        "acceptance": "block_acceptance.json",
    },
    "06_workflow": {
        "trigger": "trigger.json",
        "triggered": "triggered.json",
        "report": "workflow_report.json",
        "refreshed_edge_effects": "attribution/edge_weights.json",
        "refreshed_blocks": "partition/blocks.json",
        "refreshed_graph": "partition/weighted_graph.json",
        "partition_settings": "partition/settings.json",
    },
    "07_state": {
        "accepted_state": "accepted_state.json",
        "system_constitution": "system_constitution.json",
        "method_config": "method_config.json",
    },
}

STATUS_FIELDS = {"phase", "status", "progress", "induction_skipped", "reason", "induction_activity"}


def edge_weight_csv(path, effects):
    """The same columns as the historical attribution export."""
    path = Path(path)
    fields = ["edge_id", "source", "target", "weight", "standard_deviation", "standard_error"]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in plain(effects))
    temporary.replace(path)


def graphml(path, data, settings):
    """Export the signed directed graph, including the configured weight transform."""
    import igraph as ig

    from caiso.blocks.leiden import partition_weight

    nodes = sorted(data["nodes"])
    edges = data["edges"]
    graph = ig.Graph(n=len(nodes), directed=True)
    graph.vs["name"] = nodes
    graph.add_edges([(nodes.index(e["source"]), nodes.index(e["target"])) for e in edges])
    transform = settings.get("weight_transform", "raw_if_supported")
    graph.es["raw_weight"] = [e["weight"] for e in edges]
    graph.es["weight"] = [partition_weight(e["weight"], transform) for e in edges]
    graph.es["directed_edge_id"] = [e["edge_id"] for e in edges]
    graph["projection"] = "directed execution DAG"
    graph["weight_transform"] = transform
    temporary = Path(path).with_suffix(".graphml.tmp")
    graph.write_graphml(str(temporary))
    temporary.replace(path)


def save_validation(directory, label, result):
    """Keep metrics separate from the historical paired trajectory JSONL."""
    path = directory / "validation_metrics.json"
    metrics = read_json(path) if path.exists() else {}
    trajectories = result.get("trajectories", [])
    metrics[label] = {
        **{key: value for key, value in result.items() if key != "trajectories"},
        "trajectory_count": len(trajectories),
    }
    trace_path = directory / "validation_trajectories.jsonl"
    previous = (
        [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        if label == "candidate" and trace_path.exists()
        else []
    )
    save_jsonl(trace_path, [*previous, *trajectories])
    save_json(path, metrics)


def save_phase_artifacts(directory, phase, fields):
    """Write separate files; status updates never rewrite the phase's payloads."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fields = plain(fields)
    filenames = PHASE_FILES[phase]
    updates = {key: value for key, value in fields.items() if key in STATUS_FIELDS}
    if updates:
        path = directory / "status.json"
        status = read_json(path) if path.exists() else {}
        save_json(path, {**status, **updates})
    # Combined files may have sorted keys. Always emit the current trajectories
    # before the candidate trajectories, matching the original evaluation order.
    ordered = [
        (key, value)
        for key, value in fields.items()
        if key not in {"current_validation", "candidate_validation"}
    ]
    ordered += [
        (key, fields[key])
        for key in ("current_validation", "candidate_validation")
        if key in fields
    ]
    for key, value in ordered:
        if key in STATUS_FIELDS:
            continue
        elif key == "errors":
            save_jsonl(directory / "errors.jsonl", value)
        elif key == "inductions" and phase == "00_system_constitution":
            snapshots = directory / "inductions"
            start = max(
                (int(path.stem) for path in snapshots.glob("*.json") if path.stem.isdigit()),
                default=0,
            )
            for index, constitution in enumerate(value, start + 1):
                save_json(snapshots / f"{index:04d}.json", constitution)
                save_json(directory / "system_constitution.json", constitution)
        elif (
            key in {"current_validation", "candidate_validation"}
            and phase == "05_block_optimization"
        ):
            save_validation(directory, key.removesuffix("_validation"), value)
        elif key in filenames:
            path = directory / filenames[key]
            if path.suffix == ".jsonl":
                save_jsonl(path, value)
            else:
                save_json(path, value)
            if key == "constitution" and phase in {
                "00_system_constitution",
                "04_block_constitution",
            }:
                clustering = value.get("metadata", {}).get("clustering")
                if clustering:
                    from caiso.constitution.dicai.plotting import save_silhouette_artifacts

                    save_silhouette_artifacts(directory, clustering)
            if key in {"edge_effects", "refreshed_edge_effects"}:
                edge_weight_csv(path.with_suffix(".csv"), value)
            elif key in {"weighted_graph", "refreshed_graph"}:
                settings_path = path.parent / "settings.json"
                settings_key = "settings" if key == "weighted_graph" else "partition_settings"
                settings = fields.get(settings_key)
                if settings is None:
                    settings = read_json(settings_path) if settings_path.exists() else {}
                graphml(path.with_suffix(".graphml"), value, settings)
        else:
            raise ValueError(f"Unknown {phase} result field: {key}")
