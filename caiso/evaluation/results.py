"""Small, explicit result documents and a human-readable run index."""

import hashlib
import tempfile
from pathlib import Path

from caiso.core.serialization import read_json, save_json
from caiso.evaluation.phase_artifacts import PHASE_FILES, save_phase_artifacts

# Historical root-level files, now grouped by the phase that produced them.
LEGACY_FILES = {
    "state_before.json": "01_attribution/state_before.json",
    "diagnostic_trajectories.jsonl": "03_selection/diagnostic_trajectories.jsonl",
    "block_selection.json": "03_selection/block_selection.json",
    "preliminary_candidates.json": "04_block_constitution/preliminary_candidates.json",
    "block_preferences.json": "04_block_constitution/block_preferences.json",
    "block_preference_metrics.json": "04_block_constitution/block_preference_metrics.json",
    "block_constitution.json": "04_block_constitution/block_constitution.json",
    "block_induction_metrics.json": "04_block_constitution/block_induction_metrics.json",
    "candidates.json": "05_block_optimization/candidates.json",
    "block_candidate_state.json": "05_block_optimization/block_candidate_state.json",
    "block_acceptance.json": "05_block_optimization/block_acceptance.json",
    "validation_trajectories.jsonl": "05_block_optimization/validation_trajectories.jsonl",
    "workflow_report.json": "06_workflow/workflow_report.json",
    "accepted_state.json": "07_state/accepted_state.json",
    "system_constitution.json": "07_state/system_constitution.json",
    "method_config.json": "07_state/method_config.json",
}
LEGACY_DIRECTORIES = {
    "attribution": "01_attribution",
    "partition": "02_blocks",
    "block_constitution": "04_block_constitution/induction",
    "workflow_attribution": "06_workflow/attribution",
    "workflow_partition": "06_workflow/partition",
}


def organize_results(directory):
    """Expand combined phase documents and group the historical separate files.

    Stage and validate every destination before modifying results. Preserve the
    original combined documents under .internal/layout_backups for full provenance.
    This operation does not change checkpoints or execute any models.
    """
    directory = Path(directory)
    iterations = directory / "iterations"
    if not iterations.is_dir():
        raise ValueError(f"No iterations directory in {directory}")
    internal = directory / ".internal"
    internal.mkdir(exist_ok=True)
    combined, moves = [], []
    for iteration in sorted(p for p in iterations.iterdir() if p.is_dir()):
        for phase in PHASE_FILES:
            sources = [
                path
                for path in (iteration / f"{phase}.json", iteration / phase / "results.json")
                if path.is_file()
            ]
            if len(sources) > 1:
                raise FileExistsError(
                    f"Two combined result sources exist for {iteration.name}/{phase}"
                )
            if sources:
                combined.append((sources[0], iteration, phase))
        for old, new in LEGACY_FILES.items():
            source = iteration / old
            if source.is_file():
                moves.append((source, iteration / new))
        for old, new in LEGACY_DIRECTORIES.items():
            for source in sorted((iteration / old).rglob("*")):
                if source.is_file():
                    moves.append((source, iteration / new / source.relative_to(iteration / old)))
    with tempfile.TemporaryDirectory(prefix="layout-", dir=internal) as temporary:
        staged, backups = {}, []
        staging = Path(temporary)
        for source, iteration, phase in combined:
            destination = staging / iteration.name / phase
            save_phase_artifacts(destination, phase, read_json(source))
            for artifact in sorted(destination.rglob("*")):
                if artifact.is_file():
                    target = iteration / phase / artifact.relative_to(destination)
                    staged[target] = artifact
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            backup = internal / "layout_backups" / iteration.name / f"{phase}-{checksum}.json"
            backups.append((source, backup))
        for source, target in [*moves, *backups]:
            if target in staged and staged[target].read_bytes() != source.read_bytes():
                raise FileExistsError(f"Conflicting results for {target}")
        for source, target in (
            [(source, target) for target, source in staged.items()] + moves + backups
        ):
            if target.exists() and (
                not target.is_file() or target.read_bytes() != source.read_bytes()
            ):
                raise FileExistsError(
                    f"Destination already exists with different contents: {target}"
                )
            for parent in target.parents:
                if parent == directory:
                    break
                if parent.is_file():
                    raise FileExistsError(f"Destination parent is a file: {parent}")
        for target, source in staged.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                source.replace(target)
        for source, target in [*moves, *backups]:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                source.unlink()  # Identical content was verified before any mutations.
            else:
                source.rename(target)
        # Remove only empty historical folders, retaining any unrecognized artifacts.
        for iteration in sorted(p for p in iterations.iterdir() if p.is_dir()):
            for old in LEGACY_DIRECTORIES:
                root = iteration / old
                for path in sorted(root.rglob("*"), reverse=True):
                    if path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
                if root.is_dir() and not any(root.iterdir()):
                    root.rmdir()
    write_overview(directory)
    return len(combined) + len(moves)


def update_result(path, **fields):
    """Atomically update a phase file without discarding earlier stage results."""
    path = Path(path)
    data = read_json(path) if path.exists() else {}
    data.update(fields)
    save_json(path, data)
    return data


def run_metrics(directory):
    """Read the consolidated run result, with support for older runs."""
    directory = Path(directory)
    path = directory / "run.json"
    return read_json(path)["metrics"] if path.exists() else read_json(directory / "metrics.json")


def write_overview(directory):
    directory = Path(directory)
    lines = [
        "# Run results",
        "",
        "Start here, then open an iteration's `summary.json` or `performance.png`.",
        "Each numbered phase folder contains separate named files for its results,",
        "metrics, trajectories and diagnostics, matching the detailed historical exports.",
        "",
        "- [setup.json](setup.json): initial preferences, system constitution and validation baseline.",
        "- [run.json](run.json): final metrics, best system and constitutions (when the run stops).",
        "- [history.csv](history.csv): scores and cumulative resource usage across iterations.",
        "- [config.yaml](config.yaml): exact resolved configuration for a live experiment.",
        "- `evaluation.json`: test evaluation of the optimized system (after evaluate).",
        "- `evaluation_unoptimized.json`: test evaluation of the original system (after evaluate --unoptimized).",
        "- `checkpoint/checkpoint.json`: complete state for resuming; no manual editing needed.",
        "- `.internal/`: reusable debate caches, training assets and raw diagnostic logs.",
        "- `.internal/dicai/system/silhouette.png`: latest system-induction cluster search.",
        "",
    ]
    if (directory / "run.json").exists():
        metrics = run_metrics(directory)
        lines += [
            f"Final validation score: **{metrics.get('validation_score')}**; "
            f"stop reason: **{metrics.get('stop_reason')}**.",
            "",
        ]
    lines += [
        "| Iteration | Status | Validation | Best | Block update | Selected block | Results | Plot |",
        "|---|---|---|---|---|---|---|---|",
    ]
    latest = None
    for path in sorted(directory.glob("iterations/*/summary.json")):
        row = read_json(path)
        latest = row
        relative = path.relative_to(directory).as_posix()
        plot = path.with_name("performance.png")
        plot_link = f"[plot]({plot.relative_to(directory).as_posix()})" if plot.exists() else "—"
        lines.append(
            f"| {path.parent.name} | {row.get('status', 'running')} | {row.get('score')} | "
            f"{row.get('best_score')} | {'accepted' if row.get('accepted') else 'unchanged'} | "
            f"{row.get('selected_block') or '—'} | "
            f"[summary]({relative}) | {plot_link} |"
        )
    if latest and latest.get("terminal_summary"):
        lines += ["", "Latest iteration:", "", "```text", latest["terminal_summary"], "```"]
    lines += [
        "",
        "Phase folders inside each iteration:",
        "",
        "- `00_system_constitution/`: preferences, system_constitution.json and induction snapshots.",
        "- `01_attribution/`: state_before.json, edge_weights.json, edge_weights.csv and settings.",
        "- `02_blocks/`: blocks.json, weighted_graph.json and weighted_graph.graphml.",
        "- `03_selection/`: diagnostic_trajectories.jsonl and block_selection.json.",
        "  Each saved vote includes its justification and complete judged trajectory; blocks map IDs to components.",
        "- `04_block_constitution/`: preliminary_candidates.json, block_preferences.json,",
        "  block_preference_metrics.json, block_constitution.json and block_induction_metrics.json;",
        "  silhouette.png and silhouette_scores.json/.csv record cluster-count selection.",
        "- `05_block_optimization/`: candidates.json, block_candidate_state.json,",
        "  validation_metrics.json, validation_trajectories.jsonl and block_acceptance.json.",
        "- `06_workflow/`: trigger, workflow_report.json and refreshed attribution/partition files.",
        "- `07_state/`: method_config.json, accepted_state.json and system_constitution.json.",
        "",
        "Each active phase has status.json; errors.jsonl and model.log appear when emitted.",
        "summary.txt and performance.json are also available beside the iteration summary and plot.",
        "",
        "Keep `.internal/` and `checkpoint/` with the run to preserve resume and model assets.",
    ]
    (directory / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
