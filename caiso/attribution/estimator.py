import csv
from pathlib import Path

import numpy as np

from caiso.core.content_filter import ContentFilteredError, execute_example, record_skip
from caiso.core.parallel import ordered_map
from caiso.core.recovery import ModelResponseError
from caiso.core.serialization import Record, digest, save_json

from .interventions import execute_intervention


class EdgeEffect(Record):
    edge_id: str
    state_hash: str
    source: str
    target: str
    weight: float
    standard_deviation: float
    standard_error: float
    confidence_interval: tuple[float, float] | None = None
    deltas: list[list[float]]
    seeds: list[int]
    input_ids: list[str]


def estimate_edge_effect(
    runtime,
    state,
    edge,
    examples,
    seeds,
    normal=None,
    bootstrap_samples=0,
    bootstrap_seed=0,
    confidence_level=0.95,
):
    if not examples or not seeds:
        raise ValueError("Attribution requires nonempty inputs and repeats")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    normal = normal if normal is not None else {}
    workers = getattr(runtime, "max_workers", 1)
    # Populate shared baselines before intervention workers read them. Deduplicate
    # complete state/input/seed keys so parallelism cannot duplicate a baseline run.
    jobs = [
        (example, seed, digest([state, example, seed, runtime.providers.fingerprint()]))
        for example in examples
        for seed in seeds
    ]
    missing = {key: (example, seed) for example, seed, key in jobs if key not in normal}
    baselines = ordered_map(
        lambda job: execute_example(runtime, state, job[0], job[1]),
        missing.values(),
        max_workers=workers,
    )
    normal.update(zip(missing, baselines, strict=True))

    def difference(job):
        baseline = normal[job[2]]
        if baseline is None:
            return None
        try:
            return (
                baseline.reward
                - execute_intervention(runtime, state, job[0], edge.edge_id, job[1]).reward
            )
        except ContentFilteredError as error:
            record_skip(
                runtime.providers,
                error,
                "causal_attribution",
                example_id=job[0].example_id,
                seed=job[1],
                edge_id=edge.edge_id,
            )
            return None

    flat_deltas = ordered_map(
        difference,
        jobs,
        max_workers=workers,
    )
    rows = [flat_deltas[i : i + len(seeds)] for i in range(0, len(flat_deltas), len(seeds))]
    retained = [
        (example, row)
        for example, row in zip(examples, rows, strict=True)
        if all(value is not None for value in row)
    ]
    if not retained:
        raise ModelResponseError("No unfiltered paired examples remain for causal attribution")
    examples, deltas = zip(*retained, strict=True)
    deltas = list(deltas)
    values = np.asarray(deltas)
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    ci = None
    if bootstrap_samples:
        # Cluster bootstrap over inputs; retain repeats together to avoid pseudo-replication.
        rng = np.random.default_rng(bootstrap_seed)
        means = [
            values[rng.integers(0, len(values), len(values))].mean()
            for _ in range(bootstrap_samples)
        ]
        tail = (1 - confidence_level) / 2
        ci = tuple(float(v) for v in np.quantile(means, [tail, 1 - tail]))
    return EdgeEffect(
        edge_id=edge.edge_id,
        state_hash=state.fingerprint(),
        source=edge.source,
        target=edge.target,
        weight=float(values.mean()),
        standard_deviation=std,
        standard_error=std / np.sqrt(values.size),
        confidence_interval=ci,
        deltas=deltas,
        seeds=seeds,
        input_ids=[x.example_id for x in examples],
    )


def estimate_all_edges(
    runtime, state, examples, seeds, output=None, bootstrap_samples=0, confidence_level=0.95
):
    normal = {}
    effects = [
        estimate_edge_effect(
            runtime,
            state,
            edge,
            examples,
            seeds,
            normal,
            bootstrap_samples,
            seeds[0],
            confidence_level,
        )
        for edge in state.workflow.edges
    ]
    if output:
        path = Path(output)
        save_json(path / "edge_weights.json", effects)
        with (path / "edge_weights.csv").open("w", newline="") as stream:
            fields = [
                "edge_id",
                "source",
                "target",
                "weight",
                "standard_deviation",
                "standard_error",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: e.model_dump()[k] for k in fields} for e in effects)
    return effects
