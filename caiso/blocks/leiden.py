import math
from pathlib import Path

import igraph as ig
import leidenalg

from caiso.core.serialization import save_json


def partition_weight(weight, transform="raw_if_supported"):
    if not math.isfinite(weight):
        raise ValueError("Non-finite attribution")
    if transform == "abs":
        return abs(weight)
    if transform == "positive_only":
        return max(0.0, weight)
    # CPM explicitly supports signed weights; other partition types are not substituted.
    if transform == "raw_if_supported":
        return weight
    raise ValueError(f"Unknown weight transform: {transform}")


def discover_blocks(
    state,
    effects,
    resolution_gamma=0.1,
    weight_transform="raw_if_supported",
    seed=0,
    output=None,
    n_iterations=-1,
):
    if resolution_gamma < 0:
        raise ValueError("CPM resolution must be nonnegative")
    if {e.edge_id for e in effects} != {e.edge_id for e in state.workflow.edges}:
        raise ValueError("Attribution must cover the current workflow exactly")
    if any(e.state_hash != state.fingerprint() for e in effects):
        raise ValueError("Attribution belongs to a different state; recompute it")
    nodes = sorted(state.workflow.nodes)
    graph = ig.Graph(n=len(nodes), directed=True)
    graph.vs["name"] = nodes
    # Preserve the proposal's signed, directed causal graph for CPM.
    graph.add_edges([(nodes.index(e.source), nodes.index(e.target)) for e in effects])
    graph.es["raw_weight"] = [e.weight for e in effects]
    graph.es["weight"] = [partition_weight(e.weight, weight_transform) for e in effects]
    graph.es["directed_edge_id"] = [e.edge_id for e in effects]
    graph["projection"] = "directed execution DAG"
    graph["weight_transform"] = weight_transform
    partition = leidenalg.find_partition(
        graph,
        leidenalg.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution_gamma,
        seed=seed,
        n_iterations=n_iterations,
    )
    groups = sorted([sorted(nodes[i] for i in group) for group in partition])
    blocks = {f"B{i + 1}": group for i, group in enumerate(groups)}
    if sorted(n for members in blocks.values() for n in members) != nodes:
        raise RuntimeError("Leiden did not produce a hard partition")
    if output:
        path = Path(output)
        save_json(path / "blocks.json", blocks)
        graph.write_graphml(str(path / "weighted_graph.graphml"))
    return blocks
