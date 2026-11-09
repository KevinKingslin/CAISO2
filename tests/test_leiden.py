"""Enforce the configured size cap while partitioning a dense DAG."""

from itertools import combinations
from types import SimpleNamespace

import pytest

from caiso.blocks.leiden import discover_blocks
from caiso.method.config import validate_method


def dense_workflow(weight=1.0):
    nodes = ["a", "b", "c", "d"]
    effects = [
        SimpleNamespace(edge_id=f"{a}->{b}", source=a, target=b, weight=weight, state_hash="state")
        for a, b in combinations(nodes, 2)
    ]
    state = SimpleNamespace(
        workflow=SimpleNamespace(nodes=nodes, edges=effects), fingerprint=lambda: "state",
    )
    return state, effects


@pytest.mark.parametrize("options,bound", [({}, 2), ({"max_comm_size": 1}, 1),
                                         ({"max_comm_size": 3}, 3), ({"max_comm_size": 0}, 4)])
@pytest.mark.parametrize("weight", [1.0, 1e6])
def test_real_leiden_respects_default_and_configured_block_sizes(options, bound, weight):
    state, effects = dense_workflow(weight)
    blocks = discover_blocks(state, effects, resolution_gamma=0, **options)
    assert sorted(node for members in blocks.values() for node in members) == state.workflow.nodes
    assert all(len(members) <= bound for members in blocks.values())
    assert max(map(len, blocks.values())) == bound
    if options.get("max_comm_size") == 0:
        assert len(blocks) == 1  # The strong interactions merge without a cap.


@pytest.mark.parametrize("invalid", [-1, True, False, None, 2.5, "2", float("inf")])
def test_invalid_block_size_caps_fail_before_partitioning(invalid):
    state, effects = dense_workflow()
    with pytest.raises(ValueError, match="max_comm_size"):
        discover_blocks(state, effects, max_comm_size=invalid)
    with pytest.raises(ValueError, match="max_comm_size"):
        validate_method({"leiden": {"max_comm_size": invalid}})
