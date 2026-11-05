from .operations import (
    AddBranch,
    AddComponent,
    AddEdge,
    RemoveBranch,
    RemoveComponent,
    RemoveEdge,
    Reorder,
    ReplaceEdge,
)


def apply_operation(state, operation, library, allowed_connections=None):
    candidate = state.model_copy(deep=True)
    g = candidate.workflow
    old_hash = candidate.fingerprint()

    def add_node(cid):
        if cid not in library or cid in g.nodes:
            raise ValueError("Added component must be an inactive member of the library")
        g.nodes.append(cid)
        candidate.configurations[cid] = library[cid].config.model_copy(deep=True)

    def remove_node(cid):
        if cid not in g.nodes:
            raise ValueError("Cannot remove an inactive component")
        g.nodes.remove(cid)
        del candidate.configurations[cid]
        g.edges = [e for e in g.edges if cid not in (e.source, e.target)]

    def remove_edge(eid):
        matches = [e for e in g.edges if e.edge_id == eid]
        if len(matches) != 1:
            raise ValueError("Cannot remove unknown edge")
        g.edges = [e for e in g.edges if e.edge_id != eid]

    if isinstance(operation, AddComponent):
        add_node(operation.component_id)
        predecessors = {e.source for e in operation.edges if e.target == operation.component_id}
        successors = {e.target for e in operation.edges if e.source == operation.component_id}
        for eid in operation.remove_edges:
            old = next((e for e in g.edges if e.edge_id == eid), None)
            if old is None or old.source not in predecessors or old.target not in successors:
                raise ValueError("Insertion may only replace an edge across its new component")
            remove_edge(eid)
        if any(operation.component_id not in (e.source, e.target) for e in operation.edges):
            raise ValueError("Add-component wiring must touch that component")
        g.edges.extend(operation.edges)
    elif isinstance(operation, RemoveComponent):
        neighbors = {e.source for e in g.edges if e.target == operation.component_id} | {
            e.target for e in g.edges if e.source == operation.component_id
        }
        if any(not {e.source, e.target} <= neighbors for e in operation.reconnect):
            raise ValueError("Reconnect only the removed component neighbors")
        remove_node(operation.component_id)
        g.edges.extend(operation.reconnect)
    elif isinstance(operation, AddEdge):
        g.edges.append(operation.edge)
    elif isinstance(operation, RemoveEdge):
        remove_edge(operation.edge_id)
    elif isinstance(operation, ReplaceEdge):
        remove_edge(operation.edge_id)
        g.edges.append(operation.edge)
    elif isinstance(operation, Reorder):
        previous_order = state.workflow.topological_order()
        if operation.order == previous_order:
            raise ValueError("Reorder must change execution order")
        if set(operation.order) != set(previous_order) or len(operation.order) != len(
            previous_order
        ):
            raise ValueError("Reorder must retain every active component exactly once")
        new_position = {node: i for i, node in enumerate(operation.order)}
        reversed_pairs = [
            (a, b)
            for i, a in enumerate(previous_order)
            for b in previous_order[i + 1 :]
            if new_position[a] > new_position[b]
        ]
        if len(reversed_pairs) != 1:
            raise ValueError("One reorder proposal may reverse only one component pair")
        moved = set(reversed_pairs[0])
        if not operation.edges or {e.edge_id: e for e in operation.edges} == {
            e.edge_id: e for e in g.edges
        }:
            raise ValueError("Reorder must specify changed dataflow edges, not just scheduling")
        original_edges = {e.edge_id: e for e in g.edges}
        updated_edges = {e.edge_id: e for e in operation.edges}
        changed = {
            eid
            for eid in original_edges.keys() | updated_edges.keys()
            if original_edges.get(eid) != updated_edges.get(eid)
        }
        if any(
            not ({e.source, e.target} & moved)
            for eid in changed
            for e in (original_edges.get(eid), updated_edges.get(eid))
            if e
        ):
            raise ValueError("Reorder may rewire only the exchanged components")
        if not any(
            original_edges.get(eid) is None
            or updated_edges.get(eid) is None
            or original_edges[eid].bindings != updated_edges[eid].bindings
            for eid in changed
        ):
            raise ValueError("Reorder must change dependencies, not only edge baselines")
        g.edges = operation.edges
        g.order = operation.order
    elif isinstance(operation, AddBranch):
        if not operation.component_ids or len(set(operation.component_ids)) != len(
            operation.component_ids
        ):
            raise ValueError("A branch must add distinct library components")
        validate_branch(operation.component_ids, operation.edges)
        members = set(operation.component_ids)
        predecessors = {e.source for e in operation.edges if e.source not in members}
        successors = {e.target for e in operation.edges if e.target not in members}
        for eid in operation.remove_edges:
            old = next((e for e in g.edges if e.edge_id == eid), None)
            if old is None or old.source not in predecessors or old.target not in successors:
                raise ValueError("Branch may only replace edges across its boundary")
            remove_edge(eid)
        for cid in operation.component_ids:
            add_node(cid)
        g.edges.extend(operation.edges)
    elif isinstance(operation, RemoveBranch):
        if not operation.component_ids:
            raise ValueError("Empty branch")
        touched = [
            e
            for e in g.edges
            if e.source in operation.component_ids or e.target in operation.component_ids
        ]
        if set(operation.edge_ids) != {e.edge_id for e in touched}:
            raise ValueError("Remove branch must name exactly its incident edges")
        validate_branch(operation.component_ids, touched)
        neighbors = {e.source for e in touched} | {e.target for e in touched}
        neighbors -= set(operation.component_ids)
        if any(not {e.source, e.target} <= neighbors for e in operation.reconnect):
            raise ValueError("Reconnect only the removed branch neighbors")
        for cid in operation.component_ids:
            remove_node(cid)
        g.edges.extend(operation.reconnect)
    else:
        raise ValueError("Unknown operation")
    if not isinstance(operation, Reorder):
        g.order = []
    if allowed_connections is not None:
        allowed = {tuple(pair) for pair in allowed_connections}
        if any((e.source, e.target) not in allowed for e in g.edges):
            raise ValueError("Connection not allowed by system configuration")
    candidate.validate_state(library)
    if candidate.fingerprint() == old_hash:
        raise ValueError("Structural proposal is a no-op")
    return candidate


def validate_branch(nodes, edges):
    # One connected subgraph may fan out or merge. Global validation checks its DAG,
    # all required typed inputs, and contribution to a final output.
    members = set(nodes)
    outgoing = [e for e in edges if e.source in members and e.target not in members]
    internal = [e for e in edges if e.source in members and e.target in members]
    if (
        not members
        or not outgoing
        or any(e.source not in members and e.target not in members for e in edges)
    ):
        raise ValueError("Branch wiring must touch its members and reach an existing output path")
    reached = {next(iter(members))}
    while True:
        expanded = (
            reached
            | {e.target for e in internal if e.source in reached}
            | {e.source for e in internal if e.target in reached}
        )
        if expanded == reached:
            break
        reached = expanded
    if reached != members:
        raise ValueError("Disconnected branch")
