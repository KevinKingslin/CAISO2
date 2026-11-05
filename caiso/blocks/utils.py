def block_strength(members, effects):
    """Tie-break: sum absolute effects on edges incident to the block, counted once."""
    return sum(abs(e.weight) for e in effects if e.source in members or e.target in members)


def block_context(trajectory, members, workflow):
    """External inputs and incoming edge values only; excludes internal block outputs."""
    result = {}
    for execution in trajectory.executions:
        if execution.component_id not in members:
            continue
        internal = {
            dst
            for e in workflow.edges
            if e.target == execution.component_id and e.source in members
            for dst in e.bindings.values()
        }
        result[execution.component_id] = {
            k: v for k, v in execution.component_input.items() if k not in internal
        }
    return result
