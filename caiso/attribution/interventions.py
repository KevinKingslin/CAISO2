"""Edge intervention changes delivery only, not the producer or other edges."""


def execute_intervention(runtime, state, example, edge_id, seed):
    return runtime.execute(state, example, seed, intervention=edge_id)
