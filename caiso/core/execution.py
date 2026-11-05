from copy import deepcopy
from pathlib import Path
from time import monotonic

from .component import validate_fields
from .serialization import digest, read_json, save_json
from .trajectory import ComponentExecution, Trajectory


def component_seed(seed, component_id):
    return int(digest([seed, component_id])[:8], 16) % (2**31 - 1)


class CompoundAISystem:
    def __init__(
        self,
        library,
        handlers,
        providers,
        budget,
        metric,
        cache_dir=None,
        services=None,
        max_workers=1,
    ):
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("execution.max_workers must be a positive integer")
        self.max_workers = max_workers
        self.library, self.handlers = library, handlers
        self.providers, self.budget, self.metric = providers, budget, metric
        self.services = services or {}
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def execute_component(self, component_id, inputs, config, seed):
        comp = self.library[component_id]
        self.budget.component(comp.cost_weight)
        start = monotonic()
        try:
            with self.budget.component_calls():
                result = comp.execute(deepcopy(inputs), config.model_copy(deep=True), seed, self)
        finally:
            elapsed = monotonic() - start
            with self.budget.lock:
                times = self.budget.usage.component_seconds
                times[component_id] = times.get(component_id, 0) + elapsed
        return result, elapsed

    def execute_block(self, state, members, reference, seed):
        """Replay a block against recorded boundary inputs, without task labels or rewards.

        Unchanged bridge nodes carry changes between noncontiguous members. A model
        selector's consumers are included so judges see generated text, not model names.
        Components outside this execution scope retain their recorded outputs.
        """
        state.validate_state(self.library)
        scope = set(members)
        if not scope or not scope <= set(state.workflow.nodes):
            raise ValueError("Block must contain active components")
        scope |= {
            e.target
            for e in state.workflow.edges
            if e.source in members and self.library[e.source].implementation == "selector"
        }

        def reachable(nodes, reverse=False):
            reached = set(nodes)
            while True:
                expanded = reached | {
                    (e.source if reverse else e.target)
                    for e in state.workflow.edges
                    if (e.target if reverse else e.source) in reached
                }
                if expanded == reached:
                    return reached
                reached = expanded

        scope |= reachable(scope) & reachable(scope, reverse=True)
        recorded = {e.component_id: e for e in reference.executions}
        if set(recorded) != set(state.workflow.nodes):
            raise ValueError("Block reference must cover the candidate workflow")
        outputs = {cid: deepcopy(e.component_output) for cid, e in recorded.items()}
        executions = []
        for cid in state.workflow.topological_order():
            if cid not in scope:
                continue
            values = deepcopy(recorded[cid].component_input)
            for edge in state.workflow.edges:
                if edge.target == cid:
                    for source, target in edge.bindings.items():
                        values[target] = deepcopy(outputs[edge.source][source])
            result, _ = self.execute_component(
                cid, values, state.configurations[cid], component_seed(seed, cid)
            )
            outputs[cid] = result
            executions.append(
                {
                    "component_id": cid,
                    "component_input": values,
                    "component_output": result,
                }
            )
        return {"inputs": deepcopy(reference.inputs), "executions": executions}

    def execute(self, state, example, seed, intervention=None, use_cache=False):
        state.validate_state(self.library)
        validate_fields(example.inputs, state.workflow.input_fields, "system.input")
        if intervention is not None and intervention not in {
            e.edge_id for e in state.workflow.edges
        }:
            raise ValueError("Unknown intervention edge")
        key = digest(
            {
                "state": state,
                "library": self.library,
                "example": example,
                "intervention": intervention,
                "seed": seed,
                "providers": self.providers.fingerprint(),
                "services": {
                    k: getattr(v, "fingerprint", lambda v=v: type(v).__name__)()
                    for k, v in self.services.items()
                },
            }
        )
        path = self.cache_dir / (key + ".json") if self.cache_dir else None
        self.budget.check()
        if use_cache and path and path.exists():
            with self.budget.lock:
                self.budget.usage.cache_hits += 1
            cached = Trajectory.model_validate(read_json(path))
            cached.metadata["cache_hit"] = True
            return cached
        usage = self.budget.usage
        with self.budget.lock:
            usage.system_attempts += 1
            usage.causal_intervention_runs += int(intervention is not None)
            usage.compatibility_adaptation_runs += int(self.budget.in_compatibility)
        outputs, executions = {}, []
        for cid in state.workflow.topological_order():
            comp = self.library[cid]
            values = {
                f: deepcopy(example.inputs[f]) for f in comp.input_fields if f in example.inputs
            }
            for edge in state.workflow.edges:
                if edge.target != cid:
                    continue
                for src, dst in edge.bindings.items():
                    values[dst] = deepcopy(
                        edge.baselines[dst].resolve(example.inputs)
                        if edge.edge_id == intervention
                        else outputs[edge.source][src]
                    )
            cseed = component_seed(seed, cid)
            config = state.configurations[cid]
            result, elapsed = self.execute_component(cid, values, config, cseed)
            outputs[cid] = result
            executions.append(
                ComponentExecution(
                    component_id=cid,
                    component_input=values,
                    component_output=deepcopy(result),
                    seed=cseed,
                    config_hash=digest(config),
                    seconds=elapsed,
                )
            )
        final = {f: outputs[n][key] for f, (n, key) in state.workflow.outputs.items()}
        reward = float(self.metric(final["answer"], example.target))
        trajectory = Trajectory(
            input_id=example.example_id,
            inputs=deepcopy(example.inputs),
            executions=executions,
            final_output=final,
            reward=reward,
            seed=seed,
            state_hash=state.fingerprint(),
            metadata={
                "intervention": intervention,
                "split": example.split,
                "cache_key": key,
                "cache_hit": False,
            },
        )
        with self.budget.lock:
            usage.full_system_runs += 1
        if use_cache and path:
            save_json(path, trajectory)
        return trajectory
