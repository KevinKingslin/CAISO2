"""Bridge CAISO records to the original benchmark components and scorer.

Full-system execution calls the upstream CompoundAISystem, not a reconstructed
pipeline. Calls are serialized because upstream variables and adapter registrations
are shared mutable state. DICAI's independent LLM requests may still run in parallel.
"""

from contextlib import contextmanager
from copy import deepcopy
from itertools import product
import math
from pathlib import Path
from threading import RLock
from time import monotonic

from caiso.core.component import Component
from caiso.core.configuration import Configuration, ConfigurationType as Kind
from caiso.core.execution import CompoundAISystem
from caiso.core.serialization import digest
from caiso.core.system_state import SystemState
from caiso.core.trajectory import ComponentExecution, Example, Trajectory
from caiso.core.workflow import Baseline, Edge, Workflow
from caiso.wrappers.example import Example as BenchmarkExample


def adapt_examples(system, examples, split):
    """Keep all inputs and all ground-truth fields; never re-split or re-score data."""
    return [
        Example(
            example_id=f"{split}-{i}",
            inputs={key: deepcopy(getattr(ex, key)) for key in system.required_input_fields},
            target={key: deepcopy(getattr(ex, key)) for key in system.ground_fields},
            split=split,
        )
        for i, ex in enumerate(examples)
    ]


def zero_like(value):
    """Neutralize an observed edge while retaining its container/array shape."""
    if isinstance(value, str):
        return " "  # Blank evidence still satisfies upstream nonempty-input checks.
    if isinstance(value, bool):
        return False
    if isinstance(value, dict):
        return {key: zero_like(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(zero_like(item) for item in value)
    if value is None:
        return None
    return value * 0


def describe_system(system, edge_baselines=None):
    """Infer the same last-producer field bindings used by run_subsystem."""
    library, configurations, edges = {}, {}, []
    producers = {}
    for cid in system.execution_order:
        component = system.components[cid]
        value = component.variable
        search = component.variable_search_space
        implementation = "upstream"
        if isinstance(value, Path):
            kind = Kind.MODEL_WEIGHTS
            value = {"model": cid, "adapter": str(value)}
            options = []
        elif isinstance(value, str):
            kind, options = Kind.PROMPT, []
        elif isinstance(value, dict) and search:
            kind = Kind.MODEL_SELECTION if "model_selection" in value else Kind.HYPERPARAMETER
            implementation = "selector" if kind == Kind.MODEL_SELECTION else "upstream"
            keys = list(search)
            options = [dict(zip(keys, values)) for values in product(*(search[k] for k in keys))]
            if value not in options:
                raise ValueError(f"{cid}: upstream initial variable is outside its search space")
        elif value is None:
            kind, options = Kind.NONE, []
        else:
            raise ValueError(f"{cid}: unsupported upstream variable {type(value).__name__}")
        config = Configuration(kind=kind, value=deepcopy(value))
        configurations[cid] = config
        library[cid] = Component(
            component_id=cid,
            implementation=implementation,
            input_fields=dict.fromkeys(component.input_fields, "any"),
            output_fields=dict.fromkeys(component.output_fields, "any"),
            config=config,
            config_search_space=options,
            settings={
                "description": component.description,
                "upstream_class": f"{type(component).__module__}.{type(component).__qualname__}",
                "model": getattr(component.config, "model", None),
                "temperature": getattr(component.config, "temperature", None),
                "prompt_format": "completion" if kind == Kind.MODEL_WEIGHTS else "upstream",
            },
        )
        bindings = {}
        for field in component.input_fields:
            if field in producers:
                bindings.setdefault(producers[field], {})[field] = field
        for source, fields in bindings.items():
            baselines = {}
            for field in fields:
                override = (edge_baselines or {}).get(f"{source}->{cid}", {}).get(field)
                if override is not None:
                    baseline = Baseline.model_validate(override)
                elif library[source].config_type == Kind.MODEL_SELECTION:
                    # Empty model identifiers are not a valid inference baseline.
                    baseline = Baseline(value=system.components[source].variable["model_selection"])
                else:
                    baseline = Baseline(policy="zero_like", value=None)
                baselines[field] = baseline
            edges.append(Edge(source=source, target=cid, bindings=fields, baselines=baselines))
        producers.update(dict.fromkeys(component.output_fields, cid))
    workflow = Workflow(
        nodes=list(system.execution_order),
        order=list(system.execution_order),
        edges=edges,
        input_fields=dict.fromkeys(system.required_input_fields, "any"),
        outputs={field: (producers[field], field) for field in system.final_output_fields},
    )
    available = {edge.edge_id: set(edge.baselines) for edge in edges}
    for edge_id, fields in (edge_baselines or {}).items():
        if edge_id not in available or not set(fields) <= available[edge_id]:
            raise ValueError(f"Unknown edge or input in edge_baselines: {edge_id}")
    state = SystemState(system="upstream", workflow=workflow, configurations=configurations)
    state.validate_state(library)
    return state, library


class UpstreamRuntime(CompoundAISystem):
    def __init__(self, system, providers, budget, edge_baselines=None):
        if system.rm is not None:
            raise ValueError(
                "CAISO requires the unfiltered upstream system, without an OPTIMAS reward model"
            )
        self.engine = system
        self.initial_state, library = describe_system(system, edge_baselines)
        self._lock = RLock()
        self._component_ids = {id(comp): cid for cid, comp in system.components.items()}
        super().__init__(
            library,
            {"upstream": self._handle, "selector": self._handle},
            providers,
            budget,
            metric=None,
            max_workers=1,
        )

    def _variable(self, cid, config):
        if config.kind == Kind.MODEL_WEIGHTS:
            return Path(config.value["adapter"])
        return deepcopy(config.value)

    @contextmanager
    def _configuration(self, configs):
        """Apply and restore variables, including the real vLLM LoRA callbacks."""
        previous = {}
        try:
            for cid, config in configs.items():
                comp = self.engine.components[cid]
                value = self._variable(cid, config)
                if value != comp.variable:
                    previous[cid] = deepcopy(comp.variable)
                    comp.update(value)
            yield
        finally:
            for cid, value in reversed(list(previous.items())):
                self.engine.components[cid].update(value)

    def _handle(self, component, inputs, config, seed, runtime):
        with self._lock, self._configuration({component.component_id: config}):
            return deepcopy(self.engine.components[component.component_id](**deepcopy(inputs)))

    def execute_component(self, *args, **kwargs):
        with self._lock:
            return super().execute_component(*args, **kwargs)

    def execute_block(self, *args, **kwargs):
        with self._lock:
            return super().execute_block(*args, **kwargs)

    def apply_state(self, state):
        """Install the chosen configuration for the original evaluation script."""
        self._validate_workflow(state)
        with self._lock:
            for cid, config in state.configurations.items():
                component = self.engine.components[cid]
                value = self._variable(cid, config)
                if value != component.variable:
                    component.update(value)

    def _validate_workflow(self, state):
        state.validate_state(self.library)
        if state.workflow != self.initial_state.workflow:
            raise ValueError(
                "Benchmark parity requires the original workflow; workflow search is disabled"
            )

    def execute(self, state, example, seed, intervention=None, use_cache=False):
        self._validate_workflow(state)
        edge = None
        if intervention is not None:
            edge = next((e for e in state.workflow.edges if e.edge_id == intervention), None)
            if edge is None:
                raise ValueError(f"Unknown intervention edge: {intervention}")
        # No task-output cache and no added generation seeds: preserve upstream calls.
        with self._lock, self._configuration(state.configurations):
            self.budget.check()
            with self.budget.lock:
                self.budget.usage.system_attempts += 1
                self.budget.usage.causal_intervention_runs += int(edge is not None)
            executions = []
            original_call = self.engine._single_call

            def observed_call(component, inputs):
                cid = self._component_ids[id(component)]
                values = deepcopy(inputs)
                if edge is not None and cid == edge.target:
                    for field, baseline in edge.baselines.items():
                        values[field] = (
                            zero_like(values[field])
                            if baseline.policy == "zero_like"
                            else deepcopy(baseline.resolve(example.inputs))
                        )
                self.budget.component(self.library[cid].cost_weight)
                started = monotonic()
                with self.budget.component_calls():
                    outputs = original_call(component, values)
                elapsed = monotonic() - started
                with self.budget.lock:
                    times = self.budget.usage.component_seconds
                    times[cid] = times.get(cid, 0) + elapsed
                executions.append(
                    ComponentExecution(
                        component_id=cid,
                        component_input=values,
                        component_output=deepcopy(outputs),
                        seed=seed,
                        config_hash=digest(state.configurations[cid]),
                        seconds=elapsed,
                    )
                )
                return outputs

            self.engine._single_call = observed_call
            try:
                prediction = self.engine(**deepcopy(example.inputs))
                reference = BenchmarkExample(**example.inputs, **example.target)
                reward = float(self.engine.evaluate(reference, prediction=prediction))
            finally:
                self.engine._single_call = original_call
            if not math.isfinite(reward):
                raise ValueError("Upstream task scorer returned a non-finite score")
            with self.budget.lock:
                self.budget.usage.full_system_runs += 1
            return Trajectory(
                input_id=example.example_id,
                inputs=deepcopy(example.inputs),
                executions=executions,
                final_output={
                    key: deepcopy(getattr(prediction, key))
                    for key in self.engine.final_output_fields
                },
                reward=reward,
                seed=seed,
                state_hash=state.fingerprint(),
                metadata={
                    "split": example.split,
                    "intervention": intervention,
                    "cache_hit": False,
                    "task_generation_seeded": False,
                },
            )

    def training_prompt(self, cid, inputs, config):
        """Capture the original completion prompt without making an inference call.

        Amazon constructs this string inside forward(). Execute that code with its
        HTTP function replaced in an isolated globals dictionary; no prompt copy.
        """
        component = self.engine.components[cid]
        forward = component.forward.__func__
        if "post_http_request" not in forward.__globals__:
            raise ValueError(f"{cid}: no upstream raw-completion prompt capture available")
        import types

        captured = []

        def capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return None

        namespace = dict(forward.__globals__)
        namespace.update(post_http_request=capture, get_response=lambda response: "")
        isolated = types.FunctionType(
            forward.__code__, namespace, forward.__name__, forward.__defaults__, forward.__closure__
        )
        with self._lock, self._configuration({cid: config}):
            isolated(component, **deepcopy(inputs))
        if len(captured) != 1:
            raise ValueError(f"{cid}: expected exactly one upstream completion prompt")
        return captured[0]
