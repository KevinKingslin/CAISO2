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
from caiso.core.content_filter import ContentFilteredError
from caiso.core.configuration import Configuration, ConfigurationType as Kind
from caiso.core.execution import CompoundAISystem
from caiso.core.recovery import ModelResponseError
from caiso.core.serialization import digest
from caiso.core.system_state import SystemState
from caiso.core.trajectory import ComponentExecution, Example, Trajectory
from caiso.core.workflow import Baseline, Edge, Workflow
from caiso.wrappers.example import Example as BenchmarkExample

WIKIPEDIA_RETRIEVER = "examples.systems.hotpotqa.five_components.WikipediaRetriever"


class _SampleComplete(Exception):
    """Stop a training execution immediately after the sampled component."""


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
        self._component_cache = None
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
    def reuse_component_executions(self):
        """Share recorded training observations within one candidate search only.

        This is common evidence, not independent stochastic repetitions. Keys
        include the example, repeat ID, actual inputs and execution configuration.
        Never enable this scope around acceptance or benchmark evaluation.
        """
        with self._lock:
            if self._component_cache is not None:
                raise RuntimeError("Component reuse scopes cannot be nested")
            self._component_cache = {}
        try:
            yield
        finally:
            with self._lock:
                self._component_cache = None

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

    def sample_preference_candidate(self, state, example, reference, component_id, seed, temperature):
        """Sample one component against its recorded prefix, without a downstream run."""
        self._validate_preference_reference(example, reference)
        order = state.workflow.topological_order()
        prefix = reference.executions[:order.index(component_id)]
        return self._execute(
            state, example, seed, replay={e.component_id: e for e in prefix},
            stop_after=component_id, sample_temperature=temperature,
        )

    def continue_preference_candidate(self, state, example, reference, candidate, component_id, seed):
        """Freeze the candidate and all non-descendants; execute fresh descendants."""
        self._validate_preference_reference(example, reference)
        order = state.workflow.topological_order()
        if [e.component_id for e in candidate] != order[:order.index(component_id) + 1]:
            raise ValueError("Preference candidate must contain the complete prefix and sampled component")
        descendants = {component_id}
        for cid in state.workflow.topological_order():
            if any(e.source in descendants and e.target == cid for e in state.workflow.edges):
                descendants.add(cid)
        replay = {e.component_id: e for e in reference.executions if e.component_id not in descendants}
        replay.update({e.component_id: e for e in candidate})
        return self._execute(state, example, seed, replay=replay)

    @staticmethod
    def _validate_preference_reference(example, reference):
        if reference.input_id != example.example_id or digest(reference.inputs) != digest(example.inputs):
            raise ValueError("Preference reference must belong to the same training example")

    def execute(self, state, example, seed, intervention=None, use_cache=False):
        return self._execute(state, example, seed, intervention=intervention)

    def _execute(self, state, example, seed, intervention=None, replay=None, stop_after=None, sample_temperature=None):
        self._validate_workflow(state)
        if replay is not None and example.split != "train":
            raise ValueError("Preference sampling and replay are restricted to training examples")
        if replay is not None and self._component_cache is not None:
            raise ValueError("Preference continuations cannot reuse the training-search cache")
        if self._component_cache is not None and example.split != "train":
            raise ValueError("Component observation reuse is restricted to training searches")
        edge = None
        if intervention is not None:
            edge = next((e for e in state.workflow.edges if e.edge_id == intervention), None)
            if edge is None:
                raise ValueError(f"Unknown intervention edge: {intervention}")
        affected = {edge.target} if edge is not None else set()
        for cid in state.workflow.topological_order():
            if any(e.source in affected and e.target == cid for e in state.workflow.edges):
                affected.add(cid)
        # No task-output cache and no added generation seeds: preserve upstream calls.
        with self._lock, self._configuration(state.configurations):
            self.budget.check()
            with self.budget.lock:
                self.budget.usage.system_attempts += 1
                self.budget.usage.causal_intervention_runs += int(edge is not None)
            executions = []
            blank_inputs, empty_query_retrievals, reused_components = [], [], []
            original_call = self.engine._single_call

            def observed_call(component, inputs):
                cid = self._component_ids[id(component)]
                values = deepcopy(inputs)
                if replay is not None and cid in replay:
                    record = replay[cid]
                    if digest(values) != digest(record.component_input) or record.config_hash != digest(state.configurations[cid]):
                        raise ValueError(f"Preference replay input/configuration mismatch for {cid}")
                    outputs = deepcopy(record.component_output)
                    component.traj = {"input": values, "output": deepcopy(outputs), "variable": deepcopy(component.variable)}
                    executions.append(record.model_copy(deep=True, update={"seconds": 0}))
                    with self.budget.lock:
                        self.budget.usage.reused_component_executions += 1
                    return outputs
                if edge is not None and cid == edge.target:
                    for field, baseline in edge.baselines.items():
                        values[field] = (
                            zero_like(values[field])
                            if baseline.policy == "zero_like"
                            else deepcopy(baseline.resolve(example.inputs))
                        )
                if cid in affected:
                    # An ablated input can make later generators emit "".
                    # Preserve blank evidence using the same nonempty text
                    # representation as zero_like, rather than retrying a
                    # consumer forever with an unchanged empty input.
                    for field, value in values.items():
                        if isinstance(value, str) and value == "":
                            values[field] = " "
                            blank_inputs.append(f"{cid}.{field}")
                # Interventions always execute fresh. The training-only cache
                # shares a component draw only when every execution input agrees.
                cache = self._component_cache if edge is None and not getattr(component.config, "randomize_variable", False) else None
                key = digest([
                    example.example_id, example.inputs, seed, cid, values,
                    state.configurations[cid].kind, state.configurations[cid].value,
                    repr(vars(component.config)),
                ]) if cache is not None else None
                if cache is not None and key in cache:
                    cached = cache[key]
                    if isinstance(cached, ContentFilteredError):
                        raise cached
                    outputs = deepcopy(cached["outputs"])
                    component.traj = deepcopy(cached["trajectory"])
                    executions.append(ComponentExecution(
                        component_id=cid, component_input=values, component_output=deepcopy(outputs),
                        seed=seed, config_hash=digest(state.configurations[cid]), seconds=0,
                    ))
                    reused_components.append(cid)
                    with self.budget.lock:
                        self.budget.usage.cache_hits += 1
                        self.budget.usage.reused_component_executions += 1
                    return outputs
                self.budget.component(self.library[cid].cost_weight)
                started = monotonic()
                # A normal keyword-generation failure belongs to its producer.
                # Retry fresh generation before a consumer ever sees it. Blank
                # counterfactual outputs and explicitly masked deliveries are
                # allowed and handled as absent evidence below.
                required_queries = [
                    src
                    for binding in state.workflow.edges
                    if binding.source == cid and cid not in affected
                    and binding.edge_id != intervention
                    and self.library[binding.target].settings["upstream_class"] == WIKIPEDIA_RETRIEVER
                    for src, dst in binding.bindings.items() if dst == "search_keywords"
                ]

                def generate(attempt, previous):
                    if attempt:
                        self.budget.component(self.library[cid].cost_weight)
                    with self.budget.component_calls():
                        if (
                            cid in affected
                            and self.library[cid].settings["upstream_class"] == WIKIPEDIA_RETRIEVER
                            and isinstance(values.get("search_keywords"), str)
                            and not values["search_keywords"].strip()
                        ):
                            # No query means no retrieved evidence in this
                            # counterfactual. Never send a dummy query.
                            outputs = {"retrieve_content": ""}
                            empty_query_retrievals.append(cid)
                            component.traj = {
                                "input": deepcopy(values), "output": deepcopy(outputs),
                                "variable": deepcopy(component.variable),
                            }
                        else:
                            if cid == stop_after and sample_temperature is not None and hasattr(component.config, "temperature"):
                                # Exploration affects only this producer; downstream settings are restored.
                                temperature = max(component.config.temperature or 0, sample_temperature)
                                with component.context(temperature=temperature):
                                    outputs = original_call(component, values)
                            else:
                                outputs = original_call(component, values)
                    for field in required_queries:
                        value = outputs.get(field)
                        if not isinstance(value, str) or not value.strip():
                            raise ModelResponseError(
                                f"{cid}.{field}: keyword generation returned no usable retrieval query. "
                                "Retry the producer before calling the retriever.",
                                raw_response=repr(outputs),
                                response_context={"component_id": cid, "input_id": example.example_id, "intervention": intervention},
                            )
                    return outputs

                try:
                    outputs = (
                        self.providers.recovery.call(
                            generate, stage=f"component_output:{cid}",
                            retries=self.providers.recovery.config.get("response_retries", 2),
                        )
                        if required_queries else generate(0, None)
                    )
                    if cache is not None:
                        cache[key] = {"outputs": deepcopy(outputs), "trajectory": deepcopy(component.traj)}
                except ContentFilteredError as error:
                    if cache is not None:
                        cache[key] = error  # A policy rejection must not trigger resampling.
                    raise
                finally:
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
                if cid == stop_after:
                    raise _SampleComplete()
                return outputs

            self.engine._single_call = observed_call
            try:
                prediction = self.engine(**deepcopy(example.inputs))
                reference = BenchmarkExample(**example.inputs, **example.target)
                reward = float(self.engine.evaluate(reference, prediction=prediction))
            except _SampleComplete:
                return executions
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
                    **({"preference_replayed_components": list(replay)} if replay is not None else {}),
                    **({
                        "reused_components": reused_components,
                        "observation_reuse": "shared_training_search",
                    } if self._component_cache is not None else {}),
                    **({
                        "blank_intervention_inputs": blank_inputs,
                        "empty_query_retrievals": empty_query_retrievals,
                    } if edge is not None else {}),
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
