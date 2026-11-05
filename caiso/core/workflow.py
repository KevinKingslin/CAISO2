from pydantic import Field

from .serialization import Record


class Baseline(Record):
    policy: str = "constant"
    value: object = ""
    input_field: str | None = None

    def resolve(self, inputs):
        if self.policy == "passthrough":
            if not self.input_field:
                raise ValueError("Pass-through baseline requires an input field")
            return inputs[self.input_field]
        if self.policy != "constant":
            raise ValueError(f"Unknown baseline: {self.policy}")
        return self.value


class Edge(Record):
    source: str
    target: str
    bindings: dict[str, str]  # source output field -> destination input field
    baselines: dict[str, Baseline]  # destination input field -> baseline

    @property
    def edge_id(self):
        return f"{self.source}->{self.target}"


class Workflow(Record):
    nodes: list[str]
    edges: list[Edge]
    input_fields: dict[str, str]
    outputs: dict[str, tuple[str, str]]
    order: list[str] = Field(default_factory=list)

    def topological_order(self):
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Duplicate node IDs")
        pending = {n: set() for n in self.nodes}
        for e in self.edges:
            if e.source not in pending or e.target not in pending:
                raise ValueError("Dangling edge")
            pending[e.target].add(e.source)
        priority = self.order or self.nodes
        if set(priority) != set(self.nodes) or len(priority) != len(self.nodes):
            raise ValueError("Execution order must contain every node exactly once")
        out = []
        while pending:
            available = [n for n in priority if n in pending and not pending[n]]
            if not available:
                raise ValueError("Workflow contains a cycle")
            n = available[0]
            out.append(n)
            del pending[n]
            for parents in pending.values():
                parents.discard(n)
        if self.order and out != self.order:
            raise ValueError("Requested order violates dependencies")
        return out

    def validate_graph(self, library):
        order = self.topological_order()
        if not order or not self.outputs:
            raise ValueError("Workflow needs nodes and a final output")
        if len({e.edge_id for e in self.edges}) != len(self.edges):
            raise ValueError("Use one edge with multiple bindings per component pair")
        if any(n not in library for n in order):
            raise ValueError("Unknown library component")
        provided = {n: {} for n in order}
        for e in self.edges:
            if not e.bindings or set(e.baselines) != set(e.bindings.values()):
                raise ValueError("Every edge binding requires an explicit baseline")
            for src, dst in e.bindings.items():
                a, b = library[e.source], library[e.target]
                if src not in a.output_fields or dst not in b.input_fields:
                    raise ValueError("Invalid field binding")
                if a.output_fields[src] != b.input_fields[dst]:
                    raise ValueError("Incompatible field schemas")
                if dst in provided[e.target]:
                    raise ValueError("Ambiguous input: more than one producer")
                provided[e.target][dst] = e.source
                base = e.baselines[dst]
                from .component import TYPES

                if base.policy == "constant":
                    if not isinstance(base.value, TYPES[b.input_fields[dst]]):
                        raise ValueError("Baseline does not satisfy input schema")
                elif base.policy == "zero_like":
                    pass  # Resolved from the observed input by UpstreamRuntime.
                elif base.policy == "passthrough":
                    if self.input_fields.get(base.input_field) != b.input_fields[dst]:
                        raise ValueError("Invalid pass-through input")
                else:
                    raise ValueError("Unsupported baseline policy")
        for n in order:
            for f, typ in library[n].input_fields.items():
                if f not in provided[n] and self.input_fields.get(f) != typ:
                    raise ValueError(f"Unsatisfied input {n}.{f}")
        for _, (n, field) in self.outputs.items():
            if n not in order or field not in library[n].output_fields:
                raise ValueError("Final output is unreachable")
        # Every active node must contribute to some final output.
        useful = {n for n, _ in self.outputs.values()}
        for n in reversed(order):
            if any(e.source == n and e.target in useful for e in self.edges):
                useful.add(n)
        if useful != set(order):
            raise ValueError("Active component has no path to a final output")
        return self
