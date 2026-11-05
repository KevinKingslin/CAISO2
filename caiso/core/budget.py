"""Central accounting. Inference overhead is reported separately by default."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from time import monotonic

from pydantic import Field

from .serialization import Record


class BudgetExceeded(RuntimeError):
    pass


class BudgetUsage(Record):
    full_system_runs: int = 0
    system_attempts: int = 0
    component_executions: int = 0
    llm_calls: int = 0
    component_llm_calls: int = 0
    optimization_llm_calls: int = 0
    embedding_calls: int = 0
    local_model_inference: int = 0
    training_steps: int = 0
    training_seconds: float = 0
    causal_intervention_runs: int = 0
    compatibility_adaptation_runs: int = 0
    tokens: int = 0
    cache_hits: int = 0
    equivalent_full_system_runs: float = 0
    elapsed_seconds: float = 0
    component_seconds: dict[str, float] = Field(default_factory=dict)


class BudgetTracker:
    def __init__(
        self,
        full_cost,
        max_equivalent_full_runs=1000,
        max_wallclock_hours=24,
        max_llm_calls=100000,
        overhead_call_cost=0.0,
        usage=None,
    ):
        if full_cost <= 0:
            raise ValueError("Full-system reference cost must be positive")
        self.full_cost = full_cost
        self.max_runs = max_equivalent_full_runs
        self.max_seconds = max_wallclock_hours * 3600
        self.max_calls = max_llm_calls
        self.overhead_call_cost = overhead_call_cost
        self.usage = usage or BudgetUsage()
        self.started = monotonic()
        self.previous_seconds = self.usage.elapsed_seconds
        self.lock = RLock()
        self.caps = []
        self.in_compatibility = False
        self._component_call = ContextVar(f"component_call_{id(self)}", default=False)

    def check(self, additional=0):
        with self.lock:
            self.usage.elapsed_seconds = self.previous_seconds + monotonic() - self.started
            limit = min([self.max_runs, *self.caps])
            if self.usage.equivalent_full_system_runs + additional > limit + 1e-10:
                raise BudgetExceeded("Equivalent-run budget exhausted")
            if self.usage.elapsed_seconds >= self.max_seconds:
                raise BudgetExceeded("Wall-clock budget exhausted")

    def component(self, cost):
        with self.lock:
            units = cost / self.full_cost
            self.check(units)
            self.usage.component_executions += 1
            self.usage.equivalent_full_system_runs += units

    def call(self, local=False, embedding=False):
        with self.lock:
            units = self.overhead_call_cost / self.full_cost
            self.check(units)
            if self.usage.llm_calls + self.usage.embedding_calls >= self.max_calls:
                raise BudgetExceeded("Model-call budget exhausted")
            self.usage.equivalent_full_system_runs += units
            if embedding:
                self.usage.embedding_calls += 1
            else:
                self.usage.llm_calls += 1
                if self._component_call.get():
                    self.usage.component_llm_calls += 1
                else:
                    self.usage.optimization_llm_calls += 1
            self.usage.local_model_inference += int(local)

    def tokens(self, count):
        with self.lock:
            self.usage.tokens += int(count)

    def snapshot(self):
        with self.lock:
            self.usage.elapsed_seconds = self.previous_seconds + monotonic() - self.started
            return self.usage.model_copy(deep=True)

    @contextmanager
    def component_calls(self):
        token = self._component_call.set(True)
        try:
            yield
        finally:
            self._component_call.reset(token)

    @contextmanager
    def compatibility(self, extra_runs):
        previous = self.in_compatibility
        self.in_compatibility = True
        self.caps.append(self.usage.equivalent_full_system_runs + extra_runs)
        try:
            yield
        finally:
            self.caps.pop()
            self.in_compatibility = previous
