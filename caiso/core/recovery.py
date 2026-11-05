"""Bounded retries for model formatting errors and transient service failures."""

import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

import httpx
from pydantic import ValidationError


class ModelResponseError(ValueError):
    """A model answered, but its answer cannot be used by the algorithm."""


class RetryExhausted(RuntimeError):
    """A recoverable operation still failed after its configured retries."""


def recoverable(exc):
    if isinstance(
        exc,
        (
            ModelResponseError,
            ValidationError,
            RetryExhausted,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {
        408,
        429,
        500,
        502,
        503,
        504,
    }


class Recovery:
    def __init__(self, config=None, on_error=None, budget=None):
        self.config = config or {}
        self.on_error, self.budget = on_error, budget

    @contextmanager
    def observe_transports(self, providers):
        """Expose backend retries that happen underneath the model call retry loop."""
        previous = []
        for provider in {id(p): p for p in providers if p is not None}.values():
            if hasattr(provider, "on_error"):
                previous.append((provider, provider.on_error))
                provider.on_error = self.error
        try:
            yield
        finally:
            for provider, callback in previous:
                provider.on_error = callback

    def error(self, exc, stage, action, **fields):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "action": action,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
            **fields,
        }
        if self.on_error:
            self.on_error(event)
        return event

    def call(self, operation, *, stage, retries=None):
        """operation(attempt, previous_error); successful work is never repeated."""
        retries = self.config.get("operation_retries", 2) if retries is None else retries
        previous = None
        for attempt in range(retries + 1):
            if self.budget:
                self.budget.check()
            try:
                result = operation(attempt, previous)
            except Exception as exc:
                if not recoverable(exc):
                    raise  # Budget caps, auth/configuration errors and code bugs are not retried.
                retry = attempt < retries
                self.error(
                    exc,
                    stage,
                    "retry" if retry else "retries_exhausted",
                    attempt=attempt + 1,
                    max_attempts=retries + 1,
                )
                if not retry:
                    raise RetryExhausted(
                        f"{stage} failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                previous = exc
                delay = min(
                    self.config.get("backoff_seconds", 1) * 2**attempt,
                    self.config.get("max_backoff_seconds", 8),
                )
                if self.budget:
                    self.budget.check()
                    delay = min(
                        delay,
                        max(0, self.budget.max_seconds - self.budget.snapshot().elapsed_seconds),
                    )
                if delay:
                    time.sleep(delay)
            else:
                if previous is not None:
                    self.error(
                        previous, stage, "recovered", attempt=attempt + 1, max_attempts=retries + 1
                    )
                return result
        raise AssertionError("Unreachable retry loop")
