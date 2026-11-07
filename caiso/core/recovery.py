"""Bounded retries for model formatting errors and transient service failures."""

import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

import httpx
from pydantic import ValidationError


class ModelResponseError(ValueError):
    """A model answered, but its answer cannot be used by the algorithm."""

    def __init__(self, message, *, raw_response=None, response_context=None):
        super().__init__(message)
        self.raw_response = raw_response
        self.response_context = response_context or {}


class RetryExhausted(RuntimeError):
    """A recoverable operation still failed after its configured retries."""


class TransportExhausted(RetryExhausted):
    """Transport retries are exhausted; outer response loops must not repeat them."""


class TransportRequestError(ValueError):
    """A non-retryable service error, such as invalid credentials or deployment."""


def response_failure(error):
    """True only for invalid model output, including an exhausted format retry."""
    while isinstance(error, RetryExhausted) and not isinstance(error, TransportExhausted):
        error = error.__cause__
    return isinstance(error, (ModelResponseError, ValidationError))


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
        if isinstance(exc, ModelResponseError):
            event["response_context"] = exc.response_context
            if exc.raw_response is not None:
                event["raw_response"] = exc.raw_response[:64000]
                event["raw_response_truncated"] = len(exc.raw_response) > 64000
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
                if isinstance(exc, TransportExhausted):
                    raise
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
