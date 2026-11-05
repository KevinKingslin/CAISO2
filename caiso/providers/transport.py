"""Shared request pacing, bounded retries and explicit content-policy detection."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
from pathlib import Path
import random
from threading import BoundedSemaphore, Lock
import time

import httpx
import yaml

from caiso.core.content_filter import ContentFilteredError, filter_exception
from caiso.core.recovery import TransportExhausted, TransportRequestError
from caiso.core.serialization import append_jsonl, save_json

DEFAULTS = dict(
    requests_per_minute=60,
    concurrency=4,
    retries=4,
    rate_limit_retries=8,
    retry_backoff_seconds=2,
    max_retry_backoff_seconds=60,
    retry_jitter_seconds=1,
    timeout=120,
)
_context = ContextVar("caiso_transport_context", default=(None, None))
_terminal = ContextVar("caiso_terminal_request", default=None)
_lock, _log_lock = Lock(), Lock()
_config, _controllers, _log_path = {}, {}, None


@contextmanager
def observe_transport(budget=None, on_error=None):
    token = _context.set((budget, on_error))
    try:
        with request_scope():
            yield
    finally:
        _context.reset(token)


@contextmanager
def request_scope():
    """Keep SDK format fallbacks from resending a terminally failed component."""
    if _terminal.get() is not None:
        yield
        return
    token = _terminal.set({})
    try:
        yield
    finally:
        _terminal.reset(token)


def _fail(error):
    state = _terminal.get()
    if state is not None:
        state["error"] = error
    raise error from None


def _check_scope():
    state = _terminal.get()
    if state and "error" in state:
        raise state["error"] from None


def validate_limits(options):
    if unknown := options.keys() - DEFAULTS.keys():
        raise ValueError(f"Unknown Azure transport settings: {sorted(unknown)}")
    for key in (
        "requests_per_minute",
        "timeout",
        "retry_backoff_seconds",
        "max_retry_backoff_seconds",
        "retry_jitter_seconds",
    ):
        value = options[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if options["requests_per_minute"] == 0 or options["timeout"] == 0:
        raise ValueError("requests_per_minute and timeout must be positive")
    if options["max_retry_backoff_seconds"] < options["retry_backoff_seconds"]:
        raise ValueError("max_retry_backoff_seconds must be >= retry_backoff_seconds")
    for key, minimum in (("concurrency", 1), ("retries", 0), ("rate_limit_retries", 0)):
        if (
            isinstance(options[key], bool)
            or not isinstance(options[key], int)
            or options[key] < minimum
        ):
            raise ValueError(f"{key} must be an integer >= {minimum}")


def configure_transport(output=None, path=None):
    global _config, _log_path
    path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs/providers.yaml"
    config = yaml.safe_load(path.read_text()) if path.is_file() else {}
    defaults = {**DEFAULTS, **config.get("azure_defaults", {})}
    validate_limits(defaults)
    for limits in config.get("deployments", {}).values():
        validate_limits({**defaults, **limits})
    with _lock:
        _config = config
        _controllers.clear()
        _log_path = Path(output) / ".internal/logs/transport.jsonl" if output else None
    if output:
        save_json(
            Path(output) / ".internal/logs/transport_config.json",
            {
                "azure_defaults": defaults,
                "deployments": config.get("deployments", {}),
            },
        )
    return defaults


def get_controller(model, kwargs):
    key = (str(kwargs.get("api_base", "")).rstrip("/"), model)
    with _lock:
        if key not in _controllers:
            deployment = model.removeprefix("azure/")
            limits = {
                **DEFAULTS,
                **_config.get("azure_defaults", {}),
                **_config.get("deployments", {}).get(deployment, {}),
            }
            validate_limits(limits)
            _controllers[key] = RequestController(model, limits)
        return _controllers[key]


def _check_budget():
    budget, _ = _context.get()
    if budget is not None:
        budget.check()


def _charge(embedding):
    budget, _ = _context.get()
    if budget is not None:
        budget.call(embedding=embedding)


def _record_usage(response):
    budget, _ = _context.get()
    if budget is not None and hasattr(budget, "tokens"):
        budget.tokens(_value(_value(response, "usage", {}), "total_tokens", 0) or 0)


def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def check_response(response, model):
    for choice in _value(response, "choices", []) or []:
        if _value(choice, "finish_reason") == "content_filter":
            raise ContentFilteredError(f"Service filtered the generated response for {model}")


def emit(model, action, **fields):
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": f"azure:{model}",
        "action": action,
        **fields,
    }
    if _log_path:
        with _log_lock:
            append_jsonl(_log_path, event)
    _, callback = _context.get()
    if callback:
        callback_action = "recovered" if action == "transport_recovered" else action
        message = f"Azure transport: {action}"
        if fields.get("status_code") is not None:
            message += f" (HTTP {fields['status_code']})"
        if "retry_after_seconds" in fields:
            message += f"; retry delay {fields['retry_after_seconds']:.2f}s"
        callback(RuntimeError(message), event["stage"], callback_action, **fields)


class RequestController:
    def __init__(self, model, options):
        self.model, self.options = model, options
        self.gate = BoundedSemaphore(options["concurrency"])
        self.lock = Lock()
        self.next_request = self.cooldown_until = 0.0

    def delay(self):
        _check_budget()
        with self.lock:
            now = time.monotonic()
            delay = max(self.next_request, self.cooldown_until) - now
            if delay <= 0:
                self.next_request = now + 60 / self.options["requests_per_minute"]
                return 0
            return delay

    def retry(self, error, rates, transients):
        if filter_exception(error):
            raise ContentFilteredError(f"Service rejected content for {self.model}") from None
        response = getattr(error, "response", None)
        status = getattr(error, "status_code", None) or getattr(response, "status_code", None)
        transient = (
            status in {408, 429, 500, 502, 503, 504}
            or isinstance(
                error, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
            )
            or type(error).__name__ in {"Timeout", "APITimeoutError", "APIConnectionError"}
        )
        if not transient:
            if status is not None:
                raise TransportRequestError(
                    f"Service returned non-retryable HTTP {status} for {self.model}"
                ) from None
            raise error
        rate = status == 429
        attempt = rates if rate else transients
        maximum = self.options["rate_limit_retries" if rate else "retries"]
        seconds = self.retry_delay(response, attempt, getattr(error, "headers", None))
        if rate:
            with self.lock:
                self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)
        emit(
            self.model,
            "transport_retry" if attempt < maximum else "transport_exhausted",
            attempt=attempt + 1,
            max_attempts=maximum + 1,
            retry_after_seconds=seconds,
            status_code=status,
        )
        if attempt >= maximum:
            raise TransportExhausted(
                f"Transport retries exhausted for {self.model} (HTTP {status})"
            ) from None
        return rate, seconds

    def retry_delay(self, response, attempt, extra_headers=None):
        headers = {
            str(k).lower(): v
            for k, v in (getattr(response, "headers", None) or extra_headers or {}).items()
        }
        for key in ("retry-after-ms", "x-ms-retry-after-ms", "retry-after"):
            if key not in headers:
                continue
            try:
                seconds = float(headers[key]) / (1000 if key.endswith("-ms") else 1)
            except (ValueError, TypeError):
                try:
                    seconds = parsedate_to_datetime(headers[key]).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    continue
            if math.isfinite(seconds) and seconds >= 0:
                return seconds
        return min(
            self.options["max_retry_backoff_seconds"],
            self.options["retry_backoff_seconds"] * 2 ** min(attempt, 30)
            + random.uniform(0, self.options["retry_jitter_seconds"]),
        )


def wait(seconds):
    deadline = time.monotonic() + seconds
    while deadline > time.monotonic():
        _check_budget()
        time.sleep(min(1, max(0, deadline - time.monotonic())))


async def await_delay(seconds):
    deadline = time.monotonic() + seconds
    while deadline > time.monotonic():
        _check_budget()
        await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))


def request(function, model, args, kwargs, *, embedding=False):
    _check_scope()
    controller = get_controller(model, kwargs) if model.startswith("azure/") else None
    rates = transients = 0
    if controller:
        kwargs = {**kwargs, "num_retries": 0}
        kwargs.setdefault("timeout", controller.options["timeout"])
    while True:
        if controller:
            while not controller.gate.acquire(timeout=0.5):
                _check_budget()
        try:
            if controller:
                while delay := controller.delay():
                    wait(delay)
            _charge(embedding)
            response = function(*args, **kwargs)
            _record_usage(response)
            check_response(response, model)
            if rates or transients:
                emit(
                    model,
                    "transport_recovered",
                    rate_limit_retries=rates,
                    transient_retries=transients,
                )
            return response
        except Exception as error:
            if filter_exception(error):
                _fail(ContentFilteredError(f"Service rejected content for {model}"))
            if not controller:
                raise
            try:
                rate, seconds = controller.retry(error, rates, transients)
            except (TransportExhausted, TransportRequestError) as terminal:
                _fail(terminal)
            rates += int(rate)
            transients += int(not rate)
        finally:
            if controller:
                controller.gate.release()
        if not rate:
            wait(seconds)


async def arequest(function, model, args, kwargs, *, embedding=False):
    _check_scope()
    controller = get_controller(model, kwargs) if model.startswith("azure/") else None
    rates = transients = 0
    if controller:
        kwargs = {**kwargs, "num_retries": 0}
        kwargs.setdefault("timeout", controller.options["timeout"])
    while True:
        if controller:
            while not controller.gate.acquire(blocking=False):
                _check_budget()
                await asyncio.sleep(0.05)
        try:
            if controller:
                while delay := controller.delay():
                    await await_delay(delay)
            _charge(embedding)
            response = await function(*args, **kwargs)
            _record_usage(response)
            check_response(response, model)
            if rates or transients:
                emit(
                    model,
                    "transport_recovered",
                    rate_limit_retries=rates,
                    transient_retries=transients,
                )
            return response
        except Exception as error:
            if filter_exception(error):
                _fail(ContentFilteredError(f"Service rejected content for {model}"))
            if not controller:
                raise
            try:
                rate, seconds = controller.retry(error, rates, transients)
            except (TransportExhausted, TransportRequestError) as terminal:
                _fail(terminal)
            rates += int(rate)
            transients += int(not rate)
        finally:
            if controller:
                controller.gate.release()
        if not rate:
            await await_delay(seconds)
