"""Skip only explicit service content-filter rejections, with persistent evidence."""

import sys
from threading import RLock

from .recovery import Recovery
from .serialization import append_jsonl

_log_lock = RLock()


class ContentFilteredError(RuntimeError):
    """The service explicitly rejected this content; retrying it is inappropriate."""


def is_content_filter(body):
    if isinstance(body, dict):
        code = str(body.get("code", "")).replace("_", "").lower()
        return code in {"contentfilter", "responsibleaipolicyviolation"} or any(
            is_content_filter(value) for value in body.values()
        )
    if isinstance(body, list):
        return any(is_content_filter(value) for value in body)
    return False


def record_skip(providers, error, stage, **fields):
    recovery = getattr(providers, "recovery", None) or Recovery()
    event = recovery.error(error, stage, "skip_content_filtered", **fields)
    path = getattr(providers, "content_filter_log", None)
    if path:
        with _log_lock:
            append_jsonl(path, event)
    if recovery.on_error is None:
        message = f"Skipped content-filtered example at {stage}: {error}"
        if sys.stderr.isatty():
            message = f"\033[31m{message}\033[0m"
        print(message, file=sys.stderr)


def execute_example(runtime, state, example, seed, **kwargs):
    try:
        return runtime.execute(state, example, seed, **kwargs)
    except ContentFilteredError as error:
        record_skip(
            runtime.providers,
            error,
            "system_execution",
            example_id=example.example_id,
            split=example.split,
            seed=seed,
        )
        return None
