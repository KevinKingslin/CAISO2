"""Capture existing console handlers, including the upstream queued loggers."""

import logging
import sys
from contextlib import contextmanager


@contextmanager
def capture_console_logging(stdout, stderr, console_file, original_stdout, original_stderr):
    loggers = [logging.getLogger(), *[
        value for value in list(logging.Logger.manager.loggerDict.values())
        if isinstance(value, logging.Logger)
    ]]
    listeners, handlers = set(), set()
    for logger in loggers:
        handlers.update(logger.handlers)
        listener = getattr(logger, "listener", None)
        if listener is not None and hasattr(listener, "handlers"):
            listeners.add(listener)
            handlers.update(listener.handlers)
        for handler in logger.handlers:
            listener = getattr(handler, "listener", None)
            if listener is not None and hasattr(listener, "handlers"):
                listeners.add(listener)
                handlers.update(listener.handlers)
    changed, captured_handlers = [], set()
    try:
        for handler in handlers:
            if not isinstance(handler, logging.StreamHandler) or isinstance(handler, logging.FileHandler):
                continue
            stream = handler.stream
            if stream is stdout or stream is stderr:
                captured_handlers.add(handler)
                continue
            descriptor = getattr(type(handler), "stream", None)
            if isinstance(descriptor, property) and descriptor.fset is None:
                continue  # Dynamic stderr handlers follow redirect_stderr themselves.
            if any(stream is target for target in (original_stdout, original_stderr, sys.__stdout__, sys.__stderr__, console_file)):
                handler.setStream(stdout if stream is original_stdout else stderr)
                changed.append((handler, stream))
                captured_handlers.add(handler)
        yield
    finally:
        # Drain records submitted during capture before restoring their console
        # sinks. Otherwise delayed QueueListener writes can escape the panel.
        for listener in listeners:
            thread = getattr(listener, "_thread", None)
            drain = getattr(listener.queue, "join", None)
            if drain is not None and thread is not None and thread.is_alive() and captured_handlers.intersection(listener.handlers):
                drain()
        for handler, stream in reversed(changed):
            handler.setStream(stream)
