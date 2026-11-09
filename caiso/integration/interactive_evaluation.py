"""Read component overrides from the terminal before benchmark evaluation."""

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace


GENERATION_FIELDS = ("model", "temperature", "max_tokens", "max_new_tokens", "top_p", "top_k")


def parse_value(text, current):
    """Keep parameter types, accepting ordinary unquoted strings in the terminal."""
    if isinstance(current, str):
        return text
    if isinstance(current, Path):
        path = Path(text).expanduser()
        if not path.is_dir():
            raise ValueError("Enter an existing adapter directory")
        return path
    value = json.loads(text)
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Enter a number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Enter a finite number")
    elif current is not None and type(value) is not type(current):
        raise ValueError(f"Enter a {type(current).__name__} value")
    # Reject NaN/Infinity in nested JSON values as well.
    json.dumps(value, allow_nan=False)
    return value


def configure_interactively(system, *, read=None, write=print):
    """Collect all edits first, then install them for every evaluation worker."""
    read = read or input
    write("Configure each component. Enter keeps its current value. Ctrl+C cancels.")
    write("Enter strings without quotes; use JSON syntax for numbers, booleans and containers.")

    def parameter(label, current):
        while True:
            text = read(f"{label} [{current}]: ")
            if not text.strip():
                return current
            try:
                return parse_value(text, current)
            except (ValueError, TypeError, OverflowError) as error:
                write(f"Invalid value: {error}. Try again.")

    def prompt(current):
        write(f"Current prompt:\n{current}")
        while True:
            text = read("New prompt (Enter keeps it; /multiline starts multiple lines): ")
            if text == "/multiline":
                write("Paste/type the prompt; finish with /end on its own line.")
                lines = []
                while (line := read("")) != "/end":
                    lines.append(line)
                text = "\n".join(lines)
            if not text:
                return current
            if text.strip():
                return text
            write("The prompt must contain text.")

    records = {}
    for index, cid in enumerate(system.execution_order, 1):
        component = system.components[cid]
        write(f"\n[{index}/{len(system.components)}] {cid}")
        current = deepcopy(component.variable)
        if isinstance(current, str):
            kind, value = "PROMPT", prompt(current)
        elif isinstance(current, dict):
            kind = "MODEL_SELECTION" if "model_selection" in current else "HYPERPARAMETER"
            value = {}
            for key, old in current.items():
                choices = (component.variable_search_space or {}).get(key)
                if choices:
                    write(f"{key}: optimizer search values {choices} (custom values allowed)")
                value[key] = parameter(key, old)
        elif isinstance(current, Path):
            kind, value = "MODEL_WEIGHTS", parameter("Adapter directory", current)
        elif current is None:
            kind, value = "NONE", None
            write("No configurable component variable.")
        else:
            raise ValueError(f"{cid}: unsupported component variable {type(current).__name__}")
        generation = {
            key: parameter(key, getattr(component.config, key))
            for key in GENERATION_FIELDS if hasattr(component.config, key)
        }
        records[cid] = {"kind": kind, "variable": value, "generation": generation}

    for cid, record in records.items():
        component = system.components[cid]
        if record["variable"] != component.variable:
            component.update(record["variable"])
        if record["generation"]:
            component.update_config(**record["generation"])
            # update_config alone only changes the calling thread. New workers
            # take their configuration from default_config.
            component.default_config = SimpleNamespace(
                **{**vars(component.default_config), **record["generation"]}
            )
    return records
