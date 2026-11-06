"""Editable, packaged prompts shared by the method and benchmark systems."""

from importlib.resources import files
import re
from string import Formatter

from caiso.core.serialization import canonical
from ._catalogs import CATALOGS


def text(name):
    """Read prompt text verbatim, including significant leading/trailing spaces."""
    return files(__package__).joinpath(name + ".txt").read_text(encoding="utf-8")


def entry(catalog, key):
    """Keep existing catalog keys while loading only the requested text files."""
    reference = CATALOGS[catalog][key]
    if isinstance(reference, tuple):
        return [text(name) for name in reference]
    return text(reference)


def catalog_entries(catalog):
    """Read all catalog text, including ordered lists, for cache provenance."""
    return {key: entry(catalog, key) for key in CATALOGS[catalog]}


class _PromptFormatter(Formatter):
    def format_field(self, value, format_spec):
        if format_spec == "json":
            return canonical(value)
        return super().format_field(value, format_spec)


def render_text(catalog, key, **values):
    """Render a complete text template; :json preserves canonical evidence encoding."""
    return _PromptFormatter().vformat(entry(catalog, key), (), values)


def render_messages(catalog, key, **values):
    """Render one complete request file containing [SYSTEM] and [USER] sections.

    Parse role markers before inserting dynamic evidence, so marker-like text in
    an input remains data. Only the marker and its separator newline are removed;
    message whitespace is otherwise preserved exactly.
    """
    template = entry(catalog, key)
    markers = list(re.finditer(r"(?:\A|\n)\[(SYSTEM|USER|ASSISTANT)\]\n", template))
    if not markers or markers[0].start() != 0:
        raise ValueError(f"Prompt {catalog}/{key} must start with a message role marker")
    result = []
    formatter = _PromptFormatter()
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(template)
        result.append(
            {
                "role": marker.group(1).lower(),
                "content": formatter.vformat(template[marker.end() : end], (), values),
            }
        )
    return result
