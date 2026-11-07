"""Editable, packaged prompts shared by the method and benchmark systems."""

from importlib.resources import files
import re
from string import Formatter

from caiso.core.serialization import canonical
from ._catalogs import CATALOGS


class PromptResourceError(FileNotFoundError):
    """Report an incomplete prompt installation with all missing paths."""

    def __init__(self, root, missing):
        self.missing = tuple(missing)
        paths = "\n".join("- " + name + ".txt" for name in self.missing)
        super().__init__(
            f"CAISO prompt files are missing from {root}:\n{paths}\n"
            "Update the complete CAISO2 checkout and reinstall it with "
            "`python -m pip install --no-cache-dir --force-reinstall --no-deps -e .`. "
            "For locally deleted tracked files, restore only the listed missing files "
            "from Git. Run `python -m scripts.verify_prompts` to check the installation."
        )


def text(name):
    """Read prompt text verbatim, including significant leading/trailing spaces."""
    root = files(__package__)
    try:
        return root.joinpath(name + ".txt").read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise PromptResourceError(root, [name]) from error


def entry(catalog, key):
    """Keep existing catalog keys while loading only the requested text files."""
    reference = CATALOGS[catalog][key]
    if isinstance(reference, tuple):
        return [text(name) for name in reference]
    return text(reference)


def catalog_entries(catalog, keys=None):
    """Read selected catalog text, including ordered lists, for cache provenance."""
    return {key: entry(catalog, key) for key in (CATALOGS[catalog] if keys is None else keys)}


def validate_resources(selection=None):
    """Check selected catalog entries together, without loading unused defaults.

    With no selection, verify every catalog resource for distribution checks.
    A selection maps catalog names to entry keys, or None for the whole catalog.
    """
    selection = dict.fromkeys(CATALOGS) if selection is None else selection
    root = files(__package__)
    names = set()
    for catalog, keys in selection.items():
        for key in CATALOGS[catalog] if keys is None else keys:
            reference = CATALOGS[catalog][key]
            names.update(reference if isinstance(reference, tuple) else [reference])
    missing = [name for name in sorted(names) if not root.joinpath(name + ".txt").is_file()]
    if missing:
        raise PromptResourceError(root, missing)
    return len(names)


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
