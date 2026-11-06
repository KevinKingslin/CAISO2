"""Editable, packaged prompts shared by the method and benchmark systems."""

from importlib.resources import files

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
