"""Editable, packaged prompts shared by the method and benchmark systems."""

import json
from importlib.resources import files


def text(name):
    return files(__package__).joinpath(name + ".txt").read_text()


def entry(catalog, key):
    return catalog_entries(catalog)[key]


def catalog_entries(catalog):
    return json.loads(files(__package__).joinpath(catalog + ".json").read_text())
