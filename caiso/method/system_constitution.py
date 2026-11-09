"""Section 4.2, equations 4-5."""

from pathlib import Path

from caiso.core.serialization import save_json


def induce_system_constitution(
    dicai, preferences, seed, output=None, progress=None, cache_dir=None, activity=None, component_ids=None
):
    options = {"output_dir": Path(output).with_suffix("")} if output else {}
    if cache_dir:
        options["output_dir"] = Path(cache_dir)
    if progress:
        options["progress"] = progress
    if activity:
        options["activity"] = activity
    if component_ids is not None:
        options["component_ids"] = component_ids
    constitution = dicai.induce(preferences, seed, scope="system", **options)
    if output:
        save_json(output, constitution)
    return constitution
