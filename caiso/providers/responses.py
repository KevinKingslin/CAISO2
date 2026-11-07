"""Lossless handling of presentation wrappers around method responses."""

import json
import re

from caiso.core.recovery import ModelResponseError


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelResponseError(f"Duplicate JSON field {key!r}; return one value per field.")
        result[key] = value
    return result


def _constant(value):
    raise ModelResponseError(f"Invalid JSON numeric constant: {value}")


def unwrap(text):
    if not isinstance(text, str):
        raise ModelResponseError("Expected a textual model response")
    text = text.strip().lstrip("\ufeff")
    fence = re.fullmatch(r"```(?:json|text|markdown)?\s*\n?(.*?)\n?```", text, re.S | re.I)
    return fence.group(1).strip() if fence else text


def json_object(text):
    """Read one complete JSON object, never repair values or missing syntax."""
    value = unwrap(text)
    try:
        result = json.loads(value, object_pairs_hook=_object, parse_constant=_constant)
    except json.JSONDecodeError as error:
        start = value.find("{")
        if start < 0:
            raise ModelResponseError("Return one complete JSON object with the requested fields.") from error
        try:
            result, end = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant).raw_decode(value, start)
        except json.JSONDecodeError as error:
            raise ModelResponseError(f"Incomplete or invalid JSON: {error.msg}. Return a complete JSON object.") from error
        if "{" in value[end:] or "}" in value[:start]:
            raise ModelResponseError("Multiple or ambiguous JSON objects; return exactly one object.")
    if not isinstance(result, dict):
        raise ModelResponseError("Expected a JSON object, not an array, scalar or null.")
    return result
