"""Read published principles without depending on Markdown or list numbering."""

import json
import re

from caiso.core.recovery import ModelResponseError
from caiso.providers.responses import json_object, unwrap

PRINCIPLE_PREFIX = "Select the trajectory where"
_PREFIX = re.compile(r"^select\s+the\s+trajectory\b", re.I)
_FORMAT = re.compile(r"^select\s+the\s+trajectory\s+where\s+(\S+)\s+(.+)$", re.I)
_HEADER = re.compile(r"^(?:UPDATED\s+WORKING|FINAL|WORKING)\s+LIST(?:\s+OF\s+(?:PRINCIPLES|QUESTIONS))?(?:\s*\([^\n)]*\))?\s*:?[ \t]*$", re.I)
_LABEL = re.compile(r"^(?:(?:[-+•–—]|\d+[.):])\s+|Q\s*\d+\s*[:.)\-–—]?\s*)", re.I)


def principle_key(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def dedupe(values):
    seen, result = set(), []
    for value in values:
        key = principle_key(value)
        if key and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def plain_line(text):
    return re.sub(r"^[#*`_]+\s*|\s*[*`_]+$", "", text.strip()).strip()


def clean_item(text):
    for _ in range(3):
        cleaned = plain_line(text).strip('"\'“”‘’ ')
        if cleaned == text:
            break
        text = cleaned
    return text


def normalize_principle(text, component_ids=None):
    """Canonicalize presentation and require a component subject and condition.

    Generation callers supply execution/source IDs to enforce subject grounding.
    Standalone parsing without an ID inventory checks syntax only.
    """
    value = clean_item(unwrap(text))
    while _LABEL.match(value):
        value = clean_item(_LABEL.sub("", value, count=1))
    value = re.sub(r"\s+", " ", value).strip()
    match = _FORMAT.fullmatch(value)
    if not match:
        raise ModelResponseError(
            'Use "Select the trajectory where <component_id> <condition>." '
            'Replace both placeholders with an exact allowed component ID immediately after '
            '"where" and one nonempty, observable condition.'
        )
    component, condition = match.groups()
    if not re.fullmatch(r"[\w./:-]+", component) or not re.search(r"\w", condition):
        raise ModelResponseError("Name one exact component ID followed by an actual condition.")
    if component_ids is not None and component not in component_ids:
        raise ModelResponseError(
            f"Unknown principle subject {component!r}; use an exact allowed component ID: "
            + ", ".join(sorted(component_ids))
        )
    if component_ids is None and component.casefold() in {"the", "a", "an"}:
        raise ModelResponseError("Replace the generic subject with the actual component ID.")
    if len(re.findall(r"\bselect\s+the\s+(?:trajectory|response)\b", value, re.I)) != 1:
        raise ModelResponseError("Return one principle, not several principles in one item.")
    if re.search(r"<(?:component_id|condition|actual criterion|criterion)>", value, re.I) or value.endswith(("...", "…")):
        raise ModelResponseError("Return the actual preference criterion, not a placeholder.")
    return f"{PRINCIPLE_PREFIX} {component} {condition}"


def principle_subjects(principles):
    return {normalize_principle(p)[len(PRINCIPLE_PREFIX):].split()[0] for p in principles}


def referenced_components(principles, component_ids):
    """Find exact known IDs, including secondary components in interactions."""
    return {
        cid for cid in component_ids
        if any(re.search(r"(?<![\w./:-])" + re.escape(cid) + r"(?![\w/:-]|\.\w)", p) for p in principles)
    }


def _structured(value, component_ids):
    if isinstance(value, dict):
        for key in ("principles", "questions", "final_list", "working_list"):
            if key in value:
                return _structured(value[key], component_ids)
        if value and all(re.fullmatch(r"Q\s*\d+", str(key), re.I) for key in value):
            return _structured(list(value.values()), component_ids)
        if "select_prompt" in value:
            return [normalize_principle(value["select_prompt"], component_ids)]
        raise ModelResponseError("The principle object contains no principle list.")
    if isinstance(value, list) and value:
        return dedupe([p for item in value for p in _structured(item, component_ids)])
    if isinstance(value, str):
        return [normalize_principle(value, component_ids)]
    raise ModelResponseError("Expected a nonempty list of preference principles.")


def _items(lines, component_ids):
    items, current = [], None
    for raw in lines:
        line = plain_line(raw)
        if line.startswith("|") and line.endswith("|"):
            cells = [plain_line(cell) for cell in line.strip("|").split("|")]
            criteria = [cell for cell in cells if _PREFIX.match(cell)]
            if len(criteria) == 1:
                line = criteria[0]
        content = clean_item(line)
        while _LABEL.match(content):
            content = clean_item(_LABEL.sub("", content, count=1))
        if _PREFIX.match(content):
            if current:
                items.append(normalize_principle(current, component_ids))
            current = content
        elif current and raw.strip() and not re.search(r"[.!?][\"”']?$", current):
            if line.upper() == "DEBATE_OVER" or _HEADER.fullmatch(line) or re.match(r"^(?:reason|explanation|note|rationale)\s*:", line, re.I):
                items.append(normalize_principle(current, component_ids))
                current = None
            else:
                current += " " + line
        elif current:
            items.append(normalize_principle(current, component_ids))
            current = None
    if current:
        items.append(normalize_principle(current, component_ids))
    return dedupe(items)


def parse_principles(text, component_ids=None):
    value = unwrap(text)
    if value.startswith(("{", "[")):
        try:
            return _structured(json.loads(value), component_ids)
        except (json.JSONDecodeError, TypeError, ModelResponseError) as error:
            raise ModelResponseError("Return a complete nonempty list of preference principles: " + str(error)) from error
    # Only standalone headings delimit lists; a closing sentence such as
    # "The final list is stable" must not erase the actual published bullets.
    lines = value.splitlines()
    headers = [i for i, line in enumerate(lines) if _HEADER.fullmatch(plain_line(line))]
    if headers:
        start = headers[-1] + 1
        lines = lines[start:]
    result = _items(lines, component_ids)
    if result:
        return result
    # Some models put the requested list in an otherwise wrapped JSON object.
    if "{" in value and not headers:
        try:
            return _structured(json_object(value), component_ids)
        except ModelResponseError:
            pass
    raise ModelResponseError(
        'No published principles found. Publish the complete nonempty working list '
        f'as Q-numbered bullets beginning "{PRINCIPLE_PREFIX}". Use actual criteria '
        'from the supplied preference rationales and candidate questions. Merge '
        'redundant criteria while retaining their shared criterion. '
        'Do not return placeholders, an empty list, commentary or DEBATE_OVER alone.'
    )


def explicitly_unchanged(text):
    value = plain_line(unwrap(text)).strip(" .!").casefold()
    if value == "debate_over":
        return True
    return bool(re.fullmatch(
        r"(?:(?:the )?(?:(?:current|final|working) )?list (?:is |remains )?unchanged|"
        r"no (?:further )?changes(?: (?:are )?needed)?)(?:[.!\s]*debate_over)?",
        value,
    ))
