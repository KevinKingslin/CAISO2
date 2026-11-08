"""Read published principles without depending on Markdown or list numbering."""

import json
import re

from caiso.core.recovery import ModelResponseError
from caiso.providers.responses import json_object, unwrap

PRINCIPLE_PREFIX = "Select the trajectory"
_PREFIX = re.compile(r"^select\s+the\s+trajectory\b", re.I)
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


def normalize_principle(text):
    value = clean_item(unwrap(text))
    while _LABEL.match(value):
        value = clean_item(_LABEL.sub("", value, count=1))
    match = _PREFIX.match(value)
    if not match or not value[match.end():].strip(" .:…<>"):
        raise ModelResponseError(
            f'Return a nonempty principle beginning "{PRINCIPLE_PREFIX}" and stating an actual selection criterion.'
        )
    # Only the selection target is structural. Preserve the condition's wording
    # instead of rejecting English connectors such as "that", "which" or "with".
    prefix = "Select the trajectory"
    value = prefix + re.sub(r"^[*`_]+", "", value[match.end():])
    if value[len(prefix):].strip(" .:").lower() in {
        "that", "which", "where", "with", "whose", "in which", "for which",
        "a", "b", "(a)", "(b)", "a or b", "output(a)", "output(b)",
    }:
        raise ModelResponseError("The principle must state the actual selection criterion.")
    if len(re.findall(r"\bselect\s+the\s+(?:trajectory|response)\b", value, re.I)) != 1:
        raise ModelResponseError("Return one principle, not several principles in one item.")
    if re.search(r"<(?:condition|actual criterion|criterion)>", value, re.I) or value.strip().endswith("..."):
        raise ModelResponseError("Return the actual preference criterion, not a placeholder.")
    return re.sub(r"\s+", " ", value).strip()


def _structured(value):
    if isinstance(value, dict):
        for key in ("principles", "questions", "final_list", "working_list"):
            if key in value:
                return _structured(value[key])
        if value and all(re.fullmatch(r"Q\s*\d+", str(key), re.I) for key in value):
            return _structured(list(value.values()))
        if "select_prompt" in value:
            return [normalize_principle(value["select_prompt"])]
        raise ModelResponseError("The principle object contains no principle list.")
    if isinstance(value, list) and value:
        return dedupe([p for item in value for p in _structured(item)])
    if isinstance(value, str):
        return [normalize_principle(value)]
    raise ModelResponseError("Expected a nonempty list of preference principles.")


def _items(lines):
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
                items.append(normalize_principle(current))
            current = content
        elif current and raw.strip() and not re.search(r"[.!?][\"”']?$", current):
            if line.upper() == "DEBATE_OVER" or _HEADER.fullmatch(line) or re.match(r"^(?:reason|explanation|note|rationale)\s*:", line, re.I):
                items.append(normalize_principle(current))
                current = None
            else:
                current += " " + line
        elif current:
            items.append(normalize_principle(current))
            current = None
    if current:
        items.append(normalize_principle(current))
    return dedupe(items)


def parse_principles(text):
    value = unwrap(text)
    if value.startswith(("{", "[")):
        try:
            return _structured(json.loads(value))
        except (json.JSONDecodeError, TypeError, ModelResponseError) as error:
            raise ModelResponseError("Return a complete nonempty list of preference principles.") from error
    # Only standalone headings delimit lists; a closing sentence such as
    # "The final list is stable" must not erase the actual published bullets.
    lines = value.splitlines()
    headers = [i for i, line in enumerate(lines) if _HEADER.fullmatch(plain_line(line))]
    if headers:
        start = headers[-1] + 1
        lines = lines[start:]
    result = _items(lines)
    if result:
        return result
    # Some models put the requested list in an otherwise wrapped JSON object.
    if "{" in value and not headers:
        try:
            return _structured(json_object(value))
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
