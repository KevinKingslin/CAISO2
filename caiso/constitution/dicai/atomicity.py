"""CAISO semantic enforcement of DICAI's existing one-quality rule."""

from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator

from caiso.core.recovery import ModelResponseError, RetryExhausted, response_failure
from caiso.core.serialization import Record
from caiso.prompts import render_messages

from .parsing import dedupe, normalize_principle, referenced_components

PROMPT_KEYS = (
    "constitution_atomicity.check",
    "constitution_atomicity.split",
    "constitution_atomicity.abstract",
)


def enabled(config=None):
    return (config or {}).get("atomicity", {}).get("enabled", True)


class Assessment(Record):
    index: StrictInt = Field(ge=1)
    atomic: StrictBool
    qualities: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("qualities")
    @classmethod
    def distinct_qualities(cls, values):
        values = [value.strip() for value in values]
        if any(not value for value in values) or len({v.casefold() for v in values}) != len(values):
            raise ValueError("Name distinct, nonempty qualities")
        return values

    @model_validator(mode="after")
    def one_or_several(self):
        if (self.atomic and len(self.qualities) != 1) or (not self.atomic and len(self.qualities) < 2):
            raise ValueError("An atomic item has one quality; a compound item has at least two")
        return self


class AtomicityReview(Record):
    assessments: list[Assessment] = Field(min_length=1)


class Replacement(Record):
    index: StrictInt = Field(ge=1)
    principles: list[str] = Field(min_length=1)

    @field_validator("principles")
    @classmethod
    def actual_principles(cls, values):
        values = [normalize_principle(value) for value in values]
        if len(dedupe(values)) != len(values):
            raise ValueError("Replacement principles must be distinct")
        return values


class PrincipleRevisions(Record):
    replacements: list[Replacement] = Field(min_length=1)


class AtomicityError(ModelResponseError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def ensure_atomic(
    principles, providers, role, seed, *, config=None, kind="debate", context=None,
    component_ids=None,
):
    """Review every item; bounded repairs cannot edit unflagged principles.

    Debate repairs split independent qualities. Abstraction repairs retain one
    shared quality per cluster, preserving the cluster-to-principle mapping.
    Budget, service and policy exceptions always propagate.
    """
    allowed_by_index = {
        row["index"]: sorted(referenced_components(row["principles"], component_ids))
        for row in context
    } if kind == "abstraction" and isinstance(context, list) and component_ids is not None else {}

    def allowed(index):
        return allowed_by_index.get(index, component_ids)

    values = [normalize_principle(p, allowed(i)) for i, p in enumerate(principles, 1)]
    report = {"enabled": enabled(config), "passed": False, "calls_before_retries": 0, "trace": []}
    if not report["enabled"]:
        return values, report
    if not values or kind not in {"debate", "abstraction"}:
        raise ValueError("Atomicity checking needs principles and a valid induction stage")
    max_revisions = (config or {}).get("atomicity", {}).get("max_revisions", 2)
    batch_size = (config or {}).get("atomicity", {}).get("batch_size", 20)
    if isinstance(max_revisions, bool) or not isinstance(max_revisions, int) or max_revisions < 0:
        raise ValueError("atomicity.max_revisions must be a nonnegative integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("atomicity.batch_size must be a positive integer")

    def request(key, schema, validate, attempt, **fields):
        call_seed = seed + report["calls_before_retries"]
        report["calls_before_retries"] += 1
        result = providers.complete(
            role, render_messages(
                "method", key, component_ids=sorted(component_ids) if component_ids is not None else None,
                component_ids_by_index=allowed_by_index, **fields,
            ), seed=call_seed,
            schema=schema, validate=validate, include_schema_instruction=False,
            temperature=0.0,
            response_context={"stage": "atomicity." + kind, "revision": attempt},
        )
        report["trace"].append({"request": key, "seed": call_seed, "result": result.model_dump()})
        return result

    try:
        for attempt in range(max_revisions + 1):
            assessments = []
            for offset in range(0, len(values), batch_size):
                chunk = [{"index": i, "principle": values[i - 1]} for i in range(offset + 1, min(offset + batch_size, len(values)) + 1)]

                def full_review(result):
                    indices = [row.index for row in result.assessments]
                    if sorted(indices) != [row["index"] for row in chunk]:
                        raise ModelResponseError("Assess every supplied principle exactly once by its index")
                    return result

                partial = request(PROMPT_KEYS[0], AtomicityReview, full_review, attempt, principles=chunk)
                assessments.extend(partial.assessments)
            review = AtomicityReview(assessments=assessments)
            report.update(last_checked_principles=list(values), last_review=review.model_dump())
            issues = [row for row in review.assessments if not row.atomic]
            if not issues:
                report.update(passed=True, revisions=attempt)
                return values, report
            if attempt == max_revisions:
                break

            replacements = {}
            for offset in range(0, len(issues), batch_size):
                flagged = issues[offset:offset + batch_size]

                def valid_revisions(result):
                    indices = [row.index for row in result.replacements]
                    if sorted(indices) != sorted(row.index for row in flagged):
                        raise ModelResponseError("Replace every flagged index exactly once; leave unflagged items untouched")
                    for row in result.replacements:
                        row.principles = [normalize_principle(p, allowed(row.index)) for p in row.principles]
                        if kind == "debate" and len(row.principles) < 2:
                            raise ModelResponseError("Split each compound debate principle into at least two independent principles")
                        if kind == "abstraction" and len(row.principles) != 1:
                            raise ModelResponseError("Return exactly one shared quality for each flagged cluster abstraction")
                    return result

                revisions = request(
                    PROMPT_KEYS[1] if kind == "debate" else PROMPT_KEYS[2],
                    PrincipleRevisions, valid_revisions, attempt,
                    principles=[{"index": i, "principle": p} for i, p in enumerate(values, 1)],
                    issues=[row.model_dump() for row in flagged], context=context or {},
                )
                replacements.update({row.index: row.principles for row in revisions.replacements})
            values = dedupe([
                p for i, value in enumerate(values, 1)
                for p in replacements.get(i, [value])
            ]) if kind == "debate" else [
                replacements.get(i, [value])[0] for i, value in enumerate(values, 1)
            ]
    except (ModelResponseError, RetryExhausted) as error:
        if not response_failure(error):
            raise
        raise AtomicityError("Atomicity review or revision returned unusable output: " + str(error), report) from error
    indices = ", ".join(str(row.index) for row in issues)
    raise AtomicityError(
        f"Compound principles remain at indices {indices} after {max_revisions} revisions", report
    )
