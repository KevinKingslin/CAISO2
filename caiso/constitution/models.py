from typing import Any

from pydantic import Field

from caiso.core.serialization import Record


class PreferencePair(Record):
    pair_id: str
    context: dict[str, Any]
    candidate_a: dict[str, Any]
    candidate_b: dict[str, Any]
    preferred: str = Field(pattern="^[AB]$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Constitution(Record):
    principles: list[str] = Field(min_length=1)
    source_preference_pair_ids: list[str]
    settled_principles: list[str]
    cluster_ids: list[int]
    abstraction_outputs: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def text(self):
        return "\n".join(f"{i + 1}. {p}" for i, p in enumerate(self.principles))
