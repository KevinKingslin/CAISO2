from enum import StrEnum
from typing import Any

from pydantic import Field

from .serialization import Record


class ConfigurationType(StrEnum):
    PROMPT = "PROMPT"
    MODEL_SELECTION = "MODEL_SELECTION"
    MODEL_WEIGHTS = "MODEL_WEIGHTS"
    HYPERPARAMETER = "HYPERPARAMETER"
    NONE = "NONE"


class Configuration(Record):
    kind: ConfigurationType
    value: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
