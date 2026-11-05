"""Declarative, serializable components; one primary optimizable variable each."""

from typing import Any

from pydantic import Field

from .configuration import Configuration, ConfigurationType
from .serialization import Record

TYPES = {
    "str": str,
    "int": int,
    "float": (float, int),
    "list": list,
    "dict": dict,
    "bool": bool,
    "any": object,
}


def validate_fields(values, fields, label):
    if set(values) != set(fields):
        raise ValueError(f"{label}: expected {sorted(fields)}, got {sorted(values)}")
    for key, kind in fields.items():
        if kind not in TYPES or not isinstance(values[key], TYPES[kind]):
            raise ValueError(f"{label}.{key}: expected {kind}")


class Component(Record):
    component_id: str
    implementation: str
    input_fields: dict[str, str]
    output_fields: dict[str, str]
    config: Configuration
    config_search_space: list[Any] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    cost_weight: float = Field(default=1, gt=0)

    @property
    def config_type(self):
        return self.config.kind

    def validate_config(self, config):
        if config.kind != self.config_type:
            raise ValueError("Cannot change a component's primary configuration type")
        if config.kind == ConfigurationType.PROMPT and not isinstance(config.value, str):
            raise ValueError("Prompt must be a string")
        if config.kind in (ConfigurationType.MODEL_SELECTION, ConfigurationType.HYPERPARAMETER):
            if config.value not in self.config_search_space:
                raise ValueError(f"Value is outside search space for {self.component_id}")
        if config.kind == ConfigurationType.MODEL_WEIGHTS:
            if not isinstance(config.value, dict) or "model" not in config.value:
                raise ValueError("Weights need a model alias and optional adapter path")
        if config.kind == ConfigurationType.NONE and config != self.config:
            raise ValueError("Non-optimizable configuration changed")

    def execute(self, inputs, config, seed, runtime):
        self.validate_config(config)
        validate_fields(inputs, self.input_fields, self.component_id + ".input")
        result = runtime.handlers[self.implementation](self, inputs, config, seed, runtime)
        validate_fields(result, self.output_fields, self.component_id + ".output")
        return result
