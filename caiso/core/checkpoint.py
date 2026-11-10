from pathlib import Path

from pydantic import Field

from caiso.constitution.models import Constitution, PreferencePair
from caiso.core.budget import BudgetUsage
from caiso.core.serialization import Record, read_json, save_json
from caiso.core.system_state import SystemState
from caiso.method.joint_workflow_block import WorkflowTrigger
from caiso.method.optimization_history import OptimizationHistory


class Checkpoint(Record):
    schema_version: int = 1
    iteration: int
    current: SystemState
    best: SystemState
    current_score: float
    best_score: float
    system_constitution: Constitution
    block_constitutions: dict[str, Constitution]
    preferences: list[PreferencePair]
    budget: BudgetUsage
    trigger: WorkflowTrigger
    config_hash: str
    data_hash: str
    provider_hash: str
    library_hash: str
    blocks: dict[str, list[str]] = Field(default_factory=dict)
    attribution: list[dict] = Field(default_factory=list)
    attribution_unsuccessful_iterations: int = Field(default=0, ge=0, strict=True)
    pending_refresh: bool = False
    consecutive_errors: int = 0
    optimization_history: OptimizationHistory = Field(default_factory=OptimizationHistory)
    optimizer_state: dict = Field(default_factory=dict)

    def save(self, path):
        path = Path(path)
        # One atomic resume record contains all state; no duplicate sidecars.
        save_json(path / "checkpoint.json", self)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_dir():
            path = path / "checkpoint.json"
        checkpoint = cls.model_validate(read_json(path))
        if checkpoint.schema_version != 1:
            raise ValueError("Unsupported checkpoint schema")
        return checkpoint
