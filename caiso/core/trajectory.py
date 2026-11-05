from typing import Any

from pydantic import Field

from .serialization import Record, append_jsonl


class Example(Record):
    example_id: str
    inputs: dict[str, Any]
    target: Any
    split: str


class ComponentExecution(Record):
    component_id: str
    component_input: dict[str, Any]
    component_output: dict[str, Any]
    seed: int
    config_hash: str
    seconds: float


class Trajectory(Record):
    input_id: str
    inputs: dict[str, Any]
    executions: list[ComponentExecution]
    final_output: dict[str, Any]
    reward: float
    seed: int
    state_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def save(self, path):
        append_jsonl(path, self)


def save_parquet(path, trajectories):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .serialization import canonical

    # JSON nested columns preserve arbitrary, heterogeneous component schemas.
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "input_id": t.input_id,
                    "seed": t.seed,
                    "reward": t.reward,
                    "trajectory_json": canonical(t),
                }
                for t in trajectories
            ]
        ),
        path,
    )
