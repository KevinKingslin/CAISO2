"""Canonical hashes and atomic, JSON-only persistence (no executable checkpoints)."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return plain(value.model_dump(mode="python"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "detach"):
        return plain(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def canonical(value: Any) -> str:
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(canonical(value) + "\n")


def save_jsonl(path, values):
    """Atomically replace a complete trajectory or diagnostic collection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(canonical(value) + "\n")
    os.replace(temporary, path)
