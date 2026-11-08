"""Freeze accepted run state for evaluation in a separate process."""

from contextlib import contextmanager
from hashlib import sha256
import logging
from pathlib import Path
from typing import Literal
from unittest.mock import patch
from uuid import uuid4

from caiso.core.checkpoint import Checkpoint
from caiso.core.serialization import Record, save_json
from caiso.core.system_state import SystemState


class EvaluationSnapshot(Record):
    source: str
    selection: Literal["current", "best"]
    iteration: int
    validation_score: float
    state_fingerprint: str
    state: SystemState
    initial_system_path: str | None = None
    initial_system_sha256: str | None = None

    def provenance(self):
        return self.model_dump(exclude={"state"})


def freeze_checkpoint(path, selection, directory):
    """One atomic checkpoint read; later optimizer writes cannot alter this copy."""
    path, directory = Path(path), Path(directory)
    if path.is_dir():
        path /= "checkpoint.json"
    checkpoint = Checkpoint.load(path)
    if selection not in {"current", "best"}:
        raise ValueError("Checkpoint selection must be current or best")
    state = getattr(checkpoint, selection).model_copy(deep=True)
    snapshot = EvaluationSnapshot(
        source=str(path.resolve()), selection=selection, iteration=checkpoint.iteration,
        validation_score=getattr(checkpoint, selection + "_score"),
        state_fingerprint=state.fingerprint(), state=state,
    )
    directory.mkdir(parents=True, exist_ok=True)
    initial = path.parent.parent / "initial_system.pth"
    if initial.is_file():
        # These exports contain component settings and adapter paths, not weights.
        content = initial.read_bytes()
        frozen_initial = directory / "initial_system.pth"
        frozen_initial.write_bytes(content)
        snapshot.initial_system_path = str(frozen_initial.resolve())
        snapshot.initial_system_sha256 = sha256(content).hexdigest()
    save_json(directory / "state_snapshot.json", snapshot)
    return snapshot


def save_state_dict(state, path):
    """Publish complete .pth exports; evaluators never see a partial file."""
    import torch

    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def snapshot_models(snapshot, system_name):
    """Keep Amazon's shared vLLM aliases and adapter files owned by optimization.

    The original factory is unchanged. During snapshot evaluation its local
    clients load the selected saved adapters under unique server-side names.
    """
    if snapshot is None or system_name != "amazon_system":
        yield
        return
    from examples.systems.amazon import local_models_for_next_item_selection as amazon
    from caiso.utils.lora import unload_lora_adapter

    components = {"SessionAnalyzerModule": "session_analyzer", "CandidateProfilerModule": "candidate_profiler"}
    paths = {}
    for name, cid in components.items():
        config = snapshot.state.configurations[cid]
        adapter = config.value.get("adapter") if isinstance(config.value, dict) else None
        if not adapter or not Path(adapter).is_dir():
            raise ValueError(f"Snapshot adapter for {cid} is unavailable: {adapter}")
        paths[name] = adapter
    namespace = uuid4().hex
    aliases = {name: f"{name}-evaluation-{namespace}" for name in components}
    original_load, original_post = amazon.load_lora_adapter, amazon.post_http_request
    loaded = {}

    def load(name, path, host="localhost", port=8001, **kwargs):
        # Register cleanup before the request: a timeout may follow a successful load.
        loaded[aliases[name]] = (host, port)
        return original_load(aliases[name], paths[name], host, port, **kwargs)

    def post(*args, adapter_id=None, **kwargs):
        return original_post(*args, adapter_id=aliases[adapter_id] if adapter_id else None, **kwargs)

    try:
        with patch.multiple(amazon, initialize_adapter=lambda *args, **kwargs: None,
                            load_lora_adapter=load, post_http_request=post):
            yield
    finally:
        for name, (host, port) in loaded.items():
            try:
                unload_lora_adapter(name, host, port, silent=True, timeout=10)
            except Exception as error:
                logging.getLogger(__name__).warning("Could not release evaluation adapter %s: %s", name, error)


def apply_snapshot(system, snapshot):
    """Restore original component settings, then install the frozen CAISO state."""
    import torch
    from caiso.core.budget import BudgetTracker
    from caiso.integration.runtime import UpstreamRuntime
    from caiso.providers.base import ProviderRegistry

    if snapshot.initial_system_path:
        system.load_state_dict(torch.load(snapshot.initial_system_path, weights_only=False))
    runtime = UpstreamRuntime(system, ProviderRegistry({}), BudgetTracker(len(system.components)))
    runtime.apply_state(snapshot.state)
