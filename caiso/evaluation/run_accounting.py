import csv
import os
from pathlib import Path

from caiso.core.serialization import append_jsonl, digest


def start_wandb(directory, config, resume=False):
    """Optional opt-in mirror; local artifacts remain authoritative."""
    project = os.getenv("WANDB_PROJECT")
    mode = os.getenv("WANDB_MODE", "disabled")
    if not project or mode == "disabled":
        return None
    if mode not in {"online", "offline"}:
        raise ValueError("WANDB_MODE must be disabled, offline, or online")
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B is enabled; install caiso[logging] or set WANDB_MODE=disabled"
        ) from exc
    return wandb.init(
        project=project,
        entity=os.getenv("WANDB_ENTITY") or None,
        mode=mode,
        dir=str(directory),
        id=digest(str(Path(directory).resolve()))[:16],
        resume="allow" if resume else None,
        config=config,
    )


class RunAccounting:
    def __init__(self, directory, wandb_run=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run

    def event(self, event, **fields):
        append_jsonl(self.directory / ".internal/logs/events.jsonl", {"event": event, **fields})

    def iteration(self, record, budget):
        self.event("iteration", **record, budget=budget.model_dump())
        row = {k: record[k] for k in ["iteration", "score", "accepted", "selected_block"]}
        row.update({k: v for k, v in budget.model_dump().items() if not isinstance(v, dict)})
        self._csv("history.csv", row)
        if self.wandb_run:
            self.wandb_run.log(row)

    def _csv(self, name, row):
        path = self.directory / name
        exists = path.exists()
        fields = list(row)
        if exists:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                previous_fields = reader.fieldnames or []
                if previous_fields != fields:
                    # Older runs have fewer budget columns. Preserve their rows
                    # and expand the header before appending a new-format row.
                    fields = list(dict.fromkeys([*previous_fields, *fields]))
                    previous_rows = list(reader)
                    temporary = path.with_suffix(".csv.tmp")
                    with temporary.open("w", newline="") as target:
                        writer = csv.DictWriter(target, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(previous_rows)
                    os.replace(temporary, path)
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
