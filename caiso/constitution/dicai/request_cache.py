"""Checkpoint successful induction requests so a failed turn can resume cheaply."""

from pathlib import Path
from threading import Lock

from caiso.core.serialization import digest, read_json, save_json


class RequestCache:
    def __init__(self, providers, directory=None):
        self.providers = providers
        self.directory = Path(directory) if directory else None
        self.hits = self.requests = 0
        self.lock = Lock()

    def __getattr__(self, name):
        return getattr(self.providers, name)

    def complete(self, role, messages, *, seed, schema=None, validate=None, response_context=None, **kwargs):
        budget = getattr(getattr(self.providers, "recovery", None), "budget", None)
        if budget:
            budget.check()
        identity = digest({
            "role": role, "messages": messages, "seed": seed,
            "schema": schema.model_json_schema() if schema else None,
            "generation": kwargs,
        })
        path = self.directory / (identity + ".json") if self.directory else None
        if path and path.exists():
            record = read_json(path)
            if record.get("request") != identity or record.get("fingerprint") != digest(record.get("result")):
                raise ValueError(f"Corrupted DICAI request checkpoint: {path}")
            result = schema.model_validate(record["result"]) if schema else record["result"]
            result = validate(result) if validate else result
            with self.lock:
                self.hits += 1
            if budget:
                with budget.lock:
                    budget.usage.cache_hits += 1
            return result
        with self.lock:
            self.requests += 1
        result = self.providers.complete(
            role, messages, seed=seed, schema=schema, validate=validate,
            response_context=response_context, **kwargs,
        )
        if path:
            save_json(path, {"request": identity, "result": result, "fingerprint": digest(result)})
        return result
