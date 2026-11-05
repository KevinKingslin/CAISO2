"""Observable pair scheduling; model calls never run under this lock."""

from threading import RLock
from time import monotonic


class InductionActivity:
    def __init__(self, pairs, pair_workers, reasoning_workers, callback=None):
        self.lock, self.callback = RLock(), callback
        self.pair_workers, self.reasoning_workers = pair_workers, reasoning_workers
        self.records = [{"pair_id": pair.pair_id, "stage": "queued"} for pair in pairs]
        self.started = {}
        self.peak_active_records = self.peak_active_debates = 0

    def update(self, index, stage, **details):
        with self.lock:
            if index not in self.started:
                self.started[index] = monotonic()
            self.records[index] = {
                "pair_id": self.records[index]["pair_id"],
                "stage": stage,
                "seconds": monotonic() - self.started[index],
                **details,
            }
            state = self.snapshot()
            self.peak_active_records = max(self.peak_active_records, len(state["active_records"]))
            self.peak_active_debates = max(
                self.peak_active_debates,
                sum(record["stage"] == "debate" for record in state["active_records"]),
            )
            if self.callback:
                self.callback(self.snapshot())

    def snapshot(self):
        with self.lock:
            records = [dict(record) for record in self.records]
            return {
                "pair_workers": self.pair_workers,
                "reasoning_workers": self.reasoning_workers,
                "records_total": len(records),
                "records_completed": sum(r["stage"] in {"completed", "cached"} for r in records),
                "active_records": [r for r in records if r["stage"] in {"reasoning", "debate"}],
                "peak_active_records": self.peak_active_records,
                "peak_active_debates": self.peak_active_debates,
                "records": records,
            }
