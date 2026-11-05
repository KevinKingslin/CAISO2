"""Original task scorers with explicit failures and content-filter coverage."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from caiso.core.content_filter import ContentFilteredError, record_skip
from caiso.core.parallel import ordered_map


def eval_system(system, testset, num_repeat=1):
    """Evaluate unchanged examples/scorers; only explicit policy rejections skip.

    Skips count as zero in the requested denominator. Other failures stop the
    evaluation instead of disappearing from the average. Predictions correspond
    to input order on the last repeat, with None for filtered examples.
    """
    if not testset or num_repeat < 1:
        raise ValueError("Evaluation requires examples and a positive repetition count")
    log = getattr(system, "_caiso_skip_log", None)
    if log is None and getattr(system, "log_dir", None):
        log = Path(system.log_dir) / ".internal/logs/skipped_examples.jsonl"
    recorder = SimpleNamespace(content_filter_log=log)
    workers = getattr(system, "max_eval_workers", 4)
    if getattr(system, "rm", None) is not None:
        workers = 1
    split = getattr(system, "_caiso_eval_split", "test")
    metrics, per_repeat = {}, []
    for repeat in range(num_repeat):

        def evaluate(item):
            index, example = item
            try:
                result = system.evaluate(example, return_pred=True)
            except ContentFilteredError as error:
                record_skip(
                    recorder,
                    error,
                    "benchmark_evaluation",
                    example_id=f"{split}-{index}",
                    split=split,
                    repeat=repeat,
                    system_variant=getattr(system, "_caiso_eval_variant", "benchmark"),
                )
                return 0.0, None, True
            if result is None:
                raise RuntimeError(f"Evaluation returned no result for {split}-{index}")
            score, prediction = result if isinstance(result, tuple) else (result, None)
            if not np.isfinite(score):
                raise ValueError(f"Non-finite task score for {split}-{index}: {score}")
            return float(score), prediction, False

        results = ordered_map(evaluate, enumerate(testset), workers)
        score = sum(row[0] for row in results) / len(testset)
        skipped = sum(row[2] for row in results)
        metrics[f"trial_{repeat}"] = score
        per_repeat.append(
            {
                "repeat": repeat,
                "score": score,
                "requested": len(testset),
                "completed": len(testset) - skipped,
                "skipped_content_filter": skipped,
            }
        )
        preds = [row[1] for row in results]
    metrics.update(
        mean=float(np.mean([row["score"] for row in per_repeat])),
        std=float(np.std([row["score"] for row in per_repeat])),
        per_repeat=per_repeat,
        coverage={
            key: sum(row[key] for row in per_repeat)
            for key in ("requested", "completed", "skipped_content_filter")
        },
        failure_policy="Content-filtered examples count as zero; all other errors stop evaluation.",
    )
    return metrics, preds
