"""Original task scorers with explicit failures and content-filter coverage."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from caiso.core.content_filter import ContentFilteredError, record_skip
from caiso.core.parallel import ordered_map


def eval_system(system, testset, num_repeat=1, *, show_progress=True):
    """Evaluate unchanged examples/scorers; only explicit policy rejections skip.

    Skips count as zero in the requested denominator. Other failures stop the
    evaluation instead of disappearing from the average. Predictions correspond
    to input order on the last repeat, with None for filtered examples.
    Progress counts finished examples (including skips) as parallel jobs complete.
    Each completed repeat prints its score and coverage before the next starts.
    Set show_progress=False to suppress progress bars and repeat summaries.
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

        with tqdm(
            total=len(testset),
            desc=f"Evaluating {split} (repeat {repeat + 1}/{num_repeat})",
            unit="example",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as bar:
            results = ordered_map(
                evaluate,
                enumerate(testset),
                workers,
                progress=lambda completed, total: bar.update(completed - bar.n),
            )
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
        if show_progress:
            print(
                f"Evaluation {split} repeat {repeat + 1}/{num_repeat}: score={score:.6f} "
                f"(completed={len(testset) - skipped}/{len(testset)}, "
                f"content-policy skips={skipped})",
                file=sys.stderr,
                flush=True,
            )
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
