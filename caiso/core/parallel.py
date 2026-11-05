"""Bounded independent work; results and seeds retain their input order."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def ordered_map(function, items, max_workers=1, progress=None):
    items = list(items)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    results = [None] * len(items)
    completed = 0
    if progress:
        progress(0, len(items))
    if max_workers == 1:
        for index, item in enumerate(items):
            results[index] = function(item)
            completed += 1
            if progress:
                progress(completed, len(items))
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending, next_index = {}, 0

        def submit():
            nonlocal next_index
            future = pool.submit(function, items[next_index])
            pending[future] = next_index
            next_index += 1

        for _ in range(min(max_workers, len(items))):
            submit()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    results[pending.pop(future)] = future.result()
                    completed += 1
                    if progress:
                        progress(completed, len(items))
                while next_index < len(items) and len(pending) < max_workers:
                    submit()
        finally:
            # No new jobs after an error. Running requests finish and are accounted
            # before the caller skips the iteration or closes its logs.
            for future in pending:
                future.cancel()
    return results
