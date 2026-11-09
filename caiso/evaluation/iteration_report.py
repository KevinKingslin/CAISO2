"""One live panel and one self-contained folder per optimization iteration.

This module only observes the algorithm. It never calls a model or changes a score.
"""

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from time import monotonic, time_ns

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from caiso.core.serialization import append_jsonl, canonical, plain, read_json, save_json
from caiso.evaluation.console_logging import capture_console_logging
from caiso.evaluation.phase_artifacts import save_phase_artifacts
from caiso.evaluation.results import update_result, write_overview


def duration(seconds):
    return str(timedelta(seconds=max(0, int(seconds))))


class IterationReport:
    def __init__(self, directory, config, budget, console=None):
        self.output = Path(directory)
        self.config, self.budget = config, budget
        self.options = config.get("reporting", {})
        self.console = console or Console(file=sys.stdout, no_color=False)
        self.live = None
        self.lock = RLock()
        self.folder = None
        self.phase_dir = "setup"
        self.started = monotonic()
        self.phase_started = self.started
        self.data = {"phase": "Initializing system constitution and baseline"}
        self.component_configurations = {"original": {}, "current": {}, "candidate": None, "candidate_status": "Not proposed yet"}
        # Reading completed reports makes plot history and ETA survive a resume.
        summaries = [read_json(p) for p in sorted(self.output.glob("iterations/*/summary.json"))]
        self.history = [
            self.history_row(r) for r in summaries if r.get("status") in {"completed", "skipped"}
        ]

    @staticmethod
    def history_row(summary):
        """ETA and plotting need only scalar history, not full states and transcripts."""
        fields = (
            "iteration",
            "display_iteration",
            "status",
            "iteration_seconds",
            "budget_delta",
            "score",
            "best_score",
            "candidate_score",
        )
        row = {key: summary[key] for key in fields if key in summary}
        workflow = summary.get("workflow")
        row["workflow"] = {"candidate_score": workflow.get("candidate_score")} if workflow else None
        return row

    @contextmanager
    def capture_output(self):
        """Route Python model/training chatter to the current iteration's log."""
        if not self.options.get("capture_model_output", True):
            yield
            return
        report = self

        class ModelLog:
            encoding = "utf-8"

            def __init__(self):
                self.stream = None
                self.path = None
                self.closed = False
                self.detached = False
                self.lock = RLock()

            def current_stream(self):
                if self.closed:
                    raise ValueError("I/O operation on closed model log")
                if self.detached:
                    path = report.output / ".internal/logs/engine.log"
                else:
                    path = (
                        report.folder / report.phase_dir / "model.log"
                        if report.folder
                        else report.output / ".internal/logs/setup.log"
                    )
                if path != self.path or self.stream is None or self.stream.closed:
                    if self.stream is not None:
                        self.stream.close()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.stream = path.open("a", encoding="utf-8", buffering=1)
                    self.path = path
                return self.stream

            def write(self, text):
                with self.lock:
                    stream = self.current_stream()
                    written = stream.write(text)
                    stream.flush()
                    return written

            def fileno(self):
                # vLLM/native libraries duplicate and temporarily redirect this FD.
                # It must remain open, including before the first Python write.
                with self.lock:
                    return self.current_stream().fileno()

            def flush(self):
                with self.lock:
                    if self.stream is not None and not self.stream.closed:
                        self.stream.flush()

            def close(self):
                with self.lock:
                    if self.stream is not None:
                        self.stream.close()
                    self.closed = True

            def detach(self):
                # vLLM/logging handlers may retain this stream after capture ends.
                # Close the phase file, but allow late worker cleanup to keep logging.
                with self.lock:
                    if self.stream is not None:
                        self.stream.close()
                    self.detached = True

            def __del__(self):
                self.close()

            def isatty(self):
                return False

            def writable(self):
                return True

        # Separate FDs ensure suppressing stdout cannot hide Python stderr errors.
        stdout, stderr = ModelLog(), ModelLog()
        original_stdout, original_stderr = sys.stdout, sys.stderr
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr), capture_console_logging(stdout, stderr, self.console.file, original_stdout, original_stderr):
                yield
        finally:
            stdout.detach()
            stderr.detach()

    def start(self):
        write_overview(self.output)
        if self.options.get("cli", True) and self.console.is_terminal and not self.console.is_dumb_terminal:
            # Redirected output gets a final summary only, without terminal escape codes.
            self.live = Live(
                self,
                console=self.console,
                refresh_per_second=self.options.get("refresh_per_second", 2),
                transient=True,
                redirect_stdout=False,
                redirect_stderr=False,
                vertical_overflow="crop",
            )
            self.live.start()

    def close(self):
        if self.live:
            self.live.stop()
        if self.options.get("cli", True):
            self.console.print(self)

    def configuration_state(self, *, original=None, current=None, candidate=None, status=None, clear_candidate=False):
        """Observe canonical states, never temporary component apply/restore logs."""
        def configurations(state):
            values = getattr(state, "configurations", None)
            if values is None:
                values = state.get("configurations", state)
            return deepcopy(plain(values))

        with self.lock:
            for key, state in (("original", original), ("current", current), ("candidate", candidate)):
                if state is not None:
                    self.component_configurations[key] = configurations(state)
            if clear_candidate:
                self.component_configurations["candidate"] = None
            if status is not None:
                self.component_configurations["candidate_status"] = status
            snapshot = deepcopy(self.component_configurations)
            if self.folder:
                self.save_phase("07_state", component_configurations=snapshot)
        self.update(component_configurations=snapshot)

    def error_event(self, event):
        """Always persist the full diagnostic; show its concise version in red."""
        with self.lock:
            entry = {"iteration": self.data.get("iteration"), "phase": self.data["phase"], **event}
            append_jsonl(self.output / ".internal/logs/errors.jsonl", entry)
            if self.folder:
                append_jsonl(self.folder / self.phase_dir / "errors.jsonl", entry)
            else:
                path = self.output / "setup.json"
                setup = read_json(path) if path.exists() else {}
                update_result(path, errors=[*setup.get("errors", []), entry])
            summary = {k: entry[k] for k in ("stage", "action", "exception_type", "message")}
            summary.update({k: entry[k] for k in ("attempt", "max_attempts") if k in entry})
            self.data.setdefault("errors", []).append(summary)
            if self.folder:
                save_json(self.folder / "summary.json", self.snapshot())
        if self.live and self.console.is_terminal:
            self.live.refresh()
        else:
            color = "bold green" if summary["action"] == "recovered" else "bold red"
            self.console.print(Text(self.error_line(summary), style=color))

    @staticmethod
    def error_line(error):
        message = " ".join(error["message"].split())[:240]
        attempt = f" {error['attempt']}/{error['max_attempts']}" if "attempt" in error else ""
        label = "RECOVERED" if error["action"] == "recovered" else "ERROR"
        return f"{label} [{error['action']}{attempt}] {error['stage']}: {error['exception_type']}: {message}"

    def begin(self, iteration, score, best_score, state=None):
        self.folder = self.output / "iterations" / f"{iteration + 1:04d}"
        if self.folder.exists() and any(self.folder.iterdir()):
            # A hard interruption can leave phase results beyond the last
            # checkpoint. Keep that attempt, but do not mix its decisions into
            # the resumed attempt if a later substep fails this time.
            previous = self.output / ".internal/attempts" / self.folder.name / str(time_ns())
            previous.parent.mkdir(parents=True, exist_ok=True)
            self.folder.rename(previous)
        self.history = [row for row in self.history if row["iteration"] < iteration]
        self.folder.mkdir(parents=True, exist_ok=True)
        self.started = monotonic()
        self.phase_started = self.started
        if state is not None:
            self.component_configurations["current"] = deepcopy(plain(state.configurations))
        self.component_configurations.update(candidate=None, candidate_status="Not proposed yet")
        self.data = {
            "iteration": iteration,
            "display_iteration": iteration + 1,
            "max_iterations": self.config.get("max_iterations"),
            "status": "running",
            "phase": "Causal edge attribution",
            "score_before": score,
            "score": score,
            "best_score": best_score,
            "blocks": {},
            "phase_timings": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "component_configurations": deepcopy(self.component_configurations),
        }
        self.phase_dir = "01_attribution"
        self.save_phase("07_state", method_config=self.config, component_configurations=self.component_configurations)
        self.update()
        write_overview(self.output)
        return self.folder

    def update(self, phase=None, *, phase_dir=None, **fields):
        with self.lock:
            self.data.update(fields)
            if phase:
                if phase_dir and phase_dir != self.phase_dir:
                    self.save_phase(self.phase_dir, status="completed")
                    self.phase_dir = phase_dir
                if "progress" not in fields:
                    self.data.pop("progress", None)
                if "induction_activity" not in fields:
                    self.data.pop("induction_activity", None)
                if self.folder:
                    self.data["phase_timings"].append(
                        {
                            "phase": self.data["phase"],
                            "seconds": monotonic() - self.phase_started,
                        }
                    )
                self.phase_started = monotonic()
                self.data["phase"] = phase
            if self.folder:
                save_json(self.folder / "summary.json", self.snapshot())
            if self.folder or phase not in {
                "Loading saved checkpoint",
                "No further iterations under saved limits",
            }:
                self.save_phase(
                    self.phase_dir,
                    phase=self.data["phase"],
                    status=self.data.get("status", "running")
                    if not self.data["phase"].startswith("Stopped:")
                    else "failed",
                    progress=self.data.get("progress"),
                    induction_activity=self.data.get("induction_activity"),
                )
        if self.live:
            self.live.refresh()

    def induction_activity(self, state):
        self.update(induction_activity=state)

    def save_phase(self, name, **fields):
        """Save each result as its own named artifact inside the phase folder."""
        with self.lock:
            if self.folder:
                save_phase_artifacts(self.folder / name, name, fields)
            else:
                update_result(self.output / "setup.json", **fields)

    def save_system_induction(self, constitution, preferences):
        """Keep the latest constitution and every refresh as separate JSON files."""
        self.save_phase(
            "00_system_constitution", preferences=preferences, inductions=[constitution]
        )

    def eta(self):
        successful = [r for r in self.history if r.get("status") == "completed"]
        if not successful:
            return None, "waiting for a completed iteration"
        average = sum(r["iteration_seconds"] for r in successful) / len(successful)
        elapsed = monotonic() - self.started
        remaining_current = max(0, average - elapsed) if self.data.get("status") == "running" else 0
        candidates = [
            (
                max(0, self.budget.max_seconds - self.budget.snapshot().elapsed_seconds),
                "wall-clock cap",
            )
        ]
        total = self.config.get("max_iterations")
        completed = self.data.get("iteration", -1) + 1
        if total is not None:
            candidates.append(
                (remaining_current + max(0, total - completed) * average, "iteration rate")
            )
        for field, limit in (
            ("equivalent_full_system_runs", self.budget.max_runs),
            ("model_calls", self.budget.max_calls),
        ):
            used = sum(r["budget_delta"].get(field, 0) for r in successful)
            if used > 0:
                usage = self.budget.snapshot()
                current = (
                    usage.llm_calls + usage.embedding_calls
                    if field == "model_calls"
                    else usage.equivalent_full_system_runs
                )
                candidates.append(
                    (max(0, limit - current) / (used / len(successful)) * average, field)
                )
        return min(candidates, key=lambda item: item[0])

    def snapshot(self):
        with self.lock:
            data = deepcopy(self.data)
        seconds, basis = self.eta()
        return {
            **data,
            "iteration_seconds": monotonic() - self.started,
            "eta_seconds": seconds,
            "eta_basis": basis,
            "estimated_completion": (
                (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(
                    timespec="seconds"
                )
                if seconds is not None
                else None
            ),
            "budget": self.budget.snapshot().model_dump(),
        }

    def render_text(self, include_configurations=True):
        info = self.snapshot()
        number = info.get("display_iteration", "setup")
        total = info.get("max_iterations")
        label = str(total) if total is not None else "budget limited"
        lines = [f"CAISO | Iteration {number} / {label}", f"Phase: {info['phase']}"]
        if info.get("progress"):
            lines.append(info["progress"])
        activity = info.get("induction_activity")
        if activity:
            provider = activity.get("provider")
            if provider:
                label = provider["alias"]
                if provider.get("deployment"):
                    label += f" (deployment: {provider['deployment']})"
                lines.append(f"DICAI provider: {label}")
            active = activity["active_records"]
            lines.append(
                f"Active preference records: {len(active)}/{activity['pair_workers']} workers | "
                f"Completed: {activity['records_completed']}/{activity['records_total']} | "
                f"Peak concurrent debates: {activity['peak_active_debates']}"
            )
            for record in active:
                detail = record["stage"]
                if detail == "debate":
                    detail = f"cycle {record['cycle']}/{record['total_cycles']}, {record['turn']}"
                lines.append(f"  {record['pair_id']}: {detail}")
        eta = duration(info["eta_seconds"]) if info["eta_seconds"] is not None else "estimating"
        lines.append(
            f"Elapsed: {duration(info['iteration_seconds'])} | ETA: {eta} ({info['eta_basis']})"
        )
        if info.get("estimated_completion"):
            lines.append(f"Estimated finish (UTC): {info['estimated_completion']}")
        if "score" in info:
            lines.append(f"Validation: {info['score']:.4f} | Best: {info['best_score']:.4f}")
        if info.get("constitution_reconstruction_accuracy") is not None:
            lines.append(
                "Constitution reconstruction (training): "
                f"{info['constitution_reconstruction_accuracy']:.1%}"
            )
        for name, members in info.get("blocks", {}).items():
            selected = " [selected]" if name == info.get("selected_block") else ""
            lines.append(f"  {name}: {', '.join(members)}{selected}")
        if "accepted" in info:
            lines.append(
                f"Block update: {'accepted' if info['accepted'] else 'rejected / unchanged'}"
            )
        if info.get("candidate_score") is not None:
            lines.append(f"Block candidate validation: {info['candidate_score']:.4f}")
        if info.get("vote_fractions"):
            lines.append(
                "Selection votes: "
                + ", ".join(f"{b}={s:.0%}" for b, s in info["vote_fractions"].items())
            )
        workflow = info.get("workflow")
        if workflow:
            lines.append(f"Workflow update: {'accepted' if workflow['accepted'] else 'rejected'}")
        usage = info["budget"]
        lines.append(
            f"Budget: {usage['equivalent_full_system_runs']:.1f}/{self.budget.max_runs:g} runs | "
            f"{usage['llm_calls'] + usage['embedding_calls']}/{self.budget.max_calls} model calls"
        )
        if self.folder:
            lines.append(f"Saved: {self.folder}")
        errors = info.get("errors", [])
        if errors:
            count = sum(e["action"] != "recovered" for e in errors)
            lines.append(
                f"Errors in this iteration: {count} (full details in the phase errors.jsonl)"
            )
            lines.append(self.error_line(errors[-1]))
        if include_configurations:
            rows = self.configuration_rows(info)
            if rows:
                lines.append("Component configurations | Candidate: " + info["component_configurations"]["candidate_status"])
                for cid, original, current, candidate in rows:
                    lines.extend((f"  {cid}", f"    Original: {original}", f"    Current: {current}", f"    Candidate: {candidate}"))
        return "\n".join(lines)

    def render_styled(self, include_configurations=True):
        text = Text(self.render_text(include_configurations))
        start = text.plain.find("ERROR [")
        if start >= 0:
            text.stylize("bold red", start, len(text))
        recovered = text.plain.find("RECOVERED [")
        if recovered >= 0:
            text.stylize("bold green", recovered, len(text))
        return text

    @staticmethod
    def configuration_rows(info):
        configurations = info.get("component_configurations", {})
        original, current, candidate = (configurations.get(key) or {} for key in ("original", "current", "candidate"))

        def describe(config):
            if config is None:
                return "Inactive"
            value = config["value"]
            return "Fixed" if config["kind"] == "NONE" else (value if isinstance(value, str) else canonical(value))

        return [
            (
                cid, describe(original.get(cid)),
                "Same as original" if cid in original and current.get(cid) == original[cid] else describe(current.get(cid)),
                "Not proposed yet" if configurations.get("candidate") is None
                else "Same as current" if cid in current and candidate.get(cid) == current[cid]
                else describe(candidate.get(cid)),
            )
            for cid in dict.fromkeys([*original, *current, *candidate])
        ]

    def __rich__(self):
        info = self.snapshot()
        rows = self.configuration_rows(info)
        if not rows:
            return Panel(self.render_styled(False), title="Current iteration")
        lines = self.render_text(False).splitlines()
        priority = ("CAISO |", "Phase:", "Elapsed:", "Validation:", "Budget:", "Active preference records:", "ERROR [", "RECOVERED [")
        main = [line for line in lines if line.startswith(priority) or line == info.get("progress")]
        details = [line for line in lines if line not in main]
        # Keep every component visible even in a short terminal. Full values are
        # saved separately; terminal previews never change the actual states.
        max_lines = 2 if self.console.height >= len(main) + 2 * len(rows) + 9 else 1
        name_width = min(25, max(20, self.console.width // 5))
        value_width = max(8, (self.console.width - name_width - 18) // 3)

        def preview(value, width):
            value = " ".join(str(value).split())
            limit = width * max_lines
            if len(value) > limit:
                value = value[:limit // 2 - 1] + "…" + value[-(limit // 2):]
            wrapped = Text(value).wrap(self.console, width, overflow="fold")
            if len(wrapped) > max_lines:
                wrapped[max_lines - 1].truncate(width - 1)
                wrapped[max_lines - 1].append("…")
            return Text("\n").join(wrapped[:max_lines])

        table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
        table.add_column("Component", width=name_width)
        for name in ("Original", "Current", "Candidate"):
            table.add_column(name, ratio=1)
        for row in rows:
            table.add_row(preview(row[0], name_width), *(preview(value, value_width) for value in row[1:]))
        status = Text("Candidate: " + info["component_configurations"]["candidate_status"])
        text = Text("\n".join(main))
        if any(line.startswith("ERROR [") for line in main):
            text.highlight_regex(r"ERROR \[.*", "bold red")
        if any(line.startswith("RECOVERED [") for line in main):
            text.highlight_regex(r"RECOVERED \[.*", "bold green")
        available = max(0, self.console.height - len(main) - len(rows) * max_lines - 9)
        return Panel(Group(text, status, table, Text("Full values: 07_state/component_configurations.json"), Text("\n".join(details[:available]))), title="Current iteration")

    def finish(self, record, budget_before, state, constitution, status="completed"):
        self.configuration_state(current=state)
        self.update(
            "Completed" if status == "completed" else status, **{**record, "status": status}
        )
        summary = self.snapshot()
        before, after = budget_before.model_dump(), summary["budget"]
        summary["budget_delta"] = {
            key: after[key] - before[key] for key in after if isinstance(after[key], (int, float))
        }
        summary["budget_delta"]["model_calls"] = (
            summary["budget_delta"]["llm_calls"] + summary["budget_delta"]["embedding_calls"]
        )
        if status in {"completed", "skipped"}:
            self.history = [r for r in self.history if r["iteration"] < record["iteration"]]
            self.history.append(self.history_row(summary))
            # The first completed iteration can now contribute to the ETA.
            summary.update(self.snapshot())
        summary.update(
            accepted_state=state,
            system_principles=constitution.principles,
            system_constitution_provenance=constitution.metadata.get("provenance"),
            terminal_summary=self.render_text(),
        )
        save_json(self.folder / "summary.json", summary)
        self.save_phase(self.phase_dir, status=status)
        self.save_phase("07_state", accepted_state=state, system_constitution=constitution)
        (self.folder / "summary.txt").write_text(self.render_text() + "\n", encoding="utf-8")
        if self.options.get("plots", True):
            self.plot()
        write_overview(self.output)

    def plot(self):
        # Agg works on Linux servers without a display; no blocking plt.show().
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        setup_path = self.output / "setup.json"
        baseline_path = self.output / "baseline.json"
        baseline = (
            read_json(setup_path).get("baseline", {}).get("score") if setup_path.exists() else None
        )
        if baseline is None and baseline_path.exists():
            baseline = read_json(baseline_path)["score"]
        points = (
            [{"display_iteration": 0, "score": baseline, "best_score": baseline}]
            if baseline is not None
            else []
        )
        points += [
            {
                "display_iteration": row["display_iteration"],
                "score": row["score"],
                "best_score": row["best_score"],
                "candidate_score": row.get("candidate_score"),
                "workflow_candidate_score": (row.get("workflow") or {}).get("candidate_score"),
                "status": row["status"],
            }
            for row in self.history
        ]
        if not points:
            return
        update_result(self.folder / "summary.json", performance=points)
        save_json(self.folder / "performance.json", points)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = [p["display_iteration"] for p in points]
        ax.plot(
            x,
            [p["score"] if p["display_iteration"] == 0 else p.get("candidate_score") for p in points],
            "o-",
            label="Validation (block candidate)",
        )
        ax.plot(x, [p["best_score"] for p in points], "--", label="Best so far")
        ax.set(
            xlabel="Iteration (0 = baseline)",
            ylabel="Task metric on fixed validation batch",
            title="CAISO optimization progress",
        )
        if len(x) <= 12:
            ax.set_xticks(x)
        else:
            from matplotlib.ticker import MaxNLocator

            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.folder / "performance.png", dpi=self.options.get("plot_dpi", 150))
        plt.close(fig)
