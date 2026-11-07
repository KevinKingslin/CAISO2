"""Configuration snapshots and queued logs must stay inside one live report."""

import io
import json
import logging
import sys
from types import SimpleNamespace

import pytest
from rich.console import Console

from caiso.core.budget import BudgetTracker
from caiso.core.configuration import Configuration, ConfigurationType
from caiso.evaluation.iteration_report import IterationReport
from caiso.utils.logger import setup_logger


def state(prompt="Original instruction"):
    return SimpleNamespace(configurations={
        name: Configuration(kind=ConfigurationType.PROMPT, value=prompt)
        for name in ("question_rewriter", "info_extractor", "wikipedia_retriever", "hint_generator", "answer_generator")
    })


def report(tmp_path, console=None):
    return IterationReport(tmp_path, {"reporting": {"plots": False}, "max_iterations": 10}, BudgetTracker(5), console=console)


def test_real_queued_console_handler_is_captured_and_restored(tmp_path):
    terminal = io.StringIO()
    output = report(tmp_path, Console(file=terminal, force_terminal=False))
    logger = setup_logger("test.caiso.report." + tmp_path.name)
    handler = logger.listener.handlers[0]
    handler.setStream(terminal)
    class DynamicStderrHandler(logging.StreamHandler):
        def __init__(self):
            logging.Handler.__init__(self)

        @property
        def stream(self):
            return sys.stderr

    # Transformers also installs a read-only, dynamic stderr handler. Capture
    # must coexist with it instead of trying to assign its stream property.
    logger.addHandler(DynamicStderrHandler())
    try:
        with output.capture_output():
            output.begin(0, 0.3, 0.3)
            logger.info("HintGeneratorModule variable updated: original => candidate")
            logger.info("HintGeneratorModule variable updated: candidate => original")
            logger.listener.queue.join()
            output.update("Validating", phase_dir="05_block_optimization")
            logger.info("Next phase message")
        assert terminal.getvalue() == ""
        log = (tmp_path / "iterations/0001/01_attribution/model.log").read_text()
        assert "original => candidate" in log
        assert "candidate => original" in log
        assert "Next phase message" in (tmp_path / "iterations/0001/05_block_optimization/model.log").read_text()
        assert handler.stream is terminal
        logger.info("After capture")
        logger.listener.queue.join()
        assert "After capture" in terminal.getvalue()
    finally:
        logger.listener.stop()
        for queued in list(logger.handlers):
            logger.removeHandler(queued)
            queued.close()


def test_original_current_and_rejected_candidate_are_independent_full_snapshots(tmp_path):
    output = report(tmp_path)
    original = state()
    current = state("Previously accepted instruction")
    candidate = state("New candidate " + "complete prompt " * 150)
    output.configuration_state(original=original, current=current)
    output.begin(0, 0.3, 0.3, state=current)
    output.configuration_state(candidate=candidate, status="Selected for validation")
    candidate.configurations["hint_generator"].value = "Later mutation"
    output.configuration_state(current=current, status="Rejected")
    saved = json.loads((tmp_path / "iterations/0001/07_state/component_configurations.json").read_text())
    assert saved["original"]["hint_generator"]["value"] == "Original instruction"
    assert saved["current"]["hint_generator"]["value"] == "Previously accepted instruction"
    assert saved["candidate"]["hint_generator"]["value"] == "New candidate " + "complete prompt " * 150
    assert saved["candidate_status"] == "Rejected"
    output.begin(1, 0.3, 0.3, state=current)
    following = output.snapshot()["component_configurations"]
    assert following["original"] == saved["original"]
    assert following["current"] == saved["current"]
    assert following["candidate"] is None


@pytest.mark.parametrize("width,height", [(120, 32), (80, 24)])
def test_nonterminal_prints_one_final_box_with_all_component_columns(tmp_path, width, height):
    terminal = io.StringIO()
    output = report(tmp_path, Console(file=terminal, force_terminal=False, width=width, height=height))
    initial = state("Original full instruction " * 20)
    candidate = state("Candidate full instruction " * 20)
    output.configuration_state(original=initial, current=initial)
    output.start()
    output.begin(0, 0.3, 0.3, state=initial)
    output.configuration_state(candidate=candidate, status="Selected for validation")
    for i in range(5):
        output.update(progress=f"Candidate validation {i}/5")
    assert terminal.getvalue() == ""
    assert output.live is None
    output.close()
    rendered = terminal.getvalue()
    assert rendered.count("Current iteration") == 1
    for label in ("Original", "Current", "Candidate", *initial.configurations):
        assert label in rendered
    assert "Selected for validation" in rendered
    assert "Candidate validation 4/5" in rendered
    assert "component_configurations.json" in rendered
    assert "\x1b" not in rendered
    assert len(rendered.splitlines()) <= height


def test_dumb_terminal_never_attempts_live_redraws(tmp_path):
    terminal = io.StringIO()
    output = report(tmp_path, Console(file=terminal, force_terminal=True, _environ={"TERM": "dumb"}))
    output.start()
    assert output.live is None
    output.update("Working")
    output.close()
    assert terminal.getvalue().count("Current iteration") == 1
