import json

import pytest

from caiso.arch.base import BaseComponent
from caiso.core.content_filter import ContentFilteredError
from caiso.eval import eval_system
from scripts.results_layout import write_evaluation


class System:
    max_eval_workers = 2
    rm = None

    def __init__(self, path, error=None):
        self.log_dir, self.error = path, error

    def evaluate(self, example, return_pred=True):
        if example == "blocked":
            raise self.error or ContentFilteredError("explicit filter")
        return 1.0, example


def test_filtered_examples_count_as_zero_and_have_persistent_coverage(tmp_path):
    metrics, predictions = eval_system(System(tmp_path), ["ok", "blocked"], 2)
    assert metrics["mean"] == 0.5
    assert predictions == ["ok", None]
    assert metrics["coverage"] == {"requested": 4, "completed": 2, "skipped_content_filter": 2}
    skips = [
        json.loads(line)
        for line in (tmp_path / ".internal/logs/skipped_examples.jsonl").read_text().splitlines()
    ]
    assert len(skips) == 2
    assert {row["repeat"] for row in skips} == {0, 1}
    assert all(row["example_id"] == "test-1" for row in skips)
    report = json.loads(
        write_evaluation(tmp_path, metrics, variant="unoptimized", split="test").read_text()
    )
    assert report["coverage"] == metrics["coverage"]
    assert report["per_repeat"] == metrics["per_repeat"]


def test_all_filtered_batch_has_zero_score(tmp_path):
    metrics, _ = eval_system(System(tmp_path), ["blocked"])
    assert metrics["mean"] == 0
    assert metrics["coverage"]["completed"] == 0


def test_unrelated_failures_stop_evaluation(tmp_path):
    with pytest.raises(RuntimeError, match="connection broke"):
        eval_system(System(tmp_path, RuntimeError("connection broke")), ["blocked"])


def test_successful_evaluation_preserves_original_scores(tmp_path):
    system = System(tmp_path)
    metrics, predictions = eval_system(system, ["first", "second"], 3)
    assert metrics["mean"] == 1
    assert metrics["std"] == 0
    assert [metrics[f"trial_{i}"] for i in range(3)] == [1, 1, 1]
    assert predictions == ["first", "second"]
    assert metrics["coverage"] == {"requested": 6, "completed": 6, "skipped_content_filter": 0}


def test_original_component_does_not_retry_content_filter():
    class Filtered(BaseComponent):
        def __init__(self):
            self.calls = 0
            super().__init__(
                description="test",
                input_fields=["question"],
                output_fields=["answer"],
                variable="test",
            )

        def forward(self, **inputs):
            self.calls += 1
            raise ContentFilteredError("explicit filter")

    component = Filtered()
    with pytest.raises(ContentFilteredError):
        component(question="test")
    assert component.calls == 1


def test_real_dspy_pipeline_skips_policy_response_without_resampling(monkeypatch, tmp_path):
    import litellm
    from caiso.azure_env import install_azure_routing
    from caiso.providers.transport import configure_transport
    from caiso.wrappers.example import Example
    from examples.systems.hotpotqa.five_components import system_engine

    monkeypatch.setenv("LLM_BACKEND", "azure")
    monkeypatch.setenv("AZURE_GPT_4O_MINI_DEPLOYMENT", "test-mini")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://test.invalid/")
    monkeypatch.setenv("AZURE_API_VERSION", "2024-10-21")
    calls = []

    def reject(**kwargs):
        calls.append(kwargs)
        return litellm.ModelResponse(
            choices=[
                {"finish_reason": "content_filter", "message": {"role": "assistant", "content": ""}}
            ]
        )

    monkeypatch.setattr(litellm, "completion", reject)
    install_azure_routing()
    configure_transport(tmp_path)
    try:
        system = system_engine(log_dir=str(tmp_path))
        metrics, _ = eval_system(system, [Example(question="Question", gd_answer="answer")])
    finally:
        configure_transport()
    assert len(calls) == 1
    assert metrics["mean"] == 0
    assert metrics["coverage"]["skipped_content_filter"] == 1
