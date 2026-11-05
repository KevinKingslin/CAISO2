"""Parallel benchmark evaluation must keep each example's prediction isolated."""

from threading import Barrier

import pytest

from caiso.arch.base import BaseComponent
from caiso.arch.system import CompoundAISystem
from caiso.eval import eval_system
from caiso.wrappers.example import Example


@pytest.mark.parametrize("workers", [2, 10])
def test_parallel_evaluation_keeps_predictions_and_trajectories_in_input_order(workers):
    ready = Barrier(workers)

    class Echo(BaseComponent):
        def __init__(self):
            super().__init__("Echo input", ["question"], ["answer"], num_retry=1, retry_after=0)

        def forward(self, **inputs):
            return {"answer": inputs["question"]}

    class OverlappingSystem(CompoundAISystem):
        def _single_call(self, component, component_inputs):
            outputs = super()._single_call(component, component_inputs)
            # Finish all component calls before the system collects trajectories.
            # This forces the overlap that can otherwise depend on thread timing.
            ready.wait(timeout=10)
            return outputs

    system = OverlappingSystem(
        {"echo": Echo()},
        final_output_fields=["answer"],
        ground_fields=["gd_answer"],
        eval_func=lambda answer, gd_answer: float(answer == gd_answer),
        max_eval_workers=workers,
    )
    examples = [Example(question=str(i), gd_answer=str(i)) for i in range(workers)]
    metrics, predictions = eval_system(system, examples, num_repeat=2, show_progress=False)

    assert metrics["mean"] == 1.0
    assert [prediction.answer for prediction in predictions] == [example.question for example in examples]
    for example, prediction in zip(examples, predictions):
        assert prediction.traj["echo"]["input"] == {"question": example.question}
        assert prediction.traj["echo"]["output"] == {"answer": example.question}
