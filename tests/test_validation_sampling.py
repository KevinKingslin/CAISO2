"""Seeded acceptance samples must come from the complete validation pool."""

import random

import pytest

from caiso.method.algorithm import batch, validation_batch
from caiso.optim.optimizer import CaisoOptimizer
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example
from tests.test_runtime import make_system


@pytest.mark.parametrize("sampling,size,pool_size", [
    ("first", 5, 5),
    ("seeded", 5, 250),
    ("first", "all", 250),
    ("seeded", "all", 250),
])
def test_validation_mode_reaches_the_correct_pool(tmp_path, sampling, size, pool_size):
    optimizer = CaisoOptimizer(
        make_system(), [Example(question="1000", gd_answer="1000")],
        [Example(question=str(i), gd_answer=str(i)) for i in range(250)],
        {"dataset": "hotpotqa", "caiso": {"method": {
            "validation_sampling": sampling, "validation_batch_size": size,
        }}},
        tmp_path, providers=ProviderRegistry({}),
    )
    assert len(optimizer.validation) == pool_size
    selected = validation_batch(optimizer.validation, optimizer.config, 42)
    if size == "all":
        expected = list(range(250))
    elif sampling == "first":
        expected = list(range(size))
    else:
        expected = random.Random(42).sample(range(250), size)
        assert any(i >= size for i in expected)
        assert validation_batch(optimizer.validation, optimizer.config, 43) != selected
    assert [example.example_id for example in selected] == [f"validation-{i}" for i in expected]
    # Baseline and candidate acceptance reuse the same fixed sample on every
    # iteration. Attribution/diagnostics can draw separate mini-batches.
    assert validation_batch(optimizer.validation, optimizer.config, 42) == selected
    assert len(batch(optimizer.validation, 20, 43)) == min(20, pool_size)
