"""Bound cached attribution across rejected/no-candidate iterations and resume."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from caiso.core.budget import BudgetExceeded
from caiso.core.checkpoint import Checkpoint
from caiso.core.recovery import ModelResponseError
from caiso.core.serialization import digest
from caiso.method import algorithm as algorithm_module
from caiso.method.config import resume_config_hashes, validate_method
from caiso.optim.optimizer import CaisoOptimizer
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example
from tests.offline import OfflineEmbeddings, OfflineProvider, training_preferences
from tests.test_runtime import make_system


class VaryingProvider(OfflineProvider):
    def __init__(self):
        super().__init__(None)
        self.proposals = 0
        self.accept_at = None
        self.repeat_proposal = False

    def complete(self, messages, *, seed, schema=None, **kwargs):
        if schema and schema.__name__ == "PromptProposal":
            self.budget.call()
            self.calls.append((messages, seed, schema))
            self.proposals += 1
            number = 1 if self.repeat_proposal else self.proposals
            prompt = "double" if self.proposals == self.accept_at else f"variant-{number}"
            return json.dumps({"prompt": prompt})
        return super().complete(messages, seed=seed, schema=schema, **kwargs)


def setup_run(tmp_path, monkeypatch, iterations, *, accept_at=None, reuse=True):
    provider, embeddings, measurements = VaryingProvider(), OfflineEmbeddings(None), []
    provider.accept_at = accept_at
    registry = ProviderRegistry({"offline": provider}, dict.fromkeys(
        ["system_constitution", "constitution_judge", "block_selection", "prompt_optimizer"], "offline",
    ))
    experiment = {
        "dataset": "pubmed", "system": "synthetic-test-only", "num_iterations": iterations,
        "val_size": 1, "per_component_train_size": 1,
        "caiso": {
            "edge_baselines": {
                "source->transform": {"number": {"value": "0"}},
                "transform->sink": {"result": {"value": "0"}},
            },
            "method": {
                "execution": {"max_workers": 1},
                "attribution": {"batch_size": 2, "bootstrap_samples": 10, "reuse_unchanged": reuse},
                "block_selection": {"diagnostic_batch_size": 1},
                "block_optimization": {"prompt_candidates": 1, "validation_repeats": 1},
                "dicai": {
                    "self_consistency_samples": 1, "self_refine_steps": 0,
                    "clustering": {"min_clusters": 2, "max_clusters": 4},
                },
                "reporting": {"cli": False, "plots": False, "capture_model_output": False},
            },
        },
    }
    original_measure = algorithm_module.estimate_all_edges

    def measure(runtime, state, examples, seeds, *args, **kwargs):
        measurements.append({
            "state": state.fingerprint(), "seeds": list(seeds),
            "ids": [example.example_id for example in examples],
            "inputs": [example.inputs for example in examples],
            "split": [example.split for example in examples],
        })
        return original_measure(runtime, state, examples, seeds, *args, **kwargs)

    monkeypatch.setattr(algorithm_module, "estimate_all_edges", measure)

    def create():
        optimizer = CaisoOptimizer(
            make_system(),
            [Example(question=str(i), gd_answer=str(2 * i)) for i in range(1, 21)],
            [Example(question="100", gd_answer="200" if accept_at else "100")],
            experiment, tmp_path, providers=registry, embeddings=embeddings,
        )
        provider.budget = embeddings.budget = optimizer.budget
        return optimizer

    return create, provider, measurements


def read_refresh(directory, iteration):
    return json.loads((directory / f"iterations/{iteration:04}/01_attribution/refresh.json").read_text())


def test_five_unsuccessful_iterations_refresh_and_reset_instead_of_remeasuring_forever(tmp_path, monkeypatch):
    create, _, measured = setup_run(tmp_path, monkeypatch, 12)
    optimizer = create()
    optimizer.optimize(preferences=training_preferences(optimizer))
    assert len(measured) == 3  # Initial measurement, then immediately after attempts 5 and 10.
    for iteration in (5, 10):
        refresh = read_refresh(tmp_path, iteration)
        assert refresh["reason"] == "unsuccessful_iteration_limit"
        assert refresh["unsuccessful_iterations_before_refresh"] == 5
        assert refresh["source_split"] == "train"
    assert set(measured[0]["ids"]).isdisjoint(measured[1]["ids"])
    assert set(measured[1]["ids"]).isdisjoint(measured[2]["ids"])
    assert len({tuple(row["seeds"]) for row in measured}) == 3
    assert all(set(row["split"]) == {"train"} for row in measured)
    checkpoint = Checkpoint.load(tmp_path / "checkpoint")
    assert checkpoint.attribution_unsuccessful_iterations == 2
    assert all(effect["seeds"] == measured[-1]["seeds"] for effect in checkpoint.attribution)
    assert not (tmp_path / "iterations/0006/01_attribution/refresh.json").exists()


@pytest.mark.parametrize("iterations", [2, 3])
def test_accepted_change_is_remeasured_even_on_the_last_iteration(tmp_path, monkeypatch, iterations):
    create, _, measured = setup_run(tmp_path, monkeypatch, iterations, accept_at=2)
    optimizer = create()
    system, _ = optimizer.optimize(preferences=training_preferences(optimizer))
    assert system.components["transform"].variable == "double"
    assert len(measured) == 2  # Reuse at attempt 2, then refresh immediately after acceptance.
    assert measured[0]["state"] != measured[1]["state"]
    refresh = read_refresh(tmp_path, 2)
    checkpoint = Checkpoint.load(tmp_path / "checkpoint")
    assert refresh["reason"] == "accepted_change"
    assert refresh["state_hash"] == checkpoint.current.fingerprint()
    assert checkpoint.attribution_unsuccessful_iterations == iterations - 2
    assert checkpoint.attribution
    assert all(effect["state_hash"] == checkpoint.current.fingerprint() for effect in checkpoint.attribution)


def test_iterations_without_new_candidates_also_expire_the_cache(tmp_path, monkeypatch):
    create, provider, measured = setup_run(tmp_path, monkeypatch, 6)
    provider.repeat_proposal = True
    optimizer = create()
    optimizer.optimize(preferences=training_preferences(optimizer))
    checkpoint = Checkpoint.load(tmp_path / "checkpoint")
    assert len(measured) == 2
    assert read_refresh(tmp_path, 5)["unsuccessful_iterations_before_refresh"] == 5
    assert checkpoint.attribution_unsuccessful_iterations == 1
    assert [attempt.outcome for attempt in checkpoint.optimization_history.attempts] == [
        "validated", *["no_new_candidate"] * 5,
    ]


def test_resume_preserves_four_failed_attempts_and_refreshes_after_the_fifth(tmp_path, monkeypatch):
    create, _, measured = setup_run(tmp_path, monkeypatch, 6)
    propose = JointBlockOptimizer.propose
    calls = 0

    def interrupt_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise BudgetExceeded("interrupt before fifth proposal")
        return propose(self, *args, **kwargs)

    monkeypatch.setattr(JointBlockOptimizer, "propose", interrupt_once)
    optimizer = create()
    optimizer.optimize(preferences=training_preferences(optimizer))
    checkpoint = Checkpoint.load(tmp_path / "checkpoint")
    assert checkpoint.iteration == 4 and checkpoint.attribution_unsuccessful_iterations == 4
    assert len(measured) == 1
    restored = create()
    restored.optimize(resume=tmp_path / "checkpoint")
    assert len(measured) == 2
    assert read_refresh(tmp_path, 5)["unsuccessful_iterations_before_refresh"] == 5
    assert Checkpoint.load(tmp_path / "checkpoint").attribution_unsuccessful_iterations == 1


@pytest.mark.parametrize("failure", [BudgetExceeded("refresh cap"), ModelResponseError("bad measurement")])
def test_a_failed_post_accept_refresh_preserves_the_accepted_checkpoint(tmp_path, monkeypatch, failure):
    create, _, measured = setup_run(tmp_path, monkeypatch, 1, accept_at=1)
    measure = algorithm_module.estimate_all_edges

    def fail_second(*args, **kwargs):
        if measured:
            raise failure
        return measure(*args, **kwargs)

    monkeypatch.setattr(algorithm_module, "estimate_all_edges", fail_second)
    optimizer = create()
    system, _ = optimizer.optimize(preferences=training_preferences(optimizer))
    checkpoint = Checkpoint.load(tmp_path / "checkpoint")
    assert system.components["transform"].variable == "double"
    assert checkpoint.current.configurations["transform"].value == "double"
    assert checkpoint.best.configurations["transform"].value == "double"
    assert checkpoint.attribution == []  # No stale pre-accept measurements survive.
    assert checkpoint.attribution_unsuccessful_iterations == 0


def test_failed_periodic_refresh_keeps_the_expired_counter(tmp_path, monkeypatch):
    create, _, measured = setup_run(tmp_path, monkeypatch, 5)
    measure = algorithm_module.estimate_all_edges

    def fail_second(*args, **kwargs):
        if measured:
            raise ModelResponseError("bad periodic measurement")
        return measure(*args, **kwargs)

    monkeypatch.setattr(algorithm_module, "estimate_all_edges", fail_second)
    optimizer = create()
    optimizer.optimize(preferences=training_preferences(optimizer))
    assert Checkpoint.load(tmp_path / "checkpoint").attribution_unsuccessful_iterations == 5
    assert len(measured) == 1


def test_reuse_disabled_still_measures_each_iteration(tmp_path, monkeypatch):
    create, _, measured = setup_run(tmp_path, monkeypatch, 3, reuse=False)
    optimizer = create()
    optimizer.optimize(preferences=training_preferences(optimizer))
    assert len(measured) == 3
    assert Checkpoint.load(tmp_path / "checkpoint").attribution_unsuccessful_iterations == 1


@pytest.mark.parametrize("value", [None, False, True, 0, -1, 2.5, "5"])
def test_refresh_limit_is_a_positive_integer(value):
    with pytest.raises(ValueError, match="refresh_after_unsuccessful_iterations"):
        validate_method({"attribution": {"refresh_after_unsuccessful_iterations": value}})


def test_default_limit_can_migrate_but_explicit_changes_cannot():
    config = {"attribution": {"reuse_unchanged": True, "refresh_after_unsuccessful_iterations": 5}}
    old = {"attribution": {"reuse_unchanged": True}}
    assert digest(old) in resume_config_hashes(config)
    changed = deepcopy(config)
    changed["attribution"]["refresh_after_unsuccessful_iterations"] = 6
    assert digest(old) not in resume_config_hashes(changed)
    assert digest(changed) not in resume_config_hashes(config)


def test_old_v3_checkpoint_without_counter_forces_fresh_measurement(tmp_path, monkeypatch):
    create, _, measured = setup_run(tmp_path, monkeypatch, 3)
    propose = JointBlockOptimizer.propose
    calls = 0

    def interrupt_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise BudgetExceeded("interrupt before third proposal")
        return propose(self, *args, **kwargs)

    monkeypatch.setattr(JointBlockOptimizer, "propose", interrupt_once)
    optimizer = create()
    optimizer.optimize(preferences=training_preferences(optimizer))
    path = tmp_path / "checkpoint/checkpoint.json"
    saved = json.loads(path.read_text())
    saved.pop("attribution_unsuccessful_iterations")
    old_config = deepcopy(optimizer.config)
    old_config["attribution"].pop("refresh_after_unsuccessful_iterations")
    saved["config_hash"] = digest(old_config)
    path.write_text(json.dumps(saved))
    restored = create()
    restored.optimize(resume=path.parent)
    assert len(measured) == 2
    assert Checkpoint.load(path).attribution_unsuccessful_iterations == 1


def test_packaged_and_shared_defaults_enable_the_bound():
    import yaml

    for name in ("configs/method/default.yaml", "caiso/config/default.yaml"):
        assert yaml.safe_load(Path(name).read_text())["attribution"]["refresh_after_unsuccessful_iterations"] == 5
