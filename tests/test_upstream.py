from pathlib import Path

import pytest
import yaml

from caiso.integration.config import load_method
from scripts.verify_upstream import verify

ROOT = Path(__file__).resolve().parents[1]


def test_all_imported_benchmark_files_match_release():
    checked, commit = verify(ROOT)
    assert checked == 109
    assert commit == "71871a2dd98081231d733debd41178ba47e13528"


@pytest.mark.parametrize(
    "name,val_size,new_data", [("hotpotqa", 10, 20), ("pubmed", 10, 20), ("amazon", 20, 10)]
)
def test_original_experiment_sizes_are_resolved(name, val_size, new_data):
    original = yaml.safe_load((ROOT / f"scripts/configs/optimize/{name}.yaml").read_text())
    method = load_method(original)
    assert method["training_batch_size"] == 20
    assert method["validation_batch_size"] == val_size
    assert method["attribution"]["batch_size"] == new_data
    assert method["validation_sampling"] == "first"
    assert method["max_iterations"] == 100
    assert not method["workflow"]["enabled"]


def test_experiment_can_override_method_without_changing_other_defaults():
    method = load_method(
        {
            "val_size": 10,
            "caiso": {
                "method": {
                    "validation_batch_size": "all",
                    "dicai": {"debate_rounds": 1},
                }
            },
        }
    )
    assert method["validation_batch_size"] == "all"
    assert method["dicai"]["debate_rounds"] == 1
    assert method["dicai"]["max_workers"] == 10
    with pytest.raises(ValueError, match="original benchmark"):
        load_method({"caiso": {"method": {"workflow": {"enabled": True}}}})


def test_amazon_files_load_with_original_split():
    from examples.datasets.amazon import dataset_engine

    train, validation, test = dataset_engine()
    assert (len(train), len(validation), len(test)) == (335, 60, 99)
    assert set(train[0].inputs().keys()) == {"sequence", "choices"}


def test_pubmed_files_load_with_original_split():
    from examples.datasets.pubmed import dataset_engine

    train, validation, test = dataset_engine()
    assert (len(train), len(validation), len(test)) == (475, 25, 500)


def test_hotpot_original_artifact_loads_without_modifying_loader(monkeypatch):
    from types import SimpleNamespace
    from examples.datasets import hotpotqa

    calls = []

    def training_data(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            train=[
                SimpleNamespace(question=f"train {i}", answer="yes")
                for i in range(kwargs["train_size"])
            ],
            dev=[
                SimpleNamespace(question=f"val {i}", answer="no") for i in range(kwargs["dev_size"])
            ],
        )

    # Only remote training data is mocked; joblib reads the actual original test file.
    monkeypatch.setattr(hotpotqa, "HotPotQA", training_data)
    train, validation, test = hotpotqa.dataset_engine()
    assert calls == [{"train_seed": 1, "train_size": 5000, "dev_size": 250, "test_size": 0}]
    assert (len(train), len(validation), len(test)) == (5000, 250, 100)
    assert all(
        isinstance(example.question, str) and isinstance(example.gd_answer, str) for example in test
    )
