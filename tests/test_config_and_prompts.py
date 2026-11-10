"""Configuration migration and prompt extraction must preserve benchmark behavior."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from caiso.integration.config import load_experiment, load_method
from caiso.method.config import validate_method
from caiso.prompts import catalog_entries, entry, render_messages

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,label", [("amazon", "amazon"), ("hotpotqa", "hotpotqa"), ("pubmed", "pubmedqa")])
def test_current_profiles_upgrade_search_without_changing_benchmarks(name, label):
    original = load_experiment(ROOT / f"scripts/configs/optimize/{name}.yaml")
    current = load_experiment(ROOT / f"configs/experiments/{label}.yaml")
    active, legacy = load_method(current), load_method(original)
    assert active["validation_sampling"] == "seeded"
    assert active["validation_batch_size"] == (50 if label == "hotpotqa" else "all")
    assert active["block_optimization"]["validation_repeats"] == 2
    assert active["block_optimization"]["sample_discrete"] is False
    for method in (active, legacy):
        for key in ("validation_sampling", "validation_batch_size", "attribution", "block_selection"):
            method.pop(key)
    assert active == legacy
    assert current["num_repeat"] == 3
    for key in ("dataset", "system", "max_sample_workers", "dotenv_path", "output_dir", "wandb_run_name"):
        assert current[key] == original[key]
    assert load_experiment(ROOT / f"configs/experiments/{label}.yaml", "evaluate") == load_experiment(ROOT / f"scripts/configs/eval/{name}_vanilla.yaml", "evaluate")
    assert not any(key.startswith("ppo_") for key in current)
    assert "state_dict_path" not in current  # OPTIMAS reward checkpoint, unused by CAISO.


def test_shared_method_copy_does_not_change_any_values():
    assert yaml.safe_load((ROOT / "configs/method/default.yaml").read_text()) == yaml.safe_load((ROOT / "caiso/config/default.yaml").read_text())
    assert load_method({})["block_constitution"]["enabled"] is False
    assert load_method({})["block_optimization"]["reevaluate_current"] is True
    assert load_method({})["block_optimization"]["include_current_configuration"] is True
    assert load_method({})["block_selection"]["use_history"] is True
    assert load_method({})["leiden"]["max_comm_size"] == 2
    assert "use_history" not in load_method({})
    assert load_method({})["block_optimization"]["task_scoring"] == {
        "enabled": True, "constitution_weight": 0.0, "task_weight": 1.0,
    }


def test_experiment_overrides_method_recursively_and_paths_are_relative(tmp_path):
    method = tmp_path / "method.yaml"
    method.write_text((ROOT / "configs/method/default.yaml").read_text())
    config = yaml.safe_load((ROOT / "configs/experiments/amazon.yaml").read_text())
    config["method_config"] = "method.yaml"
    config["method"]["validation_batch_size"] = "all"
    config["method"]["dicai"] = {"debate_rounds": 1}
    config["method"]["block_constitution"] = {"enabled": True}
    config["method"]["block_selection"] = {"use_history": False}
    config["method"]["leiden"] = {"max_comm_size": 3}
    config["method"]["block_optimization"] = {"reevaluate_current": False, "include_current_configuration": False}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    resolved = load_method(load_experiment(path))
    assert resolved["validation_batch_size"] == "all"
    assert resolved["dicai"]["debate_rounds"] == 1
    assert resolved["dicai"]["max_workers"] == 10
    assert resolved["block_constitution"]["enabled"] is True
    assert resolved["block_constitution"]["preference_pairs"] == 2
    assert resolved["block_selection"]["use_history"] is False
    assert resolved["leiden"]["max_comm_size"] == 3
    assert resolved["leiden"]["resolution_gamma"] == 0.05
    assert "use_history" not in resolved
    assert resolved["block_selection"]["model_role"] == "block_selection"
    assert resolved["block_optimization"]["reevaluate_current"] is False
    assert resolved["block_optimization"]["include_current_configuration"] is False
    assert resolved["block_optimization"]["max_joint_candidates"] == 16


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", {}, [], True, False])
def test_removed_top_level_history_switch_points_to_the_selection_setting(invalid):
    with pytest.raises(ValueError, match=r"use block_selection.use_history instead"):
        validate_method({"use_history": invalid})


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", {}, []])
def test_block_selection_history_switch_requires_a_boolean(invalid):
    with pytest.raises(ValueError, match=r"block_selection.use_history"):
        validate_method({"block_selection": {"use_history": invalid}})


@pytest.mark.parametrize("invalid", [None, True, "false", []])
def test_block_selection_requires_a_mapping(invalid):
    with pytest.raises(ValueError, match="block_selection must be a configuration mapping"):
        validate_method({"block_selection": invalid})


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", {}, []])
def test_current_validation_switch_requires_a_boolean(invalid):
    with pytest.raises(ValueError, match=r"block_optimization.reevaluate_current"):
        validate_method({"block_optimization": {"reevaluate_current": invalid}})


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", {}, []])
def test_block_constitution_switch_requires_a_boolean(invalid):
    with pytest.raises(ValueError, match=r"block_constitution.enabled"):
        validate_method({"block_constitution": {"enabled": invalid}})


def test_extracted_prompts_match_pre_extraction_text_and_rendering():
    fixture = json.loads((ROOT / "tests/fixtures/prompt_parity.json").read_text())
    for catalog, expected in fixture["hashes"].items():
        actual = catalog_entries(catalog)
        assert set(actual) == set(expected)
        for key, sha256 in expected.items():
            value = actual[key]
            actual_hash = (
                [hashlib.sha256(item.encode()).hexdigest() for item in value]
                if isinstance(value, list)
                else hashlib.sha256(value.encode()).hexdigest()
            )
            assert actual_hash == sha256
    for case in fixture["rendered"].values():
        if case["catalog"] == "systems/pubmed":
            request = render_messages(
                case["catalog"], case["key"].replace(".template", ".request"), **case["arguments"]
            )
            assert request[0] == {"role": "system", "content": "You are a scientist."}
            assert request[1]["content"] == case["expected"]
        else:
            assert entry(case["catalog"], case["key"]).format(**case["arguments"]) == case["expected"]


def test_dspy_signatures_keep_instructions_and_field_metadata():
    from examples.systems.hotpotqa import five_components as hotpotqa
    from examples.systems.amazon import local_models_for_next_item_selection as amazon

    for system, module, names in [
        ("hotpotqa", hotpotqa, ["QuestionRewriter", "InfoExtractor", "HintGenerator", "AnswerGenerator"]),
        ("amazon", amazon, ["NextItemDecider"]),
    ]:
        for name in names:
            signature = getattr(module, name)
            assert signature.instructions == entry("systems/" + system, name + ".instructions")
            for field_name, field in signature.fields.items():
                for key in ("prefix", "desc"):
                    catalog_key = name + "." + field_name + "." + key
                    catalog = catalog_entries("systems/" + system)
                    if catalog_key in catalog:
                        assert field.json_schema_extra[key] == catalog[catalog_key]
