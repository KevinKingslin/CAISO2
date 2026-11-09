"""Exclude unchanged execution values from every optimizable candidate menu."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import pytest

from caiso.arch.base import BaseComponent
from caiso.arch.system import CompoundAISystem
from caiso.core.budget import BudgetTracker
from caiso.core.component import Component
from caiso.core.configuration import Configuration, ConfigurationType as Kind
from caiso.core.serialization import digest
from caiso.integration.runtime import UpstreamRuntime, adapt_examples
from caiso.method.config import resume_config_hashes, validate_method
from caiso.optimizers import weight_optimizer
from caiso.optimizers.base import OptimizationContext, filter_current_configuration
from caiso.optimizers.hyperparameter_optimizer import HyperparameterOptimizer
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer
from caiso.optimizers.model_selector import ModelSelectorOptimizer
from caiso.optimizers.prompt_optimizer import PromptOptimizer
from caiso.providers.base import ProviderRegistry
from caiso.wrappers.example import Example
from tests.prompt_requests import Recorder
from tests.test_runtime import Source


def context(tmp_path, kind, value, include, space=None, count=2):
    current = Configuration(kind=kind, value=value)
    component = Component(component_id="component", implementation="upstream", input_fields={"x": "str"},
                          output_fields={"answer": "str"}, config=current, config_search_space=space or [])
    return component, OptimizationContext(
        SimpleNamespace(providers=Recorder()), SimpleNamespace(configurations={"component": current}),
        SimpleNamespace(principles=["Select the trajectory where component computes correctly."]), None,
        [], [], 42, {"include_current_configuration": include, "prompt_candidates": count}, tmp_path, lambda _: 0.5,
    )


@pytest.mark.parametrize("include", [True, False])
@pytest.mark.parametrize("count", [0, 2])
def test_prompt_menus_exclude_the_incumbent_when_requested(tmp_path, include, count):
    component, ctx = context(tmp_path, Kind.PROMPT, "Original instruction", include, count=count)
    candidates = PromptOptimizer().propose(component, ctx)
    assert len(candidates) == count + int(include)
    assert any(c.value == "Original instruction" for c in candidates) == include
    assert len(ctx.runtime.providers.calls) == count


@pytest.mark.parametrize("include", [True, False])
@pytest.mark.parametrize("kind,optimizer,space", [
    (Kind.MODEL_SELECTION, ModelSelectorOptimizer, [{"model_selection": "old"}, {"model_selection": "new"}]),
    (Kind.HYPERPARAMETER, HyperparameterOptimizer, [{"k": 1}, {"k": 5}]),
])
def test_discrete_menus_filter_current_values(tmp_path, include, kind, optimizer, space):
    component, ctx = context(tmp_path, kind, space[0], include, space)
    candidates = optimizer().propose(component, ctx)
    assert [c.value for c in candidates] == (space if include else space[1:])


@pytest.mark.parametrize("include", [True, False])
@pytest.mark.parametrize("has_pairs", [True, False])
def test_weight_menus_exclude_current_adapter_without_changing_training_origin(monkeypatch, tmp_path, include, has_pairs):
    component, ctx = context(tmp_path, Kind.MODEL_WEIGHTS, {"model": "local", "adapter": "old-adapter"}, include)
    ctx.runtime.providers = ProviderRegistry({"local": SimpleNamespace(model_id="base-model")})
    ctx.runtime.budget = BudgetTracker(1)
    trained = []
    monkeypatch.setattr(weight_optimizer, "generate_dpo_preferences", lambda *args: [{"prompt": "P", "chosen": "A", "rejected": "B"}] if has_pairs else [])
    def train(model, current, *args):
        trained.append((model, current))
        return "new-adapter"
    monkeypatch.setattr(weight_optimizer, "train_dpo", train)
    candidates = weight_optimizer.WeightOptimizer().propose(component, ctx)
    assert [c.value["adapter"] for c in candidates] == (["old-adapter"] if include else []) + (["new-adapter"] if has_pairs else [])
    assert trained == ([("base-model", "old-adapter")] if has_pairs else [])


def test_metadata_changes_do_not_disguise_the_current_value():
    current = Configuration(kind=Kind.PROMPT, value="Same prompt")
    metadata_only = current.model_copy(deep=True)
    metadata_only.metadata = {"new_attempt": 1}
    assert filter_current_configuration(current, [metadata_only], {"include_current_configuration": False}) == []


class Retriever(BaseComponent):
    def __init__(self):
        super().__init__("Retrieve", ["number"], ["evidence"], variable={"k": 1},
                         variable_search_space={"k": [1, 5, 10]}, num_retry=1, retry_after=0)

    def forward(self, **inputs):
        return {"evidence": str(int(inputs["number"]) * self.variable["k"])}


class Answer(BaseComponent):
    def __init__(self):
        super().__init__("Answer", ["evidence"], ["answer"], variable="identity", num_retry=1, retry_after=0)

    def forward(self, **inputs):
        return {"answer": str(int(inputs["evidence"]) * {"identity": 1, "double": 2, "triple": 3}[self.variable])}


class Proposals:
    def __init__(self, values):
        self.values = values

    def propose(self, component, ctx):
        result = []
        for i, value in enumerate(self.values):
            proposal = ctx.state.configurations[component.component_id].model_copy(deep=True)
            proposal.value = value
            proposal.metadata = {"proposal_attempt": i}
            result.append(proposal)
        return result


def joint_runtime():
    system = CompoundAISystem({"source": Source(), "retriever": Retriever(), "answer_generator": Answer()},
                              final_output_fields=["answer"], ground_fields=["gd_answer"],
                              eval_func=lambda answer, gd_answer: float(answer == gd_answer))
    runtime = UpstreamRuntime(system, ProviderRegistry({}), BudgetTracker(3))
    examples = adapt_examples(system, [Example(question="2", gd_answer="4")], "train")
    return runtime, examples


def joint(runtime, tmp_path, include, limit=64, hyper=None):
    return JointBlockOptimizer(
        runtime, SimpleNamespace(choose=lambda *args: "A"),
        {"include_current_configuration": include, "max_joint_candidates": limit}, tmp_path,
        optimizers={Kind.PROMPT: Proposals(["identity", "double", "triple"]),
                    Kind.HYPERPARAMETER: Proposals(hyper if hyper is not None else [{"k": 1}, {"k": 5}, {"k": 10}])},
    )


@pytest.mark.parametrize("include", [True, False])
def test_complete_joint_options_follow_the_component_toggle(tmp_path, include):
    runtime, examples = joint_runtime()
    optimizer = joint(runtime, tmp_path, include)
    scored = optimizer.propose(runtime.initial_state, ["source", "retriever", "answer_generator"], None, examples, 42)
    # Custom generators intentionally return metadata-only copies of incumbents.
    # The legacy true mode retains them; strict false removes them by value.
    combinations = {(s.configurations["retriever"].value["k"], s.configurations["answer_generator"].value) for s in scored}
    assert combinations == set(product([1, 5, 10] if include else [5, 10], ["identity", "double", "triple"] if include else ["double", "triple"]))
    assert all(s.configurations["source"] == runtime.initial_state.configurations["source"] for s in scored)
    assert runtime.engine.components["retriever"].variable == {"k": 1}
    assert runtime.engine.components["answer_generator"].variable == "identity"


@pytest.mark.parametrize("include", [True, False])
def test_one_candidate_cap_preserves_old_mode_and_uses_changed_options_in_strict_mode(tmp_path, include):
    runtime, examples = joint_runtime()
    scored = joint(runtime, tmp_path, include, limit=1).propose(runtime.initial_state, ["retriever", "answer_generator"], None, examples, 42)
    assert len(scored) == 1
    changed = runtime.initial_state.with_configs(scored[0].configurations)
    assert (changed.configurations["retriever"].value != {"k": 1}) == (not include)
    assert (changed.configurations["answer_generator"].value != "identity") == (not include)


@pytest.mark.parametrize("hyper", [[], [{"k": 1}]])
def test_missing_changed_option_skips_the_block_without_reinserting_current(tmp_path, hyper):
    runtime, examples = joint_runtime()
    optimizer = joint(runtime, tmp_path, False, hyper=hyper)
    assert optimizer.propose(runtime.initial_state, ["retriever", "answer_generator"], None, examples, 42) == []


def test_fixed_only_blocks_have_no_strict_candidates_and_spend_no_runs(tmp_path):
    runtime, examples = joint_runtime()
    assert joint(runtime, tmp_path, False).propose(runtime.initial_state, ["source"], None, examples, 42) == []
    assert runtime.budget.usage.equivalent_full_system_runs == 0


def test_compatibility_search_handles_a_strict_empty_menu(monkeypatch, tmp_path):
    from caiso.workflow_search import compatibility_adaptation
    runtime, examples = joint_runtime()
    optimizer = joint(runtime, tmp_path, False, hyper=[{"k": 1}])
    monkeypatch.setattr(compatibility_adaptation, "affected_components", lambda *args: ["retriever"])
    state, report = compatibility_adaptation.compatibility_adapt(runtime, runtime.initial_state, runtime.initial_state,
        {}, None, optimizer, examples, [], 42, 20, 1e-6)
    assert state == runtime.initial_state
    assert report["no_candidates"] is True
    assert report["adapted"] is False


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", {}, []])
def test_candidate_toggle_requires_a_boolean(tmp_path, invalid):
    with pytest.raises(ValueError, match="include_current_configuration"):
        validate_method({"block_optimization": {"include_current_configuration": invalid}})
    with pytest.raises(ValueError, match="include_current_configuration"):
        JointBlockOptimizer(None, None, {"include_current_configuration": invalid}, tmp_path)


@pytest.mark.parametrize("include,reevaluate,use_history", list(product([True, False], repeat=3)))
def test_resume_accepts_only_equivalent_missing_true_defaults(include, reevaluate, use_history):
    cfg = {"seed": 42, "block_optimization": {"include_current_configuration": include, "reevaluate_current": reevaluate, "max_joint_candidates": 16},
           "block_selection": {"model_role": "block_selection", "use_history": use_history}}
    before = deepcopy(cfg)
    hashes = resume_config_hashes(cfg)
    paths = [("block_optimization", "reevaluate_current"), ("block_optimization", "include_current_configuration"), ("block_selection", "use_history")]
    for omitted in product([True, False], repeat=len(paths)):
        missing = [path for path, remove in zip(paths, omitted, strict=True) if remove]
        older = deepcopy(cfg)
        for section, key in missing:
            older[section].pop(key)
        assert (digest(older) in hashes) == all(cfg[section][key] for section, key in missing)
    changed = deepcopy(cfg)
    changed["seed"] = 43
    assert digest(changed) not in hashes
    assert cfg == before
