"""Deterministic method-service fixture; no benchmark scores are mocked."""

import json
import re
import numpy as np


def training_preferences(optimizer):
    """Real unequal-score arithmetic variants, supplied as labeled training data."""
    from caiso.constitution.models import PreferencePair

    runtime, current = optimizer.runtime, optimizer.runtime.initial_state
    changed = current.model_copy(deep=True)
    changed.configurations["transform"].value = "double"
    example = optimizer.train[0]
    a, b = runtime.execute(current, example, 42), runtime.execute(changed, example, 43)
    assert a.reward != b.reward
    return [PreferencePair(
        pair_id="provided-training-operation", context=example.inputs,
        candidate_a={"trajectory": a.model_dump()}, candidate_b={"trajectory": b.model_dump()},
        preferred="A" if a.reward > b.reward else "B",
        metadata={"label_source": "provided_training_task_metric"},
    )]


def atomicity_review_fixture(messages):
    """Atomic fixtures only; compound/revision cases use scripted semantic reviews."""
    values = json.loads(messages[-1]["content"].split("PRINCIPLES:\n", 1)[1])
    return {"assessments": [
        {"index": row["index"], "atomic": True, "qualities": ["operation correctness"], "reason": "One independently evaluated property."}
        for row in values
    ]}


class OfflineProvider:
    def __init__(self, budget):
        self.budget, self.calls = budget, []

    def fingerprint(self):
        return "offline-synthetic-dicai-v3-component-coverage"

    def complete(self, messages, *, seed, schema=None, **kwargs):
        self.budget.call()
        self.calls.append((messages, seed, schema))
        system = messages[0]["content"]
        user = messages[-1]["content"]
        principles = [
            "Select the trajectory where transform applies the requested arithmetic operation correctly.",
            "Select the trajectory where transform preserves the relevant input through intermediate steps.",
            "Select the trajectory where transform presents the result clearly.",
        ]
        # Full-cycle fixtures include deterministic input/output components too.
        for component, criterion in {
            "source": "preserves the question's number in its output.",
            "sink": "preserves the computed result in the final answer.",
        }.items():
            if re.search(r'"component_id"\s*:\s*"' + component + r'"', user):
                principles.append(f"Select the trajectory where {component} {criterion}")
        if schema:
            name = schema.__name__
            if name == "AtomicityReview":
                return json.dumps(atomicity_review_fixture(messages))
            if name == "Reasoning":
                return json.dumps({
                    "reasoning": "The preferred trajectory applies the requested operation and preserves its input.",
                    "dimensions": ["operation correctness", "input preservation"],
                    "questions": [{"select_prompt": p, "reason": "Evaluates the observed component behavior."} for p in principles],
                })
            if name == "PromptProposal":
                return json.dumps({"prompt": "double"})
            if name == "BlockVote":
                block_data = user.split("AVAILABLE BLOCKS AND COMPONENTS:\n", 1)[1]
                blocks = json.loads(block_data.split("\n\nALLOWED BLOCK IDS:", 1)[0])
                bid = next(k for k, v in blocks.items() if "transform" in v)
                return json.dumps(
                    {
                        "block_id": bid,
                        "brief_justification": "The transformation does not double the input.",
                    }
                )
            raise ValueError(f"Unsupported offline schema {name}")
        if "You are the Moderator/Judge." in system:
            return "FINAL LIST\n" + "\n".join(f"- Q{i}: {p}" for i, p in enumerate(principles, 1)) + "\nDEBATE_OVER"
        if "debater who must" in system:
            return "1. DEFEND:\n- Q1: Correct arithmetic matters.\n2. ATTACK:\n- None\n3. MERGE:\n- None\n4. EDIT:\n- None\n5. ADD:\n- None\n6. CONCEDE:\n- None"
        if "You abstract a cluster of preference principles" in system:
            return "Select the trajectory where transform faithfully computes the requested result from the supplied input."
        if "You are a preference judge." in system:
            first = json.JSONDecoder().raw_decode(user.split("TRAJECTORY 1:\n", 1)[1])[0]
            second = json.JSONDecoder().raw_decode(user.split("TRAJECTORY 2:\n", 1)[1])[0]

            def quality(value):
                value = value.get("trajectory", value)
                for execution in value.get("executions", []):
                    if execution["component_id"] == "transform":
                        return int(execution["component_output"]["result"]) == 2 * int(
                            execution["component_input"]["number"]
                        )
                if "input" in value and "answer" in value:
                    return int(value["answer"]) == 2 * int(value["input"])
                return 0

            return json.dumps({"choice": "1" if quality(first) >= quality(second) else "2", "reason": "Correct operation."})
        if "strict reviewer of preference rationales" in system:
            return "Retain only discriminating operation criteria."
        raise ValueError("Unrecognized offline prompt")


class OfflineEmbeddings:
    def __init__(self, budget):
        self.budget = budget

    def fingerprint(self):
        return "offline-arithmetic-embedding-v1"

    def embed(self, texts):
        self.budget.call(embedding=True)
        return np.asarray([[1.0, 0.0] if "arithmetic" in t else [0.0, 1.0] for t in texts])
