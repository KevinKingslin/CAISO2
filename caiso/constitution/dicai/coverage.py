"""Preserve explicit component coverage supported by settled debate principles.

Coverage means an exact component-ID reference, not a semantic correctness
judgment. No principle is drafted, renamed or inferred from a generic alias.
"""

import re


def observed_components(pair):
    """Read execution IDs, ignoring task text, outputs and evaluation metadata."""
    result = set()
    for candidate in (pair.candidate_a, pair.candidate_b):
        trajectories = [candidate.get("trajectory", candidate), *candidate.get("continuations", [])]
        for trajectory in trajectories:
            if not isinstance(trajectory, dict):
                continue
            for execution in trajectory.get("executions", []):
                if not isinstance(execution, dict):
                    continue
                component = execution.get("component_id")
                if isinstance(component, str) and component.strip():
                    result.add(component)
    return result


def coverage_targets(pairs, scope, component_ids=None):
    if component_ids is not None:
        if isinstance(component_ids, (str, bytes)):
            raise ValueError("Coverage component IDs must be a sequence of nonempty strings")
        components, source = list(component_ids), "provided_components"
    elif scope.startswith("block:"):
        components, source = scope.removeprefix("block:").split(","), "block_scope"
    else:
        components = set().union(*(observed_components(pair) for pair in pairs))
        source = "observed_trajectories"
    if any(not isinstance(cid, str) or not cid.strip() for cid in components):
        raise ValueError("Coverage component IDs must be nonempty strings")
    return sorted(set(components)), source


def preserve_component_coverage(principles, settled, origins, pairs, expected, *, scope, target_source):
    """Restore existing supported rules for missing IDs after final reduction.

    A source must be a settled debate result from a retained preference pair
    whose trajectory actually executed the named component. Prefer rules
    supported by more distinct pairs, then shorter rules, then source order.
    Insufficient evidence is reported without inventing a generic principle.
    """
    outputs = list(principles)
    observed_by_pair = {pair.pair_id: observed_components(pair) for pair in pairs}
    observed = set().union(*observed_by_pair.values())
    patterns = {
        cid: re.compile(r"(?<![\w./-])" + re.escape(cid) + r"(?![\w/-]|\.\w)")
        for cid in expected
    }

    def references(text):
        return {cid for cid, pattern in patterns.items() if pattern.search(text)}

    def covered(values):
        return set().union(*(references(text) for text in values)) & observed

    before = covered(outputs)
    missing = set(expected) - before
    pool = {}
    for index, (text, pair_id) in enumerate(zip(settled, origins, strict=True), 1):
        supported = references(text)
        if not supported or not supported <= observed_by_pair.get(pair_id, set()):
            continue
        entry = pool.setdefault(text, {
            "principle": text, "components": set(), "source_pair_ids": set(),
            "settled_principle_indices": [],
        })
        entry["components"].update(supported)
        entry["source_pair_ids"].add(pair_id)
        entry["settled_principle_indices"].append(index)

    restored = []
    for cid in expected:
        if cid not in missing:
            continue
        options = [entry for entry in pool.values() if cid in entry["components"]]
        if not options:
            continue
        best = min(options, key=lambda entry: (
            -len(entry["source_pair_ids"]), len(entry["principle"].split()),
            entry["settled_principle_indices"][0],
        ))
        outputs.append(best["principle"])
        restored.append({
            **best, "components": sorted(best["components"]),
            "source_pair_ids": sorted(best["source_pair_ids"]),
            "restored_for": sorted(missing & best["components"]),
        })
        missing -= best["components"]

    after = covered(outputs)
    missing = set(expected) - after
    return outputs, {
        "status": "not_applicable" if not expected else "insufficient_evidence" if missing else "complete",
        "scope": scope, "target_source": target_source,
        "coverage_definition": "exact_component_id_in_principle_and_observed_execution",
        "expected_components": expected, "observed_components": sorted(observed),
        "covered_before": sorted(before), "covered_after": sorted(after),
        "restored": restored, "missing_components": sorted(missing),
        "insufficient_evidence_components": sorted(missing),
        "unobserved_components": sorted(set(expected) - observed),
        "principle_indices_by_component": {
            cid: [i for i, text in enumerate(outputs, 1) if cid in references(text) and cid in observed]
            for cid in expected
        },
        "unattributed_principle_indices": [i for i, text in enumerate(outputs, 1) if not references(text)],
    }
