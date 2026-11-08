from caiso.core.budget import BudgetExceeded
from caiso.optimizers.joint_block_optimizer import JointBlockOptimizer


def affected_components(current, candidate, blocks):
    old = {e.edge_id: e for e in current.workflow.edges}
    new = {e.edge_id: e for e in candidate.workflow.edges}
    affected = set(current.workflow.nodes) ^ set(candidate.workflow.nodes)
    for eid in set(old) | set(new):
        if old.get(eid) != new.get(eid):
            for edge in (old.get(eid), new.get(eid)):
                if edge:
                    affected.update([edge.source, edge.target])
    if current.workflow.order != candidate.workflow.order:
        affected.update(candidate.workflow.nodes)
    while True:
        expanded = (
            affected
            | {e.target for e in candidate.workflow.edges if e.source in affected}
            | {n for members in blocks.values() if set(members) & affected for n in members}
        )
        if expanded == affected:
            break
        affected = expanded
    return sorted(affected & set(candidate.workflow.nodes))


def compatibility_adapt(
    runtime,
    current,
    candidate,
    blocks,
    constitution,
    optimizer,
    train,
    validation,
    seed,
    budget_runs,
    epsilon,
    search_config=None,
):
    affected = affected_components(current, candidate, blocks)
    working = candidate.model_copy(deep=True)
    report = {"affected": affected, "budget_runs": budget_runs, "adapted": False}
    if not affected or budget_runs <= 0:
        return working, report
    before = runtime.budget.usage.equivalent_full_system_runs
    search_config = search_config or {}
    # A deliberately small local search. The caller owns the single global reward gate;
    # a second full validation gate here previously consumed more than the entire cap.
    if isinstance(optimizer, JointBlockOptimizer):
        optimizer = JointBlockOptimizer(
            runtime,
            optimizer.judge,
            {
                **optimizer.config,
                "prompt_candidates": search_config.get("prompt_candidates", 1),
                "max_joint_candidates": search_config.get("max_joint_candidates", 4),
            },
            optimizer.output_dir,
            optimizer.optimizers,
            on_candidate=optimizer.on_candidate,
        )
    local_train = train[: search_config.get("training_batch_size", 2)]
    report["training_examples"] = len(local_train)
    with runtime.budget.compatibility(budget_runs):
        try:
            scored = optimizer.propose(working, affected, constitution, local_train, seed)
            if scored:
                best = optimizer.select(scored, working, affected, seed)
                proposed = working.with_configs(best.configurations)
                report["adapted"] = proposed != working
                report["constitution_score"] = best.constitution_score
                report["task_score"] = best.task_score
                report["selection_score"] = best.score
                working = proposed
            else:
                report["no_candidates"] = True
        except BudgetExceeded:
            report["budget_exhausted"] = True
    report["equivalent_runs_used"] = runtime.budget.usage.equivalent_full_system_runs - before
    return working, report
