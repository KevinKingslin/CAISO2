"""Section 4.8, equations 19-21: bounded adaptation then paired acceptance."""

from caiso.core.recovery import RetryExhausted
from caiso.core.serialization import Record
from caiso.workflow_search.compatibility_adaptation import compatibility_adapt
from caiso.workflow_search.validator import apply_operation

from .block_optimization import validate_candidate


class WorkflowTrigger(Record):
    accepted_since_workflow: int = 0
    plateau_iterations: int = 0

    def observe(self, accepted):
        self.accepted_since_workflow += int(accepted)
        self.plateau_iterations = 0 if accepted else self.plateau_iterations + 1

    def fires(self, k, patience):
        if k < 1 or patience < 1:
            raise ValueError("Workflow trigger thresholds must be positive")
        return self.accepted_since_workflow >= k or self.plateau_iterations >= patience

    def reset(self):
        self.accepted_since_workflow = 0
        self.plateau_iterations = 0


def optimize_workflow(
    runtime,
    state,
    constitution,
    blocks,
    generator,
    optimizer,
    train,
    validation,
    trajectories,
    performance,
    seed,
    config,
    allowed_connections=None,
):
    """Compare every configured proposal against the same accepted system."""
    candidates = config.get("candidates", 1)
    if candidates < 1:
        raise ValueError("workflow.candidates must be positive")
    best, best_report = state.model_copy(deep=True), None
    reports = []
    for index in range(candidates):
        updated, report = _optimize_workflow_candidate(
            runtime,
            state,
            constitution,
            blocks,
            generator,
            optimizer,
            train,
            validation,
            trajectories,
            performance,
            seed,
            config,
            allowed_connections,
            proposal_seed=seed + index * 1000,
        )
        reports.append(report)
        if best_report is None or (
            report.get("accepted")
            and report.get("candidate_score", -float("inf"))
            > best_report.get("candidate_score", -float("inf"))
        ):
            best, best_report = updated, report
    return best, {**best_report, "candidates": reports}


def _optimize_workflow_candidate(
    runtime,
    state,
    constitution,
    blocks,
    generator,
    optimizer,
    train,
    validation,
    trajectories,
    performance,
    seed,
    config,
    allowed_connections,
    proposal_seed,
):
    try:
        proposal = generator.propose(
            state, constitution, runtime.library, performance, trajectories, proposal_seed
        )
        candidate = apply_operation(state, proposal.operation, runtime.library, allowed_connections)
    except (ValueError, KeyError, RetryExhausted) as exc:
        recovery = getattr(runtime.providers, "recovery", None)
        if recovery:
            recovery.error(exc, "workflow_proposal", "reject_workflow")
        return state.model_copy(deep=True), {"accepted": False, "invalid_proposal": str(exc)}
    candidate, adaptation = compatibility_adapt(
        runtime,
        state,
        candidate,
        blocks,
        constitution,
        optimizer,
        train,
        validation,
        proposal_seed,
        config["compatibility_budget"],
        config.get("acceptance_epsilon", 1e-6),
        search_config=config.get("compatibility_search"),
    )
    accepted, gate = validate_candidate(
        runtime, state, candidate, validation, [seed], config.get("acceptance_epsilon", 1e-6)
    )
    return accepted, {
        "proposal": proposal.model_dump(),
        "candidate_state": candidate.model_dump(),
        "adaptation": adaptation,
        **gate.model_dump(),
    }
