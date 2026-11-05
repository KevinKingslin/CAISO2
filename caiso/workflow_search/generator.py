from caiso.core.serialization import canonical
from caiso.providers.base import messages

from .operations import WorkflowProposal


class WorkflowGenerator:
    def __init__(self, providers, role="workflow_generator"):
        self.providers, self.role = providers, role

    def propose(self, state, constitution, library, performance, trajectories, seed):
        inactive = sorted(set(library) - set(state.workflow.nodes))
        return self.providers.complete(
            self.role,
            messages(
                "Propose EXACTLY ONE structural modification to improve the compound AI workflow. "
                "Use only components in the supplied library, with each component ID at most once. "
                "AddComponent and AddBranch may use only inactive_component_ids (original "
                "components removed by prior edits). If that list is empty, do not propose either "
                "addition. Never create extra reviewers, mergers, or copies of active components. "
                "Preserve valid typed inputs, an acyclic "
                "graph and reachable final outputs. An add_branch can connect independent paths "
                "and merge them by naming old boundary edges in remove_edges. A reorder swaps one "
                "component pair and must supply changed dataflow edges; scheduling-only changes "
                "are invalid. Never invent code or components. Return one "
                "operation and a brief justification matching the supplied schema.",
                canonical(
                    {
                        "constitution": constitution.principles,
                        "state": state,
                        "component_library": library,
                        "inactive_component_ids": inactive,
                        "recent_performance": performance,
                        "trajectories": trajectories,
                        "allowed_operation_schema": WorkflowProposal.model_json_schema(),
                    }
                ),
            ),
            seed=seed,
            schema=WorkflowProposal,
        )
