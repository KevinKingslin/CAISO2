from caiso.prompts import render_messages

from .operations import WorkflowProposal


class WorkflowGenerator:
    def __init__(self, providers, role="workflow_generator"):
        self.providers, self.role = providers, role

    def propose(self, state, constitution, library, performance, trajectories, seed):
        inactive = sorted(set(library) - set(state.workflow.nodes))
        return self.providers.complete(
            self.role,
            render_messages(
                "method", "generator.request", constitution=constitution.principles,
                state=state, component_library=library, inactive_component_ids=inactive,
                recent_performance=performance, trajectories=trajectories,
                allowed_operation_schema=WorkflowProposal.model_json_schema(),
            ),
            seed=seed,
            schema=WorkflowProposal,
        )
