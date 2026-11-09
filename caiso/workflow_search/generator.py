from caiso.prompts import render_messages

from .operations import WorkflowProposal


class WorkflowGenerator:
    def __init__(self, providers, role="workflow_generator", use_history=True):
        if not isinstance(use_history, bool):
            raise ValueError("use_history must be true or false")
        self.providers, self.role = providers, role
        self.use_history = use_history

    def propose(self, state, constitution, library, performance, trajectories, seed):
        inactive = sorted(set(library) - set(state.workflow.nodes))
        return self.providers.complete(
            self.role,
            render_messages(
                "method", "generator.request", constitution=constitution.principles,
                state=state, component_library=library, inactive_component_ids=inactive,
                recent_performance=performance if self.use_history else [], trajectories=trajectories,
                allowed_operation_schema=WorkflowProposal.model_json_schema(),
            ),
            seed=seed,
            schema=WorkflowProposal,
            include_schema_instruction=False,
        )
