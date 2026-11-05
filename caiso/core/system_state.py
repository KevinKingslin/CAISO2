from .configuration import Configuration
from .serialization import Record, digest
from .workflow import Workflow


class SystemState(Record):
    system: str
    workflow: Workflow
    configurations: dict[str, Configuration]

    def validate_state(self, library):
        self.workflow.validate_graph(library)
        if set(self.configurations) != set(self.workflow.nodes):
            raise ValueError("Active configurations must match active nodes")
        for n, config in self.configurations.items():
            library[n].validate_config(config)
        return self

    def fingerprint(self):
        return digest(self)

    def with_configs(self, updates):
        if not set(updates) <= set(self.configurations):
            raise ValueError("Cannot update inactive components")
        result = self.model_copy(deep=True)
        result.configurations.update({k: v.model_copy(deep=True) for k, v in updates.items()})
        return result
