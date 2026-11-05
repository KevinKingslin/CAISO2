"""Exactly one typed structural operation per proposal."""

from typing import Annotated, Literal, Union

from pydantic import Field

from caiso.core.serialization import Record
from caiso.core.workflow import Edge


class AddComponent(Record):
    kind: Literal["add_component"] = "add_component"
    component_id: str
    edges: list[Edge]
    remove_edges: list[str]


class RemoveComponent(Record):
    kind: Literal["remove_component"] = "remove_component"
    component_id: str
    reconnect: list[Edge]


class AddEdge(Record):
    kind: Literal["add_edge"] = "add_edge"
    edge: Edge


class RemoveEdge(Record):
    kind: Literal["remove_edge"] = "remove_edge"
    edge_id: str


class ReplaceEdge(Record):
    kind: Literal["replace_edge"] = "replace_edge"
    edge_id: str
    edge: Edge


class Reorder(Record):
    kind: Literal["reorder"] = "reorder"
    order: list[str]
    edges: list[Edge] = Field(default_factory=list)


class AddBranch(Record):
    kind: Literal["add_branch"] = "add_branch"
    component_ids: list[str]
    edges: list[Edge]
    remove_edges: list[str] = Field(default_factory=list)


class RemoveBranch(Record):
    kind: Literal["remove_branch"] = "remove_branch"
    component_ids: list[str]
    edge_ids: list[str]
    reconnect: list[Edge] = Field(default_factory=list)


Operation = Annotated[
    Union[
        AddComponent,
        RemoveComponent,
        AddEdge,
        RemoveEdge,
        ReplaceEdge,
        Reorder,
        AddBranch,
        RemoveBranch,
    ],
    Field(discriminator="kind"),
]


class WorkflowProposal(Record):
    operation: Operation
    brief_justification: str
