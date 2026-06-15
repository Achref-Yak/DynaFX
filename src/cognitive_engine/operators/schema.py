"""Σ (Schema) operator — Apply domain schema.

Maps node types and edge types according to a domain-specific schema.
"""

from __future__ import annotations

from typing import Optional

from cognitive_engine.core.models import Graph, NodeType
from cognitive_engine.core.schema import Schema
from cognitive_engine.core.state import State


class SchemaOperator:
    """Σ: Apply domain schema to graph.

    Remaps node types according to the schema's type_rules.
    """
    name = "schema"

    def __call__(
        self,
        state: State,
        schema: Schema = None,
        **kwargs,
    ) -> State:
        schema = schema or state.metadata.get("schema")
        if schema is None:
            return state

        original_types = {n.type.name for n in state.graph.nodes.values()}
        state.graph = self._apply_schema(state.graph, schema)
        new_types = {n.type.name for n in state.graph.nodes.values()}
        type_changes = original_types.symmetric_difference(new_types)
        state.metadata["schema_applied"] = schema.name
        state.record(
            self.name,
            f"Applied the '{schema.name}' domain schema to restructure the graph. "
            f"Node types before: {original_types}. Node types after: {new_types}. "
            f"{'Types remapped: ' + str(type_changes) + '. ' if type_changes else 'Node types unchanged (schema matched existing types). '}"
            f"The schema normalizes proposition types to the domain ontology, enabling consistent reasoning across heterogeneous inputs.",
        )
        return state

    def _apply_schema(self, graph: Graph, schema: Schema) -> Graph:
        """Apply schema type rules to graph nodes."""
        for nid, node in graph.nodes.items():
            for rule in schema.type_rules:
                result = rule(node.text, node=node)
                if result is not None:
                    node.type = result
                    break
        return graph
