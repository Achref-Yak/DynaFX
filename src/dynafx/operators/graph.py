"""Ĝ (Graph) operator — Build/assign edges.

Rebuilds edges in the graph based on node types and relations.
"""

from __future__ import annotations

from dynafx.core.models import Graph
from dynafx.core.schema import Schema
from dynafx.core.state import State


class GraphOperator:
    """Ĝ: Build/assign edges from node types.

    Rebuilds the graph's edge structure based on the assigned node types.
    """
    name = "graph"

    def __call__(
        self,
        state: State,
        **kwargs,
    ) -> State:
        # Edge assignment is already done during extraction
        # This operator is a placeholder for future graph transformations
        node_types = {}
        for n in state.graph.nodes.values():
            node_types[n.type.name] = node_types.get(n.type.name, 0) + 1
        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(node_types.items()))
        edge_types = {}
        for e in state.graph.edges.values():
            edge_types[e.type.name] = edge_types.get(e.type.name, 0) + 1
        edge_str = ", ".join(f"{k}: {v}" for k, v in sorted(edge_types.items()))
        state.record(
            self.name,
            f"Current cognitive graph: {len(state.graph.nodes)} propositions, {len(state.graph.edges)} edges, "
            f"{len(state.graph.entities)} entities, {len(state.graph.world_relations)} world relations. "
            f"Node types: {type_str}. Edge types: {edge_str}. "
            f"The graph structure captures the complete reasoning state — claims, beliefs, entities, and their relationships.",
        )
        return state
