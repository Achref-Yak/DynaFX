from __future__ import annotations

from copy import deepcopy
from typing import Dict, Set

from cognitive_engine.core.models import Graph, EdgeType, ReasoningMode, NodeType

MODE_ACTIVE_EDGES: Dict[ReasoningMode, Set[EdgeType]] = {
    ReasoningMode.CAUSAL: {EdgeType.INFERS, EdgeType.SUPPORTS},
    ReasoningMode.CONDITIONAL: {EdgeType.QUALIFIES, EdgeType.INFERS},
    ReasoningMode.ARGUMENT: {
        EdgeType.SUPPORTS, EdgeType.CONTRADICTS,
        EdgeType.ATTACKS, EdgeType.REBUTS,
    },
    ReasoningMode.ANALOGY: {EdgeType.JUSTIFIES, EdgeType.SUPPORTS},
}

MODE_DESCRIPTIONS = {
    ReasoningMode.CAUSAL: "mechanistic cause-effect chains",
    ReasoningMode.CONDITIONAL: "IF/THEN dependencies and scope conditions",
    ReasoningMode.ARGUMENT: "claims, support, and counterarguments",
    ReasoningMode.ANALOGY: "structural parallels and justificatory relations",
}


def apply_mode(graph: Graph, mode: ReasoningMode) -> Graph:
    result = deepcopy(graph)
    result.mode = mode

    active = MODE_ACTIVE_EDGES[mode]
    result.edges = [e for e in result.edges if e.type in active]

    for node in result.nodes.values():
        node.metadata["active_edges"] = [e.type.name for e in result.edges
                                         if e.source_id == node.id or e.target_id == node.id]

    return result


def compute_mode_views(graph: Graph) -> Graph:
    for mode in ReasoningMode:
        view = apply_mode(graph, mode)
        graph.metadata.setdefault("modes", {})[mode.name] = {
            "active_edge_count": len(view.edges),
            "description": MODE_DESCRIPTIONS[mode],
        }
    return graph
