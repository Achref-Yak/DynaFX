from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Set

from cognitive_engine.core.config import Priors
from cognitive_engine.core.models import Graph, EdgeType, ReasoningMode, NodeType
from cognitive_engine.reason.mode_operators import apply_mode_operator

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
    ReasoningMode.CAUSAL: "mechanistic cause-effect chains with forward propagation",
    ReasoningMode.CONDITIONAL: "IF/THEN dependencies and scope conditions",
    ReasoningMode.ARGUMENT: "diagnostic reasoning via reverse-warrant propagation",
    ReasoningMode.ANALOGY: "structural parallels with elevated uncertainty",
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


def _compute_mode_view(
    graph: Graph,
    priors: Priors,
    mode: ReasoningMode,
) -> Graph:
    return apply_mode_operator(graph, priors, mode)


def compute_mode_views(
    graph: Graph,
    priors: Optional[Priors] = None,
) -> Graph:
    if priors is None:
        priors = Priors()

    for mode in ReasoningMode:
        view = _compute_mode_view(graph, priors, mode)
        graph.metadata.setdefault("modes", {})[mode.name] = {
            "active_edge_count": len(view.edges),
            "description": MODE_DESCRIPTIONS[mode],
        }
        # Store projected opinions per mode
        opinions = {}
        for nid, node in view.nodes.items():
            p = node.opinion[0] + node.opinion[2] * node.opinion[3]
            opinions[nid.hex] = round(p, 4)
        graph.metadata["modes"][mode.name]["opinions"] = opinions
    return graph
