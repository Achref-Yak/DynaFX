from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Set

from cognitive_engine.core.config import Priors
from cognitive_engine.core.models import Graph, EdgeType, ReasoningMode, NodeType
from cognitive_engine.reason.mode_operators import apply_mode_operator
from cognitive_engine.domain import domain as _domain


class _ModeEdgeLookup:
    """Lazily resolves mode→edge mappings from the active domain config.

    Supports dict-style access so existing callers (MODE_ACTIVE_EDGES[mode])
    continue to work, but reads the current domain config on each access,
    making it safe to switch domains at runtime (e.g. with Domain(...)).
    """

    def _resolve(self) -> dict[ReasoningMode, set[EdgeType]]:
        cfg = _domain.active().mode_active_edges
        result: dict[ReasoningMode, set[EdgeType]] = {}
        for mode in ReasoningMode:
            names = cfg.get(mode.name, set())
            result[mode] = {EdgeType[n] for n in names}
        return result

    def __getitem__(self, mode: ReasoningMode) -> set[EdgeType]:
        return self._resolve()[mode]

    def get(self, mode: ReasoningMode, default: Optional[set[EdgeType]] = None) -> Optional[set[EdgeType]]:
        try:
            return self[mode]
        except (KeyError, ValueError):
            return default

    def values(self):
        return self._resolve().values()

    def keys(self):
        return self._resolve().keys()

    def items(self):
        return self._resolve().items()

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self):
        return len(self._resolve())

    def __contains__(self, mode):
        return mode in self._resolve()


MODE_ACTIVE_EDGES: _ModeEdgeLookup = _ModeEdgeLookup()

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
    result.edges = {e.id: e for e in result.edges.values() if e.type in active}

    for node in result.nodes.values():
        node.metadata["active_edges"] = [e.type.name for e in result.edges.values()
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
