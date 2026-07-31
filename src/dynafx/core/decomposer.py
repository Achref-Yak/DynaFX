"""System decomposition API — manually define Higraph structure.

Primary usage — clean, zero-extraction-required constructor:

    d = SystemDecomposer(name="Server Pipeline")
    d.add_node("server", type="ENTITY", partition="technical")
    d.add_node("pipeline", type="PROCESS", partition="technical")
    d.add_node("queue_processor", type="PROCESS", parent="pipeline")
    d.add_edge("server", "pipeline", "CAUSES", polarity=-1)
    d.detect()
    print(d.summary())

Hybrid usage — wrap an extracted graph for annotation:

    d = SystemDecomposer(graph=extracted_graph)
    d.assign_partition("server", "technical")
    d.add_containment("pipeline", "queue_processor")
    d.add_dependency("server", "pipeline", "CAUSES", polarity=-1)
    d.detect()
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from dynafx.core.models import (
    Edge,
    EdgeType,
    EmergentProperty,
    Graph,
    Node,
    NodeType,
    ReasoningMode,
)


def _graph_depth(graph: Graph, node_id: UUID) -> int:
    depth = 0
    current = node_id
    visited: set[UUID] = set()
    while current in graph.nodes:
        if current in visited:
            break
        visited.add(current)
        parent = graph.nodes[current].container_id
        if parent is None:
            break
        depth += 1
        current = parent
    return depth

logger = logging.getLogger(__name__)


class SystemDecomposer:
    """High-level API for manually defining system structure on a graph.

    Two construction paths:
      1. ``SystemDecomposer(name="...")`` — creates a clean blank Graph.
      2. ``SystemDecomposer(graph=extracted_graph)`` — wraps an existing
         extraction output for annotation without re-extraction.
    """

    def __init__(
        self,
        graph: Optional[Graph] = None,
        name: str = "",
    ) -> None:
        if graph is not None:
            self.graph = graph
        else:
            self.graph = Graph(mode=ReasoningMode.CAUSAL)
            self.graph.source_text = name
        self._name_index: dict[str, UUID] = {}
        self._rebuild_index()

    # ── Index ─────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        self._name_index.clear()
        for n in self.graph.nodes.values():
            key = n.text.strip().lower()
            if key not in self._name_index:
                self._name_index[key] = n.id

    def _to_key(self, name: str) -> str:
        return name.strip().lower()

    def _lookup(self, name: str) -> Optional[UUID]:
        return self._name_index.get(self._to_key(name))

    # ── Nodes ─────────────────────────────────────────────────────

    def add_node(
        self,
        name: str,
        *,
        type: str = "CONCEPT",
        partition: Optional[str] = None,
        parent: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """Create a node and return its *name* for chaining.

        Idempotent — if a node with the same normalized name already
        exists, this is a no-op (existing name is returned).
        """
        key = self._to_key(name)
        if key in self._name_index:
            return name

        try:
            node_type = NodeType[type.upper()]
        except KeyError:
            node_type = NodeType.CONCEPT

        node = Node(
            text=name,
            type=node_type,
            orthogonal_partition=partition,
        )

        if parent is not None:
            parent_id = self._lookup(parent)
            if parent_id is not None:
                node.container_id = parent_id

        if confidence is not None:
            node.metadata["confidence"] = confidence

        self.graph.nodes[node.id] = node
        self._name_index[key] = node.id
        return name

    # ── Edges ─────────────────────────────────────────────────────

    def add_edge(
        self,
        source: str,
        target: str,
        type: str = "CAUSES",
        polarity: int = 1,
        confidence: float = 0.8,
    ) -> bool:
        """Add a typed edge between two named nodes."""
        src_id = self._lookup(source)
        tgt_id = self._lookup(target)

        if src_id is None:
            raise KeyError(f"Unknown source node: '{source}'")
        if tgt_id is None:
            raise KeyError(f"Unknown target node: '{target}'")
        if src_id == tgt_id:
            logger.warning("Self-loop skipped: '%s'", source)
            return False

        try:
            etype = EdgeType[type.upper()]
        except KeyError:
            logger.warning("Unknown edge type '%s', using CAUSES", type)
            etype = EdgeType.CAUSES

        edge = Edge(
            source_id=src_id,
            target_id=tgt_id,
            type=etype,
            polarity=polarity,
            metadata={"confidence": confidence},
        )
        self.graph.edges[edge.id] = edge
        logger.info(
            "Edge: '%s' --%s--> '%s' (polarity=%+d)",
            source[:30], etype.name, target[:30], polarity,
        )
        return True

    # ── Legacy helpers (graph wrapping) ───────────────────────────

    def _match_node(self, name: str) -> Optional[Node]:
        nid = self._lookup(name)
        if nid is not None:
            return self.graph.nodes.get(nid)
        name_lower = self._to_key(name)
        candidates = []
        for n in self.graph.nodes.values():
            if name_lower in n.text.strip().lower():
                candidates.append(n)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            candidates.sort(key=lambda x: len(x.text))
            return candidates[0]
        return None

    def _require_node(self, name: str) -> Node:
        n = self._match_node(name)
        if n is None:
            raise KeyError(f"No node matching '{name}' in graph")
        return n

    def assign_partition(self, node_name: str, partition: str) -> None:
        node = self._require_node(node_name)
        node.orthogonal_partition = partition
        logger.info("Assigned '%s' → partition '%s'", node.text[:40], partition)

    def create_partition(self, name: str) -> None:
        pass

    def add_containment(self, parent_name: str, child_name: str) -> bool:
        parent = self._require_node(parent_name)
        child = self._require_node(child_name)
        if child.id == parent.id:
            logger.warning("Cannot contain itself: '%s'", parent_name)
            return False
        child.container_id = parent.id
        logger.info(
            "Containment: '%s' ⊂ '%s'",
            child.text[:40], parent.text[:40],
        )
        return True

    def add_dependency(
        self,
        source_name: str,
        target_name: str,
        edge_type: str = "CAUSES",
        polarity: int = 1,
        confidence: float = 0.8,
    ) -> bool:
        return self.add_edge(source_name, target_name, edge_type, polarity, confidence)

    # ── Emergence ─────────────────────────────────────────────────

    def detect(self) -> list[EmergentProperty]:
        return []

    # ── Inspection ────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        nodes_by_partition: dict[str, list[dict]] = {}
        for n in self.graph.nodes.values():
            part = n.orthogonal_partition or "_unassigned"
            entry = {
                "id": n.id.hex[:8],
                "name": n.text[:50],
                "type": n.type.name,
                "depth": _graph_depth(self.graph, n.id),
                "children": [
                    {"id": c.id.hex[:8], "name": c.text[:40]}
                    for c in self.graph.nodes.values()
                    if c.container_id == n.id
                ],
            }
            if n.metadata.get("confidence") is not None:
                entry["confidence"] = n.metadata["confidence"]
            nodes_by_partition.setdefault(part, []).append(entry)

        result: dict[str, Any] = {
            "nodes": [
                {
                    "id": n.id.hex[:8],
                    "name": n.text[:50],
                    "type": n.type.name,
                    "partition": n.orthogonal_partition,
                    "parent": (
                        self.graph.nodes[n.container_id].text[:40]
                        if n.container_id and n.container_id in self.graph.nodes
                        else None
                    ),
                    "depth": _graph_depth(self.graph, n.id),
                }
                for n in sorted(
                    self.graph.nodes.values(),
                    key=lambda x: x.text.lower(),
                )
            ],
            "edges": [
                {
                    "source": self.graph.nodes[e.source_id].text[:40],
                    "target": self.graph.nodes[e.target_id].text[:40],
                    "type": e.type.name,
                    "polarity": e.polarity,
                }
                for e in self.graph.edges.values()
                if e.type.name != "ASSOCIATED_WITH"
            ],
            "emergent_properties": [
                {
                    "name": ep.name,
                    "condition": ep.condition,
                    "involved": [
                        {"id": nid.hex[:8], "name": self.graph.nodes[nid].text[:40]}
                        for nid in ep.involved_ids if nid in self.graph.nodes
                    ],
                }
                for ep in self.graph.emergent_properties
            ],
        }

        partitions = {
            p: [{"id": e["id"], "name": e["name"]} for e in entries]
            for p, entries in nodes_by_partition.items()
        }
        if partitions:
            result["partitions"] = partitions

        return result
