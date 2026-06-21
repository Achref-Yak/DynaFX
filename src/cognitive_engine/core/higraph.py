"""Higraph utilities — containment (ρ) and orthogonality (Π) queries.

All functions are pure queries over the existing Graph structure.
They use the container_id and orthogonal_partition fields on Node,
which are optional — argumentation graphs that never set these
fields return empty/zero for all queries.
"""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from cognitive_engine.core.models import Graph, Node


def contained_nodes(graph: Graph, blob_id: UUID) -> list[UUID]:
    """All node IDs recursively nested under *blob_id*."""
    result: list[UUID] = []
    _collect_contained(graph, blob_id, result, set())
    return result


def _collect_contained(graph: Graph, parent: UUID, result: list[UUID], visited: set[UUID]) -> None:
    for nid, node in graph.nodes.items():
        if node.container_id == parent and nid not in visited:
            visited.add(nid)
            result.append(nid)
            _collect_contained(graph, nid, result, visited)


def direct_children(graph: Graph, blob_id: UUID) -> list[UUID]:
    """Immediate children (one level deep)."""
    return [
        nid for nid, node in graph.nodes.items()
        if node.container_id == blob_id
    ]


def blob_depth(node: Node) -> int:
    """Nesting level: 0 for top-level blobs (no container)."""
    if node.container_id is None:
        return 0
    depth = 0
    # Walk up if we had the graph, but we don't — this is a local
    # estimate from the node alone. Use graph_depth() for the full walk.
    return 1 if node.container_id is not None else 0


def graph_depth(graph: Graph, node_id: UUID) -> int:
    """Full nesting depth by walking the container chain."""
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


def orthogonal_regions(graph: Graph, container_id: UUID | None = None) -> dict[str, list[UUID]]:
    """Nodes grouped by partition within a container (or top-level)."""
    groups: dict[str, list[UUID]] = defaultdict(list)
    for nid, node in graph.nodes.items():
        if node.container_id == container_id:
            partition = node.orthogonal_partition or "_default"
            groups[partition].append(nid)
    return dict(groups)


def sibling_blobs(graph: Graph, node_id: UUID) -> list[UUID]:
    """Same-container siblings excluding the node itself."""
    node = graph.nodes.get(node_id)
    if node is None:
        return []
    return [
        nid for nid, n in graph.nodes.items()
        if n.container_id == node.container_id and nid != node_id
    ]


def is_orthogonal_to(graph: Graph, a: UUID, b: UUID) -> bool:
    """True if two nodes are in different partitions of the same container."""
    na = graph.nodes.get(a)
    nb = graph.nodes.get(b)
    if na is None or nb is None:
        return False
    if na.container_id != nb.container_id:
        return False
    pa = na.orthogonal_partition
    pb = nb.orthogonal_partition
    if pa is None or pb is None:
        return False
    return pa != pb
