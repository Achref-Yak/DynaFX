"""Funnel lens — extract a single root-to-leaf inference chain.

Starting from the root claim (highest projected probability), follow the
strongest supporting edge at each node to build a dominant inference path.
The chain is stored in graph metadata as a list of node details with
confidence scores and edge information.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import UUID

from cognitive_engine.core.models import Graph, Node


def funnel_lens(graph: Graph, **params) -> Graph:
    result = deepcopy(graph)

    # Build adjacency: source_hex -> [(target_uuid, edge)]
    nodes_with_children: dict[str, list[tuple[UUID, Any]]] = {}
    for edge in result.edges:
        src_hex = edge.source_id.hex
        nodes_with_children.setdefault(src_hex, []).append(
            (edge.target_id, edge)
        )

    # Find root: source nodes that are not targets (or highest belief if none)
    targets = {e.target_id for e in result.edges}
    sources = {e.source_id for e in result.edges}
    candidate_uuids = sources - targets

    if not candidate_uuids:
        # Fallback: pick the node with highest belief
        root = max(
            result.nodes.keys(),
            key=lambda nid: result.nodes[nid].opinion[0],
        )
    else:
        # Pick the candidate with highest belief
        root = max(
            candidate_uuids,
            key=lambda nid: result.nodes.get(nid, Node()).opinion[0],
        )

    # Build chain by following strongest edge at each step
    chain: list[dict] = []
    visited: set[UUID] = {root}
    current = root

    while current.hex in nodes_with_children:
        children = [
            (tid, edge)
            for tid, edge in nodes_with_children[current.hex]
            if tid not in visited
        ]
        if not children:
            break

        # Pick child with highest belief
        target_id, edge = max(
            children,
            key=lambda pair: result.nodes.get(
                pair[0], Node()
            ).opinion[0],
        )

        child_node = result.nodes.get(target_id)
        if child_node is None:
            break

        # Get edge confidence from warrant if available
        edge_confidence = _edge_confidence(edge)

        chain.append({
            "id": current.hex,
            "text": result.nodes[current].text,
            "type": result.nodes[current].type.name,
            "belief": round(result.nodes[current].opinion[0], 4),
            "edge_type": edge.type.name if edge else None,
            "edge_confidence": edge_confidence,
        })

        visited.add(target_id)
        current = target_id

    # Add the final node (leaf)
    if current in result.nodes:
        leaf = result.nodes[current]
        chain.append({
            "id": current.hex,
            "text": leaf.text,
            "type": leaf.type.name,
            "belief": round(leaf.opinion[0], 4),
            "edge_type": None,
            "edge_confidence": None,
        })

    # Compute summary statistics
    beliefs = [step["belief"] for step in chain]
    weak_links = [
        step for step in chain
        if step["belief"] < 0.3
    ]

    result.metadata["funnel_chain"] = chain
    result.metadata["funnel_summary"] = {
        "length": len(chain),
        "min_belief": round(min(beliefs), 4) if beliefs else 0.0,
        "max_belief": round(max(beliefs), 4) if beliefs else 0.0,
        "avg_belief": round(sum(beliefs) / len(beliefs), 4) if beliefs else 0.0,
        "weak_links": [
            {"id": wl["id"], "text": wl["text"][:60], "belief": wl["belief"]}
            for wl in weak_links
        ],
    }
    result.metadata["lens"] = "funnel"
    return result


def _edge_confidence(edge) -> float | None:
    """Extract confidence score from edge warrant if available."""
    if edge is None or edge.warrant is None:
        return None
    # Warrant is (opinion_for, opinion_against) — use belief of for opinion
    if isinstance(edge.warrant, tuple) and len(edge.warrant) >= 1:
        for_opinion = edge.warrant[0]
        if isinstance(for_opinion, tuple) and len(for_opinion) >= 1:
            return round(for_opinion[0], 4)
    return None
