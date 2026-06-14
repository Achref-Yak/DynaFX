"""Decision-tree lens — branch graph on conditional edges into distinct scenarios.

For each branching edge (QUALIFIES or equivalent), create a scenario
subgraph that includes only the nodes and edges relevant to that branch.
Each scenario is stored as a JSON-serializable dict in
graph.metadata["scenarios"] with auto-generated names.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import UUID

from cognitive_engine.core.models import Graph, Node, Edge


def decision_tree_lens(graph: Graph, **params) -> Graph:
    cfg = params.get("config", None)
    if cfg is None:
        from cognitive_engine.domain import domain as _domain
        cfg = _domain.active()
    branch_types = params.get("branch_types", {"QUALIFIES"})

    result = deepcopy(graph)

    branching_edges = [
        e for e in result.edges
        if e.type.name in branch_types
    ]

    if not branching_edges:
        # Single scenario: the full graph
        scenario = _build_scenario_full(result, "Default scenario")
        result.metadata["scenarios"] = [scenario]
        result.metadata["branch_edge_count"] = 0
        result.metadata["lens"] = "decision_tree"
        return result

    scenarios = []
    for i, edge in enumerate(branching_edges):
        scenario = _build_scenario(result, edge, i + 1)
        scenarios.append(scenario)

    result.metadata["scenarios"] = scenarios
    result.metadata["branch_edge_count"] = len(branching_edges)
    result.metadata["lens"] = "decision_tree"
    return result


def _build_scenario(graph: Graph, branch_edge: Edge, index: int) -> dict:
    """Build a scenario subgraph around a branching edge."""
    source_node = graph.nodes.get(branch_edge.source_id)
    target_node = graph.nodes.get(branch_edge.target_id)

    # Auto-name the scenario
    source_text = source_node.text[:40] if source_node else "Unknown"
    target_text = target_node.text[:40] if target_node else "Unknown"
    name = f"Scenario {index}: If \"{source_text}\" qualifies \"{target_text}\""

    # Keep only nodes connected to the branch edge's subgraph
    # Start from source and target, follow edges outward
    keep_ids: set[UUID] = set()
    _collect_connected(graph, branch_edge.source_id, keep_ids)
    _collect_connected(graph, branch_edge.target_id, keep_ids)

    # Serialize nodes
    nodes = []
    for nid in keep_ids:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        nodes.append({
            "id": nid.hex,
            "text": node.text,
            "type": node.type.name,
            "belief": round(node.opinion[0], 4),
            "category": node.category,
        })

    # Serialize edges (only those connecting kept nodes)
    edges = []
    for edge in graph.edges:
        if edge.source_id in keep_ids and edge.target_id in keep_ids:
            edges.append({
                "id": edge.id.hex,
                "source_id": edge.source_id.hex,
                "target_id": edge.target_id.hex,
                "type": edge.type.name,
                "belief": round(edge.opinion[0], 4),
            })

    # Compute summary
    beliefs = [n["belief"] for n in nodes]
    node_types = {}
    for n in nodes:
        t = n["type"]
        node_types[t] = node_types.get(t, 0) + 1

    return {
        "name": name,
        "branch_edge": {
            "source_id": branch_edge.source_id.hex,
            "source_text": source_text,
            "type": branch_edge.type.name,
            "target_id": branch_edge.target_id.hex,
            "target_text": target_text,
        },
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "avg_belief": round(sum(beliefs) / len(beliefs), 4) if beliefs else 0.0,
            "min_belief": round(min(beliefs), 4) if beliefs else 0.0,
            "max_belief": round(max(beliefs), 4) if beliefs else 0.0,
            "node_types": node_types,
            "strongest_node": max(nodes, key=lambda n: n["belief"]) if nodes else None,
            "weakest_node": min(nodes, key=lambda n: n["belief"]) if nodes else None,
        },
    }


def _build_scenario_full(graph: Graph, name: str) -> dict:
    """Build a scenario from the full graph."""
    nodes = []
    for nid, node in graph.nodes.items():
        nodes.append({
            "id": nid.hex,
            "text": node.text,
            "type": node.type.name,
            "belief": round(node.opinion[0], 4),
            "category": node.category,
        })

    edges = []
    for edge in graph.edges:
        edges.append({
            "id": edge.id.hex,
            "source_id": edge.source_id.hex,
            "target_id": edge.target_id.hex,
            "type": edge.type.name,
            "belief": round(edge.opinion[0], 4),
        })

    beliefs = [n["belief"] for n in nodes]
    node_types = {}
    for n in nodes:
        t = n["type"]
        node_types[t] = node_types.get(t, 0) + 1

    return {
        "name": name,
        "branch_edge": None,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "avg_belief": round(sum(beliefs) / len(beliefs), 4) if beliefs else 0.0,
            "min_belief": round(min(beliefs), 4) if beliefs else 0.0,
            "max_belief": round(max(beliefs), 4) if beliefs else 0.0,
            "node_types": node_types,
            "strongest_node": max(nodes, key=lambda n: n["belief"]) if nodes else None,
            "weakest_node": min(nodes, key=lambda n: n["belief"]) if nodes else None,
        },
    }


def _collect_connected(graph: Graph, start: UUID, visited: set[UUID], depth: int = 3):
    """Collect all nodes reachable from start within depth hops."""
    if depth <= 0 or start in visited:
        return
    visited.add(start)
    for edge in graph.edges:
        if edge.source_id == start:
            _collect_connected(graph, edge.target_id, visited, depth - 1)
        elif edge.target_id == start:
            _collect_connected(graph, edge.source_id, visited, depth - 1)
