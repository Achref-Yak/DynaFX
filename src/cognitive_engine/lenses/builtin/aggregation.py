"""Aggregation lens — summarize opinions across the graph.

Produces per-category and per-type mean opinions, edge statistics,
and identifies the weakest/strongest nodes.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from statistics import mean

from cognitive_engine.core.models import Graph
from cognitive_engine.reason.evidence import mean_opinion


def aggregation_lens(graph: Graph, **params) -> Graph:
    result = deepcopy(graph)

    # Node statistics
    by_category: dict[int, list] = defaultdict(list)
    by_type: dict[str, list] = defaultdict(list)
    node_beliefs: list[tuple[str, str, str, float]] = []

    for node in result.nodes.values():
        by_category[node.category].append(node.opinion)
        by_type[node.type.name].append(node.opinion)
        node_beliefs.append((
            node.id.hex,
            node.text[:60],
            node.type.name,
            node.opinion[0],
        ))

    # Edge statistics
    by_edge_type: dict[str, list] = defaultdict(list)
    for edge in result.edges:
        by_edge_type[edge.type.name].append(edge.opinion[0])

    # Build type-level stats
    type_stats = {}
    for t, opinions in sorted(by_type.items()):
        beliefs = [o[0] for o in opinions]
        type_stats[t] = {
            "count": len(opinions),
            "mean_belief": round(mean(beliefs), 4),
            "min_belief": round(min(beliefs), 4),
            "max_belief": round(max(beliefs), 4),
        }

    # Build edge type stats
    edge_type_stats = {}
    for et, beliefs in sorted(by_edge_type.items()):
        edge_type_stats[et] = {
            "count": len(beliefs),
            "mean_belief": round(mean(beliefs), 4),
        }

    # Find weakest and strongest nodes
    weakest = min(node_beliefs, key=lambda x: x[3]) if node_beliefs else None
    strongest = max(node_beliefs, key=lambda x: x[3]) if node_beliefs else None

    agg = {
        "node_count": len(result.nodes),
        "edge_count": len(result.edges),
        "by_category": {
            str(cat): mean_opinion(opinions)
            for cat, opinions in sorted(by_category.items())
        },
        "by_type": type_stats,
        "by_edge_type": edge_type_stats,
        "weakest_node": {
            "id": weakest[0],
            "text": weakest[1],
            "type": weakest[2],
            "belief": round(weakest[3], 4),
        } if weakest else None,
        "strongest_node": {
            "id": strongest[0],
            "text": strongest[1],
            "type": strongest[2],
            "belief": round(strongest[3], 4),
        } if strongest else None,
    }

    result.metadata["aggregation"] = agg
    result.metadata["lens"] = "aggregation"
    return result
