from __future__ import annotations

import math
from typing import Optional

from cognitive_engine.core.models import Graph
from cognitive_engine.memory.models import LTMPattern


def retrieve_similar(
    query: Graph,
    candidates: list[LTMPattern],
    k: int = 3,
) -> list[LTMPattern]:
    """Score patterns by node-set overlap with query graph and return top-k.

    Similarity is based on:
        - Jaccard similarity of node IDs (weight 0.6)
        - Node count proximity ratio (weight 0.4)

    Args:
        query: The current graph to match against.
        candidates: List of LTM patterns to score.
        k: Maximum number of patterns to return.

    Returns:
        Top-k patterns sorted by similarity score descending.
    """
    query_nodes = set(query.nodes.keys())

    if not query_nodes or not candidates:
        return []

    scored: list[tuple[float, LTMPattern]] = []
    nq = len(query_nodes)

    for p in candidates:
        pattern_nodes = set(p.graph_snapshot.nodes.keys())

        if not pattern_nodes:
            continue

        # Jaccard similarity
        intersection = query_nodes & pattern_nodes
        union = query_nodes | pattern_nodes
        jaccard = len(intersection) / max(len(union), 1)

        # Node count proximity (1 - relative difference)
        np_count = len(pattern_nodes)
        count_proximity = 1.0 - abs(nq - np_count) / max(nq, np_count, 1)

        score = 0.6 * jaccard + 0.4 * count_proximity
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]
