from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Optional
from uuid import uuid4

from cognitive_engine.core.models import Graph
from cognitive_engine.core.state import State


def build_pattern(stm: Iterable[State]) -> "LTMPattern":
    """Compress a sequence of STM states into a single LTM pattern.

    Extracts:
        - The final graph as the snapshot
        - Average belief values per node as belief signature
        - All operator names as a trace
        - Emergence cluster labels from the final state's metadata
    """
    from cognitive_engine.memory.models import LTMPattern

    states = list(stm)
    if not states:
        graph = Graph()
        return LTMPattern(
            id=uuid4(),
            graph_snapshot=graph,
            belief_signature={},
            operator_trace=[],
            cluster_labels=[],
        )

    latest = states[-1]
    graph = latest.graph

    # Extract belief signature (avg belief per node)
    belief_sig = {}
    for nid, node in graph.nodes.items():
        if node.opinion:
            belief_sig[str(nid)] = node.opinion[0]

    # Collect operator trace
    op_trace = []
    for s in states:
        for delta in s.history:
            op_trace.append(delta.operator)

    # Extract emergence cluster labels
    emergence = latest.metadata.get("emergence", {})
    clusters = emergence.get("clusters", [])
    labels = []
    for c in clusters:
        label = _label_cluster(c)
        labels.append(label)

    return LTMPattern(
        id=uuid4(),
        graph_snapshot=graph,
        belief_signature=belief_sig,
        operator_trace=op_trace,
        cluster_labels=labels,
        last_accessed=0.0,
    )


def _label_cluster(cluster: dict) -> str:
    """Generate a human-readable label for an emergence cluster."""
    node_texts = cluster.get("node_texts", [])
    cluster_type = cluster.get("cluster_type", "unknown")

    if not node_texts:
        return f"Cluster ({cluster_type})"

    # Use first few node texts as the label
    prefix = node_texts[:3]
    label = f"{cluster_type}: {', '.join(prefix)}"
    if len(node_texts) > 3:
        label += f" +{len(node_texts) - 3} more"

    return label
