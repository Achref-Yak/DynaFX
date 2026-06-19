"""Memory consolidation — STM → LTM with Leiden community detection.

Compresses a sequence of STM states into LTM patterns, using Leiden
community detection to partition the graph into meaningful clusters.
Each community becomes a separate LTMPattern for structured retrieval.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from typing import Optional
from uuid import uuid4

from cognitive_engine.core.models import Graph
from cognitive_engine.core.state import State

logger = logging.getLogger(__name__)


def build_pattern(
    stm: Iterable[State],
    session_id: str = "",
) -> list["LTMPattern"]:
    """Compress a sequence of STM states into LTM patterns per community.

    Extracts:
        - Leiden communities from the graph
        - One LTMPattern per community with its own belief signature
        - Operator trace from all states
        - Session metadata for cross-session persistence

    Returns:
        List of LTMPattern objects, one per community.
    """
    from cognitive_engine.memory.models import LTMPattern

    states = list(stm)
    if not states:
        graph = Graph()
        return [LTMPattern(
            id=uuid4(),
            graph_snapshot=graph,
            belief_signature={},
            operator_trace=[],
            cluster_labels=[],
            session_id=session_id,
        )]

    latest = states[-1]
    graph = latest.graph

    # Collect operator trace from all states
    op_trace = []
    for s in states:
        for delta in s.history:
            op_trace.append(delta.operator)

    # Detect communities using Leiden
    communities = _detect_communities(graph)

    if not communities:
        # Fallback: single pattern for the whole graph
        belief_sig = _extract_belief_signature(graph)
        return [LTMPattern(
            id=uuid4(),
            graph_snapshot=graph,
            belief_signature=belief_sig,
            operator_trace=op_trace,
            cluster_labels=[],
            session_id=session_id,
        )]

    # Create one pattern per community
    patterns = []
    for community_id, node_ids in communities.items():
        # Extract subgraph for this community
        subgraph = _extract_subgraph(graph, node_ids)

        # Extract belief signature for community nodes
        belief_sig = {}
        for nid in node_ids:
            if nid in graph.nodes:
                node = graph.nodes[nid]
                if node.opinion:
                    belief_sig[str(nid)] = node.opinion[0]

        # Generate cluster label
        node_texts = [
            graph.nodes[nid].text[:30]
            for nid in node_ids
            if nid in graph.nodes
        ]
        label = _label_cluster({
            "node_texts": node_texts,
            "cluster_type": "Community",
        })

        patterns.append(LTMPattern(
            id=uuid4(),
            graph_snapshot=subgraph,
            belief_signature=belief_sig,
            operator_trace=list(op_trace),
            cluster_labels=[label],
            session_id=session_id,
            community_id=community_id,
        ))

    return patterns


def _detect_communities(graph: Graph) -> dict[int, list]:
    """Detect communities using Leiden algorithm.

    Returns dict mapping community_id to list of node UUIDs.
    Falls back to connected components if Leiden is unavailable.
    """
    if not graph.nodes:
        return {}

    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        logger.debug("leidenalg not available, falling back to connected components")
        return _detect_communities_fallback(graph)

    # Build igraph from our graph
    node_ids = list(graph.nodes.keys())
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    edges = []
    for edge in graph.edges.values():
        if edge.source_id in node_idx and edge.target_id in node_idx:
            edges.append((node_idx[edge.source_id], node_idx[edge.target_id]))

    if not edges:
        return _detect_communities_fallback(graph)

    g = ig.Graph(n=len(node_ids), edges=edges, directed=False)

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights=None,
        resolution_parameter=1.0,
    )

    # Convert to our format
    communities: dict[int, list] = {}
    for community_id, membership in enumerate(partition.membership):
        if membership not in communities:
            communities[membership] = []
        communities[membership].append(node_ids[community_id])

    return communities


def _detect_communities_fallback(graph: Graph) -> dict[int, list]:
    """Fallback community detection using connected components."""
    import networkx as nx

    nx_graph = nx.Graph()
    for nid in graph.nodes:
        nx_graph.add_node(nid)
    for edge in graph.edges.values():
        nx_graph.add_edge(edge.source_id, edge.target_id)

    communities = {}
    for i, component in enumerate(nx.connected_components(nx_graph)):
        communities[i] = list(component)

    return communities


def _extract_subgraph(graph: Graph, node_ids: list) -> Graph:
    """Extract a subgraph containing only the specified nodes and their edges."""
    from cognitive_engine.core.models import Node, Edge

    node_set = set(node_ids)
    subgraph = Graph()

    for nid in node_ids:
        if nid in graph.nodes:
            subgraph.nodes[nid] = graph.nodes[nid]

    for edge in graph.edges.values():
        if edge.source_id in node_set and edge.target_id in node_set:
            subgraph.edges[edge.id] = edge

    return subgraph


def _extract_belief_signature(graph: Graph) -> dict:
    """Extract belief values per node as a signature."""
    belief_sig = {}
    for nid, node in graph.nodes.items():
        if node.opinion:
            belief_sig[str(nid)] = node.opinion[0]
    return belief_sig


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
