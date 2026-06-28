"""Structural analysis post-passes — emergent property + feedback loop detection.

All functions are read-only queries on a Graph. They traverse structure,
annotate via return values or graph metadata, and never mutate nodes/edges.

Detection signatures:
  - Cascading failure: dependency chain where a node has low reliability
  - Single-point-of-failure: node with high fan-in and no backup in partition
  - Feedback loops: within-DSM cycles + cross-domain dependency deduction

Usage:
    props = detect_cascading_failure(graph)
    props += detect_spof(graph)
    loops = detect_feedback_loops(graph)
    graph.metadata["deduced_dependencies"]  # cross-domain results
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from dynafx.core.models import (
    EdgeType,
    EmergentProperty,
    FeedbackLoop,
    Graph,
    Node,
    Opinion,
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────


def _outgoing_dependency_edges(graph: Graph, node_id: UUID) -> list[UUID]:
    targets: list[UUID] = []
    for edge in graph.edges.values():
        if edge.source_id == node_id and edge.type in (EdgeType.DEPENDS, EdgeType.CAUSES):
            targets.append(edge.target_id)
    return targets


def _incoming_dependency_edges(graph: Graph, node_id: UUID) -> list[UUID]:
    sources: list[UUID] = []
    for edge in graph.edges.values():
        if edge.target_id == node_id and edge.type in (EdgeType.DEPENDS, EdgeType.CAUSES):
            sources.append(edge.source_id)
    return sources


def _node_reliability(node: Node) -> float:
    if "confidence" in node.metadata:
        return node.metadata["confidence"]
    return node.opinion.belief if node.opinion else 0.0


# ── Cascading failure ────────────────────────────────────────────


def detect_cascading_failure(
    graph: Graph,
    reliability_threshold: float = 0.3,
) -> list[EmergentProperty]:
    properties: list[EmergentProperty] = []

    for node in graph.nodes.values():
        downstream = _outgoing_dependency_edges(graph, node.id)
        if not downstream:
            continue
        chain = _walk_dependency_chain(graph, node.id, max_hops=3)
        if len(chain) < 2:
            continue
        fragile: list[UUID] = []
        for nid in chain:
            n = graph.nodes.get(nid)
            if n is None:
                continue
            if _node_reliability(n) < reliability_threshold:
                fragile.append(nid)
        if not fragile:
            continue
        properties.append(EmergentProperty(
            name="cascading_failure",
            condition=f"reliability<{reliability_threshold}",
            involved_ids=chain,
            opinion=Opinion(
                belief=0.6, disbelief=0.2, uncertainty=0.2, prior=0.5,
            ),
            detected_by="detect_cascading_failure",
            trace_ref="",
        ))

    return properties


def _walk_dependency_chain(
    graph: Graph,
    start: UUID,
    max_hops: int = 3,
) -> list[UUID]:
    chain: list[UUID] = [start]
    current = start
    visited: set[UUID] = {start}
    for _ in range(max_hops):
        next_nodes = _outgoing_dependency_edges(graph, current)
        if not next_nodes:
            break
        current = next_nodes[0]
        if current in visited:
            break
        visited.add(current)
        chain.append(current)
    return chain


# ── Single point of failure ──────────────────────────────────────


def detect_spof(
    graph: Graph,
    fan_in_threshold: int = 3,
) -> list[EmergentProperty]:
    properties: list[EmergentProperty] = []
    fan_in: dict[UUID, list[UUID]] = {}
    for edge in graph.edges.values():
        if edge.type in (EdgeType.DEPENDS, EdgeType.CAUSES):
            fan_in.setdefault(edge.target_id, []).append(edge.source_id)

    for target_id, sources in fan_in.items():
        if len(sources) < fan_in_threshold:
            continue
        target = graph.nodes.get(target_id)
        if target is None:
            continue
        partition = target.orthogonal_partition
        has_backup = False
        if partition is not None:
            for nid, node in graph.nodes.items():
                if (nid != target_id
                        and node.orthogonal_partition == partition
                        and node.container_id == target.container_id):
                    has_backup = True
                    break
        if has_backup:
            continue
        properties.append(EmergentProperty(
            name="single_point_of_failure",
            condition=f"fan_in>={fan_in_threshold}",
            involved_ids=[target_id] + sources,
            opinion=Opinion(
                belief=0.7, disbelief=0.1, uncertainty=0.2, prior=0.5,
            ),
            detected_by="detect_spof",
            trace_ref="",
        ))

    return properties


# ── Feedback loops (MDM-based) ───────────────────────────────────


def _build_mdm(graph: Graph):
    """Build a MultipleDomainMatrix from a graph.

    Returns:
        Tuple of (mdm, name_to_id).
    """
    from dynafx.mdm.matrix import MultipleDomainMatrix

    partitions: dict[str, list[UUID]] = {}
    node_partition: dict[UUID, str] = {}
    for n in graph.nodes.values():
        part = n.orthogonal_partition or "_unassigned"
        partitions.setdefault(part, []).append(n.id)
        node_partition[n.id] = part

    name_to_id: dict[str, UUID] = {}
    for n in graph.nodes.values():
        name_to_id[n.text] = n.id

    mdm = MultipleDomainMatrix()
    for part_name, node_ids in partitions.items():
        elements = [graph.nodes[nid].text for nid in node_ids]
        mdm.add_domain(part_name, elements)

    for part_name, node_ids in partitions.items():
        dsm = mdm.get_dsm(part_name)
        id_set = set(node_ids)
        for src_id in node_ids:
            src_name = graph.nodes[src_id].text
            for e in graph.edges.values():
                if e.source_id != src_id:
                    continue
                if e.type.name == "ASSOCIATED_WITH":
                    continue
                if e.target_id in id_set:
                    dsm.add_relation(src_name, graph.nodes[e.target_id].text)

    cross_edges: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for e in graph.edges.values():
        if e.type.name == "ASSOCIATED_WITH":
            continue
        src_part = node_partition.get(e.source_id)
        tgt_part = node_partition.get(e.target_id)
        if src_part and tgt_part and src_part != tgt_part:
            cross_edges.setdefault((src_part, tgt_part), []).append(
                (graph.nodes[e.source_id].text,
                 graph.nodes[e.target_id].text),
            )

    for (src_part, tgt_part), mappings in cross_edges.items():
        dmm = mdm.add_dmm(src_part, tgt_part)
        for src_name, tgt_name in mappings:
            dmm.add_mapping(src_name, tgt_name)

    return mdm, name_to_id


def detect_feedback_loops(
    graph: Graph,
    max_depth: int = 3,
) -> list[FeedbackLoop]:
    """Build the MDM and detect feedback loops + cross-domain dependencies.

    Runs within-DSM cycle detection (DFS-based) and cross-domain
    dependency deduction for all domain pairs. Deduced dependencies
    are stored in ``graph.metadata["deduced_dependencies"]``.

    Args:
        graph: The graph to analyze.
        max_depth: Max intermediate hops for cross-domain deduction.

    Returns:
        List of FeedbackLoop objects.
    """
    mdm, name_to_id = _build_mdm(graph)

    # Within-DSM cycle detection
    seen: set[frozenset[UUID]] = set()
    loops: list[FeedbackLoop] = []

    for part_name in list(mdm.domains.keys()):
        dsm = mdm.get_dsm(part_name)
        if dsm is None:
            continue
        for cycle_names in dsm.find_feedback_loops():
            node_ids = [name_to_id.get(n) for n in cycle_names if n in name_to_id]
            node_ids = [nid for nid in node_ids if nid is not None]
            if len(node_ids) < 2:
                continue
            key = frozenset(node_ids)
            if key in seen:
                continue
            seen.add(key)

            node_set = set(node_ids)
            negative_count = 0
            edge_count = 0
            for nid in node_ids:
                for e in graph.edges.values():
                    if e.source_id != nid:
                        continue
                    if e.type.name == "ASSOCIATED_WITH":
                        continue
                    if e.target_id in node_set:
                        edge_count += 1
                        if e.polarity == -1:
                            negative_count += 1

            loop_type = "reinforcing" if negative_count % 2 == 0 else "balancing"
            gain_sign = "+" if loop_type == "reinforcing" else "-"

            loops.append(FeedbackLoop(
                nodes=node_ids,
                loop_type=loop_type,
                gain_sign=gain_sign,
                edge_count=edge_count,
                negative_count=negative_count,
            ))

    if loops:
        logger.info("Detected %d feedback loops", len(loops))

    # Cross-domain dependency deduction (all pairs)
    domain_names = list(mdm.domains.keys())
    all_deduced: dict[str, list[dict]] = {}

    for src_domain in domain_names:
        for tgt_domain in domain_names:
            if src_domain == tgt_domain:
                continue
            result = mdm.deduce_dependencies(src_domain, tgt_domain, max_depth)
            if result is not None:
                key = f"{src_domain}→{tgt_domain}"
                pairs: list[dict] = []
                src_elems = mdm.domains[src_domain]
                tgt_elems = mdm.domains[tgt_domain]
                for i, src_name in enumerate(src_elems):
                    for j, tgt_name in enumerate(tgt_elems):
                        val = float(result[i][j])
                        if val > 0:
                            pairs.append({
                                "source": src_name,
                                "target": tgt_name,
                                "strength": round(val, 4),
                            })
                if pairs:
                    all_deduced[key] = pairs

    if all_deduced:
        graph.metadata["deduced_dependencies"] = all_deduced
        total = sum(len(v) for v in all_deduced.values())
        logger.info("Deduced %d indirect dependencies across %d domain pairs",
                    total, len(all_deduced))

    return loops


# ── Aggregate entry point ────────────────────────────────────────


def detect_all(graph: Graph, **kwargs: Any) -> list[EmergentProperty]:
    """Run all detection passes.

    Combines emergent property detection (cascading failure, SPOF)
    and feedback loop analysis (cycles + cross-domain deduction).
    Each pass writes its findings to the graph in addition to returning them.

    Returns:
        List of EmergentProperty objects. Feedback loops are returned
        separately via this function's side effects on graph metadata.
    """
    results: list[EmergentProperty] = []
    results.extend(detect_cascading_failure(graph, **kwargs))
    results.extend(detect_spof(graph, **kwargs))
    graph.emergent_properties.extend(results)

    detect_feedback_loops(graph)

    if results:
        logger.info("Detected %d emergent properties", len(results))
    return results
