from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from cognitive_engine.chunker import PropSpan
from cognitive_engine.models import Edge, EdgeType, NodeType
from cognitive_engine.type_mapper import Relation

logger = logging.getLogger(__name__)

SpanKey = Tuple[int, int]

_UNSET = object()


def _span_key(span: PropSpan) -> SpanKey:
    return (span.start_char, span.end_char)


_LOOKUP: Dict[Tuple[NodeType, NodeType, str], Optional[EdgeType]] = {
    (NodeType.AXIOM, NodeType.CLAIM, "Support"): EdgeType.INFERS,
    (NodeType.AXIOM, NodeType.CLAIM, "Attack"): EdgeType.CONTRADICTS,
    (NodeType.AXIOM, NodeType.EVIDENCE, "Support"): EdgeType.SUPPORTS,
    (NodeType.AXIOM, NodeType.EVIDENCE, "Attack"): EdgeType.ATTACKS,
    (NodeType.AXIOM, NodeType.CONDITION, "Support"): EdgeType.QUALIFIES,
    (NodeType.AXIOM, NodeType.CONDITION, "Attack"): EdgeType.ATTACKS,
    (NodeType.AXIOM, NodeType.COUNTERCLAIM, "Attack"): EdgeType.ATTACKS,
    (NodeType.AXIOM, NodeType.JUSTIFICATION, "Support"): EdgeType.JUSTIFIES,
    (NodeType.EVIDENCE, NodeType.CLAIM, "Support"): EdgeType.SUPPORTS,
    (NodeType.EVIDENCE, NodeType.CLAIM, "Attack"): EdgeType.ATTACKS,
    (NodeType.EVIDENCE, NodeType.EVIDENCE, "Support"): EdgeType.SUPPORTS,
    (NodeType.EVIDENCE, NodeType.EVIDENCE, "Attack"): EdgeType.CONTRADICTS,
    (NodeType.EVIDENCE, NodeType.JUSTIFICATION, "Support"): EdgeType.JUSTIFIES,
    (NodeType.EVIDENCE, NodeType.COUNTERCLAIM, "Support"): EdgeType.ATTACKS,
    (NodeType.EVIDENCE, NodeType.COUNTERCLAIM, "Attack"): EdgeType.SUPPORTS,
    (NodeType.EVIDENCE, NodeType.FALLACY, "Attack"): EdgeType.ATTACKS,
    (NodeType.CONDITION, NodeType.CLAIM, "Support"): EdgeType.QUALIFIES,
    (NodeType.CONDITION, NodeType.CLAIM, "Attack"): EdgeType.CONTRADICTS,
    (NodeType.CONDITION, NodeType.EVIDENCE, "Support"): EdgeType.QUALIFIES,
    (NodeType.CONDITION, NodeType.EVIDENCE, "Attack"): EdgeType.ATTACKS,
    (NodeType.CONDITION, NodeType.CONDITION, "Support"): EdgeType.QUALIFIES,
    (NodeType.CONDITION, NodeType.CONDITION, "Attack"): EdgeType.CONTRADICTS,
    (NodeType.CONDITION, NodeType.COUNTERCLAIM, "Attack"): EdgeType.ATTACKS,
    (NodeType.CLAIM, NodeType.EVIDENCE, "Support"): EdgeType.JUSTIFIES,
    (NodeType.CLAIM, NodeType.EVIDENCE, "Attack"): EdgeType.REBUTS,
    (NodeType.CLAIM, NodeType.CLAIM, "Support"): EdgeType.SUPPORTS,
    (NodeType.CLAIM, NodeType.CLAIM, "Attack"): EdgeType.CONTRADICTS,
    (NodeType.CLAIM, NodeType.CONDITION, "Support"): None,
    (NodeType.CLAIM, NodeType.CONDITION, "Attack"): EdgeType.ATTACKS,
    (NodeType.CLAIM, NodeType.COUNTERCLAIM, "Support"): None,
    (NodeType.CLAIM, NodeType.COUNTERCLAIM, "Attack"): EdgeType.REBUTS,
    (NodeType.CLAIM, NodeType.JUSTIFICATION, "Support"): EdgeType.SUPPORTS,
    (NodeType.CLAIM, NodeType.AXIOM, "Support"): None,
    (NodeType.CLAIM, NodeType.FALLACY, "Attack"): EdgeType.ATTACKS,
    (NodeType.COUNTERCLAIM, NodeType.CLAIM, "Attack"): EdgeType.REBUTS,
    (NodeType.COUNTERCLAIM, NodeType.CLAIM, "Support"): None,
    (NodeType.COUNTERCLAIM, NodeType.EVIDENCE, "Support"): EdgeType.SUPPORTS,
    (NodeType.COUNTERCLAIM, NodeType.EVIDENCE, "Attack"): EdgeType.ATTACKS,
    (NodeType.COUNTERCLAIM, NodeType.COUNTERCLAIM, "Support"): EdgeType.SUPPORTS,
    (NodeType.COUNTERCLAIM, NodeType.COUNTERCLAIM, "Attack"): EdgeType.CONTRADICTS,
    (NodeType.FALLACY, NodeType.CLAIM, "Support"): None,
    (NodeType.FALLACY, NodeType.CLAIM, "Attack"): EdgeType.ATTACKS,
    (NodeType.FALLACY, NodeType.EVIDENCE, "Attack"): EdgeType.ATTACKS,
    (NodeType.FALLACY, NodeType.EVIDENCE, "Support"): None,
    (NodeType.FALLACY, NodeType.JUSTIFICATION, "Attack"): EdgeType.ATTACKS,
    (NodeType.JUSTIFICATION, NodeType.CLAIM, "Support"): EdgeType.JUSTIFIES,
    (NodeType.JUSTIFICATION, NodeType.CLAIM, "Attack"): EdgeType.ATTACKS,
    (NodeType.JUSTIFICATION, NodeType.EVIDENCE, "Support"): EdgeType.SUPPORTS,
    (NodeType.JUSTIFICATION, NodeType.EVIDENCE, "Attack"): EdgeType.ATTACKS,
    (NodeType.JUSTIFICATION, NodeType.JUSTIFICATION, "Support"): EdgeType.SUPPORTS,
    (NodeType.JUSTIFICATION, NodeType.JUSTIFICATION, "Attack"): EdgeType.CONTRADICTS,
}

_NODE_TYPE_PRIORITY: Dict[NodeType, int] = {
    NodeType.AXIOM: 0,
    NodeType.CONDITION: 1,
    NodeType.JUSTIFICATION: 2,
    NodeType.EVIDENCE: 3,
    NodeType.COUNTERCLAIM: 4,
    NodeType.CLAIM: 5,
    NodeType.FALLACY: 6,
}


def _resolve_undirected(
    rel: Relation,
    typed_map: Dict[SpanKey, NodeType],
) -> Optional[Tuple[NodeType, NodeType, str]]:
    src_type = typed_map.get(_span_key(rel.source_span))
    tgt_type = typed_map.get(_span_key(rel.target_span))
    if src_type is None or tgt_type is None:
        return None
    key = (src_type, tgt_type, rel.label)
    if key in _LOOKUP:
        return key
    reversed_key = (tgt_type, src_type, rel.label)
    if reversed_key in _LOOKUP:
        return reversed_key
    src_prio = _NODE_TYPE_PRIORITY.get(src_type, 99)
    tgt_prio = _NODE_TYPE_PRIORITY.get(tgt_type, 99)
    if src_prio <= tgt_prio:
        return key
    return reversed_key


def _get_demarcation(
    node_id: UUID, nodes: Dict[UUID, Node], key: str
) -> Optional[str]:
    node = nodes.get(node_id)
    if node is None:
        return None
    return node.metadata.get("demarcation", {}).get(key)


def _resolve_edge_type(
    resolved: Tuple[NodeType, NodeType, str],
    src_id: Optional[UUID] = None,
    tgt_id: Optional[UUID] = None,
    nodes: Optional[Dict[UUID, Node]] = None,
) -> Optional[EdgeType]:
    src_type, tgt_type, label = resolved
    explicit = _LOOKUP.get(resolved, _UNSET)

    if explicit is not _UNSET:
        if explicit is None:
            return None
        edge_type = explicit
    elif label == "Support":
        edge_type = EdgeType.SUPPORTS
    elif label == "Attack":
        edge_type = EdgeType.ATTACKS
    else:
        return None

    if src_id is not None and tgt_id is not None and nodes is not None:
        refined = _refine_by_demarcation(edge_type, label, src_id, tgt_id, nodes)
        if refined != edge_type:
            return refined

    return edge_type


def _refine_by_demarcation(
    edge_type: EdgeType,
    label: str,
    src_id: UUID,
    tgt_id: UUID,
    nodes: Dict[UUID, Node],
) -> EdgeType:
    src_dem = _get_demarcation(src_id, nodes, "epistemic_vs_institutional")
    tgt_dem = _get_demarcation(tgt_id, nodes, "epistemic_vs_institutional")

    if edge_type == EdgeType.SUPPORTS:
        if src_dem == "INSTITUTIONAL" and tgt_dem == "EPISTEMIC":
            return EdgeType.QUALIFIES

        src_cog = _get_demarcation(src_id, nodes, "cognitive_vs_epistemic")
        tgt_cog = _get_demarcation(tgt_id, nodes, "cognitive_vs_epistemic")
        if {src_cog, tgt_cog} == {"EPISTEMIC", "COGNITIVE"}:
            return EdgeType.QUALIFIES

        src_aff = _get_demarcation(src_id, nodes, "affect_vs_cognition")
        if src_aff == "AFFECT":
            return EdgeType.QUALIFIES

        src_con = _get_demarcation(src_id, nodes, "constraint_vs_enablement")
        tgt_con = _get_demarcation(tgt_id, nodes, "constraint_vs_enablement")
        if {src_con, tgt_con} == {"CONSTRAINT", "ENABLEMENT"}:
            return EdgeType.QUALIFIES

        src_syn = _get_demarcation(src_id, nodes, "synchronic_vs_diachronic")
        tgt_syn = _get_demarcation(tgt_id, nodes, "synchronic_vs_diachronic")
        if {src_syn, tgt_syn} == {"SYNCHRONIC", "DIACHRONIC"}:
            return EdgeType.QUALIFIES

    if label == "Attack" and edge_type in (EdgeType.ATTACKS, EdgeType.CONTRADICTS):
        if src_dem == "INSTITUTIONAL" and tgt_dem == "EPISTEMIC":
            return EdgeType.REBUTS

        src_cog = _get_demarcation(src_id, nodes, "cognitive_vs_epistemic")
        tgt_cog = _get_demarcation(tgt_id, nodes, "cognitive_vs_epistemic")
        if {src_cog, tgt_cog} == {"EPISTEMIC", "COGNITIVE"}:
            return EdgeType.REBUTS

        src_aff = _get_demarcation(src_id, nodes, "affect_vs_cognition")
        if src_aff == "AFFECT":
            return EdgeType.REBUTS

        src_con = _get_demarcation(src_id, nodes, "constraint_vs_enablement")
        tgt_con = _get_demarcation(tgt_id, nodes, "constraint_vs_enablement")
        if {src_con, tgt_con} == {"CONSTRAINT", "ENABLEMENT"}:
            return EdgeType.REBUTS

        src_syn = _get_demarcation(src_id, nodes, "synchronic_vs_diachronic")
        tgt_syn = _get_demarcation(tgt_id, nodes, "synchronic_vs_diachronic")
        if {src_syn, tgt_syn} == {"SYNCHRONIC", "DIACHRONIC"}:
            return EdgeType.REBUTS

    return edge_type


def assign_edges(
    typed_spans: List[Tuple[PropSpan, NodeType]],
    relations: List[Relation],
    node_map: Dict[SpanKey, UUID],
    existing_nodes: Dict[UUID, Node],
) -> List[Edge]:
    typed_map: Dict[SpanKey, NodeType] = {_span_key(s): t for s, t in typed_spans}
    edges: List[Edge] = []
    seen: Set[Tuple[UUID, UUID]] = set()

    for rel in relations:
        resolved = _resolve_undirected(rel, typed_map)
        if resolved is None:
            continue

        src_type = typed_map.get(_span_key(rel.source_span))
        tgt_type = typed_map.get(_span_key(rel.target_span))
        if resolved == (src_type, tgt_type, rel.label):
            src_key, tgt_key = _span_key(rel.source_span), _span_key(rel.target_span)
        else:
            src_key, tgt_key = _span_key(rel.target_span), _span_key(rel.source_span)

        src_id = node_map.get(src_key)
        tgt_id = node_map.get(tgt_key)
        if src_id is None or tgt_id is None:
            continue
        if src_id == tgt_id:
            continue

        edge_type = _resolve_edge_type(resolved, src_id=src_id, tgt_id=tgt_id, nodes=existing_nodes)
        if edge_type is None:
            continue

        pair = (src_id, tgt_id)
        if pair in seen:
            continue
        seen.add(pair)

        edges.append(
            Edge(
                source_id=src_id,
                target_id=tgt_id,
                type=edge_type,
            )
        )

    return edges
