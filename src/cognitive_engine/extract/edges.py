from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from cognitive_engine.nlp.chunker import PropSpan
from cognitive_engine.core.models import Edge, EdgeType, NodeType
from cognitive_engine.extract.types import Relation

logger = logging.getLogger(__name__)

SpanKey = Tuple[int, int]

_UNSET = object()

# ── CAUSES edge inference patterns ───────────────────────────────

_TEMPORAL_MARKERS = {
    "last night", "last night", "earlier", "before", "previously",
    "yesterday", "this morning", "last week", "last month",
}

_CAUSAL_VERBS = {
    "cause", "caused", "leads", "led", "result", "results",
    "made", "makes", "produce", "produced", "create", "created",
    "bring", "brought", "trigger", "triggered", "start", "started",
    "begin", "began", "initiate", "initiated",
}

_CAUSAL_PARTICLES = {
    "because", "since", "due to", "from", "as a result",
    "consequently", "therefore", "thus", "hence",
}

_EFFECT_INDICATORS = {
    "wet", "wetness", "water", "moisture", "damp", "dampness",
    "puddle", "puddles", "flood", "flooding", "overflow",
}

_CAUSE_INDICATORS = {
    "rain", "rained", "raining", "sprinkler", "sprinklers",
    "pipe", "pipes", "burst", "break", "leak", "leaking",
    "hose", "water", "spill", "spilled", "spilling",
}


def _detect_causal_keywords(text: str) -> Tuple[bool, bool]:
    """Detect if text contains causal or effect keywords."""
    text_lower = text.lower()
    has_cause = any(kw in text_lower for kw in _CAUSE_INDICATORS)
    has_effect = any(kw in text_lower for kw in _EFFECT_INDICATORS)
    return has_cause, has_effect


def _detect_temporal_order(src_text: str, tgt_text: str) -> bool:
    """Check if source text likely happened before target text."""
    src_lower = src_text.lower()
    tgt_lower = tgt_text.lower()

    src_has_temporal = any(m in src_lower for m in _TEMPORAL_MARKERS)
    tgt_has_temporal = any(m in tgt_lower for m in _TEMPORAL_MARKERS)

    if src_has_temporal and not tgt_has_temporal:
        return True
    if not src_has_temporal and tgt_has_temporal:
        return False

    past_tense_indicators = {"was", "were", "had", "did", "ran", "fell"}
    present_indicators = {"is", "are", "has", "have", "does"}

    src_past = any(w in src_lower.split() for w in past_tense_indicators)
    tgt_present = any(w in tgt_lower.split() for w in present_indicators)

    if src_past and tgt_present:
        return True

    return False


def infer_causal_edges(nodes: Dict[UUID, Node]) -> Dict[UUID, Edge]:
    """Infer CAUSES edges between nodes based on causal/temporal patterns."""
    edges: Dict[UUID, Edge] = {}
    node_list = list(nodes.values())

    for i, src_node in enumerate(node_list):
        for j, tgt_node in enumerate(node_list):
            if i >= j:
                continue

            src_cause, src_effect = _detect_causal_keywords(src_node.text)
            tgt_cause, tgt_effect = _detect_causal_keywords(tgt_node.text)

            if src_cause and tgt_effect:
                if _detect_temporal_order(src_node.text, tgt_node.text):
                    e = Edge(
                        source_id=src_node.id,
                        target_id=tgt_node.id,
                        type=EdgeType.CAUSES,
                    )
                    edges[e.id] = e
            elif tgt_cause and src_effect:
                if _detect_temporal_order(tgt_node.text, src_node.text):
                    e = Edge(
                        source_id=tgt_node.id,
                        target_id=src_node.id,
                        type=EdgeType.CAUSES,
                    )
                    edges[e.id] = e

    return edges


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
    # World-model combinations (label-agnostic: match any label)
    (NodeType.AGENT, NodeType.PROCESS, "Support"): EdgeType.ENABLES,
    (NodeType.AGENT, NodeType.PROCESS, "Attack"): EdgeType.ENABLES,
    (NodeType.AGENT, NodeType.GOAL, "Support"): EdgeType.HAS_GOAL,
    (NodeType.AGENT, NodeType.GOAL, "Attack"): EdgeType.HAS_GOAL,
    (NodeType.AGENT, NodeType.ACTION, "Support"): EdgeType.INTENDS,
    (NodeType.AGENT, NodeType.ACTION, "Attack"): EdgeType.INTENDS,
    (NodeType.AGENT, NodeType.BELIEF, "Support"): EdgeType.KNOWS,
    (NodeType.AGENT, NodeType.BELIEF, "Attack"): EdgeType.KNOWS,
    (NodeType.AGENT, NodeType.KNOWLEDGE, "Support"): EdgeType.KNOWS,
    (NodeType.AGENT, NodeType.KNOWLEDGE, "Attack"): EdgeType.KNOWS,
    (NodeType.AGENT, NodeType.RESOURCE, "Support"): EdgeType.USES,
    (NodeType.AGENT, NodeType.RESOURCE, "Attack"): EdgeType.USES,
    (NodeType.PROCESS, NodeType.STATE, "Support"): EdgeType.CAUSES,
    (NodeType.PROCESS, NodeType.STATE, "Attack"): EdgeType.CAUSES,
    (NodeType.PROCESS, NodeType.RESOURCE, "Support"): EdgeType.PRODUCES,
    (NodeType.PROCESS, NodeType.RESOURCE, "Attack"): EdgeType.PRODUCES,
    (NodeType.STATE, NodeType.PROPERTY, "Support"): EdgeType.HAS_ATTRIBUTE,
    (NodeType.STATE, NodeType.PROPERTY, "Attack"): EdgeType.HAS_ATTRIBUTE,
    (NodeType.STATE, NodeType.STATE, "Support"): EdgeType.TRANSFORMS,
    (NodeType.STATE, NodeType.STATE, "Attack"): EdgeType.TRANSFORMS,
    (NodeType.ACTION, NodeType.STATE, "Support"): EdgeType.CAUSES,
    (NodeType.ACTION, NodeType.STATE, "Attack"): EdgeType.CAUSES,
    (NodeType.ACTION, NodeType.RESOURCE, "Support"): EdgeType.PRODUCES,
    (NodeType.ACTION, NodeType.RESOURCE, "Attack"): EdgeType.PRODUCES,
    (NodeType.GOAL, NodeType.ACTION, "Support"): EdgeType.ENABLES,
    (NodeType.GOAL, NodeType.ACTION, "Attack"): EdgeType.ENABLES,
    (NodeType.CONSTRAINT, NodeType.PROCESS, "Support"): EdgeType.ENABLES,
    (NodeType.CONSTRAINT, NodeType.PROCESS, "Attack"): EdgeType.ENABLES,
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
    elif label == "Causes":
        edge_type = EdgeType.CAUSES
    elif label == "Enables":
        edge_type = EdgeType.ENABLES
    elif label == "Depends":
        edge_type = EdgeType.DEPENDS
    elif label == "PartOf":
        edge_type = EdgeType.PART_OF
    else:
        return None

    if src_id is not None and tgt_id is not None and nodes is not None:
        refined = _refine_by_demarcation(edge_type, label, src_id, tgt_id, nodes)
        if refined != edge_type:
            return refined

    return edge_type


def _qualifies_by_demarcation(
    src_id: UUID, tgt_id: UUID, nodes: Dict[UUID, Node],
) -> bool:
    src_dem = _get_demarcation(src_id, nodes, "epistemic_vs_institutional")
    tgt_dem = _get_demarcation(tgt_id, nodes, "epistemic_vs_institutional")
    if src_dem == "INSTITUTIONAL" and tgt_dem == "EPISTEMIC":
        return True

    src_cog = _get_demarcation(src_id, nodes, "cognitive_vs_epistemic")
    tgt_cog = _get_demarcation(tgt_id, nodes, "cognitive_vs_epistemic")
    if {src_cog, tgt_cog} == {"EPISTEMIC", "COGNITIVE"}:
        return True

    src_aff = _get_demarcation(src_id, nodes, "affect_vs_cognition")
    if src_aff == "AFFECT":
        return True

    src_con = _get_demarcation(src_id, nodes, "constraint_vs_enablement")
    tgt_con = _get_demarcation(tgt_id, nodes, "constraint_vs_enablement")
    if {src_con, tgt_con} == {"CONSTRAINT", "ENABLEMENT"}:
        return True

    src_syn = _get_demarcation(src_id, nodes, "synchronic_vs_diachronic")
    tgt_syn = _get_demarcation(tgt_id, nodes, "synchronic_vs_diachronic")
    if {src_syn, tgt_syn} == {"SYNCHRONIC", "DIACHRONIC"}:
        return True

    return False


def _refine_by_demarcation(
    edge_type: EdgeType,
    label: str,
    src_id: UUID,
    tgt_id: UUID,
    nodes: Dict[UUID, Node],
) -> EdgeType:
    if not _qualifies_by_demarcation(src_id, tgt_id, nodes):
        return edge_type
    if edge_type == EdgeType.SUPPORTS:
        return EdgeType.QUALIFIES
    if label == "Attack" and edge_type in (EdgeType.ATTACKS, EdgeType.CONTRADICTS):
        return EdgeType.REBUTS
    return edge_type


def assign_edges(
    typed_spans: List[Tuple[PropSpan, NodeType, str]],
    relations: List[Relation],
    node_map: Dict[SpanKey, UUID],
    existing_nodes: Dict[UUID, Node],
) -> Dict[UUID, Edge]:
    typed_map: Dict[SpanKey, NodeType] = {_span_key(s): t for s, t, _ in typed_spans}
    edges: Dict[UUID, Edge] = {}
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

        e = Edge(
            source_id=src_id,
            target_id=tgt_id,
            type=edge_type,
        )
        edges[e.id] = e

    return edges
