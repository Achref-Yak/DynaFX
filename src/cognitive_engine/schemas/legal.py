"""Legal domain schemas."""

from cognitive_engine.core.models import EdgeType, NodeType
from cognitive_engine.core.schema import Schema


def _legal_type_rule(text: str, **kwargs) -> NodeType:
    """Classify legal text into node types."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["pursuant to", "statute", "regulation", "code"]):
        return NodeType.AXIOM
    if any(w in text_lower for w in ["testimony", "evidence shows", "document shows"]):
        return NodeType.EVIDENCE
    if any(w in text_lower for w in ["if", "provided that", "unless"]):
        return NodeType.CONDITION
    if any(w in text_lower for w in ["however", "but", "although", "nevertheless"]):
        return NodeType.COUNTERCLAIM
    if any(w in text_lower for w in ["therefore", "thus", "consequently"]):
        return NodeType.CLAIM
    return NodeType.EVIDENCE


LEGAL_SCHEMA = Schema(
    name="legal",
    node_types={
        "AXIOM": NodeType.AXIOM,
        "EVIDENCE": NodeType.EVIDENCE,
        "CONDITION": NodeType.CONDITION,
        "CLAIM": NodeType.CLAIM,
        "COUNTERCLAIM": NodeType.COUNTERCLAIM,
        "FALLACY": NodeType.FALLACY,
        "JUSTIFICATION": NodeType.JUSTIFICATION,
    },
    edge_types={
        (NodeType.EVIDENCE, NodeType.CLAIM, "Support"): EdgeType.SUPPORTS,
        (NodeType.EVIDENCE, NodeType.CLAIM, "Attack"): EdgeType.ATTACKS,
        (NodeType.AXIOM, NodeType.CLAIM, "Support"): EdgeType.INFERS,
        (NodeType.CONDITION, NodeType.CLAIM, "Support"): EdgeType.QUALIFIES,
        (NodeType.COUNTERCLAIM, NodeType.CLAIM, "Attack"): EdgeType.REBUTS,
    },
    type_rules=[_legal_type_rule],
    merge_strategy="keep_both",
    conflict_resolution="keep_both",
    dedup_threshold=0.85,
    metadata={
        "domain": "legal",
        "description": "Legal domain schema for litigation analysis",
    },
)
