"""Debate domain schemas."""

from cognitive_engine.core.models import EdgeType, NodeType
from cognitive_engine.core.schema import Schema


def _debate_type_rule(text: str, **kwargs) -> NodeType:
    """Classify debate text into node types."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["claim", "assertion", "position"]):
        return NodeType.CLAIM
    if any(w in text_lower for w in ["evidence", "data", "facts show"]):
        return NodeType.EVIDENCE
    if any(w in text_lower for w in ["however", "but", "counter"]):
        return NodeType.COUNTERCLAIM
    if any(w in text_lower for w in ["fallacy", "misleading", "flawed"]):
        return NodeType.FALLACY
    return NodeType.CLAIM


DEBATE_SCHEMA = Schema(
    name="debate",
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
        (NodeType.COUNTERCLAIM, NodeType.CLAIM, "Attack"): EdgeType.REBUTS,
        (NodeType.FALLACY, NodeType.CLAIM, "Attack"): EdgeType.CONTRADICTS,
    },
    type_rules=[_debate_type_rule],
    merge_strategy="keep_both",
    conflict_resolution="keep_both",
    dedup_threshold=0.75,
    metadata={
        "domain": "debate",
        "description": "Debate domain schema for argumentation analysis",
    },
)
