"""Research domain schemas."""

from cognitive_engine.core.models import EdgeType, NodeType
from cognitive_engine.core.schema import Schema


def _research_type_rule(text: str, **kwargs) -> NodeType:
    """Classify research text into node types."""
    text_lower = text.lower()

    if any(w in text_lower for w in ["study shows", "research indicates", "findings suggest"]):
        return NodeType.EVIDENCE
    if any(w in text_lower for w in ["we propose", "hypothesis", "theory"]):
        return NodeType.CLAIM
    if any(w in text_lower for w in ["however", "contrary to", "in contrast"]):
        return NodeType.COUNTERCLAIM
    if any(w in text_lower for w in ["assuming", "given that", "if"]):
        return NodeType.CONDITION
    return NodeType.EVIDENCE


RESEARCH_SCHEMA = Schema(
    name="research",
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
        (NodeType.CLAIM, NodeType.EVIDENCE, "Support"): EdgeType.JUSTIFIES,
        (NodeType.COUNTERCLAIM, NodeType.CLAIM, "Attack"): EdgeType.REBUTS,
    },
    type_rules=[_research_type_rule],
    merge_strategy="average",
    conflict_resolution="higher_confidence",
    dedup_threshold=0.8,
    metadata={
        "domain": "research",
        "description": "Research domain schema for academic analysis",
    },
)
