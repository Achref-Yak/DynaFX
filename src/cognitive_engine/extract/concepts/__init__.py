"""Concept dispatcher — routes to domain-specific matchers."""

from __future__ import annotations

from typing import Optional

from cognitive_engine.core.models import NodeType


def dispatch_concept(text_lower: str, node_type: NodeType, vn_class: str = "") -> str:
    """Try each domain matcher in order; fall back to NodeType→concept map.

    Args:
        text_lower: Lowercase proposition text.
        node_type: Classified node type.
        vn_class: Optional VerbNet class ID for verb-based concept mapping.
    """
    from cognitive_engine.extract.concepts.identity import match as match_identity
    from cognitive_engine.extract.concepts.measurement import match as match_measurement
    from cognitive_engine.extract.concepts.preference import match as match_preference
    from cognitive_engine.extract.concepts.reasoning import match as match_reasoning

    if node_type == NodeType.CONDITION:
        return "CONDITION"

    for matcher in (match_identity, match_measurement, match_preference, match_reasoning):
        result = matcher(text_lower, node_type)
        if result is not None:
            return result

    # Try VerbNet class mapping as fallback
    if vn_class:
        from cognitive_engine.extract.verbnet_roles import VN_CONCEPT_MAP
        if vn_class in VN_CONCEPT_MAP:
            return VN_CONCEPT_MAP[vn_class]

    return _FALLBACK_CONCEPT.get(node_type, "OBSERVATION")


_FALLBACK_CONCEPT: dict[NodeType, str] = {
    NodeType.CLAIM: "CLAIM",
    NodeType.COUNTERCLAIM: "CLAIM",
    NodeType.EVIDENCE: "EVIDENCE",
    NodeType.FALLACY: "OBSERVATION",
    NodeType.JUSTIFICATION: "EVIDENCE",
    NodeType.AXIOM: "EVIDENCE",
    NodeType.CONDITION: "CONDITION",
    NodeType.HYPOTHESIS: "HYPOTHESIS",
    NodeType.DECISION: "DECISION",
    NodeType.OBSERVATION: "OBSERVATION",
    NodeType.ACTION: "ACTION",
    NodeType.ENTITY: "LOCATION",
    NodeType.CONCEPT: "PREFERENCE",
    NodeType.RULE: "CONDITION",
    NodeType.DOCUMENT: "EVIDENCE",
    NodeType.AGENT: "AGENT",
    NodeType.PROCESS: "PROCESS",
    NodeType.STATE: "STATE",
    NodeType.PROPERTY: "PROPERTY",
    NodeType.RESOURCE: "RESOURCE",
    NodeType.CONSTRAINT: "CONSTRAINT",
    NodeType.GOAL: "GOAL",
    NodeType.BELIEF: "BELIEF",
    NodeType.KNOWLEDGE: "KNOWLEDGE",
    NodeType.INFORMATION: "INFORMATION",
}
