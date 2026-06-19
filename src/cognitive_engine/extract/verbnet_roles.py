"""VerbNet thematic role extraction and concept mapping.

Uses VerbNet classes to:
  1. Map verb lemmas to VerbNet classes
  2. Extract thematic roles (Agent, Theme, Patient, etc.)
  3. Classify relations based on VerbNet class semantics
  4. Enrich concept assignment with VerbNet class information
"""

from __future__ import annotations

import logging
from typing import Optional

from cognitive_engine.core.models import NodeType

logger = logging.getLogger(__name__)

# ── VerbNet class → concept mapping ──────────────────────────────

VN_CONCEPT_MAP: dict[str, str] = {
    # Transfer
    "give-13.1-1": "TRANSFER",
    "give-13.2": "TRANSFER",
    "take-13.5.1": "TRANSFER",
    "send-11.1": "TRANSFER",
    "receive-13.5.1": "TRANSFER",
    "obtain-13.5.2": "TRANSFER",
    "get-13.5.1": "TRANSFER",

    # Communication
    "say-37.7-1": "COMMUNICATION",
    "tell-37.2": "COMMUNICATION",
    "inquire-37.1.2": "COMMUNICATION",
    "indicate-78-1": "COMMUNICATION",
    "inform-37.1.2": "COMMUNICATION",
    "communicate-37.1": "COMMUNICATION",
    "transfer_mesg-37.1.1-1-1": "COMMUNICATION",

    # Cognition
    "consider-29.9-2": "KNOWLEDGE",
    "admire-31.2": "BELIEF",
    "judge-29.4": "ASSESSMENT",
    "deduce-29.9": "REASONING",
    "infer-29.9": "REASONING",

    # Desire
    "want-32.1": "GOAL",
    "need-32.1": "GOAL",
    "desire-32.1": "GOAL",

    # Motion
    "go-18.1": "MOTION",
    "run-51.3.2": "MOTION",
    "meander-47.7": "MOTION",

    # Creation
    "build-26.1-1": "CREATION",
    "construct-26.1": "CREATION",
    "produce-26.1": "CREATION",
    "engender-27": "CAUSATION",
    "destroy-44.1": "DESTRUCTION",

    # Possession
    "have-10.1": "POSSESSION",
    "own-10.1": "POSSESSION",

    # Consumption
    "eat-39.1-1": "CONSUMPTION",
    "drink-39.2": "CONSUMPTION",
    "use-105.1": "CONSUMPTION",

    # Perception
    "see-30.1": "PERCEPTION",
    "hear-30.2": "PERCEPTION",
    "observe-30.3": "PERCEPTION",

    # Competition
    "compete-36.2": "COMPETITION",
    "win-36.2.1": "COMPETITION",
}

# VerbNet class → EdgeType mapping for relation classification
VN_EdgeType_MAP: dict[str, str] = {
    # Causal
    "engender-27": "CAUSES",
    "cause-11.1": "CAUSES",
    "result-48": "CAUSES",

    # Enablement
    "enable-105.2": "ENABLES",
    "allow-105.3": "ENABLES",
    "help-105.1": "ENABLES",

    # Part-of
    "consist_of-108.1": "PART_OF",
    "comprise-108.1": "PART_OF",
    "contain-108.1": "PART_OF",

    # Communication
    "say-37.7-1": "COMMUNICATED",
    "tell-37.2": "COMMUNICATED",
    "transfer_mesg-37.1.1-1-1": "COMMUNICATED",

    # Dependency
    "rely-70": "DEPENDS",
    "depend-105.5": "DEPENDS",
}


def vn_classes_for_lemma(lemma: str) -> list[str]:
    """Get VerbNet classes for a verb lemma via SemanticResources."""
    from cognitive_engine.nlp.semantic_resources import SemanticResources
    return SemanticResources.instance().vn_classes_for_lemma(lemma)


def vn_themroles(class_id: str) -> list[dict]:
    """Get thematic roles for a VerbNet class."""
    from cognitive_engine.nlp.semantic_resources import SemanticResources
    return SemanticResources.instance().vn_themroles(class_id)


def vn_concept_for_lemma(lemma: str) -> Optional[str]:
    """Get concept label from VerbNet class for a verb lemma.

    Tries each VerbNet class for the lemma, returns the first match
    in VN_CONCEPT_MAP.
    """
    classes = vn_classes_for_lemma(lemma)
    for cid in classes:
        if cid in VN_CONCEPT_MAP:
            return VN_CONCEPT_MAP[cid]
    return None


def vn_edgetype_for_lemma(lemma: str) -> Optional[str]:
    """Get edge type from VerbNet class for a verb lemma."""
    classes = vn_classes_for_lemma(lemma)
    for cid in classes:
        if cid in VN_EdgeType_MAP:
            return VN_EdgeType_MAP[cid]
    return None


def vn_roles_for_span(lemma: str) -> dict[str, str]:
    """Get thematic role names for a verb lemma.

    Returns dict mapping role type → role name, e.g.:
    {"Agent": "Agent", "Theme": "Theme", "Goal": "Goal"}
    """
    classes = vn_classes_for_lemma(lemma)
    if not classes:
        return {}
    # Use the first (most common) class
    roles = vn_themroles(classes[0])
    return {r.get("type", ""): r.get("type", "") for r in roles if r.get("type")}


def classify_relation_by_vn(
    verb_lemma: str,
    source_type: NodeType,
    target_type: NodeType,
) -> Optional[str]:
    """Classify a relation using VerbNet class semantics.

    Args:
        verb_lemma: The verb lemma.
        source_type: NodeType of the source node.
        target_type: NodeType of the target node.

    Returns:
        EdgeType name or None.
    """
    # Try VerbNet edge type mapping first
    edge_type = vn_edgetype_for_lemma(verb_lemma)
    if edge_type:
        return edge_type

    # Use thematic roles to infer relation
    roles = vn_roles_for_span(verb_lemma)
    has_agent = "Agent" in roles
    has_theme = "Theme" in roles
    has_goal = "Goal" in roles

    if has_agent and has_theme and source_type == NodeType.AGENT:
        if target_type == NodeType.PROCESS:
            return "CAUSES"
        if target_type == NodeType.STATE:
            return "CAUSES"
    if has_goal and target_type == NodeType.GOAL:
        return "ENABLES"

    return None
