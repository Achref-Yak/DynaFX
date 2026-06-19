"""FrameNet frame-based type classification rules.

Maps FrameNet frames to NodeType. Frame rules provide higher precision
than keyword rules by matching semantic situations rather than
substring patterns.
"""

from __future__ import annotations

import logging
from typing import Optional

from cognitive_engine.core.models import NodeType

logger = logging.getLogger(__name__)

# ── Frame → NodeType mapping ─────────────────────────────────────

FRAMENodeType_MAP: dict[str, NodeType] = {
    # World-model frames
    "Personal_info": NodeType.AGENT,
    "Being_named": NodeType.AGENT,
    "Age": NodeType.AGENT,
    "Identity": NodeType.AGENT,
    "Employment": NodeType.AGENT,
    "Role": NodeType.AGENT,
    "Being_located": NodeType.STATE,
    "Change_of_state": NodeType.PROCESS,
    "Cause_change": NodeType.PROCESS,
    "Cause_to_start": NodeType.PROCESS,
    "Cause_to_end": NodeType.PROCESS,
    "Activity_start": NodeType.PROCESS,
    "Activity_end": NodeType.PROCESS,
    "Process_end": NodeType.PROCESS,
    "Intention": NodeType.GOAL,
    "Intentionally_create": NodeType.GOAL,
    "Have_as_requirement": NodeType.CONSTRAINT,
    "Prohibition": NodeType.CONSTRAINT,
    "Obligation": NodeType.CONSTRAINT,
    "Permission": NodeType.CONSTRAINT,
    "Possession": NodeType.RESOURCE,
    "Have_property": NodeType.PROPERTY,
    "Measurable_attributes": NodeType.PROPERTY,
    "Dimension": NodeType.PROPERTY,
    "Apply_heat": NodeType.PROCESS,
    "Cooking": NodeType.PROCESS,
    "Judgment": NodeType.PROCESS,
    "Assessment": NodeType.PROCESS,

    # Argumentation frames
    "Statement": NodeType.CLAIM,
    "Opinion": NodeType.CLAIM,
    "Assertion": NodeType.CLAIM,
    "Reasoning": NodeType.JUSTIFICATION,
    "Justification": NodeType.JUSTIFICATION,
    "Evidence": NodeType.EVIDENCE,
    "Proof": NodeType.EVIDENCE,
    "Concession": NodeType.COUNTERCLAIM,
    "Opposition": NodeType.COUNTERCLAIM,
    "Quibbling": NodeType.FALLACY,
    "False_equivalence": NodeType.FALLACY,
    "Conditional": NodeType.CONDITION,
    "Hypothesis": NodeType.HYPOTHESIS,
    "Consideration": NodeType.HYPOTHESIS,

    # Communication frames
    "Communication": NodeType.CLAIM,
    "Request": NodeType.CLAIM,
    "Command": NodeType.CLAIM,
}

# Frames that imply argumentation (derived from FRAMENodeType_MAP)
_ARG_TYPE_NAMES = {NodeType.CLAIM, NodeType.EVIDENCE, NodeType.COUNTERCLAIM,
                   NodeType.FALLACY, NodeType.CONDITION, NodeType.HYPOTHESIS,
                   NodeType.JUSTIFICATION}
ARGUMENTATION_FRAMES = frozenset(
    frame for frame, ntype in FRAMENodeType_MAP.items()
    if ntype in _ARG_TYPE_NAMES
)


def classify_by_frame(lemma: str, available_frames: list[str]) -> Optional[NodeType]:
    """Classify a span's NodeType based on its evoked FrameNet frames.

    Args:
        lemma: The verb lemma to look up.
        available_frames: Frame names to check (from SemanticResources).

    Returns:
        NodeType if a frame matches, None otherwise.
    """
    for frame in available_frames:
        if frame in FRAMENodeType_MAP:
            return FRAMENodeType_MAP[frame]
    return None


def get_frame_priority(frame_name: str) -> int:
    """Get priority score for a frame (higher = more specific).

    Used to rank multiple frame matches for the same span.
    """
    if frame_name in FRAMENodeType_MAP:
        node_type = FRAMENodeType_MAP[frame_name]
        # World-model types get higher priority than argumentation
        if node_type in {NodeType.AGENT, NodeType.PROCESS, NodeType.STATE,
                         NodeType.GOAL, NodeType.RESOURCE, NodeType.CONSTRAINT}:
            return 100
        if node_type in {NodeType.CONDITION, NodeType.FALLACY, NodeType.JUSTIFICATION}:
            return 60
        return 80  # argumentation types
    return 0
