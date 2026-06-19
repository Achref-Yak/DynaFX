"""Reasoning concept patterns."""

from __future__ import annotations

from typing import Optional

from cognitive_engine.core.models import NodeType

_HYPOTHESIS_KEYWORDS = {"hypothesis", "theory", "we suspect", "we hypothesize"}
_DECISION_KEYWORDS = {"decided", "chose", "selected", "we will"}
_OBSERVATION_KEYWORDS = {"observed", "noted", "found that", "saw"}


def match(text_lower: str, node_type: NodeType) -> Optional[str]:
    if any(p in text_lower for p in _HYPOTHESIS_KEYWORDS):
        return "HYPOTHESIS"
    if any(p in text_lower for p in _DECISION_KEYWORDS) and node_type == NodeType.CLAIM:
        return "DECISION"
    if any(p in text_lower for p in _OBSERVATION_KEYWORDS):
        return "OBSERVATION"
    return None
