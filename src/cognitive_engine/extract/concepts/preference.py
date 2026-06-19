"""Preference concept patterns."""

from __future__ import annotations

from typing import Optional

from cognitive_engine.core.models import NodeType

_PREFERENCE_KEYWORDS = {"i prefer", "i like", "i want", "i would like", "i rather", "i'd rather"}
_STYLE_KEYWORDS = {"dark mode", "night mode", "theme", "style"}
_LOCATION_KEYWORDS = {"i live in", "i am at", "my address", "located in", "based in"}


def match(text_lower: str, node_type: NodeType) -> Optional[str]:
    if any(p in text_lower for p in _PREFERENCE_KEYWORDS):
        return "PREFERENCE"
    if any(p in text_lower for p in _STYLE_KEYWORDS):
        return "STYLE"
    if any(p in text_lower for p in _LOCATION_KEYWORDS):
        return "LOCATION"
    return None
