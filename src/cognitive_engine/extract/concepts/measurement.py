"""Measurement concept patterns."""

from __future__ import annotations

import re
from typing import Optional

from cognitive_engine.core.models import NodeType


def match(text_lower: str, node_type: NodeType) -> Optional[str]:
    if re.search(r"\d+\s*°[cfCF]", text_lower) or "temperature" in text_lower:
        return "TEMPERATURE"
    if re.search(r"[\$£€]", text_lower) or any(p in text_lower for p in {"dollars", "cost", "price", "budget", "spend"}):
        return "BUDGET"
    if any(p in text_lower for p in {"deadline", "due date", "scheduled for"}) or re.search(
        r"\b(on|by|before)\s+(january|february|march|april|may|june|july|august|september|october|november|december)",
        text_lower,
    ):
        return "DATE"
    return None
