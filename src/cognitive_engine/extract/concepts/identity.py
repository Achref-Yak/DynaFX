"""Identity concept patterns."""

from __future__ import annotations

from typing import Optional

from cognitive_engine.core.models import NodeType

_IDENTITY_PATTERNS = {
    "PERSON_NAME": {"my name", "i am called", "my name is", "i go by"},
    "EMAIL": {"my email", "my e-mail", "reach me at"},
}


def match(text_lower: str, node_type: NodeType) -> Optional[str]:
    for concept, patterns in _IDENTITY_PATTERNS.items():
        if any(p in text_lower for p in patterns):
            return concept
    return None
