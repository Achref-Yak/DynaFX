from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph


@dataclass
class LTMPattern:
    """A compressed cognitive pattern stored in long-term memory.

    Captures a graph snapshot, belief signature, operator trace,
    emergence cluster labels, and access statistics for similarity-based
    retrieval.
    """
    id: UUID
    graph_snapshot: Graph
    belief_signature: dict
    operator_trace: list[str]
    cluster_labels: list[str]
    frequency: int = 1
    last_accessed: float = 0.0
