from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from cognitive_engine.core.concept import Appraisal, Provenance
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
    session_id: str = ""
    community_id: Optional[int] = None


@dataclass
class Fact:
    """A single versioned fact with SCD Type 2 temporal validity.

    Each fact belongs to a concept (BUDGET, NAME, etc.) and has a provenance
    (where it came from). When a fact is invalidated, valid_to is set and a
    new Fact is inserted with valid_from = now().
    """
    id: UUID = field(default_factory=uuid4)
    concept: str = ""
    value: str = ""
    original_text: str = ""
    provenance: Provenance = Provenance.AGENT_OBSERVED
    appraisal: Appraisal = field(default_factory=Appraisal)
    confidence: float = 0.5
    valid_from: float = 0.0
    valid_to: Optional[float] = None
    session_id: str = ""
    node_id: Optional[UUID] = None

    def is_active(self) -> bool:
        return self.valid_to is None

    def to_dict(self) -> dict:
        return {
            "id": self.id.hex,
            "concept": self.concept,
            "value": self.value,
            "original_text": self.original_text,
            "provenance": self.provenance.name,
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "session_id": self.session_id,
            "node_id": self.node_id.hex if self.node_id else None,
        }


@dataclass
class FactArchive:
    """An archived (invalidated) fact with reason for archival.

    Created when a fact is invalidated via MUSTIE weeding or conflict
    resolution. Preserves the full history for audit trails.
    """
    id: UUID = field(default_factory=uuid4)
    concept: str = ""
    value: str = ""
    original_text: str = ""
    provenance: Provenance = Provenance.AGENT_OBSERVED
    appraisal: Appraisal = field(default_factory=Appraisal)
    confidence: float = 0.5
    valid_from: float = 0.0
    valid_to: Optional[float] = None
    session_id: str = ""
    node_id: Optional[UUID] = None
    archived_at: float = 0.0
    archive_reason: str = ""

    @classmethod
    def from_fact(cls, fact: Fact, reason: str, archived_at: float) -> FactArchive:
        return cls(
            id=fact.id,
            concept=fact.concept,
            value=fact.value,
            original_text=fact.original_text,
            provenance=fact.provenance,
            appraisal=fact.appraisal,
            confidence=fact.confidence,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            session_id=fact.session_id,
            node_id=fact.node_id,
            archived_at=archived_at,
            archive_reason=reason,
        )
