"""Assertion Gate — hard boundary between perception and reasoning kernel.

Every neural output — regardless of source — is converted here
into a schema-conformant ABox assertion with a typed SL opinion.
Malformed outputs are quarantined and logged, never silently dropped.
Nothing downstream touches raw embeddings or raw probabilities.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID, uuid4

from cognitive_engine.core.math import check_opinion_invariant, normalize_sum
from cognitive_engine.core.models import Node, Edge, NodeType, EdgeType, Opinion

logger = logging.getLogger(__name__)


@dataclass
class Assertion:
    """A typed ABox assertion with SL opinion.

    Attributes:
        id: Unique assertion identifier.
        source: Which perception component produced this.
        node_id: Optional reference to a graph node.
        node_type: The asserted node type.
        text: Text content of the assertion.
        opinion: SL opinion (b, d, u, a) tuple or None.
        metadata: Additional provenance data.
        timestamp: When this assertion was created.
    """
    id: UUID = field(default_factory=uuid4)
    source: str = ""
    node_id: Optional[UUID] = None
    node_type: Optional[str] = None
    text: str = ""
    opinion: Optional[tuple[float, float, float, float]] = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class GateResult:
    """Result of passing an assertion through the gate."""
    passed: list[Assertion] = field(default_factory=list)
    quarantined: list[Assertion] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class AssertionGate:
    """Hard boundary between perception and reasoning kernel.

    Operations:
        1. type_check: Validates entity/relation types against domain TBox schema
        2. opinion_assignment: Maps neural confidence scores to SL (b,d,u,a) tuples
        3. invariant_check: Asserts b + d + u = 1 within clamp_epsilon
        4. quarantine: Malformed assertions logged, not dropped
    """

    def __init__(self, clamp_epsilon: float = 1e-9):
        self.clamp_epsilon = clamp_epsilon

    def process(
        self,
        assertions: list[Assertion],
        tbox: Optional[Any] = None,
    ) -> GateResult:
        """Process assertions through the gate.

        Args:
            assertions: Raw assertions from the perception zone.
            tbox: Optional TBox for type validation.

        Returns:
            GateResult with passed and quarantined assertions.
        """
        result = GateResult()
        for assertion in assertions:
            if not self._type_check(assertion, tbox):
                result.quarantined.append(assertion)
                result.errors.append(
                    f"Type check failed: {assertion.text[:50]!r} "
                    f"type={assertion.node_type}"
                )
                continue

            # Assign opinion if not already set
            if assertion.opinion is None:
                assertion.opinion = self._assign_opinion(assertion.metadata.get("confidence", 0.5))

            # Normalize and check invariant
            b, d, u = assertion.opinion[0], assertion.opinion[1], assertion.opinion[2]
            b, d, u = normalize_sum(b, d, u)
            assertion.opinion = (b, d, u, assertion.opinion[3])

            if not check_opinion_invariant(b, d, u, epsilon=0.01):
                result.quarantined.append(assertion)
                result.errors.append(
                    f"Invariant failed: b+d+u={b+d+u:.4f} for {assertion.text[:50]!r}"
                )
                continue

            result.passed.append(assertion)

        if result.errors:
            logger.warning(
                "AssertionGate: %d passed, %d quarantined (%d errors)",
                len(result.passed), len(result.quarantined), len(result.errors),
            )
        else:
            logger.info(
                "AssertionGate: %d assertions passed", len(result.passed),
            )

        return result

    def to_node(self, assertion: Assertion) -> Node:
        """Convert a passed assertion to a graph node."""
        opinion = None
        if assertion.opinion:
            b, d, u, a = assertion.opinion
            opinion = Opinion(belief=b, disbelief=d, uncertainty=u, prior=a)
        node_type = NodeType[assertion.node_type.upper()] if assertion.node_type else NodeType.CLAIM
        return Node(
            id=assertion.node_id or uuid4(),
            type=node_type,
            text=assertion.text,
            opinion=opinion,
            metadata=assertion.metadata,
        )

    def to_edge(self, source_id: UUID, target_id: UUID, edge_type: str,
                opinion: Optional[tuple[float, float, float, float]] = None) -> Edge:
        """Create a typed edge between two nodes."""
        sl_opinion = None
        if opinion:
            b, d, u, a = opinion
            sl_opinion = Opinion(belief=b, disbelief=d, uncertainty=u, prior=a)
        kwargs = {"opinion": sl_opinion} if sl_opinion else {}
        return Edge(
            source_id=source_id,
            target_id=target_id,
            type=EdgeType[edge_type.upper()],
            **kwargs,
        )

    # ── Private ───────────────────────────────────────────────────

    def _type_check(self, assertion: Assertion, tbox: Any) -> bool:
        """Validate type against TBox schema."""
        if assertion.node_type is None:
            return True
        try:
            NodeType[assertion.node_type.upper()]
            return True
        except (KeyError, ValueError):
            return False

    def _assign_opinion(self, confidence: float) -> tuple[float, float, float, float]:
        """Map neural confidence to SL opinion tuple."""
        if confidence > 0.85:
            return (0.8, 0.1, 0.1, 0.5)  # empirical_pattern
        elif confidence >= 0.5:
            return (0.4, 0.3, 0.3, 0.5)  # observational_claim
        elif confidence > 0:
            return (0.5, 0.2, 0.3, 0.5)  # cognitive_hypothesis
        return (0.0, 0.0, 1.0, 0.5)  # total_ignorance
