"""Hypothesis Generator — NLI scoring over candidate relation pairs.

Suggests candidate missing links in sparse graphs.
All outputs are candidates only — marked as cognitive_hypothesis
opinion (b=0.5, d=0.2, u=0.3, a=0.5). None enter the kernel
as facts. Hypotheses must survive conflict pass validation
before opinion upgrades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from cognitive_engine.kernel.assertion_gate import Assertion

logger = logging.getLogger(__name__)


@dataclass
class CandidateHypothesis:
    """A suggested missing link between two nodes.

    Attributes:
        id: Unique candidate identifier.
        source_id: Source node UUID.
        target_id: Target node UUID.
        relation_type: Suggested edge type (e.g., "SUPPORTS", "ATTACKS").
        score: NLI confidence score [0, 1].
        premise: Text of the source node.
        hypothesis: Text of the target node.
        metadata: Additional provenance data.
    """
    id: UUID = field(default_factory=uuid4)
    source_id: UUID = field(default_factory=uuid4)
    target_id: UUID = field(default_factory=uuid4)
    relation_type: str = "SUPPORTS"
    score: float = 0.5
    premise: str = ""
    hypothesis: str = ""
    metadata: dict = field(default_factory=dict)

    def to_assertion(self) -> Assertion:
        """Convert to an assertion for the gate."""
        return Assertion(
            id=self.id,
            source="hypothesis_generator",
            node_type=None,
            text=f"{self.premise[:50]} → {self.hypothesis[:50]}",
            opinion=(0.5, 0.2, 0.3, 0.5),  # cognitive_hypothesis
            metadata={
                "source_id": str(self.source_id),
                "target_id": str(self.target_id),
                "relation_type": self.relation_type,
                "score": self.score,
                "hypothesis": True,
            },
        )


class HypothesisGenerator:
    """NLI-based hypothesis generator for sparse graphs.

    Scores candidate (source, target, relation) triples.
    Uses heuristic scoring by default; can be backed by a
    trained NLI model when available.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def generate(
        self,
        node_texts: dict[UUID, str],
        node_types: dict[UUID, str],
        existing_edges: list,
        max_candidates: int = 10,
    ) -> list[CandidateHypothesis]:
        """Generate candidate hypotheses for missing links.

        Args:
            node_texts: Node ID → text content.
            node_types: Node ID → type name.
            existing_edges: Existing edges to avoid duplicates.
            max_candidates: Maximum number of candidates.

        Returns:
            List of candidate hypotheses, sorted by score descending.
        """
        node_ids = list(node_texts.keys())
        candidates: list[CandidateHypothesis] = []
        existing_pairs = set()
        for edge in existing_edges:
            src = edge.source_id if hasattr(edge, 'source_id') else edge[0]
            tgt = edge.target_id if hasattr(edge, 'target_id') else edge[1]
            existing_pairs.add((src, tgt))

        for src_id in node_ids:
            for tgt_id in node_ids:
                if src_id == tgt_id:
                    continue
                if (src_id, tgt_id) in existing_pairs:
                    continue

                src_text = node_texts[src_id]
                tgt_text = node_texts[tgt_id]
                score = self._score_pair(src_text, tgt_text, node_types.get(src_id, ""),
                                         node_types.get(tgt_id, ""))

                if score >= self.threshold:
                    # Determine relation type based on score and types
                    relation = self._infer_relation(src_text, tgt_text,
                                                    node_types.get(src_id, ""),
                                                    node_types.get(tgt_id, ""))
                    candidates.append(CandidateHypothesis(
                        source_id=src_id,
                        target_id=tgt_id,
                        relation_type=relation,
                        score=score,
                        premise=src_text,
                        hypothesis=tgt_text,
                    ))

        # Sort by score descending, take top-k
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:max_candidates]

    def _score_pair(
        self, text_a: str, text_b: str,
        type_a: str, type_b: str,
    ) -> float:
        """Score a candidate pair for potential relation.

        Heuristic scoring based on text overlap and type compatibility.
        Replace with NLI model call when available.
        """
        score = 0.0
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        # Word overlap suggests relatedness
        if words_a and words_b:
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            score += overlap * 0.4

        # Type-based compatibility
        compatible = self._type_compatible(type_a, type_b)
        if compatible:
            score += 0.4

        # Length similarity (short texts are more likely related)
        len_ratio = min(len(text_a), len(text_b)) / max(len(text_a), len(text_b), 1)
        score += len_ratio * 0.2

        return min(score, 1.0)

    def _infer_relation(
        self, text_a: str, text_b: str,
        type_a: str, type_b: str,
    ) -> str:
        """Infer the most likely relation type between two nodes."""
        if type_a == "EVIDENCE" and type_b in ("CLAIM", "HYPOTHESIS"):
            return "SUPPORTS"
        if type_a == "COUNTERCLAIM" and type_b == "CLAIM":
            return "ATTACKS"
        if type_a in ("CLAIM", "HYPOTHESIS") and type_b == "EVIDENCE":
            return "JUSTIFIES"
        return "SUPPORTS"

    def _type_compatible(self, type_a: str, type_b: str) -> bool:
        """Check if two node types can have a relation."""
        if not type_a or not type_b:
            return True
        if type_a == "AXIOM":
            return True
        if type_a == "FALLACY":
            return type_b in ("CLAIM", "COUNTERCLAIM")
        return True
