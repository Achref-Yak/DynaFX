from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from dynafx.core.config import Priors
from dynafx.core.models import EvidenceCounts, Graph, Opinion
from dynafx.domain import domain as _domain

logger = logging.getLogger(__name__)


def opinion_from_counts(counts: EvidenceCounts) -> tuple[float, float, float, float]:
    from dynafx.core.math import opinion_from_counts as _from_counts
    return _from_counts(
        counts.positive, counts.negative,
        counts.uncertainty_pseudocount, 0.5,
    )


def mean_opinion(opinions: list[Opinion]) -> tuple[float, float, float, float]:
    from dynafx.core.math import mean_opinion as _mean
    return _mean(list(opinions))


def mean_opinion_pair(pairs: list[tuple[Opinion, Opinion]]) -> tuple[Opinion, Opinion]:
    first = [p[0] for p in pairs]
    second = [p[1] for p in pairs]
    return (mean_opinion(first), mean_opinion(second))


def _collect_node_counts(
    graphs: list[Graph],
) -> dict[str, EvidenceCounts]:
    w = _domain.active().uncertainty_pseudocount
    counts: dict[str, EvidenceCounts] = defaultdict(lambda: EvidenceCounts(uncertainty_pseudocount=w))
    for graph in graphs:
        for node in graph.nodes.values():
            key = node.type.name
            b, d, _, _ = node.opinion
            if b > d + _domain.active().opinion_positive_threshold:
                counts[key].positive += 1
            else:
                counts[key].negative += 1
    return dict(counts)


def _collect_edge_counts(
    graphs: list[Graph],
) -> dict[str, list[tuple[Opinion, Opinion]]]:
    warrants: dict[str, list[tuple[Opinion, Opinion]]] = defaultdict(list)
    for graph in graphs:
        for edge in graph.edges.values():
            key = edge.type.name
            if edge.warrant is not None:
                warrants[key].append(edge.warrant)
    return dict(warrants)


@dataclass
class CorpusResult:
    graph_count: int = 0
    node_counts: Dict[str, EvidenceCounts] = field(default_factory=dict)
    edge_warrants: Dict[str, list[tuple[Opinion, Opinion]]] = field(default_factory=dict)

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: str | Path,
        max_files: Optional[int] = None,
        config_path: Optional[str] = None,
    ) -> CorpusResult:
        path = Path(corpus_dir)
        if not path.is_dir():
            raise NotADirectoryError(f"Corpus directory not found: {corpus_dir}")

        txt_files = sorted(path.rglob("*.txt"))
        if max_files is not None:
            txt_files = txt_files[:max_files]

        if not txt_files:
            logger.warning("No .txt files found in %s", corpus_dir)
            return cls()

        graphs: list[Graph] = []
        for f in txt_files:
            logger.warning("  Skipped %s: corpus extraction removed in this build", f.name)

        node_counts = _collect_node_counts(graphs)
        edge_warrants = _collect_edge_counts(graphs)

        return cls(
            graph_count=len(graphs),
            node_counts=node_counts,
            edge_warrants=edge_warrants,
        )

    def to_priors(self) -> Priors:
        cfg = _domain.active()
        node_type_map = cfg.source_type_map

        source_type_map: dict[str, str] = {}
        learned_opinions: dict[str, Opinion] = {}
        for node_type_name, template_name in node_type_map.items():
            counts = self.node_counts.get(node_type_name)
            if counts is not None and (counts.positive > 0 or counts.negative > 0):
                learned_opinions[template_name] = opinion_from_counts(counts)
            source_type_map[node_type_name] = template_name

        learned_opinions["total_ignorance"] = cfg.total_ignorance
        for template in list(cfg.default_opinions):
            if template not in learned_opinions:
                learned_opinions[template] = cfg.default_opinions[template]

        learned_warrants: dict[str, tuple[Opinion, Opinion]] = {}
        for edge_type_name, pairs in self.edge_warrants.items():
            if pairs:
                learned_warrants[edge_type_name] = mean_opinion_pair(pairs)

        base = Priors()
        warrants = learned_warrants if learned_warrants else base.edge_warrants
        if "SUPPORTS" not in warrants:
            warrants["SUPPORTS"] = base.edge_warrants["SUPPORTS"]
        for et_name in base.edge_warrants:
            if et_name not in warrants:
                warrants[et_name] = base.edge_warrants[et_name]

        return Priors(
            default_opinions=learned_opinions,
            source_type_map=source_type_map,
            edge_warrants=warrants,
            default_warrant=base.default_warrant,
        )

    def to_dict(self) -> dict:
        return {
            "graph_count": self.graph_count,
            "node_counts": {
                k: {"positive": v.positive, "negative": v.negative,
                    "uncertainty_pseudocount": v.uncertainty_pseudocount}
                for k, v in self.node_counts.items()
            },
            "edge_warrants": {
                k: [[list(a), list(b)] for a, b in v]
                for k, v in self.edge_warrants.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Saved corpus result to %s", p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> CorpusResult:
        p = Path(path)
        data = json.loads(p.read_text())
        node_counts = {
            k: EvidenceCounts(**v)
            for k, v in data.get("node_counts", {}).items()
        }
        edge_warrants = {
            k: [(tuple(a), tuple(b)) for a, b in v]
            for k, v in data.get("edge_warrants", {}).items()
        }
        return cls(
            graph_count=data.get("graph_count", 0),
            node_counts=node_counts,
            edge_warrants=edge_warrants,
        )


# ── Evidence Matrix ───────────────────────────────────────────────


class ConsensusLevel(Enum):
    """Classification of agreement level across sources."""
    STRONG_AGREEMENT = "strong_agreement"    # all sources agree
    MILD_AGREEMENT = "mild_agreement"        # most sources agree, minor dissent
    CONTESTED = "contested"                  # significant disagreement
    STRONG_DISAGREEMENT = "strong_disagreement"  # sources actively contradict


@dataclass
class PairwiseAgreement:
    """Agreement between two sources on a single claim."""
    source_a: str
    source_b: str
    claim: str
    opinion_a: Opinion
    opinion_b: Opinion
    agreement: float  # 0.0 = total conflict, 1.0 = total agreement
    conflicts: bool

    def to_dict(self) -> dict:
        return {
            "source_a": self.source_a,
            "source_b": self.source_b,
            "claim": self.claim,
            "opinion_a": list(self.opinion_a),
            "opinion_b": list(self.opinion_b),
            "agreement": round(self.agreement, 4),
            "conflicts": self.conflicts,
        }


@dataclass
class ClaimAssessment:
    """Assessment of a single claim across all sources."""
    claim: str
    opinions: dict[str, Opinion]  # source_name -> opinion
    fused: Opinion
    consensus: ConsensusLevel
    pairwise: list[PairwiseAgreement]

    @property
    def source_count(self) -> int:
        return len(self.opinions)

    @property
    def belief_mean(self) -> float:
        return sum(o.belief for o in self.opinions.values()) / max(1, len(self.opinions))

    @property
    def disbelief_mean(self) -> float:
        return sum(o.disbelief for o in self.opinions.values()) / max(1, len(self.opinions))

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "source_count": self.source_count,
            "opinions": {k: list(v) for k, v in self.opinions.items()},
            "fused": list(self.fused),
            "consensus": self.consensus.value,
            "belief_mean": round(self.belief_mean, 4),
            "disbelief_mean": round(self.disbelief_mean, 4),
            "pairwise": [p.to_dict() for p in self.pairwise],
        }


@dataclass
class EvidenceMatrix:
    """Structured view of where sources agree and disagree across claims.

    Aggregates opinions from multiple sources per claim, computes pairwise
    agreement, classifies consensus level, and fuses opinions. Designed to
    make reasoning transparent — no black boxes.

    Usage:
        matrix = EvidenceMatrix()
        matrix.add_source("sensor_A", {"temp": Opinion(0.8, 0.1, 0.1, 0.5)})
        matrix.add_source("news_B", {"temp": Opinion(0.2, 0.6, 0.2, 0.5)})
        result = matrix.compute()
        # result.claims["temp"].consensus == ConsensusLevel.CONTESTED
    """
    _sources: dict[str, dict[str, Opinion]] = field(default_factory=dict)
    _conflict_threshold: float = 0.5
    _strong_threshold: float = 0.6

    def add_source(self, source_name: str, opinions: dict[str, Opinion]) -> None:
        """Add a source's opinions on a set of claims."""
        self._sources[source_name] = opinions

    def remove_source(self, source_name: str) -> None:
        """Remove a source from the matrix."""
        self._sources.pop(source_name, None)

    @property
    def source_names(self) -> list[str]:
        return list(self._sources.keys())

    @property
    def claim_names(self) -> list[str]:
        claims: set[str] = set()
        for opinions in self._sources.values():
            claims.update(opinions.keys())
        return sorted(claims)

    def _compute_agreement(self, op_a: Opinion, op_b: Opinion) -> float:
        """Compute agreement score between two opinions.

        Returns 1.0 for identical opinions, 0.0 for maximal conflict.
        Uses L1 distance in belief-disbelief space: agreement = 1 - mean(|b_a-b_b|, |d_a-d_b|).
        """
        b_a, d_a, u_a, _ = op_a
        b_b, d_b, u_b, _ = op_b

        # L1 distance in belief-disbelief space
        dist = (abs(b_a - b_b) + abs(d_a - d_b)) / 2.0  # [0, 1]
        return 1.0 - dist

    def _classify_consensus(self, claim: str) -> ConsensusLevel:
        """Classify consensus level for a claim across all sources."""
        opinions = [
            src[claim]
            for src in self._sources.values()
            if claim in src
        ]
        if len(opinions) < 2:
            return ConsensusLevel.STRONG_AGREEMENT

        # Count pairwise conflicts
        n_conflicts = 0
        n_pairs = 0
        for i in range(len(opinions)):
            for j in range(i + 1, len(opinions)):
                n_pairs += 1
                b_a, d_a, _, _ = opinions[i]
                b_b, d_b, _, _ = opinions[j]
                if (b_a > self._conflict_threshold and d_b > self._conflict_threshold) or \
                   (b_b > self._conflict_threshold and d_a > self._conflict_threshold):
                    n_conflicts += 1

        conflict_ratio = n_conflicts / max(1, n_pairs)

        # Check agreement strength
        n_strong = 0
        for i in range(len(opinions)):
            for j in range(i + 1, len(opinions)):
                b_a, d_a, _, _ = opinions[i]
                b_b, d_b, _, _ = opinions[j]
                both_believe = b_a > self._strong_threshold and b_b > self._strong_threshold
                both_disbelieve = d_a > self._strong_threshold and d_b > self._strong_threshold
                if both_believe or both_disbelieve:
                    n_strong += 1

        agreement_ratio = n_strong / max(1, n_pairs)

        if conflict_ratio == 0 and agreement_ratio > 0.5:
            return ConsensusLevel.STRONG_AGREEMENT
        elif conflict_ratio == 0:
            return ConsensusLevel.MILD_AGREEMENT
        elif conflict_ratio < 0.3:
            return ConsensusLevel.MILD_AGREEMENT
        elif conflict_ratio < 0.5:
            return ConsensusLevel.CONTESTED
        else:
            return ConsensusLevel.STRONG_DISAGREEMENT

    def _fuse_claims(self) -> dict[str, Opinion]:
        """Fuse opinions across sources for each claim using cumulative fusion."""
        fused: dict[str, Opinion] = {}
        for claim in self.claim_names:
            opinions = [
                src[claim]
                for src in self._sources.values()
                if claim in src
            ]
            if not opinions:
                continue
            result = opinions[0]
            for op in opinions[1:]:
                from dynafx.epistemics.fusion import cumulative_fusion
                result = Opinion.from_tuple(cumulative_fusion(result, op))
            fused[claim] = result
        return fused

    def compute(self) -> EvidenceMatrixResult:
        """Compute the full evidence matrix analysis."""
        fused = self._fuse_claims()
        claims: dict[str, ClaimAssessment] = {}

        for claim in self.claim_names:
            source_opinions = {
                name: src[claim]
                for name, src in self._sources.items()
                if claim in src
            }

            # Pairwise agreements
            pairwise: list[PairwiseAgreement] = []
            source_names = sorted(source_opinions.keys())
            for i in range(len(source_names)):
                for j in range(i + 1, len(source_names)):
                    sa, sb = source_names[i], source_names[j]
                    op_a, op_b = source_opinions[sa], source_opinions[sb]
                    agreement = self._compute_agreement(op_a, op_b)
                    b_a, d_a, _, _ = op_a
                    b_b, d_b, _, _ = op_b
                    conflicts = (b_a > self._conflict_threshold and d_b > self._conflict_threshold) or \
                                (b_b > self._conflict_threshold and d_a > self._conflict_threshold)
                    pairwise.append(PairwiseAgreement(
                        source_a=sa, source_b=sb, claim=claim,
                        opinion_a=op_a, opinion_b=op_b,
                        agreement=agreement, conflicts=conflicts,
                    ))

            consensus = self._classify_consensus(claim)
            claims[claim] = ClaimAssessment(
                claim=claim,
                opinions=source_opinions,
                fused=fused.get(claim, Opinion()),
                consensus=consensus,
                pairwise=pairwise,
            )

        return EvidenceMatrixResult(
            source_names=self.source_names,
            claim_names=self.claim_names,
            claims=claims,
        )

    def to_dict(self) -> dict:
        """Serialize to dict (for JSON export)."""
        return {
            "sources": self.source_names,
            "claims": self.claim_names,
            "matrix": {
                name: {claim: list(op) for claim, op in src.items()}
                for name, src in self._sources.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceMatrix:
        """Deserialize from dict."""
        matrix = cls()
        for name, opinions in data.get("matrix", {}).items():
            matrix.add_source(name, {
                claim: Opinion.from_tuple(op)
                for claim, op in opinions.items()
            })
        return matrix


@dataclass
class EvidenceMatrixResult:
    """Result of computing an EvidenceMatrix."""
    source_names: list[str]
    claim_names: list[str]
    claims: dict[str, ClaimAssessment]

    @property
    def source_count(self) -> int:
        return len(self.source_names)

    @property
    def claim_count(self) -> int:
        return len(self.claim_names)

    def contested_claims(self) -> list[str]:
        """Return claims with significant disagreement."""
        return [
            name for name, c in self.claims.items()
            if c.consensus in (ConsensusLevel.CONTESTED, ConsensusLevel.STRONG_DISAGREEMENT)
        ]

    def agreed_claims(self) -> list[str]:
        """Return claims with strong agreement."""
        return [
            name for name, c in self.claims.items()
            if c.consensus == ConsensusLevel.STRONG_AGREEMENT
        ]

    def classify_fusion_situations(self) -> dict[str, "FusionSituation"]:
        """Map each claim's consensus to a FusionSituation category.

        Bridges EvidenceMatrix analysis (opinion-based) with the
        FusionSituation taxonomy (graph-based). Useful when graph
        topology is unavailable.
        """
        from dynafx.epistemics.fusion import consensus_to_fusion_situation
        return {
            name: consensus_to_fusion_situation(assessment.consensus, assessment.source_count)
            for name, assessment in self.claims.items()
        }

    def summary(self) -> str:
        """Human-readable summary of the evidence matrix."""
        lines = [f"Evidence Matrix: {self.source_count} sources, {self.claim_count} claims"]
        for level in ConsensusLevel:
            claims_at_level = [c for c in self.claims.values() if c.consensus == level]
            if claims_at_level:
                names = [c.claim for c in claims_at_level]
                lines.append(f"  {level.value}: {names}")
        contested = self.contested_claims()
        if contested:
            lines.append(f"  Action needed: {len(contested)} contested claim(s)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "source_count": self.source_count,
            "claim_count": self.claim_count,
            "claims": {name: c.to_dict() for name, c in self.claims.items()},
            "contested": self.contested_claims(),
            "agreed": self.agreed_claims(),
        }
