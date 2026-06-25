"""Confidence layer: graph fusion, argumentation filtering, and query grading."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from cognitive_engine.core.models import Opinion
from cognitive_engine.kb.model import Triple, TriplePattern
from cognitive_engine.kb.sparql import QueryResult
from cognitive_engine.kb.store import TripleStore
from cognitive_engine.reason.fusion import (
    consensus_compromise,
    cumulative_fusion,
)


# ── FusionResult ─────────────────────────────────────────────────


@dataclass
class FusionResult:
    """Result of fusing opinions across named graphs."""
    source_graphs: list[str]
    fused_count: int
    total_candidates: int
    agreement_ratio: float


# ── QueryGrade ────────────────────────────────────────────────────


@dataclass
class QueryGrade:
    """Confidence grade for a SPARQL query result."""
    avg_belief: float
    avg_disbelief: float
    avg_uncertainty: float
    consensus: str  # "high", "medium", or "low"
    cardinality: int

    def label(self) -> str:
        """Return a human-readable label."""
        return f"{self.consensus.upper()}-confidence ({self.cardinality} results)"


# ── Graph fusion ──────────────────────────────────────────────────


def fuse_graphs(
    store: TripleStore,
    source_graphs: list[str],
    target_graph: Optional[str] = None,
    method: str = "cumulative",
) -> FusionResult:
    """Fuse opinions for equivalent triples across named graphs.

    For each unique (s, p, o) triple that appears in multiple source
    graphs, the opinions from each graph are fused using the specified
    method and the fused triple is stored.

    Args:
        store: The TripleStore containing the named graphs.
        source_graphs: List of graph names to fuse.
        target_graph: If set, fused triples are written to this graph.
            Otherwise they replace the original triples in the store.
        method: Fusion method ("cumulative" or "compromise").

    Returns:
        A FusionResult with statistics.
    """
    if not source_graphs:
        return FusionResult(source_graphs=[], fused_count=0,
                            total_candidates=0, agreement_ratio=1.0)

    triples_by_key: Dict[Tuple, List[Triple]] = defaultdict(list)
    total_candidates = 0

    for graph in source_graphs:
        for triple in store.triples_in_graph(graph):
            key = (triple.subject, triple.predicate, triple.object_)
            triples_by_key[key].append(triple)
            total_candidates += 1

    fused_triples: list[Triple] = []
    multi_source_count = 0

    for key, triples in triples_by_key.items():
        if len(triples) <= 1:
            continue
        multi_source_count += 1
        opinions = [t.opinion for t in triples if t.opinion is not None]
        if len(opinions) < 2:
            continue
        fused = _fuse_opinions(opinions, method)
        fused_triple = Triple(key[0], key[1], key[2], opinion=fused)
        fused_triples.append(fused_triple)

    agreement_ratio = _compute_agreement_ratio(triples_by_key)

    if target_graph:
        for t in fused_triples:
            store.add(t, graph=target_graph)
    else:
        for t in fused_triples:
            store.add(t)

    return FusionResult(
        source_graphs=list(source_graphs),
        fused_count=len(fused_triples),
        total_candidates=total_candidates,
        agreement_ratio=agreement_ratio,
    )


def argumentative_filter(
    store: TripleStore,
    source_graphs: list[str],
    *,
    min_belief: float = 0.2,
    auto_rebut: bool = True,
    auto_undermine_low_belief: bool = True,
) -> TripleStore:
    """Filter triples through argumentation before fusion.

    Builds an argumentation framework from the named graphs, computes
    the grounded extension, and returns a new store containing only
    triples supported by acceptable arguments.

    Use this before ``fuse_graphs()`` to ensure contradictory or
    unsupported claims are removed before SL fusion.

    Args:
        store: The TripleStore containing named graphs.
        source_graphs: List of graph names to consider.
        min_belief: Belief threshold for the skeptic argument.
        auto_rebut: Enable rebut attacks between contradictory claims.
        auto_undermine_low_belief: Enable skeptic attacks on low-belief triples.

    Returns:
        A new TripleStore containing only acceptable triples.
    """
    from cognitive_engine.reason.argumentation import build_framework
    af = build_framework(
        store, source_graphs,
        min_belief=min_belief,
        auto_rebut=auto_rebut,
        auto_undermine_low_belief=auto_undermine_low_belief,
    )
    extension = af.compute_grounded()
    return af.filter_store(store, extension)


def _fuse_opinions(opinions: list[Opinion], method: str) -> Opinion:
    result_tuple = opinions[0].to_tuple()
    for opin in opinions[1:]:
        if method == "compromise":
            result_tuple = consensus_compromise(result_tuple, opin.to_tuple())
        else:
            result_tuple = cumulative_fusion(result_tuple, opin.to_tuple())
    return Opinion(result_tuple[0], result_tuple[1], result_tuple[2], result_tuple[3])


def _compute_agreement_ratio(
    triples_by_key: Dict[Tuple, List[Triple]],
) -> float:
    """Compute the ratio of triples with consistent opinions across sources."""
    if not triples_by_key:
        return 1.0
    agreement_count = 0
    total_multi = 0
    for key, triples in triples_by_key.items():
        if len(triples) < 2:
            continue
        total_multi += 1
        opinions = [t.opinion for t in triples if t.opinion is not None]
        if len(opinions) < 2:
            continue
        if _are_consistent(opinions):
            agreement_count += 1
    if total_multi == 0:
        return 1.0
    return agreement_count / total_multi


def _are_consistent(opinions: list[Opinion]) -> bool:
    """Check if opinions are consistent (not strongly conflicting)."""
    if len(opinions) < 2:
        return True
    beliefs = [o.belief for o in opinions]
    max_diff = max(beliefs) - min(beliefs)
    return max_diff < 0.5


# ── Query grading ─────────────────────────────────────────────────


def grade_query(result: QueryResult) -> QueryGrade:
    """Grade the confidence of a SPARQL query result.

    Computes aggregate opinion metrics across all result bindings.

    Args:
        result: The QueryResult from a SPARQL evaluation.

    Returns:
        A QueryGrade with aggregate confidence metrics.
    """
    if not result.opinions:
        return QueryGrade(
            avg_belief=0.5,
            avg_disbelief=0.0,
            avg_uncertainty=0.5,
            consensus="medium",
            cardinality=result.cardinality,
        )

    all_opinions: list[Opinion] = []
    for opin_map in result.opinions:
        for opin in opin_map.values():
            if opin is not None:
                all_opinions.append(opin)

    if not all_opinions:
        return QueryGrade(
            avg_belief=0.5,
            avg_disbelief=0.0,
            avg_uncertainty=0.5,
            consensus="medium",
            cardinality=result.cardinality,
        )

    avg_b = sum(o.belief for o in all_opinions) / len(all_opinions)
    avg_d = sum(o.disbelief for o in all_opinions) / len(all_opinions)
    avg_u = sum(o.uncertainty for o in all_opinions) / len(all_opinions)

    consensus = _classify_consensus(avg_b, avg_d, avg_u)

    return QueryGrade(
        avg_belief=avg_b,
        avg_disbelief=avg_d,
        avg_uncertainty=avg_u,
        consensus=consensus,
        cardinality=result.cardinality,
    )


def _classify_consensus(belief: float, disbelief: float,
                        uncertainty: float) -> str:
    if belief >= 0.7 and uncertainty < 0.3:
        return "high"
    if belief >= 0.4 or uncertainty < 0.5:
        return "medium"
    return "low"
