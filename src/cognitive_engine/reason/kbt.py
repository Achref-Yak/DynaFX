"""Knowledge-Based Trust — EM algorithm for source reliability scoring.

Implements the iterative trust estimation algorithm from:
  Dong et al. "Knowledge-Based Trust: Estimating the Trustworthiness
  of Web Sources" (VLDB 2014)

No ground truth needed — the algorithm converges to a stable fixed
point by alternating between:
  E-step: estimate which claim value is true per (s,p) group
  M-step: update each source's trust based on agreement with winners
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cognitive_engine.core.models import Opinion
from cognitive_engine.kb.model import Literal, NamedNode, Triple
from cognitive_engine.kb.store import TripleStore
from cognitive_engine.reason.argumentation import (
    PROV_NS,
    SOURCE_RELIABILITY,
)


@dataclass
class KBTResult:
    """Result of Knowledge-Based Trust computation.

    Attributes:
        source_trust: Map from source graph name to trust score [0, 1].
        iterations: Number of EM iterations run.
        converged: Whether the algorithm converged before max_iterations.
        trust_history: Per-iteration trust values for each source.
    """
    source_trust: Dict[str, float]
    iterations: int
    converged: bool
    trust_history: Dict[str, List[float]] = field(default_factory=dict)


def compute_kbt(
    store: TripleStore,
    source_graphs: list[str],
    *,
    max_iterations: int = 20,
    epsilon: float = 0.0001,
) -> KBTResult:
    """Compute source trust scores via the KBT EM algorithm.

    Args:
        store: TripleStore with named graphs per source.
        source_graphs: List of graph names to evaluate.
        max_iterations: Maximum EM iterations.
        epsilon: Convergence threshold (max change in any trust score).

    Returns:
        KBTResult with trust scores, iteration count, and history.
        Also writes ``prov:reliability`` triples to the ``"meta"`` graph
        in *store* for each source.
    """
    if not source_graphs:
        return KBTResult(
            source_trust={}, iterations=0, converged=True,
        )

    # ── 1. Collect claims per source ─────────────────────────────
    # claims_by_sp: (s, p) -> {source: [(o, belief)]}
    claims_by_sp: Dict[
        Tuple, Dict[str, List[Tuple[object, float]]]
    ] = defaultdict(lambda: defaultdict(list))

    for g in source_graphs:
        for t in store.triples_in_graph(g):
            key = (t.subject, t.predicate)
            belief = t.opinion.belief if t.opinion else 0.5
            claims_by_sp[key][g].append((t.object_, belief))

    # ── 2. Initialize trust ───────────────────────────────────────
    trust: Dict[str, float] = {g: 0.5 for g in source_graphs}
    history: Dict[str, List[float]] = {g: [0.5] for g in source_graphs}

    # ── 3. EM iterations ─────────────────────────────────────────
    for iteration in range(max_iterations):
        # ── E-step: score each distinct o per (s,p) group ────────
        # truth_scores: (s,p) -> {o: score}
        truth_scores: Dict[Tuple, Dict[object, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for sp_key, source_claims in claims_by_sp.items():
            for g, claims in source_claims.items():
                for obj, belief in claims:
                    truth_scores[sp_key][obj] += trust[g] * belief

        # ── M-step: update source trust ──────────────────────────
        new_trust: Dict[str, float] = {}
        for g in source_graphs:
            total = 0
            correct = 0
            for sp_key, source_claims in claims_by_sp.items():
                sp_scores = truth_scores[sp_key]
                if not sp_scores:
                    continue
                winner = max(sp_scores, key=sp_scores.get)
                for obj, belief in source_claims[g]:
                    total += 1
                    if _spos_equal(obj, winner):
                        correct += belief
            # Beta prior: (1 + correct) / (2 + total)
            if total == 0:
                new_trust[g] = 0.5
            else:
                new_trust[g] = (1.0 + correct) / (2.0 + total)

        # ── Check convergence ────────────────────────────────────
        max_delta = max(
            abs(new_trust[g] - trust[g]) for g in source_graphs
        )
        trust = new_trust
        for g in source_graphs:
            history[g].append(trust[g])

        if max_delta < epsilon:
            # Write reliability triples and return
            _write_reliability(store, trust)
            return KBTResult(
                source_trust=trust,
                iterations=iteration + 1,
                converged=True,
                trust_history=dict(history),
            )

    # Reached max iterations without convergence
    _write_reliability(store, trust)
    return KBTResult(
        source_trust=trust,
        iterations=max_iterations,
        converged=False,
        trust_history=dict(history),
    )


def _write_reliability(
    store: TripleStore, trust: Dict[str, float]
) -> None:
    """Write prov:reliability triples into the meta graph."""
    for source, score in trust.items():
        triple = Triple(
            NamedNode(source),
            SOURCE_RELIABILITY,
            Literal(round(score, 6)),
        )
        store.add(triple, graph="meta")


def _spos_equal(a: object, b: object) -> bool:
    """Compare two RDF node values for equality."""
    if isinstance(a, Literal) and isinstance(b, Literal):
        return a.value == b.value
    if isinstance(a, NamedNode) and isinstance(b, NamedNode):
        return a.iri == b.iri
    if isinstance(a, type(b)):
        return a == b
    return str(a) == str(b)
