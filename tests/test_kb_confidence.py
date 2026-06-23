"""Tests for kb/confidence.py — graph fusion and query grading."""

import pytest

from cognitive_engine.core.models import Opinion
from cognitive_engine.kb.confidence import (
    FusionResult,
    QueryGrade,
    _are_consistent,
    _classify_consensus,
    _compute_agreement_ratio,
    fuse_graphs,
    grade_query,
)
from cognitive_engine.kb.model import NamedNode, Triple, TriplePattern
from cognitive_engine.kb.sparql import QueryResult
from cognitive_engine.kb.store import TripleStore


# ── Helpers ───────────────────────────────────────────────────────


S1 = NamedNode("http://example.org/s1")
S2 = NamedNode("http://example.org/s2")
P = NamedNode("http://example.org/p")
O = NamedNode("http://example.org/o")
O2 = NamedNode("http://example.org/o2")
GRAPH_A = "source_a"
GRAPH_B = "source_b"
GRAPH_C = "source_c"


def _make_store():
    store = TripleStore()
    return store


# ── FusionResult ──────────────────────────────────────────────────


class TestFusionResult:
    def test_defaults(self):
        r = FusionResult(source_graphs=[], fused_count=0,
                         total_candidates=0, agreement_ratio=1.0)
        assert r.fused_count == 0
        assert r.agreement_ratio == 1.0


# ── QueryGrade ────────────────────────────────────────────────────


class TestQueryGrade:
    def test_defaults(self):
        g = QueryGrade(
            avg_belief=0.5, avg_disbelief=0.0,
            avg_uncertainty=0.5, consensus="medium", cardinality=0,
        )
        assert g.avg_belief == 0.5

    def test_label_high(self):
        g = QueryGrade(0.9, 0.05, 0.05, "high", 10)
        assert "HIGH" in g.label()
        assert "10" in g.label()

    def test_label_low(self):
        g = QueryGrade(0.2, 0.6, 0.2, "low", 1)
        assert "LOW" in g.label()


# ── _are_consistent ───────────────────────────────────────────────


class TestAreConsistent:
    def test_consistent_close_opinions(self):
        assert _are_consistent([Opinion(0.8, 0.1, 0.1),
                                Opinion(0.9, 0.05, 0.05)])

    def test_inconsistent_wide_belief_gap(self):
        assert not _are_consistent([Opinion(0.9, 0.05, 0.05),
                                    Opinion(0.2, 0.6, 0.2)])

    def test_single_opinion(self):
        assert _are_consistent([Opinion(0.8, 0.1, 0.1)])

    def test_empty(self):
        assert _are_consistent([])


# ── _classify_consensus ───────────────────────────────────────────


class TestClassifyConsensus:
    def test_high_belief_low_uncertainty(self):
        assert _classify_consensus(0.8, 0.1, 0.1) == "high"

    def test_medium_belief_moderate_uncertainty(self):
        assert _classify_consensus(0.5, 0.3, 0.2) == "medium"

    def test_low_belief_high_uncertainty(self):
        assert _classify_consensus(0.2, 0.3, 0.5) == "low"

    def test_boundary_high(self):
        assert _classify_consensus(0.7, 0.2, 0.1) == "high"

    def test_boundary_medium(self):
        assert _classify_consensus(0.4, 0.3, 0.3) == "medium"


# ── fuse_graphs: basic ────────────────────────────────────────────


class TestFuseGraphsBasic:
    def test_empty_source_graphs(self):
        store = _make_store()
        result = fuse_graphs(store, [])
        assert result.fused_count == 0
        assert result.total_candidates == 0

    def test_no_overlap(self):
        store = _make_store()
        store.add(Triple(S1, P, O, opinion=Opinion(0.8, 0.1, 0.1)),
                  graph=GRAPH_A)
        store.add(Triple(S2, P, O2, opinion=Opinion(0.9, 0.05, 0.05)),
                  graph=GRAPH_B)
        result = fuse_graphs(store, [GRAPH_A, GRAPH_B])
        assert result.fused_count == 0  # no same triple in both graphs
        assert result.agreement_ratio == 1.0

    def test_single_graph_no_fusion(self):
        store = _make_store()
        store.add(Triple(S1, P, O, opinion=Opinion(0.8, 0.1, 0.1)),
                  graph=GRAPH_A)
        result = fuse_graphs(store, [GRAPH_A])
        assert result.fused_count == 0

    def test_cumulative_fusion_two_sources_same_triple(self):
        store = _make_store()
        store.add(Triple(S1, P, O, opinion=Opinion(0.8, 0.1, 0.1)),
                  graph=GRAPH_A)
        store.add(Triple(S1, P, O, opinion=Opinion(0.6, 0.2, 0.2)),
                  graph=GRAPH_B)
        result = fuse_graphs(store, [GRAPH_A, GRAPH_B])
        assert result.fused_count == 1
        # Verify fused opinion is in store
        fused_triple = list(store.triples(TriplePattern(S1, P, O)))[0]
        assert fused_triple.opinion is not None
        assert fused_triple.opinion.belief > 0.6  # fused should be higher than lowest

    def test_compromise_fusion(self):
        store = _make_store()
        store.add(Triple(S1, P, O, opinion=Opinion(0.9, 0.05, 0.05)),
                  graph=GRAPH_A)
        store.add(Triple(S1, P, O, opinion=Opinion(0.1, 0.8, 0.1)),
                  graph=GRAPH_B)
        result = fuse_graphs(store, [GRAPH_A, GRAPH_B], method="compromise")
        assert result.fused_count == 1

    def test_three_sources(self):
        store = _make_store()
        o1 = Opinion(0.8, 0.1, 0.1)
        o2 = Opinion(0.7, 0.2, 0.1)
        o3 = Opinion(0.9, 0.05, 0.05)
        store.add(Triple(S1, P, O, opinion=o1), graph=GRAPH_A)
        store.add(Triple(S1, P, O, opinion=o2), graph=GRAPH_B)
        store.add(Triple(S1, P, O, opinion=o3), graph=GRAPH_C)
        result = fuse_graphs(store, [GRAPH_A, GRAPH_B, GRAPH_C])
        assert result.fused_count == 1

    def test_source_graphs_not_modified(self):
        store = _make_store()
        store.add(Triple(S1, P, O, opinion=Opinion(0.8, 0.1, 0.1)),
                  graph=GRAPH_A)
        store.add(Triple(S1, P, O, opinion=Opinion(0.6, 0.2, 0.2)),
                  graph=GRAPH_B)
        original_a = list(store.triples_in_graph(GRAPH_A))
        original_b = list(store.triples_in_graph(GRAPH_B))
        fuse_graphs(store, [GRAPH_A, GRAPH_B])
        assert list(store.triples_in_graph(GRAPH_A)) == original_a
        assert list(store.triples_in_graph(GRAPH_B)) == original_b


# ── fuse_graphs: target graph ─────────────────────────────────────


class TestFuseGraphsTarget:
    def test_writes_to_target_graph(self):
        store = _make_store()
        store.add(Triple(S1, P, O, opinion=Opinion(0.8, 0.1, 0.1)),
                  graph=GRAPH_A)
        store.add(Triple(S1, P, O, opinion=Opinion(0.6, 0.2, 0.2)),
                  graph=GRAPH_B)
        result = fuse_graphs(store, [GRAPH_A, GRAPH_B],
                             target_graph="fused")
        assert result.fused_count == 1
        fused = list(store.triples_in_graph("fused"))
        assert len(fused) == 1

    def test_target_graph_isolated(self):
        store = _make_store()
        store.add(Triple(S1, P, O, opinion=Opinion(0.8, 0.1, 0.1)),
                  graph=GRAPH_A)
        store.add(Triple(S1, P, O, opinion=Opinion(0.6, 0.2, 0.2)),
                  graph=GRAPH_B)
        fuse_graphs(store, [GRAPH_A, GRAPH_B], target_graph="fused")
        # Source graphs should still have their original triples
        assert len(list(store.triples_in_graph(GRAPH_A))) == 1
        assert len(list(store.triples_in_graph(GRAPH_B))) == 1


# ── grade_query ───────────────────────────────────────────────────


class TestGradeQuery:
    def test_empty_result(self):
        result = QueryResult(vars=[], bindings=[], opinions=[], cardinality=0)
        grade = grade_query(result)
        assert grade.cardinality == 0
        assert grade.consensus == "medium"

    def test_no_opinions(self):
        result = QueryResult(vars=["x"], bindings=[{"x": S1}],
                             opinions=[{}], cardinality=1)
        grade = grade_query(result)
        assert grade.consensus == "medium"

    def test_high_confidence(self):
        result = QueryResult(
            vars=["x"],
            bindings=[{"x": S1}],
            opinions=[{"x": Opinion(0.9, 0.05, 0.05)}],
            cardinality=1,
        )
        grade = grade_query(result)
        assert grade.consensus == "high"
        assert grade.avg_belief == pytest.approx(0.9)

    def test_low_confidence(self):
        result = QueryResult(
            vars=["x"],
            bindings=[{"x": S1}],
            opinions=[{"x": Opinion(0.2, 0.3, 0.5)}],
            cardinality=1,
        )
        grade = grade_query(result)
        assert grade.consensus == "low"

    def test_multiple_bindings_multiple_opinions(self):
        result = QueryResult(
            vars=["x", "y"],
            bindings=[{"x": S1, "y": O}, {"x": S2, "y": O2}],
            opinions=[
                {"x": Opinion(0.9, 0.05, 0.05), "y": Opinion(0.8, 0.1, 0.1)},
                {"x": Opinion(0.7, 0.2, 0.1), "y": Opinion(0.6, 0.3, 0.1)},
            ],
            cardinality=2,
        )
        grade = grade_query(result)
        assert grade.cardinality == 2
        assert grade.avg_belief == pytest.approx(0.75)
        assert grade.avg_disbelief == pytest.approx(0.1625, abs=0.01)
        assert grade.avg_uncertainty == pytest.approx(0.0875, abs=0.01)

    def test_mixed_none_opinions(self):
        result = QueryResult(
            vars=["x"],
            bindings=[{"x": S1}],
            opinions=[{"x": None}],
            cardinality=1,
        )
        grade = grade_query(result)
        assert grade.avg_belief == 0.5  # default when no valid opinions

    def test_cardinality_preserved(self):
        result = QueryResult(
            vars=["x"],
            bindings=[{"x": S1}, {"x": S2}, {"x": O}],
            opinions=[
                {"x": Opinion(0.9, 0.05, 0.05)},
                {"x": Opinion(0.5, 0.3, 0.2)},
                {"x": Opinion(0.3, 0.4, 0.3)},
            ],
            cardinality=3,
        )
        grade = grade_query(result)
        assert grade.cardinality == 3


# ── Integration ───────────────────────────────────────────────────


class TestIntegration:
    def test_fuse_then_grade(self):
        store = TripleStore()
        store.add(Triple(S1, P, O, opinion=Opinion(0.9, 0.05, 0.05)),
                  graph=GRAPH_A)
        store.add(Triple(S1, P, O, opinion=Opinion(0.7, 0.1, 0.2)),
                  graph=GRAPH_B)
        store.add(Triple(S2, P, O, opinion=Opinion(0.8, 0.1, 0.1)),
                  graph=GRAPH_A)
        store.add(Triple(S2, P, O, opinion=Opinion(0.6, 0.2, 0.2)),
                  graph=GRAPH_B)

        fuse_graphs(store, [GRAPH_A, GRAPH_B], target_graph="fused")

        fused = list(store.triples_in_graph("fused"))
        assert len(fused) == 2
        for t in fused:
            assert t.opinion is not None
            assert t.opinion.belief > 0.5  # fused belief should be reasonable
