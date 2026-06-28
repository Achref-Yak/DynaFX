"""Tests for reason/kbt.py — Knowledge-Based Trust EM algorithm."""

import pytest

from dynafx.core.models import Opinion
from dynafx.kb.model import Literal, NamedNode, Triple, TriplePattern
from dynafx.kb.store import TripleStore
from dynafx.reason.argumentation import PROV_NS, SOURCE_RELIABILITY
from dynafx.reason.kbt import KBTResult, compute_kbt

EX = "http://example.org/"
acme = NamedNode(f"{EX}Acme")
revenue = NamedNode(f"{EX}revenue")
hasCEO = NamedNode(f"{EX}hasCEO")
subsidiary = NamedNode(f"{EX}subsidiary")
rating = NamedNode(f"{EX}rating")


def _make_store() -> TripleStore:
    s = TripleStore()
    return s


# ── Basic KBT behavior ───────────────────────────────────────────


class TestKBT:
    def test_two_agree_one_dissents(self):
        """Two sources agree (5M), one differs (2M) →
        the pair get higher trust, the outlier drops.
        Note: TripleStore dedup means sources with identical
        (s,p,o) share the max-belief version."""
        s = _make_store()
        # a and c both claim revenue=5M → dedup to max belief
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        s.add(Triple(acme, revenue, Literal(2000000),
                     opinion=Opinion(0.5, 0.3, 0.2)), graph="c")
        r = compute_kbt(s, ["a", "b", "c"])
        # a and b (both on the 5M winner) > c (dissenter)
        assert r.source_trust["a"] > r.source_trust["c"]
        assert r.source_trust["b"] > r.source_trust["c"]
        assert r.converged

    def test_all_agree(self):
        """All sources say the same thing → all trust > 0.5."""
        s = _make_store()
        for g in ["a", "b", "c"]:
            s.add(Triple(acme, revenue, Literal(5000000),
                         opinion=Opinion(0.7, 0.1, 0.2)), graph=g)
        r = compute_kbt(s, ["a", "b", "c"])
        for trust in r.source_trust.values():
            assert trust > 0.5

    def test_all_different(self):
        """Three sources, all claim different values → no clear consensus,
        all trust converge near 0.5."""
        s = _make_store()
        s.add(Triple(acme, revenue, Literal(100),
                     opinion=Opinion(0.7, 0.1, 0.2)), graph="a")
        s.add(Triple(acme, revenue, Literal(200),
                     opinion=Opinion(0.7, 0.1, 0.2)), graph="b")
        s.add(Triple(acme, revenue, Literal(300),
                     opinion=Opinion(0.7, 0.1, 0.2)), graph="c")
        r = compute_kbt(s, ["a", "b", "c"])
        for trust in r.source_trust.values():
            assert 0.3 <= trust <= 0.7

    def test_partial_agreement(self):
        """a and b agree on both claims. c agrees on one.
        a and b end up with equal trust > c."""
        s = _make_store()
        # a and b agree on revenue=5M and hasCEO=Alice
        alice = NamedNode(f"{EX}Alice")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        s.add(Triple(acme, hasCEO, alice,
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, hasCEO, alice,
                     opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        # c agrees on hasCEO=Alice but disagrees on revenue (2M)
        s.add(Triple(acme, revenue, Literal(2000000),
                     opinion=Opinion(0.5, 0.3, 0.2)), graph="c")
        s.add(Triple(acme, hasCEO, alice,
                     opinion=Opinion(0.5, 0.3, 0.2)), graph="c")
        r = compute_kbt(s, ["a", "b", "c"])
        # a and b agreed on everything → highest
        # c agreed on hasCEO but not revenue → lower
        assert r.source_trust["a"] > r.source_trust["c"]
        assert r.source_trust["b"] > r.source_trust["c"]

    def test_empty_sources(self):
        s = _make_store()
        r = compute_kbt(s, [])
        assert r.source_trust == {}
        assert r.iterations == 0
        assert r.converged

    def test_single_source(self):
        """One source with no one to contradict → trust ~0.5."""
        s = _make_store()
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        r = compute_kbt(s, ["a"])
        assert 0.4 <= r.source_trust["a"] <= 0.6
        assert r.converged

    def test_belief_weighting(self):
        """High-belief correct source differentiates from low-belief
        correct when they use different predicates (avoiding dedup)."""
        s = _make_store()
        # a claims revenue=5M and rating=Good (b=0.95)
        alice = NamedNode(f"{EX}Alice")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.95, 0.02, 0.03)), graph="a")
        s.add(Triple(acme, rating, Literal("Good"),
                     opinion=Opinion(0.95, 0.02, 0.03)), graph="a")
        # b also claims revenue=5M (dedups to b=0.95 from a) and rating=Average
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.30, 0.40, 0.30)), graph="b")
        s.add(Triple(acme, rating, Literal("Average"),
                     opinion=Opinion(0.30, 0.40, 0.30)), graph="b")
        # c claims revenue=2M and rating=Good (b=0.7)
        s.add(Triple(acme, revenue, Literal(2000000),
                     opinion=Opinion(0.7, 0.1, 0.2)), graph="c")
        s.add(Triple(acme, rating, Literal("Good"),
                     opinion=Opinion(0.7, 0.1, 0.2)), graph="c")
        r = compute_kbt(s, ["a", "b", "c"])
        # a: agrees on revenue=5M (winner) with high belief, rating=Good vs Average
        #    revenue winner=5M, rating winner=Good (a and c say Good)
        #    a: revenue=5M correct (b=0.95), rating=Good correct (b=0.95). trust=(1+1.9)/(2+2)=2.9/4=0.725
        # b: revenue=5M correct (b=0.95 deduped), rating=Average wrong. trust=(1+0.95)/(2+2)=1.95/4=0.4875
        # c: revenue=2M wrong, rating=Good correct (b=0.7). trust=(1+0.7)/(2+2)=1.7/4=0.425
        assert r.source_trust["a"] > r.source_trust["b"]
        assert r.source_trust["a"] > r.source_trust["c"]

    def test_converges_within_iterations(self):
        """Converges within a small number of iterations."""
        s = _make_store()
        mike = NamedNode(f"{EX}Mike")
        nancy = NamedNode(f"{EX}Nancy")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.7, 0.1, 0.2)), graph="a")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.7, 0.1, 0.2)), graph="b")
        s.add(Triple(acme, revenue, Literal(2000000),
                     opinion=Opinion(0.4, 0.3, 0.3)), graph="c")
        r = compute_kbt(s, ["a", "b", "c"])
        assert r.iterations <= 10

    def test_writes_reliability_triples(self):
        """prov:reliability triples are written to the meta graph."""
        s = _make_store()
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, revenue, Literal(2000000),
                     opinion=Opinion(0.5, 0.3, 0.2)), graph="b")
        r = compute_kbt(s, ["a", "b"])
        meta_count = len(list(s.triples_in_graph("meta")))
        assert meta_count == 2
        for g in ["a", "b"]:
            tgt = list(s.triples(TriplePattern(
                NamedNode(g), SOURCE_RELIABILITY, None
            )))
            assert len(tgt) == 1
            val = float(tgt[0].object_.value)
            assert 0.0 <= val <= 1.0

    def test_multiple_properties_independent(self):
        """Different (s,p) groups are scored independently."""
        s = _make_store()
        omega = NamedNode(f"{EX}Omega")
        theta = NamedNode(f"{EX}Theta")
        # revenue: a and b agree (5M), c dissents (2M)
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        s.add(Triple(acme, revenue, Literal(2000000),
                     opinion=Opinion(0.5, 0.3, 0.2)), graph="c")
        # subsidiary: a and c agree (Omega), b dissents (Theta)
        s.add(Triple(acme, subsidiary, omega,
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, subsidiary, theta,
                     opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        s.add(Triple(acme, subsidiary, omega,
                     opinion=Opinion(0.5, 0.3, 0.2)), graph="c")
        r = compute_kbt(s, ["a", "b", "c"])
        # a agrees on both → highest trust
        assert r.source_trust["a"] > r.source_trust["b"]
        assert r.source_trust["a"] > r.source_trust["c"]

    def test_no_opinion_default(self):
        """Triples with no opinion default to 0.5 belief."""
        s = _make_store()
        # a and b same value, c different. No opinions set.
        s.add(Triple(acme, revenue, Literal(5000000)), graph="a")
        s.add(Triple(acme, revenue, Literal(5000000)), graph="b")
        s.add(Triple(acme, revenue, Literal(2000000)), graph="c")
        r = compute_kbt(s, ["a", "b", "c"])
        assert r.source_trust["a"] > r.source_trust["c"]


# ── Integration with argumentation ───────────────────────────────


class TestKBTIntegration:
    def test_kbt_feeds_argumentation(self):
        """Full pipeline: KBT → build_framework reads reliability
        → reliability values are accessible."""
        from dynafx.reason.argumentation import (
            AttackType,
            build_framework,
        )
        s = TripleStore()
        alice = NamedNode(f"{EX}Alice")
        bob = NamedNode(f"{EX}Bob")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        s.add(Triple(acme, revenue, Literal(2000000),
                     opinion=Opinion(0.4, 0.3, 0.3)), graph="c")
        s.add(Triple(acme, hasCEO, alice,
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, hasCEO, bob,
                     opinion=Opinion(0.6, 0.2, 0.2)), graph="c")

        r = compute_kbt(s, ["a", "b", "c"])
        assert r.converged

        # build_framework reads reliability from meta graph
        af = build_framework(
            s, ["a", "b", "c", "meta"],
            auto_rebut=False,
            auto_undermine_low_belief=False,
        )
        # meta graph has prov:reliability triples
        meta_triples = list(s.triples_in_graph("meta"))
        assert len(meta_triples) == 3

    def test_kbt_trust_history_tracks_iters(self):
        """trust_history records per-iteration values."""
        s = _make_store()
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        s.add(Triple(acme, revenue, Literal(2000000),
                     opinion=Opinion(0.5, 0.3, 0.2)), graph="c")
        r = compute_kbt(s, ["a", "b", "c"])
        assert "a" in r.trust_history
        assert "b" in r.trust_history
        assert "c" in r.trust_history
        assert len(r.trust_history["a"]) == r.iterations + 1

    def test_same_values_across_sources(self):
        """When all sources have same (s,p,o), dedup means all
        see the max belief → trust equal across the board."""
        s = _make_store()
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.8, 0.1, 0.1)), graph="a")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        s.add(Triple(acme, revenue, Literal(5000000),
                     opinion=Opinion(0.4, 0.3, 0.3)), graph="c")
        r = compute_kbt(s, ["a", "b", "c"])
        for trust in r.source_trust.values():
            assert trust > 0.5
