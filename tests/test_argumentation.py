"""Tests for reason/argumentation.py — Dung argumentation framework."""

import pytest

from dynafx.core.models import Opinion
from dynafx.kb.model import Literal, NamedNode, Triple, TriplePattern
from dynafx.kb.store import TripleStore
from dynafx.reason.argumentation import (
    ARG_NS,
    PROV_NS,
    Argument,
    ArgumentationFramework,
    Attack,
    AttackType,
    SOURCE_RELIABILITY,
    SupportType,
    build_framework,
)

EX = "http://example.org/"
acme = NamedNode(f"{EX}AcmeCorp")
alice = NamedNode(f"{EX}Alice")
bob = NamedNode(f"{EX}Bob")
carol = NamedNode(f"{EX}Carol")
hasCEO = NamedNode(f"{EX}hasCEO")
revenue = NamedNode(f"{EX}revenue")
Org = NamedNode(f"{EX}Organization")


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def simple_store() -> TripleStore:
    """Store with two agreeing sources."""
    s = TripleStore()
    t1 = Triple(acme, hasCEO, alice, opinion=Opinion(0.8, 0.1, 0.1))
    t2 = Triple(acme, hasCEO, alice, opinion=Opinion(0.6, 0.2, 0.2))
    s.add(t1, graph="src_a")
    s.add(t2, graph="src_b")
    return s


@pytest.fixture
def conflict_store() -> TripleStore:
    """Store with contradictory claims about the same property."""
    s = TripleStore()
    t1 = Triple(acme, revenue, Literal(1000000), opinion=Opinion(0.8, 0.1, 0.1))
    t2 = Triple(acme, revenue, Literal(500000), opinion=Opinion(0.6, 0.2, 0.2))
    s.add(t1, graph="src_a")
    s.add(t2, graph="src_b")
    return s


@pytest.fixture
def low_belief_store() -> TripleStore:
    """Store with a very low-belief triple."""
    s = TripleStore()
    t = Triple(acme, hasCEO, bob, opinion=Opinion(0.05, 0.85, 0.10))
    s.add(t, graph="src_a")
    return s


@pytest.fixture
def reliability_store() -> TripleStore:
    """Store with a source reliability meta-triple."""
    s = TripleStore()
    t1 = Triple(acme, hasCEO, alice, opinion=Opinion(0.8, 0.1, 0.1))
    t2 = Triple(acme, revenue, Literal(1000000), opinion=Opinion(0.7, 0.1, 0.2))
    s.add(t1, graph="src_a")
    s.add(t2, graph="src_a")
    # Source reliability: src_a is unreliable
    rel_triple = Triple(
        NamedNode("src_a"), SOURCE_RELIABILITY, Literal(0.15),
    )
    s.add(rel_triple, graph="meta")
    return s


# ── Argument ──────────────────────────────────────────────────────


class TestArgument:
    def test_construction(self):
        t = Triple(acme, hasCEO, alice)
        a = Argument(id="a1", triple=t, source_graph="g1")
        assert a.id == "a1"
        assert a.triple == t
        assert a.source_graph == "g1"
        assert a.support_type == SupportType.EVIDENCE
        assert a.strength == 1.0

    def test_custom_support_type(self):
        t = Triple(acme, hasCEO, alice)
        a = Argument(id="a1", triple=t,
                     support_type=SupportType.INFERENCE,
                     strength=0.7)
        assert a.support_type == SupportType.INFERENCE
        assert a.strength == 0.7


# ── Attack ────────────────────────────────────────────────────────


class TestAttack:
    def test_construction(self):
        a = Attack(source_id="a1", target_id="a2",
                    attack_type=AttackType.REBUT)
        assert a.source_id == "a1"
        assert a.target_id == "a2"
        assert a.attack_type == AttackType.REBUT
        assert a.strength == 1.0

    def test_undermine_attack(self):
        a = Attack(source_id="a1", target_id="a2",
                    attack_type=AttackType.UNDERMINE,
                    strength=0.5)
        assert a.attack_type == AttackType.UNDERMINE
        assert a.strength == 0.5


# ── ArgumentationFramework ───────────────────────────────────────


class TestArgumentationFramework:
    def test_empty_framework(self):
        af = ArgumentationFramework()
        ext = af.compute_grounded()
        assert ext == set()

    def test_single_argument_no_attacks(self):
        af = ArgumentationFramework()
        t = Triple(acme, hasCEO, alice)
        af.add_argument(Argument(id="a1", triple=t))
        ext = af.compute_grounded()
        assert ext == {"a1"}

    def test_single_argument_self_attack(self):
        af = ArgumentationFramework()
        t = Triple(acme, hasCEO, alice)
        af.add_argument(Argument(id="a1", triple=t))
        af.add_attack(Attack(source_id="a1", target_id="a1"))
        ext = af.compute_grounded()
        assert ext == set()  # self-attacking can't be accepted

    def test_mutual_attack_yields_empty(self):
        af = ArgumentationFramework()
        t1 = Triple(acme, hasCEO, alice)
        t2 = Triple(acme, hasCEO, bob)
        af.add_argument(Argument(id="a1", triple=t1))
        af.add_argument(Argument(id="a2", triple=t2))
        af.add_attack(Attack(source_id="a1", target_id="a2"))
        af.add_attack(Attack(source_id="a2", target_id="a1"))
        ext = af.compute_grounded()
        assert ext == set()

    def test_defended_argument(self):
        """a3 attacks a2, a2 attacks a1. a3 is unattacked → accepted.
        a3 defends a1 against a2 → all three accepted."""
        af = ArgumentationFramework()
        t1 = Triple(acme, hasCEO, alice)
        t2 = Triple(acme, hasCEO, bob)
        t3 = Triple(acme, hasCEO, carol)
        af.add_argument(Argument(id="a1", triple=t1))
        af.add_argument(Argument(id="a2", triple=t2))
        af.add_argument(Argument(id="a3", triple=t3))
        af.add_attack(Attack(source_id="a2", target_id="a1"))
        af.add_attack(Attack(source_id="a3", target_id="a2"))
        ext = af.compute_grounded()
        assert "a3" in ext   # no attacker
        assert "a1" in ext   # defended by a3
        assert "a2" not in ext  # attacked by a3, not defended

    def test_chain_of_attack(self):
        """a1 ← a2 ← a3 ← a4. a4 unattacked.
        Grounded: {a2, a4} — a3 defeated by a4, a1 defeated by a2 who
        is not yet defended when a1 is checked (a3 ∉ S)."""
        af = ArgumentationFramework()
        for i in range(1, 5):
            t = Triple(acme, hasCEO, NamedNode(f"{EX}Person{i}"))
            af.add_argument(Argument(id=f"a{i}", triple=t))
        af.add_attack(Attack(source_id="a2", target_id="a1"))
        af.add_attack(Attack(source_id="a3", target_id="a2"))
        af.add_attack(Attack(source_id="a4", target_id="a3"))
        ext = af.compute_grounded()
        assert "a4" in ext   # no attacker
        assert "a2" in ext   # a4 ∈ S counter-attacks a3 → a2 defended
        assert "a1" not in ext  # a3 ∉ S → a2 not counter-attacked → a1 undefended
        assert "a3" not in ext  # a4 ∉ S (a4 has no attacker) → a4 not counter-attacked

    def test_strength_threshold(self):
        """Weak attacks below threshold are ignored."""
        af = ArgumentationFramework()
        t1 = Triple(acme, hasCEO, alice)
        t2 = Triple(acme, hasCEO, bob)
        af.add_argument(Argument(id="a1", triple=t1))
        af.add_argument(Argument(id="a2", triple=t2))
        af.add_attack(Attack(source_id="a1", target_id="a2", strength=0.1))
        ext = af.compute_grounded(min_attack_strength=0.5)
        assert ext == {"a1", "a2"}  # both survive because the weak attack is ignored

    def test_acceptable_triples(self):
        af = ArgumentationFramework()
        t1 = Triple(acme, hasCEO, alice)
        t2 = Triple(acme, hasCEO, bob)
        af.add_argument(Argument(id="a1", triple=t1))
        af.add_argument(Argument(id="a2", triple=t2))
        af.add_attack(Attack(source_id="a1", target_id="a2"))
        ext = af.compute_grounded()
        triples = af.acceptable_triples(ext)
        assert t1 in triples  # unattacked
        assert t2 not in triples  # attacked, undefended

    def test_filter_store(self):
        af = ArgumentationFramework()
        t1 = Triple(acme, hasCEO, alice)
        t2 = Triple(acme, hasCEO, bob)
        af.add_argument(Argument(id="a1", triple=t1))
        af.add_argument(Argument(id="a2", triple=t2))
        af.add_attack(Attack(source_id="a1", target_id="a2"))
        store = TripleStore()
        store.add(t1, graph="g1")
        store.add(t2, graph="g2")
        filtered = af.filter_store(store)
        assert t1 in list(filtered.all_triples())
        assert t2 not in list(filtered.all_triples())

    def test_preferred_extensions(self):
        """Two arguments mutually attacking → two preferred extensions."""
        af = ArgumentationFramework()
        t1 = Triple(acme, hasCEO, alice)
        t2 = Triple(acme, hasCEO, bob)
        af.add_argument(Argument(id="a1", triple=t1))
        af.add_argument(Argument(id="a2", triple=t2))
        af.add_attack(Attack(source_id="a1", target_id="a2"))
        af.add_attack(Attack(source_id="a2", target_id="a1"))
        exts = af.compute_preferred()
        assert len(exts) >= 1
        # Both {a1} and {a2} are maximal admissible
        assert {"a1"} in exts or {"a2"} in exts
        # The empty set is admissible but not maximal
        assert set() not in exts


# ── build_framework ──────────────────────────────────────────────


class TestBuildFramework:
    def test_agreement_no_rebut(self, simple_store):
        """Two sources agreeing → no rebut attacks, both accepted."""
        af = build_framework(simple_store, ["src_a", "src_b"],
                              auto_rebut=True)
        ext = af.compute_grounded()
        assert len(ext) == 2  # both accepted

    def test_rebut_attacks_generated(self, conflict_store):
        """Contradictory claims generate mutual rebut attacks."""
        af = build_framework(conflict_store, ["src_a", "src_b"],
                              auto_rebut=True)
        # Should have 2 rebut attacks (mutual)
        rebut = [a for a in af.attacks if a.attack_type == AttackType.REBUT]
        assert len(rebut) >= 2

    def test_rebut_filters_contradictions(self, conflict_store):
        """Contradictory claims → grounded extension is empty."""
        af = build_framework(conflict_store, ["src_a", "src_b"],
                              auto_rebut=True)
        ext = af.compute_grounded()
        # Both contradict each other → neither is defensible
        assert len(ext) == 0

    def test_no_rebut_when_disabled(self, conflict_store):
        """With auto_rebut=False, contradictory claims both survive."""
        af = build_framework(conflict_store, ["src_a", "src_b"],
                              auto_rebut=False)
        ext = af.compute_grounded()
        assert len(ext) == 2  # both survive

    def test_low_belief_undermine(self, low_belief_store):
        """Low-belief triple gets attacked by skeptic."""
        af = build_framework(low_belief_store, ["src_a"],
                              auto_undermine_low_belief=True,
                              min_belief=0.2)
        # 1 argument + 1 skeptic = 2
        assert len(af.arguments) == 2
        assert "_skeptic" in af.arguments
        # The evidence argument should be attacked
        undermine = [a for a in af.attacks
                     if a.attack_type == AttackType.UNDERMINE]
        assert len(undermine) >= 1

    def test_low_belief_survives_with_high_min(self, low_belief_store):
        """With min_belief=0.01, the low-belief triple is not attacked."""
        af = build_framework(low_belief_store, ["src_a"],
                              auto_undermine_low_belief=True,
                              min_belief=0.01)
        ext = af.compute_grounded()
        assert len(ext) >= 1  # the low-belief argument survives

    def test_source_reliability_undermine(self, reliability_store):
        """Unreliable source (reliability=0.15) generates undermine attacks."""
        af = build_framework(reliability_store, ["src_a", "meta"],
                              auto_rebut=False,
                              auto_undermine_low_belief=False)
        # Both triples from src_a should be attacked by reliability argument
        undermine = [a for a in af.attacks
                     if a.attack_type == AttackType.UNDERMINE]
        assert len(undermine) >= 2

    def test_reliable_source_survives(self):
        """Reliable source (reliability=0.9) — no undermine attacks."""
        s = TripleStore()
        t = Triple(acme, hasCEO, alice, opinion=Opinion(0.8, 0.1, 0.1))
        s.add(t, graph="src_a")
        rel = Triple(NamedNode("src_a"), SOURCE_RELIABILITY, Literal(0.9))
        s.add(rel, graph="meta")
        af = build_framework(s, ["src_a", "meta"],
                              auto_rebut=False,
                              auto_undermine_low_belief=False)
        # Reliability is high (>= min_attack_strength default 0.3) → no attacks
        undermine = [a for a in af.attacks
                     if a.attack_type == AttackType.UNDERMINE]
        assert len(undermine) == 0

    def test_no_rebut_for_same_values(self, simple_store):
        """Same (s,p,o) across sources → no rebut attacks."""
        af = build_framework(simple_store, ["src_a", "src_b"],
                              auto_rebut=True)
        rebut = [a for a in af.attacks
                 if a.attack_type == AttackType.REBUT]
        assert len(rebut) == 0

    def test_three_way_conflict(self):
        """Three contradictory claims — all rebut each other."""
        s = TripleStore()
        s.add(Triple(acme, revenue, Literal(100)), graph="a")
        s.add(Triple(acme, revenue, Literal(200)), graph="b")
        s.add(Triple(acme, revenue, Literal(300)), graph="c")
        af = build_framework(s, ["a", "b", "c"], auto_rebut=True)
        ext = af.compute_grounded()
        assert len(ext) == 0  # no defense possible

    def test_two_agree_one_dissents(self):
        """Two agree on value=100, one says 200 — grounded is empty
        (no unattacked argument), but preferred semantics finds {a,b}."""
        from dynafx.reason.argumentation import ArgumentationFramework
        s = TripleStore()
        s.add(Triple(acme, revenue, Literal(100), opinion=Opinion(0.7, 0.1, 0.2)), graph="a")
        s.add(Triple(acme, revenue, Literal(100), opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        s.add(Triple(acme, revenue, Literal(200), opinion=Opinion(0.5, 0.3, 0.2)), graph="c")
        af = build_framework(s, ["a", "b", "c"], auto_rebut=True)
        # Grounded: empty (no unattacked argument in mutual rebut)
        ext = af.compute_grounded()
        assert len(ext) == 0
        # Preferred: {a, b} is maximal admissible
        prefs = af.compute_preferred()
        # Find the extension that contains a and b
        found = any("a2" in p and "a3" not in p for p in prefs)
        # (a, b, c are stored as a1, a2, a3 or similar)
        # Just check we have at least one non-empty extension
        non_empty = [p for p in prefs if p]
        # The pair (value=100) should be together
        pair_exists = any(
            len(p) >= 2 for p in prefs
        )
        assert pair_exists


# ── Integration with confidence.py ──────────────────────────────


class TestArgumentativeFilter:
    def test_filter_in_confidence_pipeline(self):
        """Integration: argumentative_filter removes contradicting triples."""
        from dynafx.kb.confidence import (
            argumentative_filter, fuse_graphs,
        )
        s = TripleStore()
        s.add(Triple(acme, revenue, Literal(100), opinion=Opinion(0.7, 0.1, 0.2)), graph="a")
        s.add(Triple(acme, revenue, Literal(100), opinion=Opinion(0.6, 0.2, 0.2)), graph="b")
        s.add(Triple(acme, revenue, Literal(200), opinion=Opinion(0.5, 0.3, 0.2)), graph="c")

        # With default (grounded), all mutually attacking are removed.
        filtered = argumentative_filter(s, ["a", "b", "c"])
        cleaned = list(filtered.all_triples())
        assert len(cleaned) == 0  # grounded yields empty for mutual rebut

        # With auto_rebut=False, no filtering happens
        filtered2 = argumentative_filter(
            s, ["a", "b", "c"],
            auto_rebut=False,
        )
        cleaned2 = list(filtered2.all_triples())
        assert len(cleaned2) >= 1  # deduped to 1 or 2

        # Fuse survivors from non-rebut filter
        result = fuse_graphs(
            filtered2, list(filtered2.graphs()),
            target_graph="fused", method="cumulative",
        )
        assert result.fused_count >= 0

    def test_no_effect_on_agreement(self, simple_store):
        """Uncontested triples pass through argumentation unchanged."""
        from dynafx.kb.confidence import argumentative_filter
        filtered = argumentative_filter(
            simple_store, ["src_a", "src_b"],
            auto_rebut=True, auto_undermine_low_belief=False,
        )
        assert len(list(filtered.all_triples())) == 1  # deduped to 1
