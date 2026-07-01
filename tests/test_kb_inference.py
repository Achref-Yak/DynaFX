"""Tests for kb/inference.py — rule engine with RDFS and OWL RL rules."""

import pytest

from dynafx.core.models import Opinion
from dynafx.knowledge.inference import (
    OWL_FUNCTIONAL_PROPERTY,
    OWL_INVERSE_OF,
    OWL_SAME_AS,
    OWL_SYMMETRIC_PROPERTY,
    OWL_TRANSITIVE_PROPERTY,
    RDF_TYPE,
    RDFS_DOMAIN,
    RDFS_RANGE,
    RDFS_SUBCLASS_OF,
    RDFS_SUBPROPERTY_OF,
    InferencePattern,
    Rule,
    RuleEngine,
    Var,
    _propagate_average,
    _propagate_min,
    _propagate_product,
    owl_rl_rules,
    propagate_opinion,
    rdfs_rules,
)
from dynafx.knowledge.model import BlankNode, Literal, NamedNode, Triple, TriplePattern
from dynafx.knowledge.store import TripleStore


# ── Var ──────────────────────────────────────────────────────────


class TestVar:
    def test_construction(self):
        v = Var("x")
        assert v.name == "x"

    def test_repr(self):
        assert repr(Var("x")) == "?x"

    def test_equality_and_hash(self):
        a = Var("x")
        b = Var("x")
        c = Var("y")
        assert a == b
        assert a != c
        assert hash(a) == hash(b)


# ── InferencePattern ─────────────────────────────────────────────


class TestInferencePattern:
    def test_all_none(self):
        p = InferencePattern()
        assert p.subject is None
        assert p.predicate is None
        assert p.object_ is None

    def test_with_var(self):
        v = Var("x")
        p = InferencePattern(v, None, None)
        assert p.subject is v

    def test_with_rdf_node(self):
        n = NamedNode("http://example.org/s")
        p = InferencePattern(n, None, None)
        assert p.subject is n

    def test_mixed(self):
        s = NamedNode("http://example.org/s")
        p = RDF_TYPE
        v = Var("o")
        ip = InferencePattern(s, p, v)
        assert ip.subject is s
        assert ip.predicate is p
        assert ip.object_ is v

    def test_equality(self):
        p1 = InferencePattern(Var("x"), RDF_TYPE, Var("c"))
        p2 = InferencePattern(Var("x"), RDF_TYPE, Var("c"))
        assert p1 == p2


# ── Rule ─────────────────────────────────────────────────────────


class TestRule:
    def test_construction(self):
        r = Rule(
            name="test-rule",
            head=[InferencePattern(Var("x"), RDF_TYPE, Var("c"))],
            body=[InferencePattern(Var("x"), RDFS_DOMAIN, Var("c"))],
        )
        assert r.name == "test-rule"
        assert len(r.head) == 1
        assert len(r.body) == 1
        assert r.confidence_fn == "min"

    def test_custom_confidence_fn(self):
        r = Rule(
            name="p-rule",
            head=[InferencePattern(Var("x"), RDF_TYPE, Var("c"))],
            body=[InferencePattern(Var("x"), RDFS_DOMAIN, Var("c"))],
            confidence_fn="product",
        )
        assert r.confidence_fn == "product"

    def test_frozen(self):
        r = Rule(name="r", head=[], body=[])
        with pytest.raises(AttributeError):
            r.name = "new-name"  # type: ignore


# ── Rule sets ────────────────────────────────────────────────────


class TestRdfsRules:
    def test_returns_list(self):
        rules = rdfs_rules()
        assert isinstance(rules, list)
        assert len(rules) == 7

    def test_all_have_names(self):
        for r in rdfs_rules():
            assert r.name.startswith("rdfs-")

    def test_rdfs_domain_rule(self):
        rules = [r for r in rdfs_rules() if r.name == "rdfs-domain"]
        assert len(rules) == 1
        r = rules[0]
        assert len(r.body) == 2
        assert len(r.head) == 1

    def test_rdfs_subclass_trans(self):
        rules = [r for r in rdfs_rules() if r.name == "rdfs-subclass-trans"]
        assert len(rules) == 1
        r = rules[0]
        assert len(r.body) == 2

    def test_rdfs_class_instance(self):
        rules = [r for r in rdfs_rules() if r.name == "rdfs-class-instance"]
        assert len(rules) == 1
        r = rules[0]
        assert len(r.body) == 1


class TestOwlRlRules:
    def test_returns_list(self):
        rules = owl_rl_rules()
        assert isinstance(rules, list)
        assert len(rules) == 4

    def test_all_have_names(self):
        for r in owl_rl_rules():
            assert r.name.startswith("owl-")

    def test_owl_inverse_rule(self):
        rules = [r for r in owl_rl_rules() if r.name == "owl-inverse"]
        assert len(rules) == 1
        r = rules[0]
        assert len(r.body) == 2

    def test_owl_functional_rule(self):
        rules = [r for r in owl_rl_rules() if r.name == "owl-functional"]
        assert len(rules) == 1
        r = rules[0]
        assert len(r.body) == 3


# ── RuleEngine ───────────────────────────────────────────────────


class TestRuleEngineConstruction:
    def test_empty_engine(self):
        eng = RuleEngine()
        assert len(eng) == 0
        assert eng.max_iterations == 10

    def test_with_rules(self):
        rules = [Rule(name="r", head=[], body=[])]
        eng = RuleEngine(rules)
        assert len(eng) == 1

    def test_custom_max_iterations(self):
        eng = RuleEngine(max_iterations=5)
        assert eng.max_iterations == 5

    def test_add_rule(self):
        eng = RuleEngine()
        assert len(eng) == 0
        eng.add_rule(Rule(name="r", head=[], body=[]))
        assert len(eng) == 1

    def test_clear_rules(self):
        eng = RuleEngine([Rule(name="r", head=[], body=[])])
        assert len(eng) == 1
        eng.clear_rules()
        assert len(eng) == 0


# ── Forward-chaining: RDFS ───────────────────────────────────────


class TestRdfsDomain:
    def test_infers_type_from_domain(self):
        store = TripleStore()
        person = NamedNode("http://example.org/Person")
        alice = NamedNode("http://example.org/Alice")
        has_age = NamedNode("http://example.org/hasAge")

        store.add(Triple(alice, has_age, Literal(30)))
        store.add(Triple(has_age, RDFS_DOMAIN, person))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(alice, RDF_TYPE, person) in store

    def test_domain_rule_dedup(self):
        store = TripleStore()
        person = NamedNode("http://example.org/Person")
        alice = NamedNode("http://example.org/Alice")
        has_age = NamedNode("http://example.org/hasAge")

        store.add(Triple(alice, has_age, Literal(30)))
        store.add(Triple(has_age, RDFS_DOMAIN, person))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count1 = eng.apply(store)
        count2 = eng.apply(store)

        assert count2 == 0


class TestRdfsRange:
    def test_infers_type_from_range(self):
        store = TripleStore()
        person = NamedNode("http://example.org/Person")
        knows = NamedNode("http://example.org/knows")
        bob = NamedNode("http://example.org/Bob")

        store.add(Triple(NamedNode("http://example.org/Alice"), knows, bob))
        store.add(Triple(knows, RDFS_RANGE, person))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(bob, RDF_TYPE, person) in store


class TestRdfsSubclassTransitivity:
    def test_infers_transitive_subclass(self):
        store = TripleStore()
        animal = NamedNode("http://example.org/Animal")
        mammal = NamedNode("http://example.org/Mammal")
        dog = NamedNode("http://example.org/Dog")

        store.add(Triple(dog, RDFS_SUBCLASS_OF, mammal))
        store.add(Triple(mammal, RDFS_SUBCLASS_OF, animal))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(dog, RDFS_SUBCLASS_OF, animal) in store


class TestRdfsSubclassUsage:
    def test_infers_type_from_subclass(self):
        store = TripleStore()
        animal = NamedNode("http://example.org/Animal")
        mammal = NamedNode("http://example.org/Mammal")
        dog = NamedNode("http://example.org/Dog")
        fido = NamedNode("http://example.org/Fido")

        store.add(Triple(dog, RDFS_SUBCLASS_OF, mammal))
        store.add(Triple(mammal, RDFS_SUBCLASS_OF, animal))
        store.add(Triple(fido, RDF_TYPE, dog))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        eng.apply(store)

        assert TriplePattern(fido, RDF_TYPE, mammal) in store
        assert TriplePattern(fido, RDF_TYPE, animal) in store


class TestRdfsSubpropertyTransitivity:
    def test_infers_transitive_subproperty(self):
        store = TripleStore()
        p1 = NamedNode("http://example.org/p1")
        p2 = NamedNode("http://example.org/p2")
        p3 = NamedNode("http://example.org/p3")

        store.add(Triple(p1, RDFS_SUBPROPERTY_OF, p2))
        store.add(Triple(p2, RDFS_SUBPROPERTY_OF, p3))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(p1, RDFS_SUBPROPERTY_OF, p3) in store


class TestRdfsSubpropertyUsage:
    def test_infers_via_subproperty(self):
        store = TripleStore()
        parent = NamedNode("http://example.org/parent")
        has_child = NamedNode("http://example.org/hasChild")
        alice = NamedNode("http://example.org/Alice")
        bob = NamedNode("http://example.org/Bob")

        store.add(Triple(alice, has_child, bob))
        store.add(Triple(has_child, RDFS_SUBPROPERTY_OF, parent))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(alice, parent, bob) in store


class TestRdfsClassInstance:
    def test_infers_type_from_subclass(self):
        store = TripleStore()
        mammal = NamedNode("http://example.org/Mammal")
        dog = NamedNode("http://example.org/Dog")

        store.add(Triple(dog, RDFS_SUBCLASS_OF, mammal))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(dog, RDF_TYPE, mammal) in store


# ── Forward-chaining: OWL RL ─────────────────────────────────────


class TestOwlInverse:
    def test_infers_inverse(self):
        store = TripleStore()
        has_child = NamedNode("http://example.org/hasChild")
        has_parent = NamedNode("http://example.org/hasParent")
        alice = NamedNode("http://example.org/Alice")
        bob = NamedNode("http://example.org/Bob")

        store.add(Triple(has_child, OWL_INVERSE_OF, has_parent))
        store.add(Triple(alice, has_child, bob))

        eng = RuleEngine(owl_rl_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(bob, has_parent, alice) in store


class TestOwlSymmetric:
    def test_infers_symmetric(self):
        store = TripleStore()
        knows = NamedNode("http://example.org/knows")
        alice = NamedNode("http://example.org/Alice")
        bob = NamedNode("http://example.org/Bob")

        store.add(Triple(knows, RDF_TYPE, OWL_SYMMETRIC_PROPERTY))
        store.add(Triple(alice, knows, bob))

        eng = RuleEngine(owl_rl_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(bob, knows, alice) in store


class TestOwlTransitive:
    def test_infers_transitive(self):
        store = TripleStore()
        ancestor_of = NamedNode("http://example.org/ancestorOf")
        alice = NamedNode("http://example.org/Alice")
        bob = NamedNode("http://example.org/Bob")
        carol = NamedNode("http://example.org/Carol")

        store.add(Triple(ancestor_of, RDF_TYPE, OWL_TRANSITIVE_PROPERTY))
        store.add(Triple(alice, ancestor_of, bob))
        store.add(Triple(bob, ancestor_of, carol))

        eng = RuleEngine(owl_rl_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(alice, ancestor_of, carol) in store


class TestOwlFunctional:
    def test_infers_same_as(self):
        store = TripleStore()
        has_ssn = NamedNode("http://example.org/hasSSN")
        x = NamedNode("http://example.org/personX")
        ssn1 = NamedNode("http://example.org/ssn1")
        ssn2 = NamedNode("http://example.org/ssn2")

        store.add(Triple(has_ssn, RDF_TYPE, OWL_FUNCTIONAL_PROPERTY))
        store.add(Triple(x, has_ssn, ssn1))
        store.add(Triple(x, has_ssn, ssn2))

        eng = RuleEngine(owl_rl_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert TriplePattern(ssn1, OWL_SAME_AS, ssn2) in store


# ── Fixpoint ─────────────────────────────────────────────────────


class TestFixpoint:
    def test_does_not_loop_forever(self):
        store = TripleStore()
        a = NamedNode("http://example.org/a")
        b = NamedNode("http://example.org/b")

        store.add(Triple(a, RDFS_SUBCLASS_OF, b))
        store.add(Triple(b, RDFS_SUBCLASS_OF, a))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)

        assert count >= 1
        assert count <= 50  # safety: no infinite loop
        # Should not have more triples than reasonable
        assert len(store) <= 50

    def test_idempotent(self):
        store = TripleStore()
        animal = NamedNode("http://example.org/Animal")
        mammal = NamedNode("http://example.org/Mammal")
        dog = NamedNode("http://example.org/Dog")

        store.add(Triple(dog, RDFS_SUBCLASS_OF, mammal))
        store.add(Triple(mammal, RDFS_SUBCLASS_OF, animal))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        c1 = eng.apply(store)
        c2 = eng.apply(store)
        c3 = eng.apply(store)

        assert c1 > 0
        assert c2 == 0
        assert c3 == 0


# ── Opinion propagation ──────────────────────────────────────────


class TestOpinionPropagation:
    def test_min(self):
        o1 = Opinion(0.8, 0.1, 0.1)
        o2 = Opinion(0.6, 0.2, 0.2)
        result = _propagate_min([o1, o2])
        assert result.belief == pytest.approx(0.6)
        assert result.disbelief == pytest.approx(0.2)
        assert result.uncertainty == pytest.approx(0.2)

    def test_product(self):
        o1 = Opinion(0.8, 0.1, 0.1)
        o2 = Opinion(0.6, 0.2, 0.2)
        result = _propagate_product([o1, o2])
        # prod_bu = (0.8+0.1)*(0.6+0.2) = 0.9*0.8 = 0.72
        # prod_u = 0.1 * 0.2 = 0.02
        # b = 0.72 - 0.02 = 0.70
        assert result.belief == pytest.approx(0.70)
        assert result.uncertainty == pytest.approx(0.02)

    def test_average(self):
        o1 = Opinion(0.8, 0.1, 0.1)
        o2 = Opinion(0.6, 0.2, 0.2)
        result = _propagate_average([o1, o2])
        assert result.belief == pytest.approx(0.7)
        assert result.disbelief == pytest.approx(0.15)
        assert result.uncertainty == pytest.approx(0.15)

    def test_propagate_opinion_min(self):
        r = Rule(name="t", head=[], body=[], confidence_fn="min")
        o1 = Opinion(0.9, 0.05, 0.05)
        o2 = Opinion(0.7, 0.1, 0.2)
        result = propagate_opinion(r, [o1, o2])
        assert result.belief == pytest.approx(0.7)

    def test_propagate_opinion_product(self):
        r = Rule(name="t", head=[], body=[], confidence_fn="product")
        o1 = Opinion(0.8, 0.1, 0.1)
        o2 = Opinion(0.6, 0.2, 0.2)
        result = propagate_opinion(r, [o1, o2])
        assert result.belief == pytest.approx(0.70, abs=1e-10)

    def test_propagate_opinion_average(self):
        r = Rule(name="t", head=[], body=[], confidence_fn="average")
        o1 = Opinion(0.8, 0.1, 0.1)
        o2 = Opinion(0.6, 0.2, 0.2)
        result = propagate_opinion(r, [o1, o2])
        assert result.belief == pytest.approx(0.7)

    def test_propagate_opinion_unknown_fn_falls_back_to_min(self):
        r = Rule(name="t", head=[], body=[], confidence_fn="unknown")
        o1 = Opinion(0.8, 0.1, 0.1)
        o2 = Opinion(0.6, 0.2, 0.2)
        result = propagate_opinion(r, [o1, o2])
        assert result.belief == pytest.approx(0.6)  # min fallback

    def test_propagate_opinion_no_opinions_default(self):
        r = Rule(name="t", head=[], body=[])
        result = propagate_opinion(r, [])
        assert result.belief == pytest.approx(0.5)
        assert result.uncertainty == pytest.approx(0.2)

    def test_propagate_opinion_ignores_none(self):
        r = Rule(name="t", head=[], body=[], confidence_fn="average")
        o = Opinion(0.8, 0.1, 0.1)
        result = propagate_opinion(r, [None, o, None])
        assert result.belief == pytest.approx(0.8)

    def test_opinions_are_clamped(self):
        o1 = Opinion(1.5, -0.2, 0.3)
        o2 = Opinion(0.5, 0.4, 0.1)
        result = _propagate_product([o1, o2])
        assert 0.0 <= result.belief <= 1.0
        assert 0.0 <= result.disbelief <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0


# ── Integration: RDFS + OWL combined ─────────────────────────────


class TestIntegration:
    def test_combined_rules(self):
        store = TripleStore()
        has_child = NamedNode("http://example.org/hasChild")
        has_parent = NamedNode("http://example.org/hasParent")
        person = NamedNode("http://example.org/Person")
        alice = NamedNode("http://example.org/Alice")
        bob = NamedNode("http://example.org/Bob")

        store.add(Triple(has_child, OWL_INVERSE_OF, has_parent))
        store.add(Triple(has_child, RDFS_DOMAIN, person))
        store.add(Triple(alice, has_child, bob))

        eng = RuleEngine(rdfs_rules() + owl_rl_rules(), max_iterations=10)
        eng.apply(store)

        assert TriplePattern(bob, has_parent, alice) in store
        assert TriplePattern(alice, RDF_TYPE, person) in store


# ── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_store(self):
        store = TripleStore()
        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)
        assert count == 0

    def test_no_matching_body(self):
        store = TripleStore()
        alice = NamedNode("http://example.org/Alice")
        store.add(Triple(alice, RDF_TYPE, NamedNode("http://example.org/Person")))

        # Domain rule won't fire since there's no domain triple
        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)
        assert count == 0

    def test_rule_with_multiple_body_patterns_shared_var(self):
        store = TripleStore()
        p = NamedNode("http://example.org/p")
        q = NamedNode("http://example.org/q")
        x = NamedNode("http://example.org/x")

        store.add(Triple(p, RDFS_SUBPROPERTY_OF, q))
        store.add(Triple(x, p, Literal(42)))

        eng = RuleEngine(rdfs_rules(), max_iterations=10)
        count = eng.apply(store)
        assert count >= 1
        assert TriplePattern(x, q, Literal(42)) in store
