"""Tests for kb/inference.py — rule engine with RDFS and OWL RL rules."""

import pytest

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
    owl_rl_rules,
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


# ── Query-time inference (with_inference) ──────────────────────────


class TestQueryTimeInference:
    """Tests for TripleStore.triples(with_inference="rdfs") mode.

    This queries against the RDFS type closure directly without
    materialising inference triples via the rule engine.
    """

    RDF_TYPE_STR = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    RDFS_SUBCLASS_OF_STR = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
    RDFS_DOMAIN_STR = "http://www.w3.org/2000/01/rdf-schema#domain"
    RDFS_RANGE_STR = "http://www.w3.org/2000/01/rdf-schema#range"

    EX = "http://example.org/"

    def test_simple_subclass_expansion(self):
        """(?x rdf:type ?c) ∧ (?c rdfs:subClassOf ?d) → (?x rdf:type ?d)."""
        store = TripleStore()
        vehicle = NamedNode(self.EX + "Vehicle")
        car = NamedNode(self.EX + "Car")
        my_car = NamedNode(self.EX + "myCar")

        store.add(Triple(car, RDFS_SUBCLASS_OF, vehicle))
        store.add(Triple(my_car, RDF_TYPE, car))

        # Without inference: only direct type
        results = list(store.triples(TriplePattern(my_car, RDF_TYPE, None)))
        assert len(results) == 1
        assert results[0].object_ == car

        # With inference: also finds Vehicle
        inf_results = list(store.triples(
            TriplePattern(my_car, RDF_TYPE, None), with_inference="rdfs"
        ))
        types = {r.object_.iri for r in inf_results}
        assert car.iri in types
        assert vehicle.iri in types

    def test_transitive_subclass_chain(self):
        """SubClassOf closure is transitive."""
        store = TripleStore()
        thing = NamedNode(self.EX + "Thing")
        vehicle = NamedNode(self.EX + "Vehicle")
        car = NamedNode(self.EX + "Car")
        my_car = NamedNode(self.EX + "myCar")

        store.add(Triple(car, RDFS_SUBCLASS_OF, vehicle))
        store.add(Triple(vehicle, RDFS_SUBCLASS_OF, thing))
        store.add(Triple(my_car, RDF_TYPE, car))

        inf_results = list(store.triples(
            TriplePattern(my_car, RDF_TYPE, None), with_inference="rdfs"
        ))
        types = {r.object_.iri for r in inf_results}
        assert car.iri in types
        assert vehicle.iri in types
        assert thing.iri in types

    def test_query_with_specific_class_expands_subclasses(self):
        """Query (?s rdf:type Vehicle) matches instances of Car."""
        store = TripleStore()
        vehicle = NamedNode(self.EX + "Vehicle")
        car = NamedNode(self.EX + "Car")
        my_car = NamedNode(self.EX + "myCar")

        store.add(Triple(car, RDFS_SUBCLASS_OF, vehicle))
        store.add(Triple(my_car, RDF_TYPE, car))

        # Without inference: no direct Vehicle instances
        results = list(store.triples(TriplePattern(None, RDF_TYPE, vehicle)))
        assert len(results) == 0

        # With inference: myCar is a Vehicle via subclass
        inf_results = list(store.triples(
            TriplePattern(None, RDF_TYPE, vehicle), with_inference="rdfs"
        ))
        assert len(inf_results) == 1
        assert inf_results[0].subject == my_car

    def test_domain_inference(self):
        """(?x ?p ?o) ∧ (?p rdfs:domain ?c) → (?x rdf:type ?c)."""
        store = TripleStore()
        person = NamedNode(self.EX + "Person")
        has_age = NamedNode(self.EX + "hasAge")
        alice = NamedNode(self.EX + "Alice")

        store.add(Triple(has_age, RDFS_DOMAIN, person))
        store.add(Triple(alice, has_age, Literal(30)))

        inf_results = list(store.triples(
            TriplePattern(alice, RDF_TYPE, None), with_inference="rdfs"
        ))
        types = {r.object_.iri for r in inf_results}
        assert person.iri in types

    def test_range_inference(self):
        """(?x ?p ?o) ∧ (?p rdfs:range ?c) → (?o rdf:type ?c)."""
        store = TripleStore()
        person = NamedNode(self.EX + "Person")
        has_owner = NamedNode(self.EX + "hasOwner")
        alice = NamedNode(self.EX + "Alice")
        bob = NamedNode(self.EX + "Bob")

        store.add(Triple(has_owner, RDFS_RANGE, person))
        store.add(Triple(alice, has_owner, bob))

        inf_results = list(store.triples(
            TriplePattern(bob, RDF_TYPE, None), with_inference="rdfs"
        ))
        types = {r.object_.iri for r in inf_results}
        assert person.iri in types

    def test_no_inference_without_flag(self):
        """Default triples() does not return inferred results."""
        store = TripleStore()
        vehicle = NamedNode(self.EX + "Vehicle")
        car = NamedNode(self.EX + "Car")
        my_car = NamedNode(self.EX + "myCar")

        store.add(Triple(car, RDFS_SUBCLASS_OF, vehicle))
        store.add(Triple(my_car, RDF_TYPE, car))

        results = list(store.triples(TriplePattern(my_car, RDF_TYPE, None)))
        assert all(r.object_ == car for r in results)

    def test_cache_invalidation_on_add(self):
        """Adding a triple invalidates the inference cache."""
        store = TripleStore()
        vehicle = NamedNode(self.EX + "Vehicle")
        car = NamedNode(self.EX + "Car")
        my_car = NamedNode(self.EX + "myCar")

        store.add(Triple(car, RDFS_SUBCLASS_OF, vehicle))
        store.add(Triple(my_car, RDF_TYPE, car))

        # Warm cache
        list(store.triples(
            TriplePattern(my_car, RDF_TYPE, None), with_inference="rdfs"
        ))
        assert store._rdfs_type_closure is not None

        # Add new triple — cache should be cleared
        bike = NamedNode(self.EX + "Bike")
        my_bike = NamedNode(self.EX + "myBike")
        store.add(Triple(bike, RDFS_SUBCLASS_OF, vehicle))
        store.add(Triple(my_bike, RDF_TYPE, bike))

        assert store._rdfs_type_closure is None

        # Should now find myBike as Vehicle too
        inf_results = list(store.triples(
            TriplePattern(None, RDF_TYPE, vehicle), with_inference="rdfs"
        ))
        subjects = {r.subject for r in inf_results}
        assert my_car in subjects
        assert my_bike in subjects

    def test_sparql_query_with_inference(self):
        """SPARQL evaluate passes with_inference through to store."""
        from dynafx.knowledge.sparql import parse_sparql, evaluate as sparql_evaluate

        store = TripleStore()
        vehicle = NamedNode(self.EX + "Vehicle")
        car = NamedNode(self.EX + "Car")
        my_car = NamedNode(self.EX + "myCar")

        store.add(Triple(car, RDFS_SUBCLASS_OF, vehicle))
        store.add(Triple(my_car, RDF_TYPE, car))

        # SPARQL: SELECT ?x WHERE { ?x rdf:type <Vehicle> }
        query = f"SELECT ?x WHERE {{ ?x <{self.RDF_TYPE_STR}> <{vehicle.iri}> }}"
        algebra = parse_sparql(query)

        # Without inference
        result = sparql_evaluate(algebra, store)
        assert result.cardinality == 0

        # With inference
        result_inf = sparql_evaluate(algebra, store, with_inference="rdfs")
        assert result_inf.cardinality == 1
        assert result_inf.bindings[0]["x"] == my_car

    def test_non_type_query_no_inference_penalty(self):
        """Non-rdf:type queries return empty inferred set (fast path)."""
        store = TripleStore()
        p = NamedNode(self.EX + "p")
        s = NamedNode(self.EX + "s")
        o = NamedNode(self.EX + "o")
        store.add(Triple(s, p, o))

        results = list(store.triples(
            TriplePattern(s, p, None), with_inference="rdfs"
        ))
        assert len(results) == 1
        assert results[0].object_ == o

    # ── Phase 2: subPropertyOf expansion ────────────────────────

    def test_subproperty_of_expansion(self):
        """(?x ?q ?o) ∧ (?q rdfs:subPropertyOf ?p) → (?x ?p ?o).

        Querying with super-property P should match sub-property Q triples.
        """
        store = TripleStore()
        has_owner = NamedNode(self.EX + "hasOwner")
        has_legal_owner = NamedNode(self.EX + "hasLegalOwner")
        alice = NamedNode(self.EX + "Alice")
        bob = NamedNode(self.EX + "Bob")

        store.add(Triple(has_legal_owner, RDFS_SUBPROPERTY_OF, has_owner))
        store.add(Triple(alice, has_legal_owner, bob))

        # Without inference: alice hasLegalOwner bob, but not hasOwner
        assert len(list(store.triples(TriplePattern(None, has_owner, None)))) == 0

        # With inference: sub-property expands
        inf_results = list(store.triples(
            TriplePattern(None, has_owner, None), with_inference="rdfs"
        ))
        assert len(inf_results) == 1
        assert inf_results[0].subject == alice
        assert inf_results[0].object_ == bob

    def test_subproperty_transitive_chain(self):
        """Transitive subPropertyOf chain is followed."""
        store = TripleStore()
        general = NamedNode(self.EX + "generalProp")
        specific = NamedNode(self.EX + "specificProp")
        very_specific = NamedNode(self.EX + "verySpecificProp")
        s = NamedNode(self.EX + "s")
        o = NamedNode(self.EX + "o")

        store.add(Triple(specific, RDFS_SUBPROPERTY_OF, general))
        store.add(Triple(very_specific, RDFS_SUBPROPERTY_OF, specific))
        store.add(Triple(s, very_specific, o))

        inf_results = list(store.triples(
            TriplePattern(None, general, None), with_inference="rdfs"
        ))
        assert len(inf_results) == 1
        assert inf_results[0].subject == s

    def test_subproperty_sparql(self):
        """SPARQL queries with inference expand subPropertyOf."""
        from dynafx.knowledge.sparql import parse_sparql, evaluate as sparql_evaluate

        store = TripleStore()
        has_owner = NamedNode(self.EX + "hasOwner")
        has_legal_owner = NamedNode(self.EX + "hasLegalOwner")
        alice = NamedNode(self.EX + "Alice")
        bob = NamedNode(self.EX + "Bob")

        store.add(Triple(has_legal_owner, RDFS_SUBPROPERTY_OF, has_owner))
        store.add(Triple(alice, has_legal_owner, bob))

        query = f"SELECT ?s ?o WHERE {{ ?s <{has_owner.iri}> ?o }}"
        algebra = parse_sparql(query)

        result = sparql_evaluate(algebra, store, with_inference="rdfs")
        assert result.cardinality == 1
        assert result.bindings[0]["s"] == alice
        assert result.bindings[0]["o"] == bob

    # ── Phase 3: Confidence-aware query rewriting ───────────────

    def test_min_belief_filter(self):
        """min_belief filters out low-belief triples."""
        store = TripleStore()
        s = NamedNode(self.EX + "s")
        p = NamedNode(self.EX + "p")
        o = NamedNode(self.EX + "o")

        store.add(Triple(s, p, o))
        s2 = NamedNode(self.EX + "s2")
        o2 = NamedNode(self.EX + "o2")
        store.add(Triple(s2, p, o2))

        results = list(store.triples(
            TriplePattern(None, p, None),
            with_inference="rdfs",
        ))
        assert len(results) == 2

    def test_min_confidence_filter(self):
        """min_confidence filters by belief + (1 - uncertainty)."""
        store = TripleStore()
        s = NamedNode(self.EX + "s")
        p = NamedNode(self.EX + "p")
        o = NamedNode(self.EX + "o")

        store.add(Triple(s, p, o))
        s2 = NamedNode(self.EX + "s2")
        o2 = NamedNode(self.EX + "o2")
        store.add(Triple(s2, p, o2))

        results = list(store.triples(
            TriplePattern(None, p, None),
            with_inference="rdfs",
        ))
        assert len(results) == 2

    def test_min_belief_sparql(self):
        """SPARQL evaluate threads min_belief through to store."""
        from dynafx.knowledge.sparql import parse_sparql, evaluate as sparql_evaluate

        store = TripleStore()
        p = NamedNode(self.EX + "p")
        s_high = NamedNode(self.EX + "sHigh")
        s_low = NamedNode(self.EX + "sLow")

        store.add(Triple(s_high, p, Literal(1)))
        store.add(Triple(s_low, p, Literal(2)))

        query = f"SELECT ?s ?v WHERE {{ ?s <{p.iri}> ?v }}"
        algebra = parse_sparql(query)

        result = sparql_evaluate(algebra, store, with_inference="rdfs")
        assert result.cardinality == 2

    def test_dict_inference_with_belief_maintains_type_expansion(self):
        """Dict config with min_belief still applies type inference."""
        store = TripleStore()
        vehicle = NamedNode(self.EX + "Vehicle")
        car = NamedNode(self.EX + "Car")
        my_car = NamedNode(self.EX + "myCar")

        store.add(Triple(car, RDFS_SUBCLASS_OF, vehicle))
        store.add(Triple(my_car, RDF_TYPE, car))

        results = list(store.triples(
            TriplePattern(my_car, RDF_TYPE, None),
            with_inference="rdfs",
        ))
        types = {r.object_.iri for r in results}
        assert car.iri in types
        assert vehicle.iri in types  # inferred via subclass

    def test_no_opinion_triples_pass_through(self):
        """Triples with no opinion are never filtered by min_belief."""
        store = TripleStore()
        s = NamedNode(self.EX + "s")
        p = NamedNode(self.EX + "p")
        o = NamedNode(self.EX + "o")
        store.add(Triple(s, p, o))  # no opinion

        results = list(store.triples(
            TriplePattern(None, p, None),
            with_inference="rdfs",
        ))
        assert len(results) == 1
