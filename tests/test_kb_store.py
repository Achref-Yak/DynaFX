"""Tests for kb/store.py — TripleStore."""

import pytest

from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
)
from dynafx.knowledge.store import TripleStore


# ── Fixtures ─────────────────────────────────────────────────────

S = NamedNode("http://example.org/s")
P = NamedNode("http://example.org/p")
O = NamedNode("http://example.org/o")
S2 = NamedNode("http://example.org/s2")
P2 = NamedNode("http://example.org/p2")
O2 = NamedNode("http://example.org/o2")
O_LIT = Literal("obj")
O_LIT2 = Literal("obj2")


@pytest.fixture
def empty_store():
    return TripleStore()


@pytest.fixture
def single_store():
    store = TripleStore()
    store.add(Triple(S, P, O))
    return store


@pytest.fixture
def populated_store():
    store = TripleStore()
    triples = [
        Triple(S, P, O),
        Triple(S, P, O2),
        Triple(S, P2, O),
        Triple(S2, P, O),
        Triple(S2, P2, O2),
        Triple(O_LIT, P, O2),  # literal as subject
    ]
    for t in triples:
        store.add(t)
    return store, triples


# ── Tests ────────────────────────────────────────────────────────


class TestAddAndTriples:
    def test_add_single(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t)
        result = list(empty_store.triples(TriplePattern(subject=S, predicate=P, object_=O)))
        assert len(result) == 1
        assert result[0] == t

    def test_add_three_bound(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t)
        pat = TriplePattern(subject=S, predicate=P, object_=O)
        assert list(empty_store.triples(pat)) == [t]

    def test_add_sp_bound(self, empty_store):
        t1 = Triple(S, P, O)
        t2 = Triple(S, P, O2)
        empty_store.add(t1)
        empty_store.add(t2)
        pat = TriplePattern(subject=S, predicate=P)
        results = list(empty_store.triples(pat))
        assert len(results) == 2
        assert t1 in results
        assert t2 in results

    def test_add_po_bound(self, populated_store):
        store, triples = populated_store
        pat = TriplePattern(predicate=P, object_=O)
        results = list(store.triples(pat))
        assert len(results) == 2

    def test_add_so_bound(self, populated_store):
        store, triples = populated_store
        pat = TriplePattern(subject=S, object_=O)
        results = list(store.triples(pat))
        assert len(results) == 2  # S-P-O, S-P2-O

    def test_add_s_bound(self, populated_store):
        store, triples = populated_store
        pat = TriplePattern(subject=S)
        results = list(store.triples(pat))
        assert len(results) == 3

    def test_add_p_bound(self, populated_store):
        store, triples = populated_store
        pat = TriplePattern(predicate=P)
        results = list(store.triples(pat))
        assert len(results) == 4  # S-P-O, S-P-O2, S2-P-O, O_LIT-P-O2

    def test_add_o_bound(self, populated_store):
        store, triples = populated_store
        pat = TriplePattern(object_=O)
        results = list(store.triples(pat))
        assert len(results) == 3  # S-P-O, S-P2-O, S2-P-O

    def test_add_all_none(self, populated_store):
        store, triples = populated_store
        pat = TriplePattern()
        results = list(store.triples(pat))
        assert len(results) == 6

    def test_add_no_match(self, empty_store):
        pat = TriplePattern(subject=S, predicate=P, object_=O)
        assert list(empty_store.triples(pat)) == []


class TestRemove:
    def test_remove_by_spo(self, single_store):
        pat = TriplePattern(subject=S, predicate=P, object_=O)
        count = single_store.remove(pat)
        assert count == 1
        assert len(single_store) == 0

    def test_remove_by_sp(self, populated_store):
        store, triples = populated_store
        pat = TriplePattern(subject=S, predicate=P)
        count = store.remove(pat)
        assert count == 2  # S-P-O, S-P-O2
        assert len(store) == 4  # 6 - 2

    def test_remove_by_s(self, populated_store):
        store, triples = populated_store
        pat = TriplePattern(subject=S)
        count = store.remove(pat)
        assert count == 3  # all S triples
        assert len(store) == 3

    def test_remove_nonexistent(self, empty_store):
        pat = TriplePattern(subject=S, predicate=P, object_=O)
        count = empty_store.remove(pat)
        assert count == 0

    def test_remove_then_add(self, single_store):
        single_store.remove(TriplePattern(subject=S, predicate=P, object_=O))
        t = Triple(S, P, O)
        single_store.add(t)
        assert len(single_store) == 1


class TestContains:
    def test_contains_true(self, single_store):
        pat = TriplePattern(subject=S, predicate=P, object_=O)
        assert pat in single_store

    def test_contains_false(self, single_store):
        pat = TriplePattern(subject=S2, predicate=P2, object_=O2)
        assert pat not in single_store

    def test_contains_empty(self, empty_store):
        pat = TriplePattern(subject=S, predicate=P, object_=O)
        assert pat not in empty_store


class TestLen:
    def test_len_empty(self, empty_store):
        assert len(empty_store) == 0

    def test_len_after_add(self, empty_store):
        empty_store.add(Triple(S, P, O))
        assert len(empty_store) == 1

    def test_len_after_remove(self, single_store):
        single_store.remove(TriplePattern(subject=S, predicate=P, object_=O))
        assert len(single_store) == 0

    def test_len_after_duplicate_spo(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t)
        empty_store.add(t)
        assert len(empty_store) == 1  # idempotent


class TestDedup:
    def test_duplicate_add_idempotent(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t)
        empty_store.add(t)
        stored = list(empty_store.triples(TriplePattern(subject=S, predicate=P, object_=O)))[0]
        assert stored == t

    def test_dedup_keeps_original(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t)
        empty_store.add(t)
        stored = list(empty_store.triples(TriplePattern(subject=S, predicate=P, object_=O)))[0]
        assert stored == t

    def test_dedup_no_opinion(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t)
        empty_store.add(t)  # same triple, no opinion
        assert len(empty_store) == 1


class TestNamedGraphs:
    def test_default_graph(self, single_store):
        assert "default" in single_store.graphs()

    def test_named_graph_isolation(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t, graph="graph_a")
        pat = TriplePattern(subject=S, predicate=P, object_=O)
        assert pat in empty_store  # exists in store
        assert len(list(empty_store.triples(pat, graph="graph_b"))) == 0
        assert len(list(empty_store.triples(pat, graph="graph_a"))) == 1

    def test_triple_in_multiple_graphs(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t, graph="g1")
        empty_store.add(t, graph="g2")
        assert len(list(empty_store.triples_in_graph("g1"))) == 1
        assert len(list(empty_store.triples_in_graph("g2"))) == 1
        assert len(empty_store) == 1  # same triple, once in store

    def test_remove_graph(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t, graph="g1")
        empty_store.add(Triple(S2, P2, O2), graph="g2")
        count = empty_store.remove_graph("g1")
        assert count == 1
        assert len(list(empty_store.triples_in_graph("g1"))) == 0
        assert len(list(empty_store.triples_in_graph("g2"))) == 1
        assert len(empty_store) == 1  # S2-P2-O2 still exists via g2

    def test_remove_graph_preserves_shared(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t, graph="g1")
        empty_store.add(t, graph="g2")
        empty_store.remove_graph("g1")
        assert len(empty_store) == 1  # still exists via g2

    def test_copy_graph(self, empty_store):
        t = Triple(S, P, O)
        empty_store.add(t, graph="src")
        empty_store.copy_graph("src", "dst")
        assert len(list(empty_store.triples_in_graph("dst"))) == 1

    def test_graphs_list(self, empty_store):
        assert empty_store.graphs() == []
        empty_store.add(Triple(S, P, O), graph="g1")
        assert "g1" in empty_store.graphs()

    def test_triples_in_graph_filter(self, empty_store):
        t1 = Triple(S, P, O)
        t2 = Triple(S2, P2, O2)
        empty_store.add(t1, graph="g1")
        empty_store.add(t2, graph="g2")
        result = list(empty_store.triples(TriplePattern(), graph="g1"))
        assert len(result) == 1
        assert result[0] == t1


class TestEdgeCases:
    def test_empty_store_operations(self, empty_store):
        assert len(empty_store) == 0
        assert empty_store.graphs() == []
        assert not (TriplePattern() in empty_store)
        assert list(empty_store.triples(TriplePattern())) == []

    def test_remove_nonexistent_graph(self, empty_store):
        assert empty_store.remove_graph("nonexistent") == 0

    def test_copy_nonexistent_graph(self, empty_store):
        empty_store.copy_graph("nonexistent", "dst")
        assert "dst" not in empty_store.graphs()

    def test_pattern_with_all_none(self, empty_store):
        assert list(empty_store.triples(TriplePattern())) == []

    def test_mixed_node_types(self, empty_store):
        s = NamedNode("http://example.org/s")
        p = NamedNode("http://example.org/p")
        o_lit = Literal("hello", lang_tag="en")
        t = Triple(s, p, o_lit)
        empty_store.add(t)
        assert len(empty_store) == 1
        result = list(empty_store.triples(TriplePattern(subject=s, predicate=p, object_=o_lit)))
        assert len(result) == 1

    def test_literals_as_subject(self):
        """RDF allows literals only as objects, but our store is flexible."""
        store = TripleStore()
        s = Literal("literal_subject")
        p = NamedNode("http://example.org/p")
        o = NamedNode("http://example.org/o")
        store.add(Triple(s, p, o))
        assert len(store) == 1
        assert TriplePattern(subject=s) in store

    def test_remove_only_matching(self, populated_store):
        store, triples = populated_store
        store.remove(TriplePattern(subject=S, predicate=P))
        remaining = list(store.triples(TriplePattern()))
        assert len(remaining) == 4
        # Verify S-P-* are gone
        sp_remain = list(store.triples(TriplePattern(subject=S, predicate=P)))
        assert len(sp_remain) == 0
