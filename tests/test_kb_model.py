"""Tests for kb/model.py — RDF data model."""

import pytest

from dynafx.core.models import Opinion
from dynafx.kb.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
    xsd,
)


class TestNamedNode:
    def test_construction(self):
        n = NamedNode("http://example.org/test")
        assert n.iri == "http://example.org/test"

    def test_equality(self):
        a = NamedNode("http://example.org/a")
        b = NamedNode("http://example.org/a")
        c = NamedNode("http://example.org/c")
        assert a == b
        assert a != c

    def test_hash(self):
        a = NamedNode("http://example.org/a")
        b = NamedNode("http://example.org/a")
        assert hash(a) == hash(b)

    def test_n3(self):
        n = NamedNode("http://example.org/test")
        assert n.n3() == "<http://example.org/test>"


class TestBlankNode:
    def test_explicit_id(self):
        bn = BlankNode(id="_:b1")
        assert bn.id == "_:b1"

    def test_auto_id(self):
        bn = BlankNode()
        assert bn.id.startswith("_:b")
        assert len(bn.id) > 3

    def test_unique_ids(self):
        a = BlankNode()
        b = BlankNode()
        assert a.id != b.id

    def test_equality(self):
        a = BlankNode(id="_:x")
        b = BlankNode(id="_:x")
        c = BlankNode(id="_:y")
        assert a == b
        assert a != c

    def test_hash(self):
        a = BlankNode(id="_:x")
        b = BlankNode(id="_:x")
        assert hash(a) == hash(b)

    def test_n3(self):
        bn = BlankNode(id="_:b1")
        assert bn.n3() == "_:b1"

    def test_is_rdfnode(self):
        assert isinstance(BlankNode(), RDFNode)


class TestLiteral:
    def test_string(self):
        lit = Literal("hello")
        assert lit.value == "hello"
        assert lit.datatype is None
        assert lit.lang_tag is None

    def test_integer(self):
        lit = Literal(42)
        assert lit.value == 42

    def test_float(self):
        lit = Literal(3.14)
        assert lit.value == 3.14

    def test_bool(self):
        lit = Literal(True)
        assert lit.value is True

    def test_datatype(self):
        lit = Literal(42, datatype=xsd("integer").iri)
        assert lit.datatype == xsd("integer").iri

    def test_lang_tag(self):
        lit = Literal("hello", lang_tag="en")
        assert lit.lang_tag == "en"

    def test_equality(self):
        a = Literal("hello")
        b = Literal("hello")
        c = Literal("world")
        assert a == b
        assert a != c

    def test_hash(self):
        a = Literal("hello")
        b = Literal("hello")
        assert hash(a) == hash(b)

    def test_n3_plain_string(self):
        lit = Literal("hello")
        assert lit.n3() == '"hello"'

    def test_n3_escaped_string(self):
        lit = Literal('hello "world"')
        assert lit.n3() == '"hello \\"world\\""'

    def test_n3_integer(self):
        lit = Literal(42)
        assert lit.n3() == "42"

    def test_n3_float(self):
        lit = Literal(3.14)
        assert lit.n3() == "3.14"

    def test_n3_boolean(self):
        lit = Literal(True)
        assert "true" in lit.n3()
        assert "boolean" in lit.n3()

    def test_n3_lang_tag(self):
        lit = Literal("hello", lang_tag="en")
        assert lit.n3() == '"hello"@en'

    def test_n3_datatype(self):
        lit = Literal(42, datatype=xsd("integer").iri)
        assert lit.n3() == '"42"^^<http://www.w3.org/2001/XMLSchema#integer>'

    def test_is_rdfnode(self):
        assert isinstance(Literal("x"), RDFNode)


class TestTriple:
    def test_construction(self):
        s = NamedNode("http://example.org/s")
        p = NamedNode("http://example.org/p")
        o = NamedNode("http://example.org/o")
        t = Triple(s, p, o)
        assert t.subject == s
        assert t.predicate == p
        assert t.object_ == o
        assert t.opinion is None

    def test_with_opinion(self):
        s = NamedNode("http://example.org/s")
        p = NamedNode("http://example.org/p")
        o = NamedNode("http://example.org/o")
        t = Triple(s, p, o)
        op = Opinion(0.8, 0.1, 0.1)
        t2 = t.with_opinion(op)
        assert t2.opinion == op
        assert t2.subject == s
        assert t2.predicate == p
        assert t2.object_ == o

    def test_equality_ignores_opinion(self):
        s = NamedNode("http://example.org/s")
        p = NamedNode("http://example.org/p")
        o = NamedNode("http://example.org/o")
        t1 = Triple(s, p, o, opinion=Opinion(0.8, 0.1, 0.1))
        t2 = Triple(s, p, o, opinion=Opinion(0.5, 0.3, 0.2))
        assert t1 == t2

    def test_inequality_different_subject(self):
        t1 = Triple(NamedNode("http://example.org/a"),
                     NamedNode("http://example.org/p"),
                     NamedNode("http://example.org/o"))
        t2 = Triple(NamedNode("http://example.org/b"),
                     NamedNode("http://example.org/p"),
                     NamedNode("http://example.org/o"))
        assert t1 != t2

    def test_spo_property(self):
        s = NamedNode("http://example.org/s")
        p = NamedNode("http://example.org/p")
        o = Literal(42)
        t = Triple(s, p, o)
        assert t.spo == (s, p, o)

    def test_hash_based_on_spo(self):
        s = NamedNode("http://example.org/s")
        p = NamedNode("http://example.org/p")
        o = Literal("x")
        t1 = Triple(s, p, o)
        t2 = Triple(s, p, o)
        assert hash(t1) == hash(t2)

    def test_frozen(self):
        s = NamedNode("http://example.org/s")
        p = NamedNode("http://example.org/p")
        o = NamedNode("http://example.org/o")
        t = Triple(s, p, o)
        with pytest.raises(AttributeError):
            t.subject = NamedNode("http://example.org/other")  # type: ignore


class TestTriplePattern:
    def test_all_none(self):
        p = TriplePattern()
        assert p.subject is None
        assert p.predicate is None
        assert p.object_ is None

    def test_partial_bound(self):
        s = NamedNode("http://example.org/s")
        p = TriplePattern(subject=s, predicate=None, object_=None)
        assert p.subject == s
        assert p.predicate is None
        assert p.object_ is None

    def test_fully_bound(self):
        s = NamedNode("http://example.org/s")
        p = NamedNode("http://example.org/p")
        o = Literal("o")
        pattern = TriplePattern(subject=s, predicate=p, object_=o)
        assert pattern.subject == s
        assert pattern.predicate == p
        assert pattern.object_ == o


class TestXsd:
    def test_xsd_integer(self):
        ns = xsd("integer")
        assert ns.iri == "http://www.w3.org/2001/XMLSchema#integer"

    def test_xsd_string(self):
        ns = xsd("string")
        assert ns.iri == "http://www.w3.org/2001/XMLSchema#string"

    def test_xsd_is_named_node(self):
        assert isinstance(xsd("integer"), NamedNode)

    def test_common_constants(self):
        from dynafx.kb.model import XSD_INTEGER, XSD_STRING
        assert XSD_INTEGER == xsd("integer")
        assert XSD_STRING == xsd("string")
