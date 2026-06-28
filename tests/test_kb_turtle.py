"""Tests for kb/turtle.py — Turtle/N-Triples parser and serializer."""

import pytest

from dynafx.kb.model import (
    BlankNode,
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
    xsd,
)
from dynafx.kb.turtle import (
    parse_ntriples,
    parse_turtle,
    serialize_ntriples,
    serialize_turtle,
    tokenize,
)


# ── Tokenizer tests ──────────────────────────────────────────────


class TestTokenizer:
    def test_basic_tokens(self):
        tokens = tokenize("<a> <b> <c> .")
        kinds = [t[0] for t in tokens]
        assert kinds == ["IRI", "IRI", "IRI", "DOT", "EOF"]

    def test_prefix_directive(self):
        tokens = tokenize("@prefix ex: <http://ex.org/> .")
        kinds = [t[0] for t in tokens]
        assert "PREFIX" in kinds
        assert "PNAME_NS" in kinds
        assert "IRI" in kinds

    def test_a_keyword(self):
        tokens = tokenize(":s a :Person .")
        kinds = [t[0] for t in tokens]
        assert "A" in kinds

    def test_literal(self):
        tokens = tokenize('"hello" .')
        kinds = [t[0] for t in tokens]
        assert "STRING" in kinds
        assert tokens[0][1] == "hello"

    def test_langtag(self):
        tokens = tokenize('"hello"@en .')
        kinds = [t[0] for t in tokens]
        assert "LANGTAG" in kinds

    def test_blank_node(self):
        tokens = tokenize("_:b1 :p :o .")
        kinds = [t[0] for t in tokens]
        assert "BLANK_NODE" in kinds

    def test_integer(self):
        tokens = tokenize(":s :p 42 .")
        assert tokens[2][0] == "INTEGER"

    def test_decimal(self):
        tokens = tokenize(":s :p 3.14 .")
        assert tokens[2][0] == "DECIMAL"

    def test_comment(self):
        tokens = tokenize("# comment\n:s :p :o .")
        kinds = [t[0] for t in tokens]
        assert "COMMENT" not in kinds
        assert "PNAME_LN" in kinds


# ── Parse tests ──────────────────────────────────────────────────


class TestParse:
    def test_simple_triple(self):
        turtle = "<http://ex.org/s> <http://ex.org/p> <http://ex.org/o> ."
        store = parse_turtle(turtle)
        assert len(store) == 1

    def test_a_keyword(self):
        turtle = "<http://ex.org/s> a <http://ex.org/Type> ."
        store = parse_turtle(turtle)
        pat = TriplePattern(
            subject=NamedNode("http://ex.org/s"),
            predicate=NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
            object_=NamedNode("http://ex.org/Type"),
        )
        assert pat in store

    def test_prefix(self):
        turtle = "@prefix ex: <http://ex.org/> .\nex:s ex:p ex:o ."
        store = parse_turtle(turtle)
        assert len(store) == 1

    def test_string_literal(self):
        turtle = '<http://ex.org/s> <http://ex.org/p> "hello" .'
        store = parse_turtle(turtle)
        pat = TriplePattern(
            subject=NamedNode("http://ex.org/s"),
            predicate=NamedNode("http://ex.org/p"),
            object_=Literal("hello"),
        )
        assert pat in store

    def test_integer_literal(self):
        turtle = '<http://ex.org/s> <http://ex.org/p> 42 .'
        store = parse_turtle(turtle)
        pat = TriplePattern(object_=Literal(42))
        assert pat in store

    def test_decimal_literal(self):
        turtle = '<http://ex.org/s> <http://ex.org/p> 3.14 .'
        store = parse_turtle(turtle)
        pat = TriplePattern(object_=Literal(3.14))
        assert pat in store

    def test_boolean_literal(self):
        turtle = '<http://ex.org/s> <http://ex.org/p> true .'
        store = parse_turtle(turtle)
        pat = TriplePattern(object_=Literal(True))
        assert pat in store

    def test_lang_tagged_literal(self):
        turtle = '<http://ex.org/s> <http://ex.org/p> "hello"@en .'
        store = parse_turtle(turtle)
        pat = TriplePattern(object_=Literal("hello", lang_tag="en"))
        assert pat in store

    def test_typed_literal(self):
        turtle = '<http://ex.org/s> <http://ex.org/p> "42"^^<http://www.w3.org/2001/XMLSchema#integer> .'
        store = parse_turtle(turtle)
        pat = TriplePattern(
            object_=Literal(42, datatype="http://www.w3.org/2001/XMLSchema#integer")
        )
        assert pat in store

    def test_blank_node(self):
        turtle = '<http://ex.org/s> <http://ex.org/p> _:b1 .'
        store = parse_turtle(turtle)
        pat = TriplePattern(object_=BlankNode(id="_:b1"))
        assert pat in store

    def test_semicolon_grouping(self):
        turtle = """
        @prefix : <http://ex.org/> .
        :s :p1 :o1 ;
            :p2 :o2 .
        """
        store = parse_turtle(turtle)
        assert len(store) == 2

    def test_comma_grouping(self):
        turtle = """
        @prefix : <http://ex.org/> .
        :s :p :o1, :o2 .
        """
        store = parse_turtle(turtle)
        assert len(store) == 2

    def test_multiple_triples(self):
        turtle = """
        @prefix : <http://ex.org/> .
        :s1 :p :o1 .
        :s2 :p :o2 .
        """
        store = parse_turtle(turtle)
        assert len(store) == 2

    def test_base_iri(self):
        turtle = '<s> <p> <o> .'
        store = parse_turtle(turtle, base_iri="http://ex.org/")
        pat = TriplePattern(
            subject=NamedNode("http://ex.org/s"),
            predicate=NamedNode("http://ex.org/p"),
            object_=NamedNode("http://ex.org/o"),
        )
        assert pat in store

    def test_empty_input(self):
        store = parse_turtle("")
        assert len(store) == 0

    def test_default_prefixes(self):
        turtle = "rdf:type rdfs:subClassOf owl:sameAs ."
        store = parse_turtle(turtle)
        assert len(store) == 1

    def test_comment(self):
        turtle = "# This is a comment\n<http://ex.org/s> <http://ex.org/p> <http://ex.org/o> .\n# Another comment"
        store = parse_turtle(turtle)
        assert len(store) == 1


# ── Serializer tests ─────────────────────────────────────────────


class TestSerialize:
    def test_simple_triple(self):
        s = NamedNode("http://ex.org/s")
        p = NamedNode("http://ex.org/p")
        o = NamedNode("http://ex.org/o")
        t = Triple(s, p, o)
        output = serialize_turtle([t])
        assert "<http://ex.org/s>" in output
        assert "<http://ex.org/p>" in output
        assert "<http://ex.org/o>" in output

    def test_prefix_compression(self):
        s = NamedNode("http://ex.org/subject")
        p = NamedNode("http://ex.org/predicate")
        o = NamedNode("http://ex.org/object")
        t = Triple(s, p, o)
        prefixes = {"ex": "http://ex.org/"}
        output = serialize_turtle([t], prefixes=prefixes)
        assert "ex:subject" in output
        assert "ex:predicate" in output
        assert "ex:object" in output
        assert "@prefix" in output

    def test_opinion_comment(self):
        from dynafx.core.models import Opinion
        s = NamedNode("http://ex.org/s")
        p = NamedNode("http://ex.org/p")
        o = NamedNode("http://ex.org/o")
        t = Triple(s, p, o, opinion=Opinion(0.8, 0.1, 0.1))
        output = serialize_turtle([t])
        assert "b=0.80" in output

    def test_serialize_roundtrip(self):
        turtle = '@prefix : <http://ex.org/> .\n:s :p "hello" ;\n  :q 42 .\n'
        store1 = parse_turtle(turtle)
        triples = list(store1.triples(TriplePattern()))
        output = serialize_turtle(triples)
        store2 = parse_turtle(output)
        assert len(store1) == len(store2)
        for t in triples:
            pat = TriplePattern(subject=t.subject, predicate=t.predicate, object_=t.object_)
            assert pat in store2

    def test_mixed_types(self):
        triples = [
            Triple(NamedNode("http://ex.org/s"), NamedNode("http://ex.org/p"), Literal("str")),
            Triple(NamedNode("http://ex.org/s"), NamedNode("http://ex.org/p"), Literal(42)),
            Triple(NamedNode("http://ex.org/s"), NamedNode("http://ex.org/p"), Literal(True)),
            Triple(NamedNode("http://ex.org/s"), NamedNode("http://ex.org/p"), Literal("hello", lang_tag="en")),
            Triple(NamedNode("http://ex.org/s"), NamedNode("http://ex.org/p"), BlankNode(id="_:b1")),
        ]
        output = serialize_turtle(triples)
        for t in triples:
            assert _n3_contains(output, t)

    def test_empty_list(self):
        assert serialize_turtle([]) == ""


# ── N-Triples tests ──────────────────────────────────────────────


class TestNTriples:
    def test_parse(self):
        nt = '<http://ex.org/s> <http://ex.org/p> <http://ex.org/o> .\n'
        store = parse_ntriples(nt)
        assert len(store) == 1

    def test_parse_blank_node(self):
        nt = '<http://ex.org/s> <http://ex.org/p> _:b1 .\n'
        store = parse_ntriples(nt)
        assert len(store) == 1

    def test_parse_literal(self):
        nt = '<http://ex.org/s> <http://ex.org/p> "hello" .\n'
        store = parse_ntriples(nt)
        assert len(store) == 1

    def test_parse_literal_with_lang(self):
        nt = '<http://ex.org/s> <http://ex.org/p> "hello"@en .\n'
        store = parse_ntriples(nt)
        assert len(store) == 1

    def test_parse_literal_with_datatype(self):
        nt = '<http://ex.org/s> <http://ex.org/p> "42"^^<http://www.w3.org/2001/XMLSchema#integer> .\n'
        store = parse_ntriples(nt)
        assert len(store) == 1

    def test_serialize(self):
        t = Triple(
            NamedNode("http://ex.org/s"),
            NamedNode("http://ex.org/p"),
            NamedNode("http://ex.org/o"),
        )
        output = serialize_ntriples([t])
        assert "<http://ex.org/s>" in output
        assert "<http://ex.org/p>" in output
        assert "<http://ex.org/o>" in output

    def test_serialize_opinion(self):
        from dynafx.core.models import Opinion
        t = Triple(
            NamedNode("http://ex.org/s"),
            NamedNode("http://ex.org/p"),
            NamedNode("http://ex.org/o"),
            opinion=Opinion(0.9, 0.05, 0.05),
        )
        output = serialize_ntriples([t])
        assert "b=0.90" in output

    def test_serialize_roundtrip(self):
        t = Triple(
            NamedNode("http://ex.org/s"),
            NamedNode("http://ex.org/p"),
            Literal("hello", lang_tag="en"),
        )
        output = serialize_ntriples([t])
        store = parse_ntriples(output)
        assert len(store) == 1
        pat = TriplePattern(subject=t.subject, predicate=t.predicate, object_=t.object_)
        assert pat in store


# ── Helpers ──────────────────────────────────────────────────────


def _n3_contains(output: str, triple: Triple) -> bool:
    """Check if triple appears in Turtle output."""
    s_str = _node_in(triple.subject)
    p_str = _node_in(triple.predicate)
    o_str = _node_in(triple.object_)
    if triple.opinion:
        o_str += "  # b="
    return s_str in output and p_str in output


def _node_in(node) -> str:
    if isinstance(node, NamedNode):
        return node.iri
    if isinstance(node, BlankNode):
        return node.id
    if isinstance(node, Literal):
        return str(node.value)
    return str(node)
