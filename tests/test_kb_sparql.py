"""SPARQL query tests — 30 tests covering SELECT, ASK, CONSTRUCT,
FILTER, OPTIONAL, UNION, DISTINCT, ORDER BY, LIMIT, OFFSET."""

from __future__ import annotations

import pytest

from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
)
from dynafx.knowledge.store import TripleStore
from dynafx.knowledge.sparql import (
    parse_sparql,
    evaluate,
    QueryResult,
    Variable,
    SPARQLTriplePattern,
    BGP,
    Project,
    Ask,
    Construct,
    Filter,
    Optional_,
    Union,
    OrderBy,
    Slice,
    Comparison,
    And,
    Or,
    Not,
    VarRef,
    Constant,
    RegexFunc,
    BoundFunc,
)
from dynafx.core.models import Opinion


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def store():
    s = TripleStore()
    ex = NamedNode("http://ex.org/")
    s.add(Triple(ex, NamedNode("http://ex.org/p"), Literal("hello")))
    s.add(Triple(ex, NamedNode("http://ex.org/q"), Literal(42)))
    s.add(Triple(ex, NamedNode("http://ex.org/r"), Literal(True)))
    s.add(Triple(NamedNode("http://ex.org/s2"), NamedNode("http://ex.org/p"), Literal("world")))
    s.add(Triple(NamedNode("http://ex.org/s2"), NamedNode("http://ex.org/q"), Literal(99)))
    s.add(Triple(NamedNode("http://ex.org/s3"), NamedNode("http://ex.org/p"), Literal(3.14)))
    return s


@pytest.fixture
def people_store():
    s = TripleStore()
    ex = "http://ex.org/"
    alice = NamedNode(f"{ex}alice")
    bob = NamedNode(f"{ex}bob")
    charlie = NamedNode(f"{ex}charlie")
    name = NamedNode(f"{ex}name")
    age = NamedNode(f"{ex}age")
    knows = NamedNode(f"{ex}knows")
    s.add(Triple(alice, name, Literal("Alice")))
    s.add(Triple(alice, age, Literal(30)))
    s.add(Triple(alice, knows, bob))
    s.add(Triple(bob, name, Literal("Bob")))
    s.add(Triple(bob, age, Literal(25)))
    s.add(Triple(bob, knows, charlie))
    s.add(Triple(charlie, name, Literal("Charlie")))
    s.add(Triple(charlie, age, Literal(35)))
    s.add(Triple(charlie, knows, alice))
    return s


# ── 1. SELECT with BGP → bindings ───────────────────────────────


def test_select_bgp(store):
    query = "SELECT ?x ?y WHERE { ?x <http://ex.org/p> ?y . }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store)
    assert len(result.bindings) == 3
    assert "x" in result.bindings[0]
    assert "y" in result.bindings[0]
    assert all(b["x"] is not None for b in result.bindings)


# ── 2. SELECT with no matches → empty ────────────────────────────


def test_select_no_matches(store):
    query = "SELECT ?x WHERE { ?x <http://ex.org/nonexistent> ?y . }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store)
    assert len(result.bindings) == 0
    assert result.cardinality == 0


# ── 3. ASK true ──────────────────────────────────────────────────


def test_ask_true(store):
    query = "ASK WHERE { ?x <http://ex.org/p> ?y . }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store)
    assert result.cardinality == 1


# ── 4. ASK false ─────────────────────────────────────────────────


def test_ask_false(store):
    query = "ASK WHERE { ?x <http://ex.org/nope> ?y . }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store)
    assert result.cardinality == 0


# ── 5. FILTER numeric comparison ─────────────────────────────────


def test_filter_numeric(store):
    query = "SELECT ?x ?y WHERE { ?x <http://ex.org/q> ?y . FILTER(?y > 30) }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=store)
    assert len(result.bindings) == 2  # 42 and 99 are both > 30
    assert all(b["y"].value > 30 for b in result.bindings)


def test_filter_eq(people_store):
    query = "SELECT ?x WHERE { ?x <http://ex.org/age> ?age . FILTER(?age = 25) }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 1
    assert result.bindings[0]["x"] == NamedNode("http://ex.org/bob")


def test_filter_ne(people_store):
    query = "SELECT ?x WHERE { ?x <http://ex.org/age> ?age . FILTER(?age != 25) }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 2


def test_filter_le(people_store):
    query = "SELECT ?x WHERE { ?x <http://ex.org/age> ?age . FILTER(?age <= 25) }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 1


# ── 6. FILTER &&, ||, ! ─────────────────────────────────────────


def test_filter_and_or(people_store):
    query = "SELECT ?x ?age WHERE { ?x <http://ex.org/age> ?age . FILTER(?age > 20 && ?age < 40) }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3


def test_filter_or(people_store):
    query = "SELECT ?x WHERE { ?x <http://ex.org/age> ?age . FILTER(?age = 25 || ?age = 35) }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 2


def test_filter_not(people_store):
    query = "SELECT ?x WHERE { ?x <http://ex.org/age> ?age . FILTER(!(?age = 25)) }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 2


# ── 7. FILTER REGEX ──────────────────────────────────────────────


def test_filter_regex(people_store):
    query = 'SELECT ?x WHERE { ?x <http://ex.org/name> ?name . FILTER regex(?name, "lice") }'
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 1
    assert result.bindings[0]["x"] == NamedNode("http://ex.org/alice")


# ── 8. OPTIONAL (match) ──────────────────────────────────────────


def test_optional_match(people_store):
    query = """
    SELECT ?name ?age WHERE {
        ?x <http://ex.org/name> ?name .
        OPTIONAL { ?x <http://ex.org/age> ?age . }
    }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3
    for b in result.bindings:
        assert "age" in b  # all people have age


# ── 9. OPTIONAL (no match → unbound) ─────────────────────────────


def test_optional_no_match(people_store):
    query = """
    SELECT ?name ?w WHERE {
        ?x <http://ex.org/name> ?name .
        OPTIONAL { ?x <http://ex.org/unknown> ?w . }
    }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3
    for b in result.bindings:
        assert "w" not in b or b["w"] is None


# ── 10. UNION ────────────────────────────────────────────────────


def test_union(store):
    query = """
    SELECT ?x WHERE {
        { ?x <http://ex.org/p> ?y . }
        UNION
        { ?x <http://ex.org/q> ?z . }
    }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=store)
    assert len(result.bindings) == 5  # 3 p-values + 2 q-values (but s3 has p only)


# ── 11. DISTINCT ─────────────────────────────────────────────────


def test_distinct(store):
    query = "SELECT DISTINCT ?x WHERE { ?x <http://ex.org/p> ?y . }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=store)
    # 3 subjects with p
    assert len(result.bindings) == 3


# ── 12. ORDER BY ASC ─────────────────────────────────────────────


def test_order_by_asc(people_store):
    query = """
    SELECT ?name ?age WHERE {
        ?x <http://ex.org/name> ?name .
        ?x <http://ex.org/age> ?age .
    } ORDER BY ?age
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3
    ages = [b["age"].value for b in result.bindings]
    assert ages == [25, 30, 35]


# ── 13. ORDER BY DESC ────────────────────────────────────────────


def test_order_by_desc(people_store):
    query = """
    SELECT ?name ?age WHERE {
        ?x <http://ex.org/name> ?name .
        ?x <http://ex.org/age> ?age .
    } ORDER BY DESC(?age)
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3
    ages = [b["age"].value for b in result.bindings]
    assert ages == [35, 30, 25]


# ── 14. LIMIT ────────────────────────────────────────────────────


def test_limit(people_store):
    query = """
    SELECT ?name ?age WHERE {
        ?x <http://ex.org/name> ?name .
        ?x <http://ex.org/age> ?age .
    } ORDER BY ?age LIMIT 2
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 2
    ages = [b["age"].value for b in result.bindings]
    assert ages == [25, 30]


# ── 15. OFFSET ───────────────────────────────────────────────────


def test_offset(people_store):
    query = """
    SELECT ?name ?age WHERE {
        ?x <http://ex.org/name> ?name .
        ?x <http://ex.org/age> ?age .
    } ORDER BY ?age OFFSET 1
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 2
    ages = [b["age"].value for b in result.bindings]
    assert ages == [30, 35]


# ── 16. CONSTRUCT → new triples in store ─────────────────────────


def test_construct(people_store):
    query = """
    PREFIX ex: <http://ex.org/>
    CONSTRUCT { ?x ex:hasName ?name . }
    WHERE { ?x ex:name ?name . }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert result.cardinality == 3


# ── 17. Variable prefix (?x, $x) ─────────────────────────────────


def test_dollar_variable(store):
    query = "SELECT $x $y WHERE { $x <http://ex.org/p> $y . }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=store)
    assert len(result.bindings) == 3
    assert "x" in result.bindings[0]
    assert "y" in result.bindings[0]


# ── 18. Prefix resolution in query ───────────────────────────────


def test_prefix_resolution(people_store):
    query = """
    PREFIX ex: <http://ex.org/>
    SELECT ?name WHERE { ?x ex:name ?name . }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3


# ── 19. BGP joins across patterns ────────────────────────────────


def test_bgp_join(people_store):
    query = """
    SELECT ?name ?age WHERE {
        ?x <http://ex.org/name> ?name .
        ?x <http://ex.org/age> ?age .
    }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3
    for b in result.bindings:
        assert "name" in b
        assert "age" in b


# ── 20. Multiple FILTERs ─────────────────────────────────────────


def test_multiple_filters(people_store):
    query = """
    SELECT ?name ?age WHERE {
        ?x <http://ex.org/name> ?name .
        ?x <http://ex.org/age> ?age .
        FILTER(?age > 20)
        FILTER(?age < 40)
    }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3


# ── 21. Nested OPTIONAL + FILTER ─────────────────────────────────


def test_nested_optional_filter(people_store):
    query = """
    SELECT ?name ?age WHERE {
        ?x <http://ex.org/name> ?name .
        OPTIONAL {
            ?x <http://ex.org/age> ?age .
            FILTER(?age > 30)
        }
    }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3
    matched_age = [b for b in result.bindings if "age" in b and b["age"] is not None]
    assert len(matched_age) == 1  # only charlie (age 35) matches


# ── 22. Empty query → error ──────────────────────────────────────


def test_empty_query():
    with pytest.raises(SyntaxError):
        parse_sparql("")


# ── 23. Syntax error → informative message ───────────────────────


def test_syntax_error():
    with pytest.raises(SyntaxError) as exc:
        parse_sparql("SELECT ?x WHERE { ?x @bad . }")
    assert "position" in str(exc.value) or "Unexpected" in str(exc.value)


# ── 24. Unbound variable in filter → no error ────────────────────


def test_unbound_in_filter(store):
    # ?unbound is not bound in the BGP, filter should not error
    query = "SELECT ?x WHERE { ?x <http://ex.org/p> ?y . FILTER(?unbound = 42) }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=store)
    # The filter with unbound variable should evaluate to false
    # so no bindings should be returned
    assert len(result.bindings) == 0


# ── 25. QueryResult.opinions populated ───────────────────────────


def test_opinions_populated(store):
    query = "SELECT ?x ?y WHERE { ?x <http://ex.org/p> ?y . }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=store)
    assert len(result.opinions) == len(result.bindings)
    for opin in result.opinions:
        assert isinstance(opin, dict)


# ── 26. SELECT with no WHERE clause ──────────────────────────────


def test_select_simple(store):
    query = "SELECT ?x ?y WHERE { ?x <http://ex.org/p> ?y . }"
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=store)
    assert len(result.bindings) == 3


# ── 27. Complex 3-pattern BGP ────────────────────────────────────


def test_three_pattern_bgp(people_store):
    query = """
    SELECT ?name ?age ?friend WHERE {
        ?x <http://ex.org/name> ?name .
        ?x <http://ex.org/age> ?age .
        ?x <http://ex.org/knows> ?friend .
    }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 3
    for b in result.bindings:
        assert "name" in b
        assert "age" in b
        assert "friend" in b


# ── 28. ORDER BY + LIMIT combination ─────────────────────────────


def test_order_by_limit(people_store):
    query = """
    SELECT ?name ?age WHERE {
        ?x <http://ex.org/name> ?name .
        ?x <http://ex.org/age> ?age .
    } ORDER BY DESC(?age) LIMIT 2
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert len(result.bindings) == 2
    ages = [b["age"].value for b in result.bindings]
    assert ages == [35, 30]


# ── 29. CONSTRUCT with multiple templates ────────────────────────


def test_construct_multiple(people_store):
    query = """
    PREFIX ex: <http://ex.org/>
    CONSTRUCT {
        ?x ex:hasName ?name .
        ?x ex:hasAge ?age .
    }
    WHERE {
        ?x ex:name ?name .
        ?x ex:age ?age .
    }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    assert result.cardinality == 6  # 3 people × 2 templates each


# ── 30. UNION inside OPTIONAL (complex) ──────────────────────────


def test_union_inside_optional(people_store):
    query = """
    SELECT ?name ?info WHERE {
        ?x <http://ex.org/name> ?name .
        OPTIONAL {
            { ?x <http://ex.org/age> ?info . }
            UNION
            { ?x <http://ex.org/knows> ?info . }
        }
    }
    """
    algebra = parse_sparql(query)
    result = evaluate(algebra, store=people_store)
    # Each person has at least age, plus knows
    assert len(result.bindings) >= 3


# ── 31. DECIMAL tokenization (regression) ──────────────────────


def test_decimal_tokenization():
    """FILTER with DECIMAL numbers must tokenize correctly (0.4 not INTEGER+WS+DOT)."""
    query = 'SELECT ?x WHERE { ?x <http://ex.org/v> ?y . FILTER(?y >= 0.4) }'
    algebra = parse_sparql(query)
    assert algebra is not None


def test_decimal_edge_cases():
    """DECIMAL patterns with trailing dot must not match (0.x should fail)."""
    from dynafx.knowledge.sparql import tokenize
    with pytest.raises(SyntaxError):
        tokenize("0.x")
    # Valid DECIMAL numbers
    tokens = tokenize("0.5")
    assert any(t[0] == "DECIMAL" for t in tokens)
    tokens = tokenize("1.0")
    assert any(t[0] == "DECIMAL" for t in tokens)
    # Integer + dot = two tokens, not one DECIMAL
    tokens = tokenize("0.")
    assert not any(t[0] == "DECIMAL" for t in tokens)
    assert any(t[0] == "INTEGER" for t in tokens)
    assert any(t[0] == "DOT" for t in tokens)
