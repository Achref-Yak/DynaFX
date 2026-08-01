# Tutorial 5 — Knowledge Graphs

DynaFX's knowledge layer is an **RDF triple store**: facts are `(subject,
predicate, object)` triples, organized into named graphs, with Turtle I/O,
SPARQL querying, and forward-chaining inference.

## Triples

```python
from dynafx.knowledge import (
    TripleStore, NamedNode, Literal, Triple, TriplePattern,
)

store = TripleStore()
store.add(Triple(
    NamedNode("http://ex.org/alice"),
    NamedNode("http://ex.org/age"),
    Literal(30),
), graph="people")
```

- **Subject/predicate** are IRIs (`NamedNode`).
- **Object** is either an IRI (a link) or a `Literal` (a value).
- Triples are tagged with a **named graph** (here `"people"`).

## Reading triples back

Match with a pattern; `None` is a wildcard:

```python
pattern = TriplePattern(
    subject=NamedNode("http://ex.org/alice"),
    predicate=NamedNode("http://ex.org/age"),
    object_=None,
)
for t in store.triples(pattern, graph="people"):
    print(t.object_.value)   # 30
```

## Named graphs

Named graphs let you separate *sources* or *layers* of knowledge. In the
flagship twin, suppliers, projects, logistics, workforce, meta-ontology, and
simulation evidence each live in their own graph.

```python
store.add(Triple(
    NamedNode("http://ex.org/bob"),
    NamedNode("http://ex.org/age"),
    Literal(25),
), graph="people")

print(store.graphs())                       # ['people']
print(len(list(store.triples_in_graph("people"))))   # 2
```

## Loading Turtle

Turtle is the on-disk format. Parse a string or a file into a store:

```python
from dynafx.knowledge import parse_turtle

turtle = """
@prefix ex: <http://ex.org/> .
ex:alice a ex:Person .
ex:alice ex:age 30 .
ex:alice ex:knows ex:bob .
"""
store = parse_turtle(turtle)
print(len(list(store.all_triples())))    # 3
```

`parse_turtle(text, base_iri=...)` and `serialize_turtle(triples, prefixes=...)`
round-trip between stores and text.

## RDFS inference

The forward-chaining engine derives implicit triples. The built-in `rdfs_rules()`
derive types from domain/range declarations:

```python
from dynafx.knowledge import RuleEngine, rdfs_rules

turtle = """
@prefix ex: <http://ex.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
ex:hasAge a rdf:Property ; rdfs:domain ex:Person .
ex:alice ex:hasAge 30 .
"""
store = parse_turtle(turtle)

engine = RuleEngine(rdfs_rules())
added = engine.apply(store)
print(added)   # 1 — ex:alice rdf:type ex:Person inferred
```

## The enterprise CSV pipeline

Real data flows in via YAML-declared CSV mappings (`data/mappings/*.yaml`)
through `ingest_csv`. Each column maps to a predicate, and the result lands in
a target graph:

```python
from dynafx.knowledge import ingest_csv, MappingDef, ColumnMapping

mapping = MappingDef(
    csv="suppliers.csv",
    target_graph="http://ex.org/suppliers",
    entity_class="http://ex.org/Supplier",
    id_column="id",
    id_prefix="http://ex.org/",
    columns={
        "name": ColumnMapping(predicate="http://ex.org/name", col_type="string"),
        "reliability": ColumnMapping(predicate="http://ex.org/reliability", col_type="float"),
    },
)

import tempfile, os
path = os.path.join(tempfile.gettempdir(), "suppliers.csv")
with open(path, "w") as f:
    f.write("id,name,reliability\ns1,Acme,0.9\ns2,Beta,0.75\n")

report = ingest_csv(mapping, path, store, strict=True)
print(report.rows_parsed)     # 2
print(report.triples_added)   # 6
```

## What's next

- Query this knowledge with SPARQL, including from inside simulations, in [Semantic Queries](06-semantic-queries.md).
- Define your own inference rules and type systems in [Custom Ontology](09-custom-ontology.md).
