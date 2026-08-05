# Tutorial 6 — Semantic Queries

SPARQL is how DynaFX asks questions of its knowledge base — and it can be
called *from inside* a running simulation via `KB_QUERY(...)`.

## Querying a store

```python
from dynafx.knowledge import parse_sparql, sparql_evaluate, parse_turtle

turtle = """
@prefix ex: <http://ex.org/> .
ex:alice ex:age 30 .
ex:bob ex:age 25 .
ex:charlie ex:age 35 .
"""
store = parse_turtle(turtle)

query = "SELECT ?person ?age WHERE { ?person <http://ex.org/age> ?age }"
result = sparql_evaluate(parse_sparql(query), store)

print(result.cardinality)   # 3
for row in result.bindings:
    print(row["person"].iri, row["age"].value)   # IRIs and literal values
```

`QueryResult` carries `.bindings` (a list of dicts mapping variable → RDFNode)
and `.cardinality`. Bindings hold `NamedNode` objects (read `.iri`) or `Literal`
objects (read `.value`).

## ASK — boolean questions

```python
is_old = sparql_evaluate(
    parse_sparql("ASK WHERE { ?p <http://ex.org/age> ?a FILTER(?a > 30) }"),
    store,
)
print(is_old.cardinality)   # 1 → true
```

## FILTER, ORDER BY, LIMIT

The evaluator supports the common SPARQL 1.1 operators:

```python
query = """
SELECT ?person ?age WHERE {
  ?person <http://ex.org/age> ?age .
  FILTER(?age >= 25)
}
ORDER BY DESC(?age)
LIMIT 2
"""
result = sparql_evaluate(parse_sparql(query), store)
print([r["person"].iri for r in result.bindings])
```

## Inside a simulation: KB_QUERY

`KB_QUERY(query_param)` evaluates a SPARQL query every step. The query string
is passed as a **parameter** (the expression parser has no string literals):

```python
from dynafx.dynamics import parse_sysd
from dynafx.knowledge import TripleStore, NamedNode, Literal, Triple

kb = TripleStore()
kb.add(Triple(
    NamedNode("http://ex.org/Portfolio"),
    NamedNode("http://ex.org/supplierReliability"),
    Literal(0.9),
), graph="enterprise")

model = parse_sysd("""
T
dt 1.0
from 0 to 5

stock X: 100
  - Out: X * kb_reliability

aux "kb_reliability": KB_QUERY(reliability_q)
""")

result = model.simulate(
    params={
        "reliability_q": "SELECT ?v WHERE { <http://ex.org/Portfolio> <http://ex.org/supplierReliability> ?v }",
    },
    kb=kb,
)

print(result.aux_values["kb_reliability"][0])   # 0.9 — read from KB each step
```

The stock `X` now drains by 90% per step instead of the (unset) default.

## KB_QUERY in agents and DES

The same builtin works in ABM rule conditions and DES arrival rates. The
flagship twin gates its port outflow on an `ASK` query:

```python
DISRUPTION_Q = "ASK WHERE { <http://epc.org/GlobalDisruption> <http://epc.org/active> true }"
```

When the triple exists in the KB, `KB_QUERY(DISRUPTION_Q)` returns 1.0, which
cuts port throughput. See [Closed-Loop Simulation](07-closed-loop-simulation.md) for the
full pattern.

## KB_ASSERT — writing back from inside

Agents can also write triples mid-run with `KB_ASSERT(subject, predicate, value)`:

```python
model = parse_sysd("""
T
dt 1.0
from 0 to 3

agent "Probe": 1
  property "v": 0.0
  rule "report": always
    v += 1
    KB_ASSERT("http://ex.org/Probe", "http://ex.org/level", v)
""")

kb = TripleStore()
model.simulate(kb=kb)

# Read back what the agent wrote
for t in kb.triples(TriplePattern(
    subject=NamedNode("http://ex.org/Probe"),
    predicate=NamedNode("http://ex.org/level"),
)):
    print(t.object_.value)
```

## What's next

- Close the loop: KB facts in, simulation evidence out, in [Closed-Loop Simulation](07-closed-loop-simulation.md).
- Add inference rules so the KB derives more than it was told, in [Custom Ontology](09-custom-ontology.md).
