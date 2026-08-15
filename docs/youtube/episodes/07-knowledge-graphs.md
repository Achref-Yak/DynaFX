# Video 7: Knowledge Graphs - Your Simulation's Memory

**What it does:** Store facts as RDF, query with SPARQL, load from Turtle and CSV.

**Duration:** 15-18 minutes

**Hook:** "What if your simulation could remember facts, reason about them, and learn from results?"

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Simulations forget everything when they end. Knowledge graphs remember."

### 0:30 - RDF Basics (3 min)
- Subject, predicate, object
- "Jane worksAt Acme" is a triple
- IRIs identify things globally

### 3:30 - TripleStore (4 min)
```python
from dynafx.knowledge import TripleStore, NamedNode, Literal, Triple

store = TripleStore()
store.add(Triple(
    NamedNode("http://ex.org/jane"),
    NamedNode("http://ex.org/worksAt"),
    NamedNode("http://ex.org/acme")
), graph="people")
```
- Create store
- Add triples
- Named graphs for organization

### 7:30 - Turtle Format (3 min)
```python
from dynafx.knowledge import parse_turtle

store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:jane ex:worksAt ex:acme .
    ex:jane ex:age 30 .
    ex:bob ex:worksAt ex:acme .
""")
```
- Human-readable format
- Prefix declarations
- Load directly into store

### 10:30 - SPARQL Queries (4 min)
```python
from dynafx.knowledge import parse_sparql, sparql_evaluate

query = """
PREFIX ex: <http://ex.org/>
SELECT ?person ?age
WHERE {
    ?person ex:worksAt ex:acme .
    ?person ex:age ?age .
}
"""
result = sparql_evaluate(parse_sparql(query), store)
for row in result.bindings:
    print(row["person"].iri, row["age"].value)
```
- SELECT queries
- FILTER, ORDER BY, LIMIT
- ASK queries (boolean)

### 14:30 - Challenge (1 min)
- "Build a supplier database with 5 suppliers. Query for those with reliability > 0.8."

### 15:30 - Outro (30 sec)
- "Next time: the closed loop - connecting KB to simulation."

---

## Code Samples

### Basic Store
```python
from dynafx.knowledge import TripleStore, NamedNode, Literal, Triple

store = TripleStore()

# Add facts
store.add(Triple(
    NamedNode("http://ex.org/supplier_a"),
    NamedNode("http://ex.org/reliability"),
    Literal(0.95)
), graph="suppliers")

store.add(Triple(
    NamedNode("http://ex.org/supplier_b"),
    NamedNode("http://ex.org/reliability"),
    Literal(0.70)
), graph="suppliers")

# Query
pattern = TriplePattern(
    subject=NamedNode("http://ex.org/supplier_a"),
    predicate=NamedNode("http://ex.org/reliability"),
    object_=None
)
for t in store.triples(pattern):
    print(t.object_.value)  # 0.95
```

### Turtle Loading
```python
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:supplier_a ex:reliability 0.95 .
    ex:supplier_a ex:capacity 1000 .
    ex:supplier_b ex:reliability 0.70 .
    ex:supplier_b ex:capacity 2000 .
""")
```

### SPARQL Queries
```python
query = """
PREFIX ex: <http://ex.org/>
SELECT ?supplier ?reliability
WHERE {
    ?supplier ex:reliability ?reliability .
    FILTER (?reliability > 0.8)
}
"""
result = sparql_evaluate(parse_sparql(query), store)
for row in result.bindings:
    print(f"{row['supplier'].iri}: {row['reliability'].value}")
```

### CSV Ingestion
```python
from dynafx.knowledge import ingest_csv, MappingDef, ColumnMapping

mapping = MappingDef(
    csv="suppliers.csv",
    target_graph="enterprise",
    entity_class="http://ex.org/Supplier",
    id_column="id",
    id_prefix="http://ex.org/supplier/",
    columns={
        "name":        ColumnMapping(predicate="http://ex.org/name",        col_type="string"),
        "reliability": ColumnMapping(predicate="http://ex.org/reliability", col_type="float"),
    },
)
report = ingest_csv(mapping, "suppliers.csv", store)
```

---

## Thumbnail
- Left: Graph nodes and edges
- Right: SPARQL query result
- Bottom: "Your simulation's memory"

---

## Description
```
Build knowledge graphs in DynaFX - RDF, Turtle, SPARQL queries.

Docs: https://achref-yak.github.io/DynaFX/api/knowledge/TripleStore/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Challenge #7: Build a supplier database with 5 suppliers. Each has:
- reliability (0.5-1.0)
- capacity (500-2000)
Query: find suppliers with reliability > 0.8 AND capacity > 1000
```
