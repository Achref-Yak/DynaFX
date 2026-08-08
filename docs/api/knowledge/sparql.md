# SPARQL Querying

**What it is:** Parse and evaluate SPARQL queries against a TripleStore.

**When to use it:** When you need to query your knowledge graph with SPARQL.

**Quick start:**
```python
from dynafx.knowledge import parse_sparql, sparql_evaluate

query = "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:revenue ?v }"
ast = parse_sparql(query)
result = sparql_evaluate(ast, store)
print(result.bindings)  # [{'v': Literal(950.0)}]
```

---

## Import

```python
from dynafx.knowledge import parse_sparql, sparql_evaluate, QueryResult
```

## Functions

### `parse_sparql(query)`

Parse a SPARQL query string into an AST.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | — | SPARQL query string (required) |

**Returns:** Parsed AST

**Example:**
```python
ast = parse_sparql("PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:a ex:b ?v }")
```

---

### `sparql_evaluate(ast, store)`

Evaluate a parsed query against a store.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ast` | `Any` | — | Parsed AST from `parse_sparql()` |
| `store` | `TripleStore` | — | The triple store to query |

**Returns:** `QueryResult`

**Example:**
```python
ast = parse_sparql(query)
result = sparql_evaluate(ast, store)
```

---

## QueryResult

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `bindings` | `list[dict]` | List of variable → node mappings |
| `cardinality` | `int` | Number of results |

### Example

```python
result = sparql_evaluate(ast, store)
print(result.bindings)  # [{'v': Literal(950.0)}]
print(result.cardinality)  # 1
```

---

## Query Examples

### SELECT Query

```python
query = """
PREFIX ex: <http://ex.org/>
SELECT ?revenue ?cost
WHERE {
    ex:portfolio ex:revenue ?revenue .
    ex:portfolio ex:cost ?cost .
}
"""
ast = parse_sparql(query)
result = sparql_evaluate(ast, store)
for row in result.bindings:
    print(f"Revenue: {row['revenue'].value}, Cost: {row['cost'].value}")
```

### ASK Query

```python
query = """
PREFIX ex: <http://ex.org/>
ASK { ex:portfolio ex:revenue ?v }
"""
ast = parse_sparql(query)
result = sparql_evaluate(ast, store)
print(result.cardinality > 0)  # True if matches exist
```

---

## See Also

- [`TripleStore`](TripleStore.md) — Where to query
- [Tutorial 6: Semantic Queries](../../tutorials/06-semantic-queries.md) — SPARQL deep dive
