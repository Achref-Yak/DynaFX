# Turtle Parsing

**What it is:** Parse Turtle format RDF text into a TripleStore, or serialize triples back to Turtle.

**When to use it:** When loading data from Turtle strings or exporting to Turtle format.

**Quick start:**
```python
from dynafx.knowledge import parse_turtle, serialize_turtle

store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:portfolio ex:revenue 950.0 .
    ex:portfolio ex:cost 600.0 .
""")
```

---

## Import

```python
from dynafx.knowledge import parse_turtle, serialize_turtle
```

## Functions

### `parse_turtle(text, base_iri=None)`

Parse Turtle text into a TripleStore.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | — | Turtle format text (required) |
| `base_iri` | `str \| None` | `None` | Base IRI for relative IRIs |

**Returns:** `TripleStore` with parsed triples

**Example:**
```python
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ex:portfolio ex:revenue "950.0"^^xsd:double .
    ex:portfolio ex:cost "600.0"^^xsd:double .
    ex:supplier ex:reliability "0.85"^^xsd:double .
""")
```

---

### `serialize_turtle(triples, prefixes=None)`

Serialize triples to Turtle text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `triples` | `list[Triple]` | — | Triples to serialize (required) |
| `prefixes` | `dict[str, str] \| None` | `None` | Namespace prefixes |

**Returns:** `str` — Turtle format text

**Example:**
```python
turtle_text = serialize_turtle(
    list(store.all_triples()),
    prefixes={"ex": "http://ex.org/"}
)
print(turtle_text)
```

---

## See Also

- [`TripleStore`](TripleStore.md) — Where to store parsed data
- [Tutorial 5: Knowledge Graphs](../../tutorials/05-knowledge-graph.md) — Getting started
