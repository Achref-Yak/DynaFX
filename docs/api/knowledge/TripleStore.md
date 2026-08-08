# TripleStore

**What it is:** An in-memory RDF triple store — a database for facts stored as (subject, predicate, object) triples. Supports named graphs, fast pattern matching, and event callbacks.

**When to use it:** When you need to store structured facts, query them by pattern, or connect a knowledge graph to a simulation.

**Quick start:**
```python
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, Triple

store = TripleStore()
store.add(Triple(
    NamedNode("http://ex.org/portfolio"),
    NamedNode("http://ex.org/revenue"),
    Literal(950.0)
), graph="enterprise")
```

---

## Import

```python
from dynafx.knowledge import TripleStore
```

## Constructor

### `TripleStore()`

Creates an empty triple store.

**Returns:** `TripleStore` instance

**Example:**
```python
store = TripleStore()
```

---

## Methods

### `add(triple, graph="default")`

Add a triple to the store. Idempotent — adding the same triple twice has no effect (last write wins).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `triple` | `Triple` | — | The triple to add (required) |
| `graph` | `str` | `"default"` | Named graph to store in |

**Returns:** `None`

**Example:**
```python
from dynafx.knowledge.model import NamedNode, Literal, Triple

store = TripleStore()

# Add a triple
store.add(Triple(
    NamedNode("http://ex.org/jane"),
    NamedNode("http://ex.org/worksAt"),
    NamedNode("http://ex.org/acme"),
), graph="enterprise")

# Add to specific graph
store.add(Triple(
    NamedNode("http://ex.org/portfolio"),
    NamedNode("http://ex.org/revenue"),
    Literal(950.0)
), graph="financial")
```

---

### `remove(pattern, graph=None)`

Remove triples matching a pattern.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `TriplePattern` | — | Pattern to match (required) |
| `graph` | `str \| None` | `None` | Restrict to this graph (None = all) |

**Returns:** `int` — number of triples removed

**Example:**
```python
from dynafx.knowledge.model import TriplePattern, NamedNode

# Remove all triples with a specific subject
pattern = TriplePattern(
    subject=NamedNode("http://ex.org/jane"),
    predicate=None,
    object_=None
)
count = store.remove(pattern)
print(f"Removed {count} triples")
```

---

### `triples(pattern, graph=None, with_inference=None)`

Iterate over triples matching a pattern.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `TriplePattern` | — | Pattern to match (required) |
| `graph` | `str \| None` | `None` | Restrict to this graph (None = all) |
| `with_inference` | `str \| dict \| None` | `None` | Enable inference: `"rdfs"` or config dict |

**Returns:** `Iterator[Triple]` — matching triples

**Example:**
```python
from dynafx.knowledge.model import TriplePattern, NamedNode

# Find all triples with a specific predicate
pattern = TriplePattern(
    subject=None,
    predicate=NamedNode("http://ex.org/revenue"),
    object_=None
)
for triple in store.triples(pattern):
    print(f"{triple.subject.iri}: {triple.object_.value}")

# Find triples in a specific graph
for triple in store.triples(pattern, graph="enterprise"):
    print(triple)
```

---

### `triples_in_graph(graph)`

Iterate over all triples in a named graph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `str` | — | Graph name (required) |

**Returns:** `Iterator[Triple]` — triples in the graph

**Example:**
```python
for triple in store.triples_in_graph("enterprise"):
    print(triple)
```

---

### `graphs()`

List all named graphs in the store.

**Returns:** `list[str]` — graph names

**Example:**
```python
print(store.graphs())  # ["default", "enterprise", "financial"]
```

---

### `all_triples()`

Iterate over all triples in the store.

**Returns:** `Iterator[Triple]` — all triples

**Example:**
```python
for triple in store.all_triples():
    print(triple)
```

---

### `copy_graph(source, dest)`

Copy all triples from one graph to another.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | — | Source graph name (required) |
| `dest` | `str` | — | Destination graph name (required) |

**Returns:** `None`

**Example:**
```python
store.copy_graph("enterprise", "archive")
```

---

### `remove_graph(graph)`

Remove all triples in a named graph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `str` | — | Graph name (required) |

**Returns:** `None`

**Example:**
```python
store.remove_graph("archive")
```

---

### `on_add(fn)`

Register a callback for when triples are added.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable[[Triple, str], None]` | — | Callback function (required) |

**Returns:** `None`

**Example:**
```python
def on_triple_added(triple, graph):
    print(f"Added to {graph}: {triple}")

store.on_add(on_triple_added)
```

---

### `on_remove(fn)`

Register a callback for when triples are removed.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fn` | `Callable[[TriplePattern, str \| None], None]` | — | Callback function (required) |

**Returns:** `None`

**Example:**
```python
def on_triple_removed(pattern, graph):
    print(f"Removed from {graph}: {pattern}")

store.on_remove(on_triple_removed)
```

---

### `suppress_callbacks()`

Context manager to temporarily suppress event callbacks.

**Returns:** Context manager

**Example:**
```python
with store.suppress_callbacks():
    # Callbacks won't fire during this block
    store.add(triple)
```

---

### `__contains__(pattern)`

Check if any triples match a pattern.

**Example:**
```python
from dynafx.knowledge.model import TriplePattern, NamedNode

pattern = TriplePattern(
    subject=NamedNode("http://ex.org/jane"),
    predicate=None,
    object_=None
)
if pattern in store:
    print("Jane has triples in the store")
```

---

### `__len__()`

Get the total number of triples.

**Returns:** `int`

**Example:**
```python
print(len(store))  # 42
```

---

## Common Patterns

### Pattern 1: Basic CRUD

```python
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, Triple, TriplePattern

store = TripleStore()

# Create
store.add(Triple(NamedNode("http://ex.org/jane"), NamedNode("http://ex.org/age"), Literal(30)))

# Read
pattern = TriplePattern(NamedNode("http://ex.org/jane"), None, None)
for t in store.triples(pattern):
    print(t)

# Update (add with same subject/predicate, different object)
store.add(Triple(NamedNode("http://ex.org/jane"), NamedNode("http://ex.org/age"), Literal(31)))

# Delete
store.remove(pattern)
```

### Pattern 2: Named Graphs

```python
# Separate data by source
store.add(triple1, graph="enterprise")
store.add(triple2, graph="simulation")
store.add(triple3, graph="financial")

# Query specific graph
for t in store.triples(pattern, graph="enterprise"):
    print(t)

# List graphs
print(store.graphs())  # ["enterprise", "simulation", "financial"]
```

### Pattern 3: Event-Driven Updates

```python
store.on_add(lambda t, g: print(f"Added: {t} to {g}"))
store.on_remove(lambda p, g: print(f"Removed: {p} from {g}"))

# These will trigger callbacks
store.add(triple)
store.remove(pattern)
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Triple not found | Wrong graph name | Check `store.graphs()` |
| Callbacks not firing | Using `suppress_callbacks()` | Remove context manager |
| Memory usage high | Too many triples | Use named graphs to organize data |

---

## See Also

- [`NamedNode`](NamedNode.md) — RDF node types
- [`Triple`](Triple.md) — Triple class
- [`TriplePattern`](Triple.md#triplepattern) — Query patterns
- [Tutorial 5: Knowledge Graphs](../../tutorials/05-knowledge-graph.md) — Getting started
