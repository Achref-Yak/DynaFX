# Triple / TriplePattern

**What they are:** A `Triple` is a (subject, predicate, object) fact. A `TriplePattern` is a query template where `None` means "match anything."

**When to use them:** When adding data to or querying a `TripleStore`.

**Quick start:**
```python
from dynafx.knowledge.model import NamedNode, Literal, Triple, TriplePattern

# Create a triple
triple = Triple(
    subject=NamedNode("http://ex.org/jane"),
    predicate=NamedNode("http://ex.org/age"),
    object_=Literal(30)
)

# Create a pattern to query
pattern = TriplePattern(
    subject=NamedNode("http://ex.org/jane"),
    predicate=None,  # match any predicate
    object_=None     # match any object
)
```

---

## Import

```python
from dynafx.knowledge.model import Triple, TriplePattern
```

---

## Triple

**What it is:** An RDF triple — a statement of the form (subject, predicate, object).

### Constructor

#### `Triple(subject, predicate, object_)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subject` | `NamedNode \| BlankNode` | — | The subject (required) |
| `predicate` | `NamedNode` | — | The predicate (required) |
| `object_` | `NamedNode \| BlankNode \| Literal` | — | The object (required) |

**Returns:** `Triple` instance (frozen dataclass)

**Example:**
```python
from dynafx.knowledge.model import NamedNode, Literal, Triple

triple = Triple(
    subject=NamedNode("http://ex.org/jane"),
    predicate=NamedNode("http://ex.org/worksAt"),
    object_=NamedNode("http://ex.org/acme")
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `subject` | `NamedNode \| BlankNode` | The subject |
| `predicate` | `NamedNode` | The predicate |
| `object_` | `NamedNode \| BlankNode \| Literal` | The object |

### Properties

#### `spo: tuple`

The identity tuple `(subject, predicate, object_)` used for hashing and equality.

```python
print(triple.spo)  # (NamedNode("..."), NamedNode("..."), NamedNode("..."))
```

### Methods

#### `__hash__()`

Hash based on `(subject, predicate, object_)`.

#### `__eq__(other)`

Equality based on `(subject, predicate, object_)`.

### Usage Examples

```python
from dynafx.knowledge.model import NamedNode, Literal, Triple, xsd

# Fact: Jane works at Acme
triple1 = Triple(
    subject=NamedNode("http://ex.org/jane"),
    predicate=NamedNode("http://ex.org/worksAt"),
    object_=NamedNode("http://ex.org/acme")
)

# Fact: Jane is 30 years old
triple2 = Triple(
    subject=NamedNode("http://ex.org/jane"),
    predicate=NamedNode("http://ex.org/age"),
    object_=Literal(30, datatype=xsd("integer"))
)

# Fact: Acme has revenue of 950.0
triple3 = Triple(
    subject=NamedNode("http://ex.org/acme"),
    predicate=NamedNode("http://ex.org/revenue"),
    object_=Literal(950.0, datatype=xsd("double"))
)
```

---

## TriplePattern

**What it is:** A query template for matching triples. Use `None` for positions you want to match anything.

### Constructor

#### `TriplePattern(subject=None, predicate=None, object_=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subject` | `Any \| None` | `None` | Subject to match (None = wildcard) |
| `predicate` | `Any \| None` | `None` | Predicate to match (None = wildcard) |
| `object_` | `Any \| None` | `None` | Object to match (None = wildcard) |

**Returns:** `TriplePattern` instance (frozen dataclass)

**Example:**
```python
from dynafx.knowledge.model import TriplePattern, NamedNode

# Match all triples with Jane as subject
pattern = TriplePattern(
    subject=NamedNode("http://ex.org/jane"),
    predicate=None,
    object_=None
)

# Match all triples with a specific predicate
pattern = TriplePattern(
    subject=None,
    predicate=NamedNode("http://ex.org/worksAt"),
    object_=None
)

# Match all triples with a specific object
pattern = TriplePattern(
    subject=None,
    predicate=None,
    object_=NamedNode("http://ex.org/acme")
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `subject` | `Any \| None` | Subject to match or `None` |
| `predicate` | `Any \| None` | Predicate to match or `None` |
| `object_` | `Any \| None` | Object to match or `None` |

### Common Patterns

```python
from dynafx.knowledge.model import TriplePattern, NamedNode, Literal

# Find all triples about Jane
pattern = TriplePattern(subject=NamedNode("http://ex.org/jane"))

# Find all "worksAt" relationships
pattern = TriplePattern(predicate=NamedNode("http://ex.org/worksAt"))

# Find all triples with a specific object
pattern = TriplePattern(object_=NamedNode("http://ex.org/acme"))

# Find all triples (empty pattern)
pattern = TriplePattern()
```

---

## Common Workflows

### Workflow 1: Add and Query

```python
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, Triple, TriplePattern

store = TripleStore()

# Add triples
store.add(Triple(
    NamedNode("http://ex.org/jane"),
    NamedNode("http://ex.org/age"),
    Literal(30)
))

# Query
pattern = TriplePattern(
    subject=NamedNode("http://ex.org/jane"),
    predicate=NamedNode("http://ex.org/age"),
    object_=None
)
for triple in store.triples(pattern):
    print(triple.object_.value)  # 30
```

### Workflow 2: Wildcard Queries

```python
from dynafx.knowledge.model import TriplePattern, NamedNode

# Find all properties of Jane
pattern = TriplePattern(subject=NamedNode("http://ex.org/jane"))
for triple in store.triples(pattern):
    print(f"{triple.predicate.iri}: {triple.object_}")

# Find all subjects with a specific property
pattern = TriplePattern(predicate=NamedNode("http://ex.org/worksAt"))
for triple in store.triples(pattern):
    print(f"{triple.subject.iri} works at {triple.object_.iri}")
```

### Workflow 3: Graph-Specific Queries

```python
# Query only in a specific graph
pattern = TriplePattern(subject=NamedNode("http://ex.org/jane"))
for triple in store.triples(pattern, graph="enterprise"):
    print(triple)
```

---

## See Also

- [`TripleStore`](TripleStore.md) — Where to store and query triples
- [`NamedNode`](NamedNode.md) — RDF node types
- [Tutorial 5: Knowledge Graphs](../../tutorials/05-knowledge-graph.md) — Getting started
