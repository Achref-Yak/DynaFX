# NamedNode / Literal / BlankNode

**What it are:** The three types of RDF nodes — the building blocks of triples. `NamedNode` is identified by an IRI, `Literal` holds a value, and `BlankNode` is anonymous.

**When to use them:** When creating triples to store in a `TripleStore`.

**Quick start:**
```python
from dynafx.knowledge.model import NamedNode, Literal, BlankNode

# Subject and predicate (identified by IRI)
subject = NamedNode("http://ex.org/jane")
predicate = NamedNode("http://ex.org/age")

# Object (literal value)
object_ = Literal(30)
```

---

## Import

```python
from dynafx.knowledge.model import NamedNode, Literal, BlankNode, xsd
```

---

## NamedNode

**What it is:** An RDF node identified by an IRI (Internationalized Resource Identifier). Used for subjects and predicates.

### Constructor

#### `NamedNode(iri)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iri` | `str` | — | The IRI identifying this node (required) |

**Returns:** `NamedNode` instance (frozen dataclass)

**Example:**
```python
from dynafx.knowledge.model import NamedNode

# Create nodes with full IRIs
person = NamedNode("http://ex.org/jane")
works_at = NamedNode("http://ex.org/worksAt")
company = NamedNode("http://ex.org/acme")
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `iri` | `str` | The IRI |

### Methods

#### `n3()`

Return the N3 representation: `<http://ex.org/jane>`

```python
node = NamedNode("http://ex.org/jane")
print(node.n3())  # <http://ex.org/jane>
```

---

## Literal

**What it is:** An RDF literal value with optional datatype and language tag. Used for objects that represent data values.

### Constructor

#### `Literal(value, datatype=None, lang_tag=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | `Any` | — | The literal value (required) |
| `datatype` | `str \| None` | `None` | XSD datatype IRI |
| `lang_tag` | `str \| None` | `None` | Language tag (e.g., `"en"`) |

**Returns:** `Literal` instance (frozen dataclass)

**Example:**
```python
from dynafx.knowledge.model import Literal, xsd

# Numeric values
revenue = Literal(950.0, datatype=xsd("double"))
count = Literal(42, datatype=xsd("integer"))

# String values
name = Literal("Jane Smith", datatype=xsd("string"))

# Language-tagged strings
greeting = Literal("Hello", lang_tag="en")
salut = Literal("Bonjour", lang_tag="fr")

# Boolean
active = Literal(True, datatype=xsd("boolean"))
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `value` | `Any` | The literal value |
| `datatype` | `str \| None` | XSD datatype IRI |
| `lang_tag` | `str \| None` | Language tag |

### Methods

#### `n3()`

Return the N3 representation.

```python
literal = Literal(950.0, datatype=xsd("double"))
print(literal.n3())  # "950.0"^^<http://www.w3.org/2001/XMLSchema#double>

greeting = Literal("Hello", lang_tag="en")
print(greeting.n3())  # "Hello"@en
```

---

## BlankNode

**What it is:** An anonymous RDF node with no IRI. Used when the node doesn't need a global identifier.

### Constructor

#### `BlankNode(id=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | `str \| None` | `None` | Optional identifier (auto-generated if None) |

**Returns:** `BlankNode` instance (frozen dataclass)

**Example:**
```python
from dynafx.knowledge.model import BlankNode

# Auto-generated ID
node1 = BlankNode()  # _:b12345678

# Explicit ID
node2 = BlankNode(id="_:b001")
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | The blank node identifier |

### Methods

#### `n3()`

Return the N3 representation.

```python
node = BlankNode(id="_:b001")
print(node.n3())  # _:b001
```

---

## XSD Type Helpers

### `xsd(typename)`

Create an XSD type NamedNode.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `typename` | `str` | — | XSD type name (required) |

**Returns:** `NamedNode`

**Example:**
```python
from dynafx.knowledge.model import xsd

xsd("double")   # NamedNode("http://www.w3.org/2001/XMLSchema#double")
xsd("string")   # NamedNode("http://www.w3.org/2001/XMLSchema#string")
xsd("integer")  # NamedNode("http://www.w3.org/2001/XMLSchema#integer")
xsd("boolean")  # NamedNode("http://www.w3.org/2001/XMLSchema#boolean")
```

### Common XSD Constants

```python
from dynafx.knowledge.model import xsd, XSD_STRING, XSD_INTEGER, XSD_DOUBLE, XSD_BOOLEAN

# These are pre-defined:
XSD_STRING   = xsd("string")
XSD_INTEGER  = xsd("integer")
XSD_DOUBLE   = xsd("double")
XSD_BOOLEAN  = xsd("boolean")
XSD_DATE_TIME = xsd("dateTime")
```

---

## Common Patterns

### Pattern 1: Creating Triples

```python
from dynafx.knowledge.model import NamedNode, Literal, Triple, xsd

triple = Triple(
    subject=NamedNode("http://ex.org/jane"),
    predicate=NamedNode("http://ex.org/age"),
    object_=Literal(30, datatype=xsd("integer"))
)
```

### Pattern 2: Namespace Helpers

```python
from dynafx.knowledge.model import NamedNode, Literal, Triple

# Define namespace prefixes for cleaner code
EX = "http://ex.org/"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# Create triples with namespaces
triple = Triple(
    subject=NamedNode(f"{EX}jane"),
    predicate=NamedNode(f"{RDF}type"),
    object_=NamedNode(f"{EX}Person")
)
```

### Pattern 3: Working with Literals

```python
from dynafx.knowledge.model import Literal, xsd

# Auto-detect type (no explicit datatype)
auto = Literal(950.0)  # No datatype specified

# Explicit datatype
explicit = Literal(950.0, datatype=xsd("double"))

# Language tag
named = Literal("Jane Smith", lang_tag="en")
```

---

## See Also

- [`Triple`](Triple.md) — How to use these nodes in triples
- [`TripleStore`](TripleStore.md) — Where to store triples
- [Tutorial 5: Knowledge Graphs](../../tutorials/05-knowledge-graph.md) — Getting started
