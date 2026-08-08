# Type Hierarchy

**What it is:** OWL2-style subtype lattice for type inference and validation.

**When to use it:** When you need to reason about class hierarchies or validate type relationships.

**Quick start:**
```python
from dynafx.knowledge import TypeHierarchy, GENERAL_TBOX

hierarchy = TypeHierarchy(GENERAL_TBOX)
print(hierarchy.is_subtype("Dog", "Animal"))  # True
```

---

## Import

```python
from dynafx.knowledge import TypeHierarchy, TypeNode, TBox, load_tbox, GENERAL_TBOX, MDM_TYPE_HIERARCHY
```

## Classes

### TypeHierarchy

```python
TypeHierarchy(tbox: TBox)
```

#### Methods

```python
hierarchy.is_subtype(child, parent)  # Check subtype relationship
hierarchy.get_ancestors(type_name)   # Get all ancestors
hierarchy.get_descendants(type_name) # Get all descendants
```

### TypeNode

A node in the hierarchy.

```python
TypeNode(
    name: str,
    parents: list[str] = [],
    properties: dict = {},
)
```

### TBox

Load a named TBox.

```python
tbox = load_tbox("general")
```

### Built-in Hierarchies

```python
GENERAL_TBOX        # General type hierarchy
MDM_TYPE_HIERARCHY  # Master data management types
```

---

## See Also

- [`Inference`](inference.md) — Forward-chaining rules
- [Tutorial 9: Custom Ontology](../../tutorials/09-custom-ontology.md) — Ontology deep dive
