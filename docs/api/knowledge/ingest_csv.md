# CSV Ingestion

**What it is:** Declarative CSV → RDF ingestion — load tabular data into a TripleStore using mapping definitions.

**When to use it:** When loading data from CSV files into your knowledge graph.

**Quick start:**
```python
from dynafx.knowledge import ingest_csv, MappingDef, ColumnMapping

mapping = MappingDef(
    target_graph="enterprise",
    entity_class="http://ex.org/Portfolio",
    columns=[
        ColumnMapping(csv_column="id", predicate="http://ex.org/id", datatype="string"),
        ColumnMapping(csv_column="revenue", predicate="http://ex.org/revenue", datatype="double"),
    ],
)
report = ingest_csv(mapping, "data/portfolio.csv", store)
```

---

## Import

```python
from dynafx.knowledge import ingest_csv, MappingDef, ColumnMapping, IngestReport, load_all_mappings
```

## Functions

### `ingest_csv(mapping, path, store, strict=False)`

Ingest a CSV file into a TripleStore.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mapping` | `MappingDef` | — | Mapping definition (required) |
| `path` | `str` | — | CSV file path (required) |
| `store` | `TripleStore` | — | Target store (required) |
| `strict` | `bool` | `False` | Raise on errors |

**Returns:** `IngestReport`

---

### `load_all_mappings(dir)`

Load all YAML mapping files from a directory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dir` | `str` | — | Directory path (required) |

**Returns:** `list[MappingDef]`

---

## Classes

### MappingDef

```python
MappingDef(
    target_graph: str,              # Graph to store in
    entity_class: str,              # RDF class for entities
    columns: list[ColumnMapping],   # Column mappings
    base_iri: str = "",             # Base IRI for subjects
)
```

### ColumnMapping

```python
ColumnMapping(
    csv_column: str,                # CSV column name
    predicate: str,                 # Predicate IRI
    datatype: str = "string",       # XSD datatype
    required: bool = False,         # Require non-empty
)
```

### IngestReport

```python
IngestReport(
    rows_parsed: int,
    triples_added: int,
    errors: list[str],
)
```

---

## See Also

- [`TripleStore`](TripleStore.md) — Where data is loaded
- [Tutorial 5: Knowledge Graphs](../../tutorials/05-knowledge-graph.md) — Getting started
