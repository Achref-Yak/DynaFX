# CSV Ingestion

**What it is:** Declarative CSV → RDF ingestion — load tabular data into a TripleStore using mapping definitions.

**When to use it:** When loading data from CSV files into your knowledge graph.

**Quick start:**
```python
from dynafx.knowledge import ingest_csv, MappingDef, ColumnMapping

mapping = MappingDef(
    csv="data/portfolio.csv",
    target_graph="enterprise",
    entity_class="http://ex.org/Portfolio",
    id_column="id",
    id_prefix="http://ex.org/portfolio/",
    columns={
        "revenue": ColumnMapping(predicate="http://ex.org/revenue", col_type="float"),
        "name":    ColumnMapping(predicate="http://ex.org/name",    col_type="string"),
    },
)
report = ingest_csv(mapping, "data/portfolio.csv", store)
print(f"{report.triples_added} triples added, {report.rows_skipped} rows skipped")
```

---

## Import

```python
from dynafx.knowledge import ingest_csv, MappingDef, ColumnMapping, IngestReport, load_all_mappings
```

## Functions

### `ingest_csv(mapping, csv_source, store, strict=False, encoding="utf-8")`

Ingest a CSV file into a TripleStore.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mapping` | `MappingDef \| str \| Path` | — | Mapping definition, or path to a YAML mapping file (required) |
| `csv_source` | `str \| Path \| list[dict]` | — | CSV file path, or pre-parsed list of dicts (required) |
| `store` | `TripleStore` | — | Target store (required) |
| `strict` | `bool` | `False` | If `True`, raise on first conversion error. If `False`, skip bad rows with a warning |
| `encoding` | `str` | `"utf-8"` | CSV file encoding |

**Returns:** `IngestReport`

**Example:**
```python
# From YAML mapping file
report = ingest_csv("data/mappings/suppliers.yaml", "data/suppliers.csv", store)

# With pre-parsed data and strict mode
rows = [{"id": "1", "name": "Acme", "revenue": "50000"}]
report = ingest_csv(mapping, rows, store, strict=True)

# Non-UTF-8 file
report = ingest_csv(mapping, "data/legacy.csv", store, encoding="latin-1")
```

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
    csv: str,                              # CSV file path
    target_graph: str,                     # RDF graph to store triples in
    entity_class: str,                     # RDF class IRI for each entity row
    id_column: str,                        # CSV column name used as entity ID
    id_prefix: str,                        # IRI prefix prepended to entity IDs
    columns: dict[str, ColumnMapping],     # Column name → mapping (dict, not list)
    prefixes: dict[str, str] = {},         # Namespace prefixes for CURIE expansion
)
```

**Validation:** `__post_init__` raises `ValueError` if any column has `col_type="iri"` but no `iri_prefix` set.

**Loading from YAML:**
```python
mapping = MappingDef.from_yaml("data/mappings/suppliers.yaml")
```

YAML files support CURIE-style prefix expansion (e.g., `foaf:name` expands using the `prefixes` dict).

### ColumnMapping

```python
ColumnMapping(
    predicate: str,                 # Predicate IRI for this column
    col_type: str = "string",       # Type coercion: "string", "float", "integer", "boolean", "iri"
    iri_prefix: str | None = None,  # Required when col_type="iri" — prefix for IRI values
)
```

### IngestReport

```python
IngestReport(
    csv: str,                    # Source CSV path (or "<list[dict]>" for in-memory data)
    mapping: str,                # Mapping source (YAML path or "<MappingDef>")
    target_graph: str,           # Target RDF graph name
    rows_parsed: int = 0,        # Successfully parsed rows
    rows_skipped: int = 0,       # Rows skipped due to errors or empty IDs
    triples_added: int = 0,      # RDF triples written to store
    errors: list[str] = [],      # Conversion/ingestion errors (actively populated when strict=False)
    warnings: list[str] = [],    # Non-fatal warnings (e.g., empty ID columns)
)
```

When `strict=False` (default), bad rows are skipped and the error message is recorded in `report.errors`. When `strict=True`, the first error raises immediately.

---

## See Also

- [`TripleStore`](TripleStore.md) — Where data is loaded
- [Tutorial 5: Knowledge Graphs](../../tutorials/05-knowledge-graph.md) — Getting started
