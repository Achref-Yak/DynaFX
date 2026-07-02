"""Ingest CSV data into a TripleStore via declarative YAML mappings.

Usage:
    >>> from dynafx.knowledge.ingest_csv import ingest_csv
    >>> from dynafx.knowledge.store import TripleStore
    >>> store = TripleStore()
    >>> report = ingest_csv("data/mappings/suppliers.yaml", "data/epc_suppliers.csv", store)
    >>> print(f"{report.triples_added} triples added")
"""

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from dynafx.knowledge.model import (
    RDF_TYPE,
    XSD_BOOLEAN,
    XSD_DOUBLE,
    XSD_INTEGER,
    XSD_STRING,
    BlankNode,
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
)


@dataclass
class ColumnMapping:
    """How a single CSV column maps to an RDF predicate."""

    predicate: str
    col_type: str  # "string" | "float" | "integer" | "boolean" | "iri"
    iri_prefix: Optional[str] = None  # required when col_type == "iri"


@dataclass
class MappingDef:
    """Declarative mapping from a CSV file to RDF triples."""

    csv: str
    target_graph: str
    entity_class: str
    id_column: str
    id_prefix: str
    columns: dict[str, ColumnMapping]
    prefixes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MappingDef":
        raw = yaml.safe_load(Path(path).read_text())
        prefixes = raw.get("prefixes", {})

        def _expand(val: str) -> str:
            if not isinstance(val, str) or ":" not in val:
                return val
            if val.startswith("http://") or val.startswith("https://"):
                return val
            prefix, local = val.split(":", 1)
            if prefix in prefixes:
                return prefixes[prefix] + local
            return val

        target_graph = _expand(raw["target_graph"])
        entity_class = _expand(raw["entity"]["class"])
        id_prefix = _expand(raw["entity"]["id_prefix"])

        columns: dict[str, ColumnMapping] = {}
        for csv_col, cfg in raw.get("columns", {}).items():
            col_type = cfg.get("type", "string")
            predicate = _expand(cfg["predicate"])
            iri_prefix = _expand(cfg["iri_prefix"]) if cfg.get("iri_prefix") else None
            columns[csv_col] = ColumnMapping(
                predicate=predicate,
                col_type=col_type,
                iri_prefix=iri_prefix,
            )

        return cls(
            csv=raw["csv"],
            target_graph=target_graph,
            entity_class=entity_class,
            id_column=raw["entity"]["id_column"],
            id_prefix=id_prefix,
            columns=columns,
            prefixes=prefixes,
        )


@dataclass
class IngestReport:
    """Result of a single ingest_csv() call."""

    csv: str
    mapping: str
    target_graph: str
    rows_parsed: int = 0
    rows_skipped: int = 0
    triples_added: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_TYPE_CONVERTERS = {
    "string": lambda v: Literal(str(v), datatype=XSD_STRING),
    "float": lambda v: Literal(str(float(v)), datatype=XSD_DOUBLE),
    "integer": lambda v: Literal(str(int(v)), datatype=XSD_INTEGER),
    "boolean": lambda v: Literal(
        "true" if str(v).lower() in ("1", "true", "yes") else "false",
        datatype=XSD_BOOLEAN,
    ),
}


def _convert_value(col: ColumnMapping, value: str) -> Optional[Any]:
    """Convert a CSV string to an RDF node based on column type."""
    stripped = value.strip()
    if not stripped:
        return None
    if col.col_type == "iri":
        return NamedNode(col.iri_prefix + stripped)
    converter = _TYPE_CONVERTERS.get(col.col_type)
    if converter is None:
        raise ValueError(f"Unknown column type: {col.col_type}")
    return converter(stripped)


def ingest_csv(
    mapping: str | Path | MappingDef,
    csv_source: str | Path | list[dict],
    store: "TripleStore",
    strict: bool = False,
) -> IngestReport:
    """Apply a YAML mapping to a CSV and write result triples into *store*.

    Parameters
    ----------
    mapping:
        Path to a YAML mapping file, or a pre-parsed MappingDef.
    csv_source:
        Path to a CSV file, or pre-parsed list of dicts.
    store:
        Target TripleStore to write triples into.
    strict:
        If True, raise on first conversion error. If False (default),
        skip bad rows with a warning.

    Returns
    -------
    IngestReport
        Counts of rows parsed/skipped, triples added, and any errors.
    """
    # 1. Parse mapping
    if isinstance(mapping, (str, Path)):
        mapping_def = MappingDef.from_yaml(mapping)
        mapping_path = str(mapping)
    else:
        mapping_def = mapping
        mapping_path = "<MappingDef>"

    # 2. Load CSV
    if isinstance(csv_source, (str, Path)):
        with open(csv_source, newline="") as f:
            rows = list(csv.DictReader(f))
        csv_path = str(csv_source)
    else:
        rows = csv_source
        csv_path = "<list[dict]>"

    report = IngestReport(
        csv=csv_path,
        mapping=mapping_path,
        target_graph=mapping_def.target_graph,
    )
    entity_iri_parts = []
    type_ns, type_local = (
        mapping_def.entity_class.rsplit("/", 1)
        if "/" in mapping_def.entity_class
        else ("", mapping_def.entity_class)
    )
    class_node = NamedNode(mapping_def.entity_class)

    for row_idx, row in enumerate(rows):
        # Extract entity ID
        raw_id = row.get(mapping_def.id_column, "").strip()
        if not raw_id:
            report.rows_skipped += 1
            msg = f"Row {row_idx}: empty id column '{mapping_def.id_column}'"
            report.warnings.append(msg)
            if strict:
                raise ValueError(msg)
            continue

        entity_iri = NamedNode(mapping_def.id_prefix + raw_id)
        triple_count = 0

        def _add(p: str, o: Any) -> None:
            nonlocal triple_count
            store.add(Triple(entity_iri, NamedNode(p), o), mapping_def.target_graph)
            triple_count += 1

        try:
            # rdf:type triple
            _add(RDF_TYPE.iri, class_node)

            # Column triples
            for csv_col, col_mapping in mapping_def.columns.items():
                raw_val = row.get(csv_col, "")
                obj = _convert_value(col_mapping, raw_val)
                if obj is None:
                    continue
                _add(col_mapping.predicate, obj)

            report.rows_parsed += 1
            report.triples_added += triple_count
        except Exception as exc:
            report.rows_skipped += 1
            msg = f"Row {row_idx} (id={raw_id}): {exc}"
            report.warnings.append(msg)
            if strict:
                raise ValueError(msg) from exc

    return report


def load_all_mappings(directory: str | Path) -> list[MappingDef]:
    """Load all YAML mapping files from a directory, sorted by filename."""
    mappings = []
    for yaml_path in sorted(Path(directory).glob("*.yaml")):
        mappings.append(MappingDef.from_yaml(str(yaml_path)))
    return mappings
