from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from cognitive_engine.core.config import Priors
from cognitive_engine.core.models import EvidenceCounts, Graph, Opinion
from cognitive_engine.domain import domain as _domain
from cognitive_engine.reason.evidence import (
    mean_opinion,
    mean_opinion_pair,
    opinion_from_counts,
)

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL DEFAULT '',
    processed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS node_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    node_type TEXT NOT NULL,
    positive INTEGER NOT NULL DEFAULT 0,
    negative INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edge_warrants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    b1 REAL NOT NULL DEFAULT 0.5,
    d1 REAL NOT NULL DEFAULT 0.0,
    u1 REAL NOT NULL DEFAULT 0.5,
    a1 REAL NOT NULL DEFAULT 0.5,
    b2 REAL NOT NULL DEFAULT 0.5,
    d2 REAL NOT NULL DEFAULT 0.0,
    u2 REAL NOT NULL DEFAULT 0.5,
    a2 REAL NOT NULL DEFAULT 0.5
);

CREATE INDEX IF NOT EXISTS idx_node_counts_source
    ON node_counts(source_id);
CREATE INDEX IF NOT EXISTS idx_node_counts_type
    ON node_counts(node_type);
CREATE INDEX IF NOT EXISTS idx_edge_warrants_source
    ON edge_warrants(source_id);
CREATE INDEX IF NOT EXISTS idx_edge_warrants_type
    ON edge_warrants(edge_type);
"""


class CorpusStore:
    """SQLite-backed persistent evidence store.

    Accumulates node-type counts and edge warrants across multiple
    graph runs so priors can be learned incrementally.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CorpusStore:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _source_exists(self, source_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sources WHERE id = ?", (source_id,)
        ).fetchone()
        return row is not None

    def store_graph(
        self,
        graph: Graph,
        source_id: str,
        filename: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO sources (id, filename, processed_at) VALUES (?, ?, ?)",
            (source_id, filename, now),
        )

        for node in graph.nodes.values():
            b, d, _, _ = node.opinion
            pos = 1 if b > d + _domain.active().opinion_positive_threshold else 0
            neg = 1 if d + _domain.active().opinion_positive_threshold >= b else 0
            self._conn.execute(
                "INSERT INTO node_counts (source_id, node_type, positive, negative) VALUES (?, ?, ?, ?)",
                (source_id, node.type.name, pos, neg),
            )

        for edge in graph.edges:
            if edge.warrant is None:
                continue
            (b1, d1, u1, a1), (b2, d2, u2, a2) = edge.warrant
            self._conn.execute(
                "INSERT INTO edge_warrants (source_id, edge_type, b1, d1, u1, a1, b2, d2, u2, a2) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source_id, edge.type.name, b1, d1, u1, a1, b2, d2, u2, a2),
            )

        self._conn.commit()

    def accumulate_node_counts(self) -> dict[str, EvidenceCounts]:
        rows = self._conn.execute(
            "SELECT node_type, SUM(positive) AS pos, SUM(negative) AS neg "
            "FROM node_counts GROUP BY node_type ORDER BY node_type"
        ).fetchall()
        w = _domain.active().uncertainty_pseudocount
        result: dict[str, EvidenceCounts] = {}
        for row in rows:
            result[row["node_type"]] = EvidenceCounts(
                positive=int(row["pos"]),
                negative=int(row["neg"]),
                uncertainty_pseudocount=w,
            )
        return result

    def accumulate_edge_warrants(self) -> dict[str, list[tuple[Opinion, Opinion]]]:
        rows = self._conn.execute(
            "SELECT edge_type, b1, d1, u1, a1, b2, d2, u2, a2 "
            "FROM edge_warrants ORDER BY edge_type"
        ).fetchall()
        result: dict[str, list[tuple[Opinion, Opinion]]] = defaultdict(list)
        for row in rows:
            op1: Opinion = (row["b1"], row["d1"], row["u1"], row["a1"])
            op2: Opinion = (row["b2"], row["d2"], row["u2"], row["a2"])
            result[row["edge_type"]].append((op1, op2))
        return dict(result)

    def to_priors(self, w: float | None = None) -> Priors:
        if w is None:
            w = _domain.active().uncertainty_pseudocount

        cfg = _domain.active()
        node_type_map = cfg.source_type_map

        source_type_map: dict[str, str] = {}
        learned_opinions: dict[str, Opinion] = {}
        node_counts = self.accumulate_node_counts()
        for node_type_name, template_name in node_type_map.items():
            counts = node_counts.get(node_type_name)
            if counts is not None and (counts.positive > 0 or counts.negative > 0):
                ec = EvidenceCounts(
                    positive=counts.positive,
                    negative=counts.negative,
                    uncertainty_pseudocount=w,
                )
                learned_opinions[template_name] = opinion_from_counts(ec)
            source_type_map[node_type_name] = template_name

        learned_opinions["total_ignorance"] = cfg.total_ignorance
        for template in list(cfg.default_opinions):
            if template not in learned_opinions:
                learned_opinions[template] = cfg.default_opinions[template]

        learned_warrants: dict[str, tuple[Opinion, Opinion]] = {}
        edge_warrants = self.accumulate_edge_warrants()
        for edge_type_name, pairs in edge_warrants.items():
            if pairs:
                learned_warrants[edge_type_name] = mean_opinion_pair(pairs)

        base = Priors()
        warrants = learned_warrants if learned_warrants else base.edge_warrants
        if "SUPPORTS" not in warrants:
            warrants["SUPPORTS"] = base.edge_warrants["SUPPORTS"]
        for et_name in base.edge_warrants:
            if et_name not in warrants:
                warrants[et_name] = base.edge_warrants[et_name]

        return Priors(
            default_opinions=learned_opinions,
            source_type_map=source_type_map,
            edge_warrants=warrants,
            default_warrant=base.default_warrant,
        )

    def list_sources(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, filename, processed_at FROM sources ORDER BY processed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def remove_source(self, source_id: str) -> None:
        self._conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        self._conn.commit()

    def clear(self) -> None:
        self._conn.executescript("DELETE FROM edge_warrants; DELETE FROM node_counts; DELETE FROM sources;")
        self._conn.commit()

    @property
    def source_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM sources").fetchone()
        return int(row["cnt"]) if row else 0
