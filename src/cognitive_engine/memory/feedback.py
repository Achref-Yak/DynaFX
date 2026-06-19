"""Feedback operator — SL opinion fusion for relevancy and trust.

Uses Beta distribution belief updates for combining feedback from
multiple sources. Integrates with InferenceCycle for iterative
belief refinement.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from cognitive_engine.core.math import cumulative_fusion, opinion_from_counts

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    node_id: str
    source: str
    belief: float
    uncertainty: float
    timestamp: float
    session_id: str
    cycle: int


class FeedbackStore:
    """SL opinion fusion store for feedback aggregation.

    Uses Beta distribution (cumulative_fusion) to combine feedback
    from multiple sources. Supports time-decay weighting.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._entries: dict[str, list[FeedbackEntry]] = {}

        if db_path:
            self._init_db()

    def _init_db(self) -> None:
        """Create feedback table."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                source TEXT NOT NULL,
                belief REAL NOT NULL,
                uncertainty REAL NOT NULL,
                timestamp REAL NOT NULL,
                session_id TEXT,
                cycle INTEGER,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fb_node ON feedback(node_id)"
        )
        self._conn.commit()

    def add(
        self,
        node_id: str,
        source: str,
        belief: float,
        uncertainty: float,
        timestamp: float,
        session_id: str = "",
        cycle: int = 0,
    ) -> None:
        """Record a feedback entry."""
        entry = FeedbackEntry(
            node_id=node_id,
            source=source,
            belief=belief,
            uncertainty=uncertainty,
            timestamp=timestamp,
            session_id=session_id,
            cycle=cycle,
        )
        self._entries.setdefault(node_id, []).append(entry)

        if self._conn:
            self._conn.execute(
                "INSERT INTO feedback "
                "(node_id, source, belief, uncertainty, timestamp, session_id, cycle) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (node_id, source, belief, uncertainty, timestamp, session_id, cycle),
            )
            self._conn.commit()

    def fuse(self, node_id: str) -> Optional[tuple[float, float, float, float]]:
        """Fuse all feedback for a node using SL cumulative fusion.

        Returns (belief, disbelief, uncertainty, apriori) opinion tuple,
        or None if no feedback exists.
        """
        entries = self._entries.get(node_id, [])
        if not entries and self._conn:
            cursor = self._conn.execute(
                "SELECT belief, uncertainty, timestamp FROM feedback "
                "WHERE node_id = ? ORDER BY timestamp",
                (node_id,),
            )
            rows = cursor.fetchall()
            if rows:
                entries = [
                    FeedbackEntry(
                        node_id=node_id,
                        source="",
                        belief=r[0],
                        uncertainty=r[1],
                        timestamp=r[2],
                        session_id="",
                        cycle=0,
                    )
                    for r in rows
                ]

        if not entries:
            return None

        # Weight by recency (time decay)
        if len(entries) == 1:
            e = entries[0]
            return (e.belief, 1.0 - e.belief - e.uncertainty, e.uncertainty, 0.5)

        # Sort by timestamp, apply cumulative fusion
        entries.sort(key=lambda e: e.timestamp)
        fused = None
        for entry in entries:
            opinion = opinion_from_counts(
                belief=entry.belief * (1 - entry.uncertainty),
                disbelief=(1 - entry.belief) * (1 - entry.uncertainty),
                sample_size=10,
            )
            if fused is None:
                fused = opinion
            else:
                fused = cumulative_fusion(fused, opinion)

        return fused

    def fuse_all(self) -> dict[str, tuple[float, float, float, float]]:
        """Fuse feedback for all nodes with entries."""
        node_ids = set(self._entries.keys())
        if self._conn:
            cursor = self._conn.execute(
                "SELECT DISTINCT node_id FROM feedback"
            )
            node_ids.update(r[0] for r in cursor.fetchall())

        results = {}
        for nid in node_ids:
            opinion = self.fuse(nid)
            if opinion:
                results[nid] = opinion
        return results

    def get_entries(
        self, node_id: str, limit: int = 100,
    ) -> list[FeedbackEntry]:
        """Get raw entries for a node."""
        entries = self._entries.get(node_id, [])
        if not entries and self._conn:
            cursor = self._conn.execute(
                "SELECT node_id, source, belief, uncertainty, timestamp, session_id, cycle "
                "FROM feedback WHERE node_id = ? ORDER BY timestamp DESC LIMIT ?",
                (node_id, limit),
            )
            return [
                FeedbackEntry(
                    node_id=r[0], source=r[1], belief=r[2], uncertainty=r[3],
                    timestamp=r[4], session_id=r[5] or "", cycle=r[6] or 0,
                )
                for r in cursor.fetchall()
            ]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
