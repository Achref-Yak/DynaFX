"""Blackboard agent — shared result pool with optional SQLite persistence.

The SP-agent provides a shared workspace for inter-agent communication.
Supports both in-memory and SQLite-backed persistence for cross-session
state sharing.

Integrates with the global EventBus for model change notifications.
When a value is published, both the internal key-based subscribers
and the global EventBus are notified.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cognitive_engine.core.events import Event, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class BlackboardEntry:
    """A single entry on the blackboard."""
    key: str
    value: Any
    publisher: str
    timestamp: float
    session_id: str


class BlackboardAgent:
    """SP-agent: shared result pool with optional SQLite persistence.

    Provides publish/subscribe communication between agents.
    When db_path is provided, entries are persisted to SQLite for
    cross-session state sharing.
    """

    def __init__(
        self, db_path: Optional[str] = None, session_id: str = "",
    ) -> None:
        self._db_path = db_path
        self._session_id = session_id
        self._memory: dict[str, list[BlackboardEntry]] = {}
        self._subscribers: dict[str, list[Callable]] = {}
        self._conn: Optional[sqlite3.Connection] = None

        if db_path:
            self._init_db()

    def _init_db(self) -> None:
        """Create blackboard table if needed."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS blackboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                publisher TEXT NOT NULL,
                timestamp REAL NOT NULL,
                session_id TEXT,
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bb_key ON blackboard(key)
        """)
        self._conn.commit()

    def publish(self, key: str, value: Any, publisher: str = "") -> None:
        """Publish a value to the blackboard.

        Also publishes a 'blackboard_updated' event to the global EventBus.
        """
        entry = BlackboardEntry(
            key=key,
            value=value,
            publisher=publisher,
            timestamp=time.time(),
            session_id=self._session_id,
        )
        self._memory.setdefault(key, []).append(entry)

        if self._conn:
            self._conn.execute(
                "INSERT INTO blackboard (key, value_json, publisher, timestamp, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, json.dumps(value, default=str), publisher, entry.timestamp, self._session_id),
            )
            self._conn.commit()

        # Notify internal subscribers
        for callback in self._subscribers.get(key, []):
            try:
                callback(value)
            except Exception:
                pass

        # Publish to global EventBus
        try:
            bus = get_event_bus()
            bus.publish(Event(
                event_type="blackboard_updated",
                source=self,
                data={"key": key, "value": value, "publisher": publisher},
            ))
        except Exception:
            pass

    def get(self, key: str, latest: bool = True) -> Optional[Any]:
        """Get value(s) from blackboard."""
        entries = self._memory.get(key, [])
        if not entries:
            return None
        if latest:
            return entries[-1].value
        return [e.value for e in entries]

    def get_all(self) -> dict[str, Any]:
        """Get latest value for each key."""
        return {key: entries[-1].value for key, entries in self._memory.items() if entries}

    def subscribe(self, key: str, callback: Callable) -> None:
        """Subscribe to updates on a key."""
        self._subscribers.setdefault(key, []).append(callback)

    def query(
        self,
        key: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query blackboard entries."""
        if self._conn:
            sql = "SELECT key, value_json, publisher, timestamp FROM blackboard WHERE 1=1"
            params: list[Any] = []
            if key:
                sql += " AND key = ?"
                params.append(key)
            if since:
                sql += " AND timestamp >= ?"
                params.append(since)
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            cursor = self._conn.execute(sql, params)
            return [
                {"key": r[0], "value": json.loads(r[1]), "publisher": r[2], "timestamp": r[3]}
                for r in cursor.fetchall()
            ]
        else:
            results: list[dict] = []
            for k, entries in self._memory.items():
                if key and k != key:
                    continue
                for e in entries:
                    if since and e.timestamp < since:
                        continue
                    results.append({
                        "key": k,
                        "value": e.value,
                        "publisher": e.publisher,
                        "timestamp": e.timestamp,
                    })
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            return results[:limit]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
