from __future__ import annotations

import sqlite3
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StateDelta:
    timestamp: float
    operator: str
    description: str
    node_count: int = 0
    edge_count: int = 0
    effect_type: str = "deterministic"
    metadata: dict = field(default_factory=dict)


class TraceBuffer:
    """Formal monoid Σ: concatenation of cognitive effect traces.

    Stores StateDelta entries with bounded capacity and optional
    SQLite persistence. Supports monoid concatenation via __add__.

    Properties:
        - Capacity-bounded (configurable maxlen)
        - Iterable, indexable, length-reporting
        - Optional SQLite persistence when db_path provided
        - Monoid append/merge via __add__
    """

    def __init__(
        self,
        capacity: int = 100,
        db_path: Optional[str] = None,
    ):
        self._entries: deque[StateDelta] = deque(maxlen=capacity)
        self._capacity = capacity
        self._db: Optional[sqlite3.Connection] = None
        if db_path is not None:
            p = Path(db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(p))
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS trace ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  timestamp REAL,"
                "  operator TEXT,"
                "  description TEXT,"
                "  node_count INTEGER,"
                "  edge_count INTEGER,"
                "  effect_type TEXT,"
                "  metadata TEXT"
                ")"
            )
            self._db.commit()

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, entry: StateDelta) -> None:
        self._entries.append(entry)
        if self._db is not None:
            import json
            self._db.execute(
                "INSERT INTO trace (timestamp, operator, description, "
                "node_count, edge_count, effect_type, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.timestamp,
                    entry.operator,
                    entry.description,
                    entry.node_count,
                    entry.edge_count,
                    entry.effect_type,
                    json.dumps(entry.metadata),
                ),
            )
            self._db.commit()

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            result = TraceBuffer(capacity=self._capacity)
            entries = list(self._entries)[index]
            for e in entries:
                result._entries.append(e)
            return result
        return list(self._entries)[index]

    def __add__(self, other: TraceBuffer) -> TraceBuffer:
        merged = TraceBuffer(
            capacity=max(self._capacity, other._capacity),
        )
        for e in self._entries:
            merged._entries.append(e)
        for e in other._entries:
            merged._entries.append(e)
        return merged

    def copy(self) -> TraceBuffer:
        cp = TraceBuffer(capacity=self._capacity)
        cp._entries = deque(self._entries, maxlen=self._capacity)
        return cp

    @property
    def llm_call_count(self) -> int:
        return sum(1 for e in self._entries if e.effect_type == "llm")

    @property
    def total_tokens(self) -> int:
        return sum(
            e.metadata.get("tokens", 0)
            for e in self._entries
            if e.effect_type == "llm"
        )
