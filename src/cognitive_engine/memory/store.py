from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from cognitive_engine.core.models import Graph
from cognitive_engine.core.state import State
from cognitive_engine.memory.models import LTMPattern
from cognitive_engine.memory.retrieval import retrieve_similar


class MemoryStore:
    """LTM ⊣ STM adjunction (spec §8) with SQLite persistence.

    Short-term memory is an in-memory deque of recent states.
    Long-term memory persists patterns to SQLite.

    Usage:
        mem = MemoryStore("memory.db")
        mem.store(state)           # push to STM, auto-evict to LTM
        state = mem.retrieve(state)  # LTM → STM augmentation
        count = mem.consolidate()     # STM → LTM pattern extraction
    """

    def __init__(
        self,
        ltm_path: str = ":memory:",
        stm_capacity: int = 10,
    ):
        self.stm: deque[State] = deque(maxlen=stm_capacity)
        self._conn = sqlite3.connect(str(ltm_path))
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS ltm_patterns ("
            "  id TEXT PRIMARY KEY,"
            "  graph_json TEXT,"
            "  belief_signature TEXT,"
            "  operator_trace TEXT,"
            "  cluster_labels TEXT,"
            "  frequency INTEGER DEFAULT 1,"
            "  last_accessed REAL"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS node_embeddings ("
            "  node_id TEXT PRIMARY KEY,"
            "  embedding TEXT,"
            "  created_at REAL"
            ")"
        )
        self._conn.commit()

    def __len__(self) -> int:
        rows = self._conn.execute("SELECT COUNT(*) FROM ltm_patterns").fetchone()
        return rows[0] if rows else 0

    def __call__(self, state: State) -> State:
        self.store(state)
        return self.retrieve(state)

    def store(self, state: State) -> None:
        """Push state into STM; auto-consolidate to LTM when full."""
        self.stm.append(state)
        if len(self.stm) == self.stm.maxlen:
            self.consolidate()

    def retrieve(self, state: State, k: int = 3) -> State:
        """LTM → STM: find similar patterns and augment current graph."""
        patterns = self._similar(state.graph, k)
        for p in patterns:
            for nid, node in p.graph_snapshot.nodes.items():
                if nid not in state.graph.nodes:
                    state.graph.nodes[nid] = node
            p.frequency += 1
            p.last_accessed = time.time()
            self._update_pattern(p)
        state.metadata["ltm_retrieved"] = len(patterns)
        if patterns:
            state.metadata["ltm_pattern_ids"] = [str(p.id) for p in patterns]
        return state

    def consolidate(self) -> int:
        """STM → LTM: compress STM trace into an LTM pattern."""
        if not self.stm:
            return 0
        from cognitive_engine.memory.consolidate import build_pattern
        pattern = build_pattern(self.stm)
        self._insert_pattern(pattern)
        self.stm.clear()
        return 1

    def store_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Persist a node embedding vector to the DB for out-of-band retrieval."""
        self._conn.execute(
            "INSERT OR REPLACE INTO node_embeddings (node_id, embedding, created_at) VALUES (?, ?, ?)",
            (node_id, json.dumps(embedding), time.time()),
        )
        self._conn.commit()

    def get_embedding(self, node_id: str) -> Optional[list[float]]:
        """Retrieve a previously stored embedding vector by node ID."""
        row = self._conn.execute(
            "SELECT embedding FROM node_embeddings WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self._conn.close()

    def _similar(self, graph: Graph, k: int) -> list[LTMPattern]:
        """Find top-k similar patterns by node count proximity."""
        rows = self._conn.execute(
            "SELECT id, graph_json, belief_signature, operator_trace, "
            "cluster_labels, frequency, last_accessed "
            "FROM ltm_patterns ORDER BY last_accessed DESC LIMIT 100"
        ).fetchall()

        candidates = []
        for row in rows:
            p = LTMPattern(
                id=UUID(row[0]),
                graph_snapshot=Graph.from_dict(json.loads(row[1])),
                belief_signature=json.loads(row[2]) if row[2] else {},
                operator_trace=json.loads(row[3]) if row[3] else [],
                cluster_labels=json.loads(row[4]) if row[4] else [],
                frequency=row[5],
                last_accessed=row[6],
            )
            candidates.append(p)

        scored = retrieve_similar(graph, candidates, k)
        return scored

    def _insert_pattern(self, pattern: LTMPattern) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ltm_patterns "
            "(id, graph_json, belief_signature, operator_trace, "
            "cluster_labels, frequency, last_accessed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(pattern.id),
                json.dumps(pattern.graph_snapshot.to_dict()),
                json.dumps(pattern.belief_signature),
                json.dumps(pattern.operator_trace),
                json.dumps(pattern.cluster_labels),
                pattern.frequency,
                pattern.last_accessed or time.time(),
            ),
        )
        self._conn.commit()

    def _update_pattern(self, pattern: LTMPattern) -> None:
        self._conn.execute(
            "UPDATE ltm_patterns SET frequency=?, last_accessed=? WHERE id=?",
            (pattern.frequency, pattern.last_accessed, str(pattern.id)),
        )
        self._conn.commit()
