from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from cognitive_engine.core.concept import Provenance
from cognitive_engine.core.models import Graph
from cognitive_engine.core.state import State
from cognitive_engine.memory.models import Fact, LTMPattern
from cognitive_engine.memory.retrieval import retrieve_similar


class MemoryStore:
    """LTM ⊣ STM adjunction (spec §8) with SQLite persistence.

    Short-term memory is an in-memory deque of recent states.
    Long-term memory persists patterns to SQLite.
    FactStore provides cross-session fact persistence with SCD Type 2.

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
        session_id: str = "",
    ):
        self.stm: deque[State] = deque(maxlen=stm_capacity)
        self.session_id = session_id or uuid.uuid4().hex
        self._conn = sqlite3.connect(str(ltm_path))
        self._init_db()

        # Initialize FactStore for cross-session persistence
        from cognitive_engine.memory.fact_store import FactStore
        self.fact_store = FactStore(
            db_path=ltm_path if ltm_path != ":memory:" else ":memory:",
            session_id=self.session_id,
        )

    def _init_db(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS ltm_patterns ("
            "  id TEXT PRIMARY KEY,"
            "  graph_json TEXT,"
            "  belief_signature TEXT,"
            "  operator_trace TEXT,"
            "  cluster_labels TEXT,"
            "  frequency INTEGER DEFAULT 1,"
            "  last_accessed REAL,"
            "  session_id TEXT,"
            "  community_id INTEGER"
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

        # Extract and persist facts from the state
        self._extract_and_store_facts(state)

        if len(self.stm) == self.stm.maxlen:
            self.consolidate()

    def _extract_and_store_facts(self, state: State) -> None:
        """Extract facts from state graph nodes and persist to FactStore."""
        for nid, node in state.graph.nodes.items():
            # Skip nodes without meaningful text
            if not node.text or len(node.text.strip()) < 3:
                continue

            # Determine concept from node metadata or type
            concept = node.metadata.get("concept", node.type.name)

            # Determine provenance from node metadata
            prov_name = node.metadata.get("provenance", "AGENT_OBSERVED")
            try:
                provenance = Provenance[prov_name]
            except KeyError:
                provenance = Provenance.AGENT_OBSERVED

            # Create fact
            fact = Fact(
                id=uuid4(),
                concept=concept,
                value=node.text,
                original_text=node.text,
                provenance=provenance,
                confidence=node.opinion[0] if node.opinion else 0.5,
                valid_from=time.time(),
                session_id=self.session_id,
                node_id=nid,
            )

            self.fact_store.store(fact)

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
        """STM → LTM: compress STM trace into LTM patterns using Leiden."""
        if not self.stm:
            return 0
        from cognitive_engine.memory.consolidate import build_pattern
        patterns = build_pattern(self.stm, session_id=self.session_id)
        for pattern in patterns:
            self._insert_pattern(pattern)
        self.stm.clear()
        return len(patterns)

    def query_facts(
        self,
        concept: Optional[str] = None,
        provenance_min: Optional[Provenance] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[Fact]:
        """Query facts from FactStore using structured retrieval."""
        return self.fact_store.query(
            concept=concept,
            provenance_min=provenance_min,
            active_only=active_only,
            limit=limit,
        )

    def close(self) -> None:
        self._conn.close()
        self.fact_store.close()

    def _similar(self, graph: Graph, k: int) -> list[LTMPattern]:
        """Find top-k similar patterns by node count proximity."""
        rows = self._conn.execute(
            "SELECT id, graph_json, belief_signature, operator_trace, "
            "cluster_labels, frequency, last_accessed, session_id, community_id "
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
                session_id=row[7] or "",
                community_id=row[8],
            )
            candidates.append(p)

        scored = retrieve_similar(graph, candidates, k)
        return scored

    def _insert_pattern(self, pattern: LTMPattern) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO ltm_patterns "
            "(id, graph_json, belief_signature, operator_trace, "
            "cluster_labels, frequency, last_accessed, session_id, community_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(pattern.id),
                json.dumps(pattern.graph_snapshot.to_dict()),
                json.dumps(pattern.belief_signature),
                json.dumps(pattern.operator_trace),
                json.dumps(pattern.cluster_labels),
                pattern.frequency,
                pattern.last_accessed or time.time(),
                pattern.session_id,
                pattern.community_id,
            ),
        )
        self._conn.commit()

    def _update_pattern(self, pattern: LTMPattern) -> None:
        self._conn.execute(
            "UPDATE ltm_patterns SET frequency=?, last_accessed=? WHERE id=?",
            (pattern.frequency, pattern.last_accessed, str(pattern.id)),
        )
        self._conn.commit()
