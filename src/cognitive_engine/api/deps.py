from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from cognitive_engine.api.config import settings
from cognitive_engine.api.schemas import SessionInfo, TraceEntry
from cognitive_engine.core.diff import CycleDiff
from cognitive_engine.core.state import State
from cognitive_engine.core.models import Graph
from cognitive_engine.core.workflow import (
    WorkflowDefinition, WorkflowEngine, PrimitiveRegistry,
)

logger = logging.getLogger(__name__)


class SessionState:
    def __init__(self, session_id: UUID):
        self.id = session_id
        self.created_at = time.time()
        self.last_active = time.time()
        self.state = State(graph=Graph())
        self.status = "ready"
        self.trace: list[TraceEntry] = []
        self.subscribers: list[Callable[[CycleDiff], None]] = []
        self._workflow_engine = WorkflowEngine(PrimitiveRegistry.get_instance())

    def touch(self) -> None:
        self.last_active = time.time()

    def is_expired(self, ttl: int) -> bool:
        return (time.time() - self.last_active) > ttl

    def push_diff(self, diff: CycleDiff) -> None:
        entry = TraceEntry(
            step_id=diff.step_id,
            cycle_number=diff.cycle_number,
            operator_name=diff.step_id,
            summary=diff.summary,
            convergence_delta=diff.convergence_delta,
            timestamp=time.time(),
            diff={
                "nodes_added": [n.id for n in diff.nodes_added],
                "nodes_removed": diff.nodes_removed,
                "nodes_modified": [n.id for n in diff.nodes_modified],
                "edges_added": [e.id for e in diff.edges_added],
                "edges_removed": diff.edges_removed,
                "edges_modified": [e.id for e in diff.edges_modified],
                "contradictions": len(diff.contradictions),
                "opinion_shifts": len(diff.opinion_shifts),
            },
        )
        self.trace.append(entry)

        for sub in self.subscribers:
            try:
                sub(diff)
            except Exception:
                logger.warning("Subscriber failed, removing")
                self.subscribers.remove(sub)

    def snapshot(self) -> SessionInfo:
        self.touch()
        return SessionInfo(
            id=self.id,
            created_at=self.created_at,
            last_active=self.last_active,
            status=self.status,
            node_count=len(self.state.graph.nodes),
            edge_count=len(self.state.graph.edges),
            step_count=len(self.trace),
        )


class SessionManager:
    def __init__(self):
        self._sessions: dict[UUID, SessionState] = {}
        self._ttl = settings.session_ttl
        self._max_sessions = settings.max_sessions

    def create(self) -> SessionState:
        self._evict_expired()
        if len(self._sessions) >= self._max_sessions:
            oldest = min(self._sessions.keys(), key=lambda k: self._sessions[k].last_active)
            del self._sessions[oldest]
        sid = uuid4()
        session = SessionState(sid)
        self._sessions[sid] = session
        return session

    def get(self, session_id: UUID) -> Optional[SessionState]:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired(self._ttl):
            del self._sessions[session_id]
            return None
        session.touch()
        return session

    def list_active(self) -> list[SessionInfo]:
        self._evict_expired()
        return [s.snapshot() for s in self._sessions.values()]

    def _evict_expired(self) -> None:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired(self._ttl)]
        for sid in expired:
            del self._sessions[sid]


_sessions: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global _sessions
    if _sessions is None:
        _sessions = SessionManager()
    return _sessions
