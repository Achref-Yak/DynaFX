from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID, uuid4


@dataclass
class SessionInfo:
    id: UUID
    created_at: float
    last_active: float
    status: str
    node_count: int
    edge_count: int
    step_count: int


@dataclass
class SessionCreateResponse:
    session_id: str
    status: str = "ready"


@dataclass
class RunRequest:
    workflow_id: Optional[str] = None
    workflow: Optional[dict] = None
    text: Optional[str] = None
    params: dict = field(default_factory=dict)


@dataclass
class RunResponse:
    session_id: str
    status: str
    step_count: int
    convergence_delta: float = 0.0
    summary: str = ""


@dataclass
class PrimitiveCatalog:
    id: str
    name: str
    kind: str
    description: str
    params_schema: dict


@dataclass
class WorkflowSummary:
    id: str
    name: str
    description: str
    step_count: int
    metadata: dict


@dataclass
class TraceEntry:
    step_id: str
    cycle_number: int
    operator_name: str
    summary: str
    convergence_delta: float
    timestamp: float
    diff: Optional[dict] = None


@dataclass
class TraceResponse:
    session_id: str
    entries: list[TraceEntry]
    total_steps: int
    converged: bool
