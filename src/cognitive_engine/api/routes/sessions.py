from __future__ import annotations

import json
import logging
import time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cognitive_engine.api.deps import get_session_manager
from cognitive_engine.api.schemas import (
    RunRequest, RunResponse, SessionCreateResponse, SessionInfo, TraceResponse, TraceEntry,
)
from cognitive_engine.core.state import State
from cognitive_engine.core.models import Graph
from cognitive_engine.core.workflow import WorkflowDefinition, WorkflowEngine

logger = logging.getLogger(__name__)

router = APIRouter()


class SessionCreateRequest(BaseModel):
    text: Optional[str] = None


class SessionRunRequest(BaseModel):
    workflow_id: Optional[str] = None
    workflow: Optional[dict] = None
    text: Optional[str] = None
    params: dict = {}


@router.post("", status_code=201)
async def create_session(req: SessionCreateRequest = None) -> SessionCreateResponse:
    mgr = get_session_manager()
    session = mgr.create()
    if req and req.text:
        session.state.graph.source_text = req.text
        session.state.metadata["text"] = req.text
    return SessionCreateResponse(session_id=session.id.hex, status="ready")


@router.get("")
async def list_sessions() -> list[SessionInfo]:
    mgr = get_session_manager()
    return mgr.list_active()


@router.get("/{session_id}")
async def get_session(session_id: str) -> SessionInfo:
    mgr = get_session_manager()
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    session = mgr.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session.snapshot()


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    mgr = get_session_manager()
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    session = mgr.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    del mgr._sessions[sid]
    return {"status": "deleted", "session_id": session_id}


@router.post("/{session_id}/run")
async def run_workflow(session_id: str, req: SessionRunRequest) -> RunResponse:
    mgr = get_session_manager()
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    session = mgr.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    if req.text or (req.params and req.params.get("text")):
        text = req.text or req.params["text"]
        session.state.graph.source_text = text
        session.state.metadata["text"] = text

    workflow_def = None
    if req.workflow_id:
        from cognitive_engine.api.routes.workflows import _workflow_store
        workflow_def = _workflow_store.get(req.workflow_id)
        if workflow_def is None:
            raise HTTPException(status_code=404, detail=f"Workflow '{req.workflow_id}' not found")
    elif req.workflow:
        wf = req.workflow
        if "steps" not in wf and "id" not in wf and "name" not in wf:
            wf = {"steps": wf}
        workflow_def = WorkflowDefinition.from_dict(wf)
    else:
        raise HTTPException(status_code=400, detail="Either workflow_id or workflow body required")

    session.status = "running"

    engine = WorkflowEngine()

    def on_diff(diff):
        session.push_diff(diff)

    try:
        final_state = engine.run_sync(workflow_def, session.state, on_diff=on_diff)
        session.state = final_state
        session.status = "completed"
    except Exception as e:
        logger.exception("Workflow execution failed")
        session.status = "failed"
        raise HTTPException(status_code=500, detail=str(e))

    last_entry = session.trace[-1] if session.trace else None
    return RunResponse(
        session_id=session_id,
        status=session.status,
        step_count=len(session.trace),
        convergence_delta=last_entry.convergence_delta if last_entry else 0.0,
        summary=last_entry.summary if last_entry else "",
    )


@router.get("/{session_id}/trace")
async def get_trace(session_id: str) -> TraceResponse:
    mgr = get_session_manager()
    try:
        sid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID")
    session = mgr.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return TraceResponse(
        session_id=session_id,
        entries=session.trace,
        total_steps=len(session.trace),
        converged=session.status == "completed",
    )
