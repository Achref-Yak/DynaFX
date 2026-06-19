from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from cognitive_engine.api.deps import get_session_manager
from cognitive_engine.core.diff import CycleDiff

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/{session_id}/ws")
async def websocket_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        sid = UUID(session_id)
    except ValueError:
        await websocket.send_json({"error": "Invalid session ID"})
        await websocket.close()
        return

    mgr = get_session_manager()
    session = mgr.get(sid)
    if session is None:
        await websocket.send_json({"error": "Session not found or expired"})
        await websocket.close()
        return

    queue: asyncio.Queue[CycleDiff] = asyncio.Queue()

    def subscriber(diff: CycleDiff):
        asyncio.ensure_future(queue.put(diff))

    session.subscribers.append(subscriber)

    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "status": session.status,
        })

        while True:
            try:
                diff = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json({
                    "type": "diff",
                    "step_id": diff.step_id,
                    "cycle_number": diff.cycle_number,
                    "nodes_added": [n.__dict__ for n in diff.nodes_added],
                    "nodes_removed": diff.nodes_removed,
                    "nodes_modified": [n.__dict__ for n in diff.nodes_modified],
                    "edges_added": [e.__dict__ for e in diff.edges_added],
                    "edges_removed": diff.edges_removed,
                    "edges_modified": [e.__dict__ for e in diff.edges_modified],
                    "contradictions": [c.__dict__ for c in diff.contradictions],
                    "convergence_delta": diff.convergence_delta,
                    "opinion_shifts": {k: [list(v[0]), list(v[1])] for k, v in diff.opinion_shifts.items()},
                    "summary": diff.summary,
                })
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    except Exception:
        logger.exception("WebSocket error for session %s", session_id)
    finally:
        if subscriber in session.subscribers:
            session.subscribers.remove(subscriber)
