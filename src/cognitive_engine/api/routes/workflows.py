from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cognitive_engine.api.schemas import WorkflowSummary
from cognitive_engine.core.workflow import WorkflowDefinition

logger = logging.getLogger(__name__)

router = APIRouter()

_workflow_store: dict[str, WorkflowDefinition] = {}


class WorkflowCreateRequest(BaseModel):
    id: str
    name: str
    description: str = ""
    steps: list[dict] | dict
    entry_points: list[str] = []
    metadata: dict = {}


class WorkflowUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[list[dict] | dict] = None
    entry_points: Optional[list[str]] = None
    metadata: Optional[dict] = None


@router.get("")
async def list_workflows() -> list[WorkflowSummary]:
    return [
        WorkflowSummary(
            id=wd.id,
            name=wd.name,
            description=wd.description,
            step_count=len(wd.steps),
            metadata=wd.metadata,
        )
        for wd in _workflow_store.values()
    ]


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str) -> dict:
    wd = _workflow_store.get(workflow_id)
    if wd is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    return wd.to_dict()


@router.post("", status_code=201)
async def create_workflow(req: WorkflowCreateRequest) -> WorkflowSummary:
    if req.id in _workflow_store:
        raise HTTPException(status_code=409, detail=f"Workflow '{req.id}' already exists")
    data = {
        "id": req.id,
        "name": req.name,
        "description": req.description,
        "steps": req.steps,
        "entry_points": req.entry_points,
        "metadata": req.metadata,
    }
    wd = WorkflowDefinition.from_dict(data)
    _workflow_store[wd.id] = wd
    return WorkflowSummary(
        id=wd.id,
        name=wd.name,
        description=wd.description,
        step_count=len(wd.steps),
        metadata=wd.metadata,
    )


@router.put("/{workflow_id}")
async def update_workflow(workflow_id: str, req: WorkflowUpdateRequest) -> WorkflowSummary:
    wd = _workflow_store.get(workflow_id)
    if wd is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    if req.name is not None:
        wd.name = req.name
    if req.description is not None:
        wd.description = req.description
    if req.steps is not None:
        wd = WorkflowDefinition.from_dict({
            "id": wd.id,
            "name": wd.name,
            "description": wd.description,
            "steps": req.steps,
            "entry_points": req.entry_points or wd.entry_points,
            "metadata": req.metadata or wd.metadata,
        })
    if req.entry_points is not None:
        wd.entry_points = req.entry_points
    if req.metadata is not None:
        wd.metadata = req.metadata
    _workflow_store[workflow_id] = wd
    return WorkflowSummary(
        id=wd.id,
        name=wd.name,
        description=wd.description,
        step_count=len(wd.steps),
        metadata=wd.metadata,
    )


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str) -> dict:
    if workflow_id not in _workflow_store:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    del _workflow_store[workflow_id]
    return {"status": "deleted", "id": workflow_id}
