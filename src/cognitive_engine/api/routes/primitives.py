from __future__ import annotations

from fastapi import APIRouter

from cognitive_engine.api.schemas import PrimitiveCatalog
from cognitive_engine.core.workflow import PrimitiveRegistry

router = APIRouter()


@router.get("")
async def list_primitives() -> list[PrimitiveCatalog]:
    registry = PrimitiveRegistry.get_instance()
    return [
        PrimitiveCatalog(
            id=p.id,
            name=p.name,
            kind=p.kind,
            description=p.description,
            params_schema=p.params_schema,
        )
        for p in registry.all()
    ]


@router.get("/{kind}")
async def list_primitives_by_kind(kind: str) -> list[PrimitiveCatalog]:
    registry = PrimitiveRegistry.get_instance()
    return [
        PrimitiveCatalog(
            id=p.id,
            name=p.name,
            kind=p.kind,
            description=p.description,
            params_schema=p.params_schema,
        )
        for p in registry.list_by_kind(kind)
    ]
