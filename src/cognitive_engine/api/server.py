from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cognitive_engine.api.config import settings
from cognitive_engine.api.routes.primitives import router as primitives_router
from cognitive_engine.api.routes.workflows import router as workflows_router
from cognitive_engine.api.routes.sessions import router as sessions_router
from cognitive_engine.api.routes.stream import router as stream_router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cognitive Engine API",
        version="0.1.0",
        description="Reasoning engine with composable operator workflows",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(primitives_router, prefix="/primitives", tags=["primitives"])
    app.include_router(workflows_router, prefix="/workflows", tags=["workflows"])
    app.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
    app.include_router(stream_router, prefix="/sessions", tags=["stream"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
