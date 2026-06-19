from __future__ import annotations

import logging
import sys

import uvicorn

from cognitive_engine.api.config import settings
from cognitive_engine.api.server import create_app

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(levelname)s %(name)s %(message)s",
)

app = create_app()


def run() -> None:
    uvicorn.run(
        "cognitive_engine.api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    run()
