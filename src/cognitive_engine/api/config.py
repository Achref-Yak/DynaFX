from __future__ import annotations

import os


class Settings:
    host: str = os.getenv("COGNITIVE_ENGINE_HOST", "0.0.0.0")
    port: int = int(os.getenv("COGNITIVE_ENGINE_PORT", "8000"))
    session_ttl: int = int(os.getenv("COGNITIVE_ENGINE_SESSION_TTL", "3600"))
    max_sessions: int = int(os.getenv("COGNITIVE_ENGINE_MAX_SESSIONS", "100"))
    log_level: str = os.getenv("COGNITIVE_ENGINE_LOG_LEVEL", "info")
    cors_origins: list[str] = os.getenv("COGNITIVE_ENGINE_CORS_ORIGINS", "*").split(",")


settings = Settings()
