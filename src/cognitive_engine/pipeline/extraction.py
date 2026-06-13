from __future__ import annotations

import logging
from typing import Optional

from cognitive_engine.core.models import Graph, ReasoningMode
from cognitive_engine.pipeline.pipeline import run as deterministic_run

logger = logging.getLogger(__name__)


def extract_graph(
    text: str,
    max_tokens: int = 512,
    overlap: int = 128,
    merge_margin: int = 20,
    mode: ReasoningMode = ReasoningMode.ARGUMENT,
) -> Graph:
    return deterministic_run(text, max_tokens=max_tokens, overlap=overlap, merge_margin=merge_margin, mode=mode)
