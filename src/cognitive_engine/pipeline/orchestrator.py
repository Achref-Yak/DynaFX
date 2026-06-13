from __future__ import annotations

import logging
from typing import Optional

from cognitive_engine.core.config import Priors, load_priors
from cognitive_engine.pipeline.extraction import extract_graph
from cognitive_engine.core.models import ConversationTree, Graph, ReasoningMode, Severity
from cognitive_engine.reason.modes import apply_mode, compute_mode_views
from cognitive_engine.reason.sl_operators import compute_opinions
from cognitive_engine.reason.validators import validate_all

logger = logging.getLogger(__name__)


def run(
    text: str,
    config_path: Optional[str] = None,
    mode: Optional[str] = None,
    max_tokens: int = 512,
    overlap: int = 128,
    merge_margin: int = 20,
) -> Graph:
    priors = load_priors(config_path)

    graph = extract_graph(text, max_tokens=max_tokens, overlap=overlap, merge_margin=merge_margin)

    compute_opinions(graph, priors)

    violations = validate_all(graph)
    errors = [v for v in violations if v.severity == Severity.ERROR]
    if errors:
        logger.warning("%d validation error(s) found", len(errors))
        for v in errors:
            logger.warning("  %s", v.description)

    compute_mode_views(graph)

    if mode is not None:
        resolved = ReasoningMode[mode.upper()]
        graph = apply_mode(graph, resolved)
        compute_opinions(graph, priors)

    graph.cta = ConversationTree.from_graph(graph)

    return graph
