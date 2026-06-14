from __future__ import annotations

import logging
from typing import Optional

from cognitive_engine.core.config import Priors, load_priors
from cognitive_engine.core.models import ConversationTree, EdgeType, Graph, ReasoningMode, Severity
from cognitive_engine.reason.modes import apply_mode, compute_mode_views
from cognitive_engine.reason.mode_operators import apply_mode_operator
from cognitive_engine.reason.sl_operators import compute_opinions
from cognitive_engine.reason.validators import validate_all

logger = logging.getLogger(__name__)


def _break_cycles(graph: Graph) -> None:
    """Remove minimum positive edges to make CTA parent subgraph acyclic."""
    import networkx as nx

    positive_edges = [
        e for e in graph.edges
        if e.type in (EdgeType.INFERS, EdgeType.SUPPORTS, EdgeType.JUSTIFIES)
    ]
    if not positive_edges:
        return

    nx_graph = nx.DiGraph()
    for e in positive_edges:
        nx_graph.add_edge(e.source_id.hex, e.target_id.hex)

    try:
        list(nx.topological_sort(nx_graph))
        return
    except nx.NetworkXUnfeasible:
        pass

    to_remove = nx.minimum_feedback_arc_set(nx_graph)
    edge_by_key = {(e.source_id.hex, e.target_id.hex): e for e in positive_edges}
    remove_ids = {edge_by_key[k].id for k in to_remove if k in edge_by_key}

    removed = [e for e in graph.edges if e.id in remove_ids]
    if removed:
        logger.info(
            "Removed %d edge(s) to resolve %d CTA cycle(s)",
            len(removed), len(to_remove),
        )
        graph.edges = [e for e in graph.edges if e.id not in remove_ids]


def run(
    text: str,
    config_path: Optional[str] = None,
    mode: Optional[str] = None,
    max_tokens: int = 512,
    overlap: int = 128,
    merge_margin: int = 20,
    coefficients_path: Optional[str] = None,
    use_unified: bool = True,
) -> Graph:
    """Run the full reasoning pipeline.

    Args:
        text: Input text to reason about.
        config_path: Path to priors config file.
        mode: Reasoning mode (causal, conditional, argument, analogy).
        max_tokens: Maximum tokens per chunk.
        overlap: Token overlap between chunks.
        merge_margin: Character margin for merging propositions.
        coefficients_path: Path to coefficients JSON (for unified framework).
        use_unified: Whether to use the new unified framework (default True).
            Falls back to legacy SL operators if False.

    Returns:
        Graph with reasoned beliefs and metadata.
    """
    priors = load_priors(config_path)

    from cognitive_engine.pipeline.extraction import extract_graph as _extract_graph
    graph = _extract_graph(text, max_tokens=max_tokens, overlap=overlap, merge_margin=merge_margin)

    if use_unified:
        # New unified framework
        try:
            from cognitive_engine.unified.reasoner import UnifiedReasoner
            from cognitive_engine.unified.coefficients import Coefficients

            # Load coefficients
            if coefficients_path:
                coefficients = Coefficients.load(coefficients_path)
            else:
                coefficients = Coefficients()

            # Create reasoner and run
            reasoner = UnifiedReasoner(coefficients)
            result = reasoner.reason(graph, priors=priors)

            # Update graph with unified beliefs
            for node_id, belief in result.beliefs.items():
                if node_id in graph.nodes:
                    # Convert belief to SL opinion for backward compat
                    b = max(0.0, min(1.0, belief))
                    d = max(0.0, min(1.0, 1.0 - b))
                    u = 0.0
                    a = 0.5
                    graph.nodes[node_id].opinion = (b, d, u, a)

            # Store unified metadata
            graph.metadata["unified_reasoning"] = {
                "beliefs": {str(k): v for k, v in result.beliefs.items()},
                "truth_values": {str(k): v for k, v in result.truth_values.items()},
                "objective": result.objective,
                "violations": {str(k): v for k, v in result.violations.items()},
                "coefficients": result.metadata.get("coefficients", {}),
            }

            logger.info(
                "Unified reasoning: %d nodes, objective=%.4f",
                len(result.beliefs), result.objective,
            )

        except ImportError as e:
            logger.warning(
                "Unified framework not available, falling back to legacy: %s", e
            )
            use_unified = False

    if not use_unified:
        # Legacy SL operators
        compute_opinions(graph, priors)

    violations = validate_all(graph)
    errors = [v for v in violations if v.severity == Severity.ERROR]
    if errors:
        logger.warning("%d validation error(s) found", len(errors))
        for v in errors:
            logger.warning("  %s", v.description)

    compute_mode_views(graph, priors)

    if mode is not None:
        resolved = ReasoningMode[mode.upper()]
        graph = apply_mode_operator(graph, priors, resolved)
        if not use_unified:
            compute_opinions(graph, priors)

    _break_cycles(graph)

    graph.cta = ConversationTree.from_graph(graph)

    return graph
