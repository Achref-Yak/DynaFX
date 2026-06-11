from __future__ import annotations

import logging
from copy import deepcopy

from cognitive_engine.config import Priors, load_priors, sweep_priors
from cognitive_engine.extraction import extract_graph
from cognitive_engine.models import Graph, Severity
from cognitive_engine.reviewers import review_graph
from cognitive_engine.sl_operators import compute_opinions
from cognitive_engine.validators import validate_all

logger = logging.getLogger(__name__)

MAX_ROUNDS = 5


class CafGenError(Exception):
    pass


async def _run_single(
    text: str,
    priors: Priors,
    max_rounds: int = MAX_ROUNDS,
    model_name: str | None = None,
    best_effort: bool = False,
) -> Graph:
    feedback: str | None = None
    last_graph: Graph | None = None

    for r in range(1, max_rounds + 1):
        logger.info("Round %d/%d: extracting graph...", r, max_rounds)
        graph = await extract_graph(text, feedback=feedback, model_name=model_name)
        last_graph = graph

        compute_opinions(graph, priors)

        violations = validate_all(graph)
        errors = [v for v in violations if v.severity == Severity.ERROR]

        logger.info("Round %d: %d violations (%d errors)", r, len(violations), len(errors))

        result = await review_graph(
            graph=graph,
            source_text=text,
            pre_computed_violations=violations,
            model_name=model_name,
        )

        logger.info("Round %d: reviewer says '%s'", r, result.status)

        if result.status == "accept" and not errors:
            logger.info("Graph accepted after %d round(s)", r)
            return graph

        feedback = result.feedback or "Fix the violations identified above."
        logger.info("Round %d feedback: %s", r, feedback[:200])

    if best_effort and last_graph is not None:
        logger.warning("Returning best-effort graph after %d rounds", max_rounds)
        return last_graph

    raise CafGenError(
        f"Graph not accepted after {max_rounds} rounds. Last feedback: {feedback}"
    )


async def run(
    text: str,
    max_rounds: int = MAX_ROUNDS,
    model_name: str | None = None,
    best_effort: bool = False,
    config_path: str | None = None,
    sweep: bool = False,
) -> Graph:
    priors = load_priors(config_path)

    if not sweep:
        return await _run_single(text, priors, max_rounds, model_name, best_effort)

    variants = [priors] + sweep_priors(priors, delta=0.1)[1:]
    logger.info("Running sensitivity sweep: %d prior variants", len(variants))

    all_graphs: list[Graph] = []
    for i, variant in enumerate(variants):
        logger.info("Sweep %d/%d: running with varied priors", i + 1, len(variants))
        g = await _run_single(text, variant, max_rounds, model_name, best_effort=True)
        all_graphs.append(g)

    return _aggregate_sweep(all_graphs)


def _aggregate_sweep(graphs: list[Graph]) -> Graph:
    if not graphs:
        raise CafGenError("No graphs produced during sweep")

    base = deepcopy(graphs[0])

    for nid in base.nodes:
        opinions = [g.nodes[nid].opinion for g in graphs if nid in g.nodes]
        if not opinions:
            continue
        bs = [o[0] for o in opinions]
        ds = [o[1] for o in opinions]
        us = [o[2] for o in opinions]
        base.nodes[nid].opinion = (
            sum(bs) / len(bs),
            sum(ds) / len(ds),
            sum(us) / len(us),
            0.5,
        )
        base.nodes[nid].metadata["sweep"] = {
            "b_min": min(bs),
            "b_max": max(bs),
            "d_min": min(ds),
            "d_max": max(ds),
            "u_min": min(us),
            "u_max": max(us),
        }

    base.metadata["sweep_count"] = len(graphs)
    base.metadata["sweep_priors"] = [g.metadata.get("priors", {}) for g in graphs]

    return base
