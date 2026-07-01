from __future__ import annotations

from copy import deepcopy

from dynafx.core.config import Priors
from dynafx.core.models import (
    Edge,
    Graph,
    Opinion,
    ReasoningMode,
)
from dynafx.epistemics.fusion import _clamp
from dynafx.epistemics.sl_operators import (
    compute_opinions,
    conditional_deduction,
    projected_probability,
)


def reverse_warrant(
    warrant: tuple[Opinion, Opinion],
    base_rate_source: float,
    base_rate_target: float,
) -> tuple[Opinion, Opinion]:
    """Compute inverse conditional opinions via Bayes' theorem.

    Given forward warrant (omega_target|source, omega_target|not_source)
    and base rates, compute (omega_source|target, omega_source|not_target).
    """
    (b_t_s, d_t_s, u_t_s, a_t), (b_t_ns, d_t_ns, u_t_ns, _) = warrant
    a_s = base_rate_source
    a_t_val = base_rate_target

    p_t_s = projected_probability((b_t_s, d_t_s, u_t_s, a_t_val))
    p_t_ns = projected_probability((b_t_ns, d_t_ns, u_t_ns, a_t_val))
    p_s = a_s
    p_t = p_s * p_t_s + (1 - p_s) * p_t_ns

    if p_t <= 0 or p_t >= 1:
        return ((0.0, 0.0, 1.0, a_s), (0.0, 0.0, 1.0, a_s))

    p_s_t = p_t_s * p_s / p_t
    p_s_nt = p_t_ns * p_s / (1 - p_t)

    b_s_t = p_s_t * (1 - u_t_s)
    d_s_t = (1 - p_s_t) * (1 - u_t_s)
    u_s_t = u_t_s

    b_s_nt = p_s_nt * (1 - u_t_ns)
    d_s_nt = (1 - p_s_nt) * (1 - u_t_ns)
    u_s_nt = u_t_ns

    return (
        _clamp((b_s_t, d_s_t, u_s_t, a_s)),
        _clamp((b_s_nt, d_s_nt, u_s_nt, a_s)),
    )


def subjective_abduction(
    omega_effect: Opinion,
    warrant: tuple[Opinion, Opinion],
    base_rate_cause: float,
) -> Opinion:
    """Infer cause from effect using subjective abduction.

    Given observed omega_effect and the causal link
    warrant = (omega_cause|effect, omega_cause|not_effect),
    compute the abduced omega_cause.
    """
    b_e, d_e, u_e, _ = omega_effect
    (b_c_e, d_c_e, u_c_e, a_c), (b_c_ne, d_c_ne, u_c_ne, _) = warrant
    a_c_val = base_rate_cause

    b = b_e * b_c_e + d_e * b_c_ne + u_e * a_c_val
    d = b_e * d_c_e + d_e * d_c_ne + u_e * (1 - a_c_val)
    u = b_e * u_c_e + d_e * u_c_ne + u_e

    return _clamp((b, d, u, a_c_val))


def _reverse_graph_for_diagnostic(graph: Graph) -> Graph:
    """Reverse all edges and invert warrants for diagnostic ARGUMENT mode."""
    new_edges: list[Edge] = []
    for edge in graph.edges.values():
        source = graph.nodes.get(edge.source_id)
        target = graph.nodes.get(edge.target_id)
        if source is None or target is None:
            continue

        if edge.warrant is not None:
            rev = reverse_warrant(
                edge.warrant,
                source.opinion[3],
                target.opinion[3],
            )
        else:
            rev = None

        new_edges.append(Edge(
            source_id=edge.target_id,
            target_id=edge.source_id,
            type=edge.type,
            warrant=rev,
        ))
    graph.edges = {e.id: e for e in new_edges}
    return graph


def _apply_analogy_warrants(graph: Graph, priors: Priors) -> Graph:
    """Increase uncertainty in warrants for ANALOGY mode."""
    for edge in graph.edges.values():
        if edge.warrant is not None:
            (b1, d1, u1, a1), (b2, d2, u2, a2) = edge.warrant
            delta1 = b1 * 0.2
            delta2 = b2 * 0.2
            edge.warrant = (
                _clamp((b1 - delta1, d1, u1 + delta1, a1)),
                _clamp((b2 - delta2, d2, u2 + delta2, a2)),
            )
    return graph


def apply_mode_operator(
    graph: Graph,
    priors: Priors,
    mode: ReasoningMode,
) -> Graph:
    """Compute mode-specific opinions on a copy of the graph."""
    from dynafx.epistemics.modes import MODE_ACTIVE_EDGES

    view = deepcopy(graph)
    view.mode = mode

    active = MODE_ACTIVE_EDGES[mode]
    view.edges = {e.id: e for e in view.edges.values() if e.type in active}

    if mode == ReasoningMode.ARGUMENT:
        view = _reverse_graph_for_diagnostic(view)

    if mode == ReasoningMode.ANALOGY:
        view = _apply_analogy_warrants(view, priors)

    return compute_opinions(view, priors)
