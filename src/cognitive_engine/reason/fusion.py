from __future__ import annotations

from uuid import UUID

from cognitive_engine.domain import domain as _domain
from cognitive_engine.core.models import (
    Edge,
    FusionSituation,
    Graph,
    Node,
    NodeType,
    Opinion,
)
def _clamp(op: Opinion) -> Opinion:
    b, d, u, a = op
    b = max(0.0, min(1.0, b))
    d = max(0.0, min(1.0, d))
    u = max(0.0, min(1.0, u))
    total = b + d + u
    if abs(total - 1.0) > 1e-9 and total > 0:
        b /= total
        d /= total
        u /= total
    elif total == 0:
        u = 1.0
    a = max(0.0, min(1.0, a))
    return (b, d, u, a)


def cumulative_fusion(omega_a: Opinion, omega_b: Opinion) -> Opinion:
    b_a, d_a, u_a, a_a = omega_a
    b_b, d_b, u_b, _ = omega_b
    kappa = u_a + u_b - u_a * u_b
    if kappa == 0:
        b = (b_a + b_b) / 2
        d = (d_a + d_b) / 2
        u = 0.0
        return _clamp((b, d, u, a_a))
    b = (b_a * u_b + b_b * u_a) / kappa
    d = (d_a * u_b + d_b * u_a) / kappa
    u = (u_a * u_b) / kappa
    return _clamp((b, d, u, a_a))


def consensus_compromise(omega_a: Opinion, omega_b: Opinion) -> Opinion:
    """Consensus & Compromise Fusion (Jøsang 2016 §8.3).

    For conflicting sources: belief mass from disagreement is
    redirected to uncertainty (vague belief).
    """
    b_a, d_a, u_a, a_a = omega_a
    b_b, d_b, u_b, a_b = omega_b

    conflict = b_a * d_b + d_a * b_b
    denom = 1.0 - conflict

    if denom <= 1e-12:
        return (0.0, 0.0, 1.0, (a_a + a_b) / 2.0)

    b = (b_a * b_b + b_a * u_b + b_b * u_a) / denom
    d = (d_a * d_b + d_a * u_b + d_b * u_a) / denom
    u = (u_a * u_b) / denom
    a = (a_a + a_b) / 2.0

    return _clamp((b, d, u, a))


def weighted_belief_fusion(
    omega_a: Opinion,
    omega_b: Opinion,
    weight_a: float,
    weight_b: float,
) -> Opinion:
    """Weighted Belief Fusion (Jøsang 2016 §8.4).

    Fuses opinions weighted by source trust/authority.
    weights must sum to 1.
    """
    total = weight_a + weight_b
    if total <= 0:
        return omega_a
    w_a, w_b = weight_a / total, weight_b / total

    b_a, d_a, u_a, a_a = omega_a
    b_b, d_b, u_b, a_b = omega_b

    b = w_a * b_a + w_b * b_b
    d = w_a * d_a + w_b * d_b
    u = w_a * u_a + w_b * u_b
    a = w_a * a_a + w_b * a_b

    return _clamp((b, d, u, a))


def trust_transfer(
    omega_source: Opinion,
    omega_recommendation: Opinion,
) -> Opinion:
    """Trust Transfer (Jøsang 2016 §9.2).

    Given trust in a source and the source's recommendation,
    derive the resulting opinion.
    """
    b_s, d_s, u_s, _ = omega_source
    b_r, d_r, u_r, a_r = omega_recommendation

    b = b_s * b_r
    d = b_s * d_r
    u = 1.0 - b_s + b_s * u_r
    a = a_r

    return _clamp((b, d, u, a))


def _opinions_conflict(
    omega_a: Opinion,
    omega_b: Opinion,
    threshold: float | None = None,
) -> bool:
    """Two opinions conflict when one believes and the other disbelieves."""
    if threshold is None:
        threshold = _domain.active().conflict_threshold
    b_a, d_a, _, _ = omega_a
    b_b, d_b, _, _ = omega_b
    return (b_a > threshold and d_b > threshold) or (
        b_b > threshold and d_a > threshold
    )


def _shared_ancestor(
    uid_a: UUID,
    uid_b: UUID,
    graph: Graph,
    visited: set[UUID] | None = None,
) -> bool:
    if visited is None:
        visited = set()
    ancestors_a: set[UUID] = set()
    stack = [uid_a]
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        ancestors_a.add(nid)
        for e in graph.edges.values():
            if e.target_id == nid:
                stack.append(e.source_id)
    stack = [uid_b]
    visited_b: set[UUID] = set()
    while stack:
        nid = stack.pop()
        if nid in visited_b:
            continue
        visited_b.add(nid)
        if nid in ancestors_a:
            return True
        for e in graph.edges.values():
            if e.target_id == nid:
                stack.append(e.source_id)
    return False


def classify_fusion_situation(
    contributions: list[Opinion],
    incoming_edges: list[Edge],
    graph: Graph,
) -> FusionSituation:
    if len(contributions) < 2:
        return FusionSituation.INDEPENDENT_SOURCES

    conflict_detected = False
    for i in range(len(contributions)):
        for j in range(i + 1, len(contributions)):
            if _opinions_conflict(contributions[i], contributions[j]):
                conflict_detected = True
                break
        if conflict_detected:
            break

    if conflict_detected:
        return FusionSituation.CONFLICTING_VIEWS

    source_ids = {e.source_id for e in incoming_edges if e.source_id in graph.nodes}

    if len(source_ids) < len(incoming_edges):
        return FusionSituation.SAME_SOURCE

    src_list = list(source_ids)
    for i in range(len(src_list)):
        for j in range(i + 1, len(src_list)):
            if _shared_ancestor(src_list[i], src_list[j], graph):
                return FusionSituation.DEPENDENT_SOURCES

    return FusionSituation.INDEPENDENT_SOURCES
