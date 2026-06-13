from typing import Optional
from uuid import UUID

from cognitive_engine.core.config import Priors
from cognitive_engine.core.models import (
    Edge,
    FusionSituation,
    Graph,
    NodeType,
    EdgeType,
    Opinion,
)
from cognitive_engine.reason.fusion import (
    classify_fusion_situation,
    consensus_compromise,
    cumulative_fusion,
    weighted_belief_fusion,
)


def conjunction(omega_x: Opinion, omega_y: Opinion) -> Opinion:
    b_x, d_x, u_x, a_x = omega_x
    b_y, d_y, u_y, a_y = omega_y
    denom = 1 - a_x * a_y
    if denom == 0:
        denom = 1.0
    b = (
        b_x * b_y
        + (a_x * b_y * (1 - a_y) * u_x + a_y * b_x * (1 - a_x) * u_y) / denom
    )
    d = d_x + d_y - d_x * d_y
    u = (
        u_x * u_y
        + (b_x * (1 - a_y) * u_y + b_y * (1 - a_x) * u_x) / denom
    )
    a = a_x * a_y
    return _clamp((b, d, u, a))


def disjunction(omega_x: Opinion, omega_y: Opinion) -> Opinion:
    b_x, d_x, u_x, a_x = omega_x
    b_y, d_y, u_y, a_y = omega_y
    denom = 1 - a_x * a_y
    if denom == 0:
        denom = 1.0
    b = b_x + b_y - b_x * b_y
    d = (
        d_x * d_y
        + (a_x * d_y * (1 - a_y) * u_x + a_y * d_x * (1 - a_x) * u_y) / denom
    )
    u = (
        u_x * u_y
        + (d_x * (1 - a_y) * u_y + d_y * (1 - a_x) * u_x) / denom
    )
    a = a_x * a_y
    return _clamp((b, d, u, a))


def conditional_deduction(
    omega_p: Opinion, warrant: tuple[Opinion, Opinion]
) -> Opinion:
    b_p, d_p, u_p, _ = omega_p
    omega_c_given_p, omega_c_given_not_p = warrant
    b_c_given_p, d_c_given_p, u_c_given_p, a_c = omega_c_given_p
    b_c_given_not_p, d_c_given_not_p, u_c_given_not_p, _ = omega_c_given_not_p
    b = b_p * b_c_given_p + d_p * b_c_given_not_p + u_p * a_c
    d = b_p * d_c_given_p + d_p * d_c_given_not_p + u_p * (1 - a_c)
    u = u_p + b_p * u_c_given_p + d_p * u_c_given_not_p
    return _clamp((b, d, u, a_c))


def projected_probability(omega: Opinion) -> float:
    b, d, u, a = omega
    return b + a * u


def dirichlet_strength(omega: Opinion) -> float:
    b, d, u, _ = omega
    if u == 0:
        return float("inf")
    return (b + d) / u


def attach_opinions(graph: Graph, priors: Priors) -> Graph:
    for node in graph.nodes.values():
        key = priors.source_type_map.get(node.type.name, "observational_claim")
        node.opinion = priors.default_opinions.get(key, priors.default_opinions["total_ignorance"])
    for edge in graph.edges:
        edge.opinion = priors.default_opinions["total_ignorance"]
    return graph


def _topological_order(graph: Graph) -> list[UUID]:
    indeg: dict[UUID, int] = {nid: 0 for nid in graph.nodes}
    for e in graph.edges:
        if e.target_id in indeg:
            indeg[e.target_id] += 1
    queue = [nid for nid, d in indeg.items() if d == 0]
    order: list[UUID] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for e in graph.edges:
            if e.source_id == nid and e.target_id in indeg:
                indeg[e.target_id] -= 1
                if indeg[e.target_id] == 0:
                    queue.append(e.target_id)
    remaining = [nid for nid in graph.nodes if nid not in order]
    order.extend(remaining)
    return order


def _select_fusion_fn(situation: FusionSituation):
    match situation:
        case FusionSituation.CONFLICTING_VIEWS:
            return consensus_compromise
        case FusionSituation.DEPENDENT_SOURCES:
            return weighted_belief_fusion
        case FusionSituation.SAME_SOURCE:
            return cumulative_fusion
        case FusionSituation.INDEPENDENT_SOURCES:
            return cumulative_fusion


def _fusion_strategy(
    contributions: list[Opinion],
    incoming_edges: list,
    graph: Graph,
) -> Opinion:
    if len(contributions) == 1:
        return contributions[0]

    situation = classify_fusion_situation(contributions, incoming_edges, graph)
    fusion_fn = _select_fusion_fn(situation)

    if fusion_fn is weighted_belief_fusion:
        weights = _compute_trust_weights(incoming_edges, graph)
        fused = contributions[0]
        for idx, op in enumerate(contributions[1:], start=1):
            fused = weighted_belief_fusion(
                fused, op, weights[0], weights[idx],
            )
        return fused

    fused = contributions[0]
    for op in contributions[1:]:
        fused = fusion_fn(fused, op)
    return fused


def _compute_trust_weights(
    incoming_edges: list[Edge],
    graph: Graph,
) -> list[float]:
    weights: list[float] = []
    for edge in incoming_edges:
        source = graph.nodes.get(edge.source_id)
        if source is None:
            weights.append(1.0)
        else:
            b, d, u, _ = source.opinion
            weights.append(b + 0.5 * u)
    total = sum(weights)
    if total > 0:
        return [w / total for w in weights]
    return [1.0 / len(weights)] * len(weights)


def _compute_node_opinion(
    nid: UUID, incoming: list[Edge], graph: Graph, priors: Priors,
) -> Optional[Opinion]:
    contributions: list[Opinion] = []
    for edge in incoming:
        source = graph.nodes.get(edge.source_id)
        if source is None:
            continue
        warrant = edge.warrant or priors.edge_warrants.get(
            edge.type.name, priors.default_warrant
        )
        result = conditional_deduction(source.opinion, warrant)
        contributions.append(result)

    if not contributions:
        return None

    return _fusion_strategy(contributions, incoming, graph)


def compute_opinions(graph: Graph, priors: Priors | None = None) -> Graph:
    if priors is None:
        priors = Priors()

    attach_opinions(graph, priors)

    order = _topological_order(graph)

    for nid in order:
        incoming = [e for e in graph.edges if e.target_id == nid]
        if not incoming:
            continue
        opinion = _compute_node_opinion(nid, incoming, graph, priors)
        if opinion is not None:
            graph.nodes[nid].opinion = opinion

    graph.metadata["priors"] = priors.to_dict()

    return graph


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
