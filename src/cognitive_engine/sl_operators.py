from uuid import UUID

from cognitive_engine.config import Priors
from cognitive_engine.models import Graph, NodeType, Opinion


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


def compute_opinions(graph: Graph, priors: Priors | None = None) -> Graph:
    if priors is None:
        priors = Priors()

    attach_opinions(graph, priors)

    order = _topological_order(graph)

    for nid in order:
        incoming = [e for e in graph.edges if e.target_id == nid]
        if not incoming:
            continue

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
            continue

        fused = contributions[0]
        for op in contributions[1:]:
            fused = cumulative_fusion(fused, op)
        graph.nodes[nid].opinion = fused

    graph.metadata["priors"] = priors.to_dict()

    return graph


def _clamp(op: Opinion) -> Opinion:
    b, d, u, a = op
    b = max(0.0, min(1.0, b))
    d = max(0.0, min(1.0, d))
    u = max(0.0, min(1.0, u))
    total = b + d + u
    if total > 1.0 and total > 0:
        b /= total
        d /= total
        u /= total
    elif total == 0:
        u = 1.0
    a = max(0.0, min(1.0, a))
    return (b, d, u, a)
