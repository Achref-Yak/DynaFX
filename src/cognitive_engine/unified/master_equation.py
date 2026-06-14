"""Master Equation: R(x) = αP(x) + βG(x) + γL(x) - δA(x).

Pure function that combines all level outputs into final beliefs.

Usage:
    from cognitive_engine.unified.master_equation import master_equation
    from cognitive_engine.unified.coefficients import Coefficients
    beliefs = master_equation(
        graph, graph_beliefs, probabilistic_beliefs,
        logic_consistency, attack_strengths, violations,
        Coefficients(),
    )
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph


def master_equation(
    graph: Graph,
    beliefs: dict[UUID, float],
    probabilities: dict[UUID, float],
    logic_consistency: dict[UUID, float],
    attack_strengths: dict[UUID, float],
    violations: dict[UUID, int],
    coefficients,
) -> dict[UUID, float]:
    """Compute the master equation for all nodes.

    R(x) = αP(x) + βG(x) + γL(x) - δA(x)

    Where:
        P(x) = probability evidence (Level 2 output)
        G(x) = graph propagation (Level 4 output)
        L(x) = logic consistency (Level 0/6 output)
        A(x) = adversarial contradiction (Level 5 output)

    Args:
        graph: The reasoning graph.
        beliefs: Graph propagation beliefs (Level 4).
        probabilities: Bayesian probabilities (Level 2).
        logic_consistency: Logic consistency scores (Level 0/6).
        attack_strengths: Attack strengths from argumentation (Level 5).
        violations: Constraint violation counts per node.
        coefficients: Coefficients dataclass with α, β, γ, δ.

    Returns:
        Dict mapping node ID → final belief score in [0, 1].
    """
    truth_values = {}

    for node_id in graph.nodes:
        P_x = probabilities.get(node_id, 0.5)
        G_x = beliefs.get(node_id, 0.5)
        L_x = logic_consistency.get(node_id, 1.0)
        A_x = attack_strengths.get(node_id, 0.0)
        v = violations.get(node_id, 0)

        truth_values[node_id] = (
            coefficients.alpha * P_x +
            coefficients.beta * G_x +
            coefficients.gamma * L_x -
            coefficients.delta * A_x -
            coefficients.level7_lambda_violations * v
        )

    return truth_values


def compute_support_sum(
    node_id: UUID,
    graph: Graph,
    beliefs: dict[UUID, float],
) -> float:
    """Compute weighted support sum: Σ_{j∈Supp(i)} W_ji * B_j."""
    total = 0.0
    support_weights = {
        "SUPPORTS": 0.85, "INFERS": 0.9, "JUSTIFIES": 0.8,
        "DIRECT": 0.95, "CIRCUMSTANTIAL": 0.6, "QUALIFIES": 0.5,
    }
    for edge in graph.edges:
        if edge.target_id == node_id and edge.source_id in beliefs:
            weight = support_weights.get(edge.type.name, 0.5)
            total += weight * beliefs[edge.source_id]
    return total


def compute_attack_sum(
    node_id: UUID,
    graph: Graph,
    beliefs: dict[UUID, float],
) -> float:
    """Compute weighted attack sum: Σ_{k∈Att(i)} B_k."""
    total = 0.0
    attack_types = {"ATTACKS", "CONTRADICTS", "REBUTS"}
    for edge in graph.edges:
        if edge.target_id == node_id and edge.source_id in beliefs:
            if edge.type.name in attack_types:
                total += beliefs[edge.source_id]
    return total
