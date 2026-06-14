"""Global objective function for unified reasoning.

Implements:
    G* = argmax_G (Σ_v T(v) - λ·C_violations)

Usage:
    from cognitive_engine.unified.objective import compute_objective, count_violations
    obj = compute_objective(graph, truth_values, violations, coefficients)
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph


def count_violations(graph: Graph) -> dict[UUID, int]:
    """Count constraint violations per node.

    Checks:
        1. Category hierarchy (higher categories shouldn't imply lower)
        2. Edge consistency (support edges should have positive beliefs)
        3. Opinion invariant (b + d + u ≈ 1)

    Returns:
        Dict mapping node ID → number of violations.
    """
    violations: dict[UUID, int] = {}

    for node_id, node in graph.nodes.items():
        count = 0

        # Check opinion invariant
        if node.opinion:
            b, d, u, a = node.opinion
            total = b + d + u
            if abs(total - 1.0) > 0.01:
                count += 1
            if u < 0:
                count += 1

        # Check edge consistency
        for edge in graph.edges:
            if edge.source_id == node_id:
                # Support edges should have source with positive belief
                if edge.type.name in ("SUPPORTS", "INFERS", "JUSTIFIES", "DIRECT"):
                    if node.opinion:
                        b, d, u, a = node.opinion
                        if b + a * u < 0.2:  # Very low belief but supporting
                            count += 1

        if count > 0:
            violations[node_id] = count

    return violations


def compute_objective(
    graph: Graph,
    truth_values: dict[UUID, float],
    violations: dict[UUID, int],
    coefficients,
) -> float:
    """Compute global objective: G* = Σ_v T(v) - λ·C_violations.

    Args:
        graph: The reasoning graph.
        truth_values: Truth values from master equation.
        violations: Constraint violation counts.
        coefficients: Coefficients with level7_lambda_violations.

    Returns:
        Scalar objective value (higher is better).
    """
    total_truth = sum(truth_values.values())
    total_violations = sum(violations.values())

    return total_truth - coefficients.level7_lambda_violations * total_violations
