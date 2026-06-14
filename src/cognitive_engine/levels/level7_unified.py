"""Level 7: Unified Graph Truth.

Implements the master truth function that combines all level outputs:
    T(v) = αP(v) + βΣW_jv*B_j - γΣ_{k∈Att(v)}B_k - δ·Violations(v)

And the global objective:
    G* = argmax_G (Σ_v T(v) - λ·C_violations)

And fixed-point reasoning:
    B* = F(B*)

Usage:
    from cognitive_engine.levels.level7_unified import UnifiedLevel
    from cognitive_engine.unified.coefficients import Coefficients
    level = UnifiedLevel(Coefficients())
    output = level.compute(graph, context)
"""
from __future__ import annotations

import logging
import math
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph, EdgeType
from cognitive_engine.levels.base import BaseLevel, LevelOutput, ReasoningContext

logger = logging.getLogger(__name__)


# ── Edge type → weight for support/attack aggregation ─────────────
_SUPPORT_WEIGHTS: dict[EdgeType, float] = {
    EdgeType.SUPPORTS: 0.85, EdgeType.INFERS: 0.9,
    EdgeType.JUSTIFIES: 0.8, EdgeType.DIRECT: 0.95,
    EdgeType.CIRCUMSTANTIAL: 0.6, EdgeType.QUALIFIES: 0.5,
}
_ATTACK_WEIGHTS: dict[EdgeType, float] = {
    EdgeType.ATTACKS: 0.8, EdgeType.CONTRADICTS: 0.85,
    EdgeType.REBUTS: 0.6, EdgeType.HEARSAY: 0.4,
}


class UnifiedLevel(BaseLevel):
    """Level 7: Unified Graph Truth.

    Combines outputs from all levels into a single truth value per node,
    using the master equation and fixed-point iteration.
    """

    @property
    def name(self) -> str:
        return "Unified Graph Truth"

    @property
    def level_number(self) -> int:
        return 7

    def __init__(self, coefficients=None) -> None:
        from cognitive_engine.unified.coefficients import Coefficients
        self.coeffs = coefficients or Coefficients()

    def compute(
        self, graph: Graph, context: ReasoningContext,
    ) -> LevelOutput:
        """Run unified truth computation on the graph.

        Uses outputs from previous levels stored in context.previous_outputs
        to compute final truth values via the master equation.
        """
        if not graph.nodes:
            return LevelOutput(beliefs={}, metadata={})

        # Extract previous level outputs
        beliefs = self._extract_beliefs(graph, context)
        probabilities = self._extract_probabilities(graph, context)
        logic_consistency = self._extract_logic_consistency(graph, context)
        attack_strengths = self._extract_attack_strengths(graph, context)
        violations = self._extract_violations(graph, context)

        # Compute truth values via master equation
        truth_values = self.compute_truth_values(
            graph, beliefs, probabilities, logic_consistency,
            attack_strengths, violations,
        )

        # Compute global objective
        objective = self.compute_global_objective(graph, truth_values, violations)

        return LevelOutput(
            beliefs=truth_values,
            metadata={
                "objective": objective,
                "alpha": self.coeffs.alpha,
                "beta": self.coeffs.beta,
                "gamma": self.coeffs.gamma,
                "delta": self.coeffs.delta,
                "num_nodes": len(graph.nodes),
                "mean_truth": sum(truth_values.values()) / len(truth_values) if truth_values else 0.0,
            },
        )

    def compute_truth_value(
        self,
        node_id: UUID,
        graph: Graph,
        beliefs: dict[UUID, float],
        probabilities: dict[UUID, float],
        logic_consistency: dict[UUID, float],
        attack_strengths: dict[UUID, float],
        violations: dict[UUID, int],
    ) -> float:
        """T(v) = αP(v) + βΣW_jv*B_j - γΣ_{k∈Att(v)}B_k - δ·Violations(v)."""
        P_x = probabilities.get(node_id, 0.5)
        G_x = beliefs.get(node_id, 0.5)
        L_x = logic_consistency.get(node_id, 1.0)
        A_x = attack_strengths.get(node_id, 0.0)
        v = violations.get(node_id, 0)

        return (
            self.coeffs.alpha * P_x +
            self.coeffs.beta * G_x +
            self.coeffs.gamma * L_x -
            self.coeffs.delta * A_x -
            self.coeffs.level7_lambda_violations * v
        )

    def compute_truth_values(
        self,
        graph: Graph,
        beliefs: dict[UUID, float],
        probabilities: dict[UUID, float],
        logic_consistency: dict[UUID, float],
        attack_strengths: dict[UUID, float],
        violations: dict[UUID, int],
    ) -> dict[UUID, float]:
        """Compute truth values for all nodes."""
        truth_values = {}
        for nid in graph.nodes:
            truth_values[nid] = self.compute_truth_value(
                nid, graph, beliefs, probabilities,
                logic_consistency, attack_strengths, violations,
            )
        return truth_values

    def compute_global_objective(
        self,
        graph: Graph,
        truth_values: dict[UUID, float],
        violations: dict[UUID, int],
    ) -> float:
        """G* = argmax_G (Σ_v T(v) - λ·C_violations)."""
        total_truth = sum(truth_values.values())
        total_violations = sum(violations.values())
        return total_truth - self.coeffs.level7_lambda_violations * total_violations

    def fixed_point_reasoning(
        self, graph: Graph, context: ReasoningContext,
    ) -> dict[UUID, float]:
        """Iterate until B* = F(B*)."""
        beliefs = {}

        for iteration in range(self.coeffs.level7_max_iterations):
            old_beliefs = dict(beliefs)

            # Get updated outputs from context
            new_beliefs = self._extract_beliefs(graph, context)
            probabilities = self._extract_probabilities(graph, context)
            logic_consistency = self._extract_logic_consistency(graph, context)
            attack_strengths = self._extract_attack_strengths(graph, context)
            violations = self._extract_violations(graph, context)

            # Compute new truth values
            beliefs = self.compute_truth_values(
                graph, new_beliefs, probabilities,
                logic_consistency, attack_strengths, violations,
            )

            # Check convergence
            change = math.sqrt(
                sum((beliefs[nid] - old_beliefs.get(nid, 0.5)) ** 2
                    for nid in beliefs)
            )

            if change < self.coeffs.convergence_threshold:
                logger.debug(
                    "Fixed-point reasoning converged at iteration %d (Δ=%.6f)",
                    iteration + 1, change,
                )
                break

        return beliefs

    # ── Private helpers for extracting level outputs ──────────────

    def _extract_beliefs(
        self, graph: Graph, context: ReasoningContext,
    ) -> dict[UUID, float]:
        """Extract beliefs from Level 4 (Graph Propagation)."""
        level4_output = context.previous_outputs.get("level4")
        if level4_output and level4_output.beliefs:
            return level4_output.beliefs
        # Fallback: use node opinions
        beliefs = {}
        for nid, node in graph.nodes.items():
            if node.opinion:
                b, d, u, a = node.opinion
                beliefs[nid] = b + a * u
            else:
                beliefs[nid] = 0.5
        return beliefs

    def _extract_probabilities(
        self, graph: Graph, context: ReasoningContext,
    ) -> dict[UUID, float]:
        """Extract probabilities from Level 2 (Probabilistic)."""
        level2_output = context.previous_outputs.get("level2")
        if level2_output and level2_output.beliefs:
            return level2_output.beliefs
        # Fallback
        return {nid: 0.5 for nid in graph.nodes}

    def _extract_logic_consistency(
        self, graph: Graph, context: ReasoningContext,
    ) -> dict[UUID, float]:
        """Extract logic consistency from Level 0 (Symbolic)."""
        level0_output = context.previous_outputs.get("level0")
        if level0_output and level0_output.beliefs:
            # Invert violations: high violations → low consistency
            return {nid: 1.0 - min(1.0, belief) for nid, belief in level0_output.beliefs.items()}
        return {nid: 1.0 for nid in graph.nodes}

    def _extract_attack_strengths(
        self, graph: Graph, context: ReasoningContext,
    ) -> dict[UUID, float]:
        """Extract attack strengths from Level 5 (Argumentation)."""
        level5_output = context.previous_outputs.get("level5")
        if level5_output and level5_output.metadata:
            attack = level5_output.metadata.get("attack", {})
            return {UUID(k): v for k, v in attack.items()}
        return {nid: 0.0 for nid in graph.nodes}

    def _extract_violations(
        self, graph: Graph, context: ReasoningContext,
    ) -> dict[UUID, int]:
        """Extract violations from Level 0 or Level 6."""
        # Try Level 0 first
        level0_output = context.previous_outputs.get("level0")
        if level0_output and level0_output.metadata:
            violations = level0_output.metadata.get("violations", {})
            return {UUID(k): v for k, v in violations.items()}
        # Try Level 6
        level6_output = context.previous_outputs.get("level6")
        if level6_output and level6_output.metadata:
            violations = level6_output.metadata.get("violations", {})
            return {UUID(k): v for k, v in violations.items()}
        return {nid: 0 for nid in graph.nodes}
