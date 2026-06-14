"""Level 6: Neuro-Symbolic Fusion.

Combines neural and symbolic reasoning via weighted fusion:
    F(x) = λ * f_neural(x) + (1-λ) * f_symbolic(x)

Also computes logic penalty for constraint violations:
    L_logic = Σ ||A ∧ ¬B||

And joint loss:
    L_total = L_neural + α * L_logic

Usage:
    from cognitive_engine.levels.level6_neurosymbolic import NeuroSymbolicLevel
    from cognitive_engine.levels.level3_neural import NeuralLevel
    from cognitive_engine.levels.level0_symbolic import SymbolicLevel
    level = NeuroSymbolicLevel(neural_level, symbolic_level, lambda_neural=0.5)
    output = level.compute(graph, context)
"""
from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph
from cognitive_engine.levels.base import BaseLevel, LevelOutput, ReasoningContext
from cognitive_engine.levels.level0_symbolic import SymbolicLevel
from cognitive_engine.levels.level3_neural import NeuralLevel

logger = logging.getLogger(__name__)


class NeuroSymbolicLevel(BaseLevel):
    """Level 6: Neuro-Symbolic Fusion.

    Fuses neural and symbolic predictions via weighted combination,
    enforcing logical consistency while leveraging learned representations.
    """

    @property
    def name(self) -> str:
        return "Neuro-Symbolic Fusion"

    @property
    def level_number(self) -> int:
        return 6

    def __init__(
        self,
        neural_level: Optional[NeuralLevel] = None,
        symbolic_level: Optional[SymbolicLevel] = None,
        lambda_neural: float = 0.5,
        logic_penalty_weight: float = 0.1,
    ) -> None:
        self.neural = neural_level or NeuralLevel()
        self.symbolic = symbolic_level or SymbolicLevel()
        self.lambda_neural = lambda_neural
        self.logic_penalty_weight = logic_penalty_weight

    def compute(
        self, graph: Graph, context: ReasoningContext,
    ) -> LevelOutput:
        """Run neuro-symbolic fusion on the graph.

        1. Get neural predictions
        2. Get symbolic predictions
        3. Fuse via F(x) = λ * neural + (1-λ) * symbolic
        4. Compute logic penalty
        5. Adjust beliefs for consistency
        """
        if not graph.nodes:
            return LevelOutput(beliefs={}, metadata={})

        # Apply coefficient overrides
        if context.coefficients:
            self.lambda_neural = context.coefficients.level6_lambda_neural
            self.logic_penalty_weight = context.coefficients.level6_logic_penalty_weight

        # Get neural predictions
        neural_output = self.neural.compute(graph, context)
        neural_beliefs = neural_output.beliefs

        # Get symbolic predictions
        symbolic_output = self.symbolic.compute(graph, context)
        symbolic_beliefs = symbolic_output.beliefs

        # Fuse predictions
        fused_beliefs = self.fuse(graph, neural_beliefs, symbolic_beliefs)

        # Compute logic penalty
        violations = self.compute_logic_penalty(graph)

        # Adjust beliefs for constraint violations
        for nid, violation_count in violations.items():
            if nid in fused_beliefs:
                # Reduce belief proportionally to violations
                penalty = self.logic_penalty_weight * violation_count
                fused_beliefs[nid] = max(0.0, fused_beliefs[nid] - penalty)

        return LevelOutput(
            beliefs=fused_beliefs,
            metadata={
                "lambda_neural": self.lambda_neural,
                "logic_penalty_weight": self.logic_penalty_weight,
                "neural_beliefs": {str(k): v for k, v in neural_beliefs.items()},
                "symbolic_beliefs": {str(k): v for k, v in symbolic_beliefs.items()},
                "violations": {str(k): v for k, v in violations.items()},
                "num_violations": len(violations),
            },
        )

    def fuse(
        self,
        graph: Graph,
        neural_beliefs: dict[UUID, float],
        symbolic_beliefs: dict[UUID, float],
    ) -> dict[UUID, float]:
        """F(x) = λ * f_neural(x) + (1-λ) * f_symbolic(x)."""
        fused = {}
        for nid in graph.nodes:
            neural = neural_beliefs.get(nid, 0.5)
            symbolic = symbolic_beliefs.get(nid, 0.5)
            fused[nid] = self.lambda_neural * neural + (1 - self.lambda_neural) * symbolic
        return fused

    def compute_logic_penalty(self, graph: Graph) -> dict[UUID, int]:
        """Compute L_logic = Σ ||A ∧ ¬B|| for constraint violations.

        Counts how many edges have source belief True but target belief False
        (i.e., implication violations).
        """
        violations: dict[UUID, int] = {}

        # Get symbolic beliefs for checking
        context = ReasoningContext()
        symbolic_output = self.symbolic.compute(graph, context)
        beliefs = symbolic_output.beliefs

        for edge in graph.edges:
            source_belief = beliefs.get(edge.source_id, 0.5)
            target_belief = beliefs.get(edge.target_id, 0.5)

            # Check for implication violation: A=True but B=False
            if source_belief > 0.7 and target_belief < 0.3:
                # Count as violation for the target
                violations[edge.target_id] = violations.get(edge.target_id, 0) + 1

        return violations

    def compute_total_loss(
        self,
        predictions: dict[UUID, float],
        targets: dict[UUID, float],
        graph: Graph,
    ) -> float:
        """L_total = L_neural + α * L_logic."""
        # Neural loss (MSE)
        neural_loss = 0.0
        count = 0
        for nid, pred in predictions.items():
            if nid in targets:
                neural_loss += (pred - targets[nid]) ** 2
                count += 1
        if count > 0:
            neural_loss /= count

        # Logic penalty
        violations = self.compute_logic_penalty(graph)
        logic_penalty = sum(violations.values())

        return neural_loss + self.logic_penalty_weight * logic_penalty

    def update_lambda(self, validation_loss: float, target_loss: float = 0.1) -> None:
        """Adaptively update λ based on validation performance.

        If neural is doing well, increase λ (trust neural more).
        If symbolic is doing better, decrease λ.
        """
        if validation_loss < target_loss:
            # Neural is good, trust it more
            self.lambda_neural = min(0.9, self.lambda_neural + 0.05)
        else:
            # Neural needs help, trust symbolic more
            self.lambda_neural = max(0.1, self.lambda_neural - 0.05)
