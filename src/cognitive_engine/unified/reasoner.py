"""UnifiedReasoner — main entry point for the 8-level reasoning framework.

Orchestrates all levels and produces final beliefs via the Master Equation.

Usage:
    from cognitive_engine.unified.reasoner import UnifiedReasoner
    from cognitive_engine.unified.coefficients import Coefficients

    reasoner = UnifiedReasoner(Coefficients())
    result = reasoner.reason(graph)
    # result.beliefs contains final beliefs per node
    # result.truth_values contains T(v) per node
    # result.objective is the global objective value
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph, NodeType
from cognitive_engine.levels.base import LevelOutput, ReasoningContext
from cognitive_engine.levels.level0_symbolic import SymbolicLevel
from cognitive_engine.levels.level1_cognitive import CognitiveLevel
from cognitive_engine.levels.level2_probabilistic import ProbabilisticLevel
from cognitive_engine.levels.level3_neural import NeuralLevel
from cognitive_engine.levels.level4_graph import GraphLevel
from cognitive_engine.levels.level5_argumentation import ArgumentationLevel
from cognitive_engine.levels.level6_neurosymbolic import NeuroSymbolicLevel
from cognitive_engine.levels.level7_unified import UnifiedLevel
from cognitive_engine.unified.coefficients import Coefficients
from cognitive_engine.unified.master_equation import master_equation
from cognitive_engine.unified.objective import compute_objective, count_violations

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result from UnifiedReasoner.reason().

    Attributes:
        beliefs: Final beliefs per node (from master equation).
        truth_values: T(v) per node (from Level 7).
        level_outputs: Raw output from each level.
        support: Support strengths from Level 5.
        attack: Attack strengths from Level 5.
        violations: Constraint violation counts.
        objective: Global objective value G*.
        metadata: Additional computation metadata.
    """
    beliefs: dict[UUID, float] = field(default_factory=dict)
    truth_values: dict[UUID, float] = field(default_factory=dict)
    level_outputs: dict[str, LevelOutput] = field(default_factory=dict)
    support: dict[UUID, float] = field(default_factory=dict)
    attack: dict[UUID, float] = field(default_factory=dict)
    violations: dict[UUID, int] = field(default_factory=dict)
    objective: float = 0.0
    metadata: dict = field(default_factory=dict)


class UnifiedReasoner:
    """Main entry point for the 8-level reasoning framework.

    Orchestrates all levels and produces final beliefs via the Master Equation:
        R(x) = αP(x) + βG(x) + γL(x) - δA(x)
    """

    def __init__(self, coefficients: Optional[Coefficients] = None) -> None:
        self.coeffs = coefficients or Coefficients()

        # Initialize all levels
        self.level0 = SymbolicLevel()
        self.level1 = CognitiveLevel(
            temperature=self.coeffs.level1_temperature,
            decay_rate=self.coeffs.level1_decay_rate,
            firing_threshold=self.coeffs.level1_firing_threshold,
        )
        self.level2 = ProbabilisticLevel()
        self.level3 = NeuralLevel(
            embedding_dim=self.coeffs.level3_embedding_dim,
            num_heads=self.coeffs.level3_attention_heads,
            num_layers=self.coeffs.level3_num_layers,
            dropout=self.coeffs.level3_dropout,
        )
        self.level4 = GraphLevel(
            max_iterations=self.coeffs.level4_max_iterations,
            convergence_threshold=self.coeffs.level4_convergence_threshold,
        )
        self.level5 = ArgumentationLevel(
            discount_factor=self.coeffs.level5_discount_factor,
        )
        self.level6 = NeuroSymbolicLevel(
            neural_level=self.level3,
            symbolic_level=self.level0,
            lambda_neural=self.coeffs.level6_lambda_neural,
            logic_penalty_weight=self.coeffs.level6_logic_penalty_weight,
        )
        self.level7 = UnifiedLevel(coefficients=self.coeffs)

    def reason(
        self, graph: Graph, domain_config=None, priors=None,
    ) -> ReasoningResult:
        """Run all 8 levels and produce final beliefs.

        Args:
            graph: The reasoning graph to process.
            domain_config: Optional DomainConfig for domain-specific rules.
            priors: Optional Priors for backward compatibility.

        Returns:
            ReasoningResult with beliefs, truth_values, and metadata.
        """
        if not graph.nodes:
            return ReasoningResult()

        # Build reasoning context
        context = ReasoningContext(
            coefficients=self.coeffs,
            priors=priors,
            domain_config=domain_config,
            graph=graph,
        )

        level_outputs = {}

        # Level 0: Symbolic Logic
        logger.debug("Running Level 0: Symbolic Logic")
        level_outputs["level0"] = self.level0.compute(graph, context)
        context.previous_outputs = level_outputs

        # Level 1: Cognitive Architecture
        logger.debug("Running Level 1: Cognitive Architecture")
        level_outputs["level1"] = self.level1.compute(graph, context)
        context.previous_outputs = level_outputs

        # Level 2: Probabilistic Reasoning
        logger.debug("Running Level 2: Probabilistic Reasoning")
        level_outputs["level2"] = self.level2.compute(graph, context)
        context.previous_outputs = level_outputs

        # Level 4: Graph Propagation (before neural for embeddings)
        logger.debug("Running Level 4: Graph Propagation")
        level_outputs["level4"] = self.level4.compute(graph, context)
        context.previous_outputs = level_outputs

        # Level 5: Argumentation Theory
        logger.debug("Running Level 5: Argumentation Theory")
        level_outputs["level5"] = self.level5.compute(graph, context)
        context.previous_outputs = level_outputs

        # Level 3: Neural Reasoning (needs graph structure)
        logger.debug("Running Level 3: Neural Reasoning")
        level_outputs["level3"] = self.level3.compute(graph, context)
        context.previous_outputs = level_outputs

        # Level 6: Neuro-Symbolic Fusion
        logger.debug("Running Level 6: Neuro-Symbolic Fusion")
        level_outputs["level6"] = self.level6.compute(graph, context)
        context.previous_outputs = level_outputs

        # Level 7: Unified Graph Truth
        logger.debug("Running Level 7: Unified Graph Truth")
        level_outputs["level7"] = self.level7.compute(graph, context)

        # Extract final outputs
        beliefs = level_outputs["level4"].beliefs  # Graph propagation
        probabilities = level_outputs["level2"].beliefs  # Bayesian
        logic_consistency = {
            nid: 1.0 - min(1.0, b)
            for nid, b in level_outputs["level0"].beliefs.items()
        }
        attack_data = level_outputs["level5"].metadata.get("attack", {})
        attack_strengths = {UUID(k): v for k, v in attack_data.items()}
        violations = count_violations(graph)

        # Run master equation
        final_beliefs = master_equation(
            graph, beliefs, probabilities,
            logic_consistency, attack_strengths, violations,
            self.coeffs,
        )

        # Compute truth values (Level 7)
        truth_values = level_outputs["level7"].beliefs

        # Compute global objective
        objective = compute_objective(graph, truth_values, violations, self.coeffs)

        # Extract support/attack from Level 5
        support_data = level_outputs["level5"].metadata.get("support", {})
        support = {UUID(k): v for k, v in support_data.items()}
        attack = attack_strengths

        return ReasoningResult(
            beliefs=final_beliefs,
            truth_values=truth_values,
            level_outputs=level_outputs,
            support=support,
            attack=attack,
            violations=violations,
            objective=objective,
            metadata={
                "coefficients": self.coeffs.to_dict(),
                "num_nodes": len(graph.nodes),
                "num_edges": len(graph.edges),
                "levels_run": list(level_outputs.keys()),
            },
        )

    def reason_with_mode(
        self, graph: Graph, mode: str, domain_config=None, priors=None,
    ) -> ReasoningResult:
        """Run reasoning with a specific mode applied after.

        Modes (CAUSAL, CONDITIONAL, ARGUMENT, ANALOGY) filter edges
        and apply mode-specific transformations.
        """
        from cognitive_engine.core.models import ReasoningMode
        from cognitive_engine.reason.mode_operators import apply_mode_operator

        # Run base reasoning
        result = self.reason(graph, domain_config, priors)

        # Apply mode operator
        try:
            resolved_mode = ReasoningMode[mode.upper()]
            graph = apply_mode_operator(graph, priors, resolved_mode)
            # Re-run on mode-filtered graph
            result = self.reason(graph, domain_config, priors)
        except (KeyError, Exception) as e:
            logger.warning("Failed to apply mode %s: %s", mode, e)

        return result
