"""Level 0: Symbolic Logic Engine.

Implements formal logic: propositions, implications, Modus Ponens,
universal/existential quantification, inference closure, and
constraint satisfaction.

Core formulas:
    P → Q ≡ ¬P ∨ Q
    (P→Q) ∧ P ⇒ Q          (Modus Ponens)
    ∀x P(x)                 (Universal quantification)
    ∃x P(x)                 (Existential quantification)
    C(S) = {q | S ⊢ q}     (Inference closure)

Usage:
    from cognitive_engine.levels.level0_symbolic import SymbolicLevel
    level = SymbolicLevel()
    level.add_fact("red_light", True)
    level.add_rule(["red_light"], "should_stop", strength=0.9)
    result = level.modus_ponens()
    # result now contains {"should_stop": True}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph, NodeType
from cognitive_engine.levels.base import BaseLevel, LevelOutput, ReasoningContext

logger = logging.getLogger(__name__)


@dataclass
class LogicalProposition:
    """A proposition with a truth value.

    Attributes:
        name: Unique identifier (e.g. "red_light", "witness_hearsay").
        value: Truth value (True/False) or None if unknown.
        source_node_id: Optional Graph node ID this proposition maps to.
        strength: Confidence in the truth value [0, 1]. Default 1.0.
    """
    name: str
    value: Optional[bool] = None
    source_node_id: Optional[UUID] = None
    strength: float = 1.0


@dataclass
class LogicalRule:
    """An implication rule: IF all antecedents THEN consequent.

    Attributes:
        antecedents: List of proposition names that must all be true.
        consequent: Proposition name that becomes true when rule fires.
        strength: Rule confidence [0, 1]. Used for probabilistic firing.
        negated_antecedents: Propositions that must be FALSE (negation).
    """
    antecedents: list[str] = field(default_factory=list)
    consequent: str = ""
    strength: float = 1.0
    negated_antecedents: list[str] = field(default_factory=list)


@dataclass
class Contradiction:
    """A detected logical contradiction.

    Attributes:
        proposition: Name of the contradictory proposition.
        true_value: The value that was asserted.
        conflicting_value: The value inferred or observed.
        source: Where the conflict came from.
    """
    proposition: str
    true_value: bool
    conflicting_value: bool
    source: str = ""


class SymbolicLevel(BaseLevel):
    """Level 0: Formal Logic Engine.

    Maintains a knowledge base of propositions and rules.
    Applies Modus Ponens until fixpoint to infer new facts.
    Checks consistency and computes constraint satisfaction.
    """

    @property
    def name(self) -> str:
        return "Symbolic Logic"

    @property
    def level_number(self) -> int:
        return 0

    def __init__(self) -> None:
        self.propositions: dict[str, LogicalProposition] = {}
        self.rules: list[LogicalRule] = []
        self._inferred: list[str] = []

    def add_fact(self, name: str, value: bool, strength: float = 1.0,
                 source_node_id: Optional[UUID] = None) -> None:
        """Assert a known truth value for a proposition."""
        self.propositions[name] = LogicalProposition(
            name=name, value=value, strength=strength,
            source_node_id=source_node_id,
        )

    def add_rule(self, antecedents: list[str], consequent: str,
                 strength: float = 1.0,
                 negated_antecedents: Optional[list[str]] = None) -> None:
        """Add an implication rule: IF antecedents THEN consequent."""
        self.rules.append(LogicalRule(
            antecedents=antecedents,
            consequent=consequent,
            strength=strength,
            negated_antecedents=negated_antecedents or [],
        ))

    def get_fact(self, name: str) -> Optional[bool]:
        """Get truth value of a proposition, or None if unknown."""
        prop = self.propositions.get(name)
        return prop.value if prop else None

    def modus_ponens(self, max_iterations: int = 100) -> dict[str, bool]:
        """Apply Modus Ponens until fixpoint.

        Returns:
            Dict of newly inferred propositions.
        """
        self._inferred.clear()
        newly_inferred: dict[str, bool] = {}

        for _ in range(max_iterations):
            progress = False
            for rule in self.rules:
                if self._can_fire(rule) and rule.consequent not in self.propositions:
                    # Fire the rule
                    strength = min(
                        rule.strength,
                        *[self.propositions[a].strength
                          for a in rule.antecedents
                          if a in self.propositions],
                    )
                    self.add_fact(rule.consequent, True, strength=strength)
                    newly_inferred[rule.consequent] = True
                    self._inferred.append(rule.consequent)
                    progress = True
                    logger.debug(
                        "Modus Ponens fired: %s → %s (strength=%.2f)",
                        rule.antecedents, rule.consequent, strength,
                    )
            if not progress:
                break

        return newly_inferred

    def check_consistency(self) -> list[Contradiction]:
        """Detect contradictions: propositions with both True and False.

        Returns:
            List of Contradiction objects found.
        """
        contradictions = []
        for name, prop in self.propositions.items():
            if prop.value is None:
                continue
            # Check if any rule inferred the opposite
            for rule in self.rules:
                if (rule.consequent == name and
                        self._can_fire_negated(rule, prop.value)):
                    contradictions.append(Contradiction(
                        proposition=name,
                        true_value=prop.value,
                        conflicting_value=not prop.value,
                        source=f"rule: {rule.antecedents} → {rule.consequent}",
                    ))
        return contradictions

    def constraint_satisfaction(self, graph: Graph) -> dict[UUID, int]:
        """Check all rules against graph nodes, return violation counts.

        For each node, counts how many rules that should apply to it
        are violated (antecedent true but consequent false).

        Returns:
            Dict mapping node ID → number of constraint violations.
        """
        violations: dict[UUID, int] = {}

        for node_id, node in graph.nodes.items():
            count = 0
            # Map node text/type to propositions
            node_props = self._node_to_propositions(node)

            for rule in self.rules:
                # Check if rule's antecedents match this node
                if self._rule_applies_to_node(rule, node_props):
                    # Check if consequent holds
                    if not node_props.get(rule.consequent, False):
                        count += 1

            if count > 0:
                violations[node_id] = count

        return violations

    def compute(
        self, graph: Graph, context: ReasoningContext,
    ) -> LevelOutput:
        """Run symbolic logic on the graph.

        1. Extract propositions from graph nodes
        2. Extract rules from domain config or context
        3. Apply Modus Ponens
        4. Check consistency
        5. Compute constraint satisfaction
        """
        # Load rules from context if available
        if context.domain_config and hasattr(context.domain_config, 'logic_rules'):
            for rule_data in context.domain_config.logic_rules:
                self.add_rule(**rule_data)

        # Map graph nodes to propositions
        for node_id, node in graph.nodes.items():
            prop_name = self._node_to_prop_name(node)
            if prop_name:
                # Initialize with high uncertainty if not already set
                if prop_name not in self.propositions:
                    self.add_fact(
                        prop_name, value=None, strength=0.5,
                        source_node_id=node_id,
                    )

        # Apply Modus Ponens
        inferred = self.modus_ponens()

        # Check consistency
        contradictions = self.check_consistency()

        # Compute constraint satisfaction
        violations = self.constraint_satisfaction(graph)

        # Build beliefs: map proposition truth values back to nodes
        beliefs = {}
        for node_id, node in graph.nodes.items():
            prop_name = self._node_to_prop_name(node)
            if prop_name and prop_name in self.propositions:
                prop = self.propositions[prop_name]
                if prop.value is True:
                    beliefs[node_id] = prop.strength
                elif prop.value is False:
                    beliefs[node_id] = 1.0 - prop.strength
                else:
                    beliefs[node_id] = 0.5  # unknown

        return LevelOutput(
            beliefs=beliefs,
            metadata={
                "inferred": inferred,
                "contradictions": [
                    {"proposition": c.proposition,
                     "true_value": c.true_value,
                     "conflicting_value": c.conflicting_value,
                     "source": c.source}
                    for c in contradictions
                ],
                "violations": {str(k): v for k, v in violations.items()},
                "num_rules": len(self.rules),
                "num_propositions": len(self.propositions),
                "num_inferred": len(inferred),
            },
        )

    # ── Private helpers ───────────────────────────────────────────

    def _can_fire(self, rule: LogicalRule) -> bool:
        """Check if all antecedents are true and negated are false."""
        for ant in rule.antecedents:
            prop = self.propositions.get(ant)
            if prop is None or prop.value is not True:
                return False
        for neg in rule.negated_antecedents:
            prop = self.propositions.get(neg)
            if prop is not None and prop.value is True:
                return False
        return True

    def _can_fire_negated(self, rule: LogicalRule, target_value: bool) -> bool:
        """Check if a rule could infer the opposite of target_value."""
        if target_value:
            # Rule would need to infer False — we don't support that directly
            return False
        return self._can_fire(rule)

    def _node_to_prop_name(self, node) -> Optional[str]:
        """Convert a graph node to a proposition name."""
        text = node.text.strip().lower() if node.text else ""
        if not text:
            return None
        # Simple: use first 50 chars as proposition name
        return text[:50].replace(" ", "_").replace('"', "")

    def _node_to_propositions(self, node) -> dict[str, bool]:
        """Extract proposition-like facts from a node."""
        props = {}
        text = node.text.strip().lower() if node.text else ""

        # Map node type to generic propositions
        type_name = node.type.name if hasattr(node, 'type') else ""
        if type_name:
            props[f"type_{type_name.lower()}"] = True

        # Map text content to proposition
        prop_name = self._node_to_prop_name(node)
        if prop_name:
            props[prop_name] = True

        return props

    def _rule_applies_to_node(self, rule: LogicalRule, node_props: dict) -> bool:
        """Check if a rule's antecedents match a node's propositions."""
        if not rule.antecedents:
            return False
        return all(ant in node_props for ant in rule.antecedents)

    def clear(self) -> None:
        """Reset all propositions and rules."""
        self.propositions.clear()
        self.rules.clear()
        self._inferred.clear()
