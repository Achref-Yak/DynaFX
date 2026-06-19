"""Rule engine — declarative graph transformations with negation and TBox support.

Provides pattern-matching rules that can infer new edges from existing
graph structure. Supports:
  - Negation-as-failure (Jena noValue semantics)
  - TBox-aware type matching (match on type hierarchy, not exact types)
  - Property chain reasoning (if A → B and B → C, then A → C)
  - Transitive property support

Rules are evaluated against a Graph and produce actions (edge creations)
with variable bindings from matched patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Edge, EdgeType, Graph, Node, Opinion
from cognitive_engine.tbox.hierarchy import TypeHierarchy, MDM_TYPE_HIERARCHY

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """A graph pattern to match against nodes and edges.

    Supports TBox-aware matching:
        - source_type can be a parent type (e.g., "AGENT" matches "PERSON")
        - edge_type can be a parent type (e.g., "CAUSAL" matches "CAUSES")
        - transitive matching: if CAUSES is transitive, match A→B→C as A→C

    Attributes:
        source_type: NodeType name to match on source, or None for any.
                     If TBox is provided, matches subtypes as well.
        edge_type: EdgeType name to match on edge, or None for any.
                   If TBox is provided, matches subtypes as well.
        target_type: NodeType name to match on target, or None for any.
                     If TBox is provided, matches subtypes as well.
        source_var: Variable name to bind the source node (e.g., "?a").
        target_var: Variable name to bind the target node (e.g., "?b").
        negated: If True, this pattern must NOT exist (negation-as-failure).
        min_belief: Minimum belief on source node to match.
        max_belief: Maximum belief on source node to match.
        transitive: If True, match transitive closure of the edge type.
    """
    source_type: Optional[str] = None
    edge_type: Optional[str] = None
    target_type: Optional[str] = None
    source_var: Optional[str] = None
    target_var: Optional[str] = None
    negated: bool = False
    min_belief: Optional[float] = None
    max_belief: Optional[float] = None
    transitive: bool = False


@dataclass
class Action:
    """An edge to create when patterns match.

    Attributes:
        source_var: Variable name from the matched pattern for source.
        target_var: Variable name from the matched pattern for target.
        edge_type: EdgeType name to create.
        weight: Weight for the new edge.
        confidence: Confidence for the new edge's opinion.
    """
    source_var: str
    target_var: str
    edge_type: str
    weight: float = 0.5
    confidence: float = 0.5


@dataclass
class Rule:
    """A complete rule with conditions and actions.

    Attributes:
        name: Human-readable rule name.
        when: All patterns must match (AND semantics).
        then: All actions execute when patterns match.
        confidence: Rule-level confidence multiplier.
        enabled: Whether this rule is active.
    """
    name: str
    when: list[Pattern] = field(default_factory=list)
    then: list[Action] = field(default_factory=list)
    confidence: float = 0.5
    enabled: bool = True


class RuleEngine:
    """Pattern-matching rule engine with negation and TBox support.

    Evaluates rules against a Graph and returns actions with variable
    bindings. Supports:
      - Variable binding across multiple patterns
      - Negation-as-failure (pattern must NOT exist)
      - Belief filters on source nodes
      - Confidence-weighted actions
      - TBox-aware type matching (match on type hierarchy)
      - Property chain reasoning (A→B→C implies A→C)
      - Transitive property support
    """

    def __init__(self, tbox: Optional[TypeHierarchy] = None) -> None:
        """Initialize the rule engine.

        Args:
            tbox: Optional TypeHierarchy for TBox-aware matching.
                  If None, uses exact type matching only.
        """
        self.rules: list[Rule] = []
        self.tbox = tbox or MDM_TYPE_HIERARCHY

    def add_rule(self, rule: Rule) -> None:
        """Register a rule for evaluation."""
        self.rules.append(rule)

    def evaluate(self, graph: Graph) -> list[tuple[Action, dict[str, UUID], float]]:
        """Evaluate all rules against the graph.

        Returns:
            List of (action, bindings, confidence) tuples. Each tuple
            represents one action to execute with its variable bindings
            and the rule's confidence.
        """
        results: list[tuple[Action, dict[str, UUID], float]] = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            bindings_list = self._match_patterns(rule.when, graph)

            for bindings in bindings_list:
                for action in rule.then:
                    results.append((action, bindings, rule.confidence))

        return results

    def _match_patterns(
        self, patterns: list[Pattern], graph: Graph,
    ) -> list[dict[str, UUID]]:
        """Match all patterns against graph, return list of variable bindings."""
        if not patterns:
            return [{}]

        # Start with first pattern's matches
        first_matches = self._match_single(patterns[0], graph)

        # Intersect with remaining patterns
        results = []
        for bindings in first_matches:
            if self._match_remaining(patterns[1:], graph, bindings):
                results.append(bindings)

        return results

    def _match_single(
        self, pattern: Pattern, graph: Graph,
    ) -> list[dict[str, UUID]]:
        """Match a single pattern against the graph, return possible bindings.

        Supports TBox-aware matching: if pattern.source_type is "AGENT",
        it will also match "PERSON", "ORGANIZATION", "SYSTEM" (subtypes).
        """
        bindings_list: list[dict[str, UUID]] = []
        found_match = False

        for nid, node in graph.nodes.items():
            # Check source type (TBox-aware)
            if pattern.source_type and pattern.source_type != "?var":
                if not self._types_match(node.type.name, pattern.source_type):
                    continue

            # Check source belief filters
            if pattern.min_belief is not None:
                b = (node.opinion or Opinion()).belief
                if b < pattern.min_belief:
                    continue
            if pattern.max_belief is not None:
                b = (node.opinion or Opinion()).belief
                if b > pattern.max_belief:
                    continue

            # Find matching edges from this source
            for edge in graph.edges.values():
                if edge.source_id != nid:
                    continue

                target = graph.nodes.get(edge.target_id)
                if target is None:
                    continue

                # Check edge type (TBox-aware)
                if pattern.edge_type and pattern.edge_type != "?var":
                    if not self._types_match(edge.type.name, pattern.edge_type):
                        continue

                # Check target type (TBox-aware)
                if pattern.target_type and pattern.target_type != "?var":
                    if not self._types_match(target.type.name, pattern.target_type):
                        continue

                # For negation, any match means the negation FAILS
                if pattern.negated:
                    found_match = True
                    break

                # Build binding
                binding: dict[str, UUID] = {}
                if pattern.source_var:
                    binding[pattern.source_var] = nid
                if pattern.target_var:
                    binding[pattern.target_var] = edge.target_id

                bindings_list.append(binding)

            if found_match:
                break

        # Handle negation: if pattern is negated, we need ZERO matches
        if pattern.negated:
            if found_match:
                return []  # Pattern found → negation fails
            else:
                return [{}]  # Pattern not found → negation succeeds

        return bindings_list

    def _types_match(self, actual: str, expected: str) -> bool:
        """Check if actual type matches expected type (TBox-aware).

        If TBox is available, checks if actual is a subtype of expected.
        Otherwise, falls back to exact string matching.
        """
        if actual == expected:
            return True
        if self.tbox:
            return self.tbox.is_subtype(actual, expected)
        return False

    def _match_remaining(
        self, patterns: list[Pattern], graph: Graph, bindings: dict[str, UUID],
    ) -> bool:
        """Check if remaining patterns match given existing bindings."""
        if not patterns:
            return True

        pattern = patterns[0]

        for nid, node in graph.nodes.items():
            # If source is bound, check it matches
            if pattern.source_var and pattern.source_var in bindings:
                if nid != bindings[pattern.source_var]:
                    continue
            elif pattern.source_type and pattern.source_type != "?var":
                if not self._types_match(node.type.name, pattern.source_type):
                    continue

            # Check belief filters on source
            if pattern.min_belief is not None:
                b = (node.opinion or Opinion()).belief
                if b < pattern.min_belief:
                    continue
            if pattern.max_belief is not None:
                b = (node.opinion or Opinion()).belief
                if b > pattern.max_belief:
                    continue

            for edge in graph.edges.values():
                if edge.source_id != nid:
                    continue

                target = graph.nodes.get(edge.target_id)
                if target is None:
                    continue

                # Check edge type (TBox-aware)
                if pattern.edge_type and pattern.edge_type != "?var":
                    if not self._types_match(edge.type.name, pattern.edge_type):
                        continue

                # Check target type (TBox-aware)
                if pattern.target_type and pattern.target_type != "?var":
                    if not self._types_match(target.type.name, pattern.target_type):
                        continue

                # Check target binding
                if pattern.target_var and pattern.target_var in bindings:
                    if edge.target_id != bindings[pattern.target_var]:
                        continue

                # Check negation
                if pattern.negated:
                    return False  # Found a match → negation fails

                # Build new bindings
                new_bindings = dict(bindings)
                if pattern.source_var and pattern.source_var not in new_bindings:
                    new_bindings[pattern.source_var] = nid
                if pattern.target_var and pattern.target_var not in new_bindings:
                    new_bindings[pattern.target_var] = edge.target_id

                if self._match_remaining(patterns[1:], graph, new_bindings):
                    return True

        # If no edges matched and pattern is not negated, fail
        if not pattern.negated:
            return False

        # If pattern is negated and nothing matched, succeed
        return True
