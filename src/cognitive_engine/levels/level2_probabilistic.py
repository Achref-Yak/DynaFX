"""Level 2: Probabilistic Reasoning.

Implements Bayesian networks with conditional probability tables,
MAP inference, Bayes rule, and expectation computation.

Core formulas:
    P(H|E) = P(E|H) * P(H) / P(E)     (Bayes rule)
    P(X_1,...,X_n) = Π_i P(X_i|Parents(X_i))  (joint distribution)
    H* = argmax_H P(H|E)               (MAP inference)
    E[X] = Σ_x x * P(x)               (expectation)

Usage:
    from cognitive_engine.levels.level2_probabilistic import ProbabilisticLevel
    level = ProbabilisticLevel()
    level.add_variable("guilt", [], {True: 0.3, False: 0.7})
    level.add_variable("fingerprint", ["guilt"],
                       {(True, True): 0.9, (True, False): 0.1,
                        (False, True): 0.05, (False, False): 0.95})
    level.observe("fingerprint", True)
    p_guilt = level.infer("guilt")  # P(guilt | fingerprint=True)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional
from uuid import UUID

from cognitive_engine.core.models import Graph, NodeType
from cognitive_engine.levels.base import BaseLevel, LevelOutput, ReasoningContext

logger = logging.getLogger(__name__)


@dataclass
class RandomVariable:
    """A random variable in the Bayesian network.

    Attributes:
        name: Unique identifier.
        parents: Names of parent variables.
        cpt: Conditional probability table.
            Keys are tuples of (parent_values..., self_value).
            Values are probabilities P(self_value | parent_values).
        domain: Possible values (default: [True, False]).
    """
    name: str
    parents: list[str] = field(default_factory=list)
    cpt: dict[tuple, float] = field(default_factory=dict)
    domain: list[Any] = field(default_factory=lambda: [True, False])


class ProbabilisticLevel(BaseLevel):
    """Level 2: Probabilistic Reasoning.

    Maintains a Bayesian network with variables, CPTs, and evidence.
    Supports exact inference via variable enumeration for small networks.
    """

    @property
    def name(self) -> str:
        return "Probabilistic Reasoning"

    @property
    def level_number(self) -> int:
        return 2

    def __init__(self) -> None:
        self.variables: dict[str, RandomVariable] = {}
        self.evidence: dict[str, Any] = {}

    def add_variable(
        self, name: str, parents: list[str], cpt: dict,
        domain: Optional[list] = None,
    ) -> None:
        """Add a random variable with its conditional probability table.

        CPT keys are tuples: (parent_val_1, ..., parent_val_n, self_val)
        CPT values are probabilities: P(self_val | parent_vals).
        """
        self.variables[name] = RandomVariable(
            name=name, parents=parents, cpt=cpt,
            domain=domain or [True, False],
        )

    def observe(self, name: str, value: Any) -> None:
        """Set evidence: variable name is observed to be value."""
        self.evidence[name] = value

    def clear_evidence(self) -> None:
        """Remove all evidence."""
        self.evidence.clear()

    def infer(self, target: str) -> float:
        """Compute P(target | evidence) via enumeration.

        For small networks, enumerates all configurations.
        """
        if target not in self.variables:
            return 0.5

        # Get all variable names in topological order
        var_names = self._topological_order()

        total_prob_true = 0.0
        total_prob_all = 0.0

        # Enumerate all configurations of non-evidence variables
        non_evidence_vars = [
            v for v in var_names
            if v != target and v not in self.evidence
        ]

        for vals in self._enumerate_configs(non_evidence_vars):
            config = dict(self.evidence)
            config.update(vals)

            # Compute P(config, target=True)
            config[target] = True
            prob_true = self._joint_probability(config, var_names)
            total_prob_true += prob_true

            # Compute P(config, target=False)
            config[target] = False
            prob_false = self._joint_probability(config, var_names)
            total_prob_all += prob_true + prob_false

        if total_prob_all == 0:
            return 0.5

        return total_prob_true / total_prob_all

    def map_inference(self) -> dict[str, Any]:
        """Find H* = argmax_H P(H | evidence).

        Returns the most probable configuration of all variables.
        """
        var_names = self._topological_order()
        non_evidence_vars = [v for v in var_names if v not in self.evidence]

        best_config = dict(self.evidence)
        best_prob = 0.0

        for vals in self._enumerate_configs(non_evidence_vars):
            config = dict(self.evidence)
            config.update(vals)
            prob = self._joint_probability(config, var_names)

            if prob > best_prob:
                best_prob = prob
                best_config = dict(config)

        return best_config

    def expectation(self, name: str) -> float:
        """Compute E[X] = Σ_x x * P(x).

        For boolean variables, returns P(X=True).
        """
        if name not in self.variables:
            return 0.5

        # Save and restore evidence
        old_evidence = dict(self.evidence)
        self.clear_evidence()

        p_true = self.infer(name)
        self.evidence = old_evidence

        return p_true

    def to_beliefs(self, graph: Graph) -> dict[UUID, float]:
        """Convert Bayesian beliefs to node belief dict.

        Maps graph nodes to their probability based on variable names.
        """
        beliefs = {}

        for node_id, node in graph.nodes.items():
            # Try to find a matching variable
            var_name = self._node_to_variable_name(node)
            if var_name and var_name in self.variables:
                beliefs[node_id] = self.infer(var_name)
            elif node.opinion:
                # Fall back to SL opinion projected probability
                b, d, u, a = node.opinion
                beliefs[node_id] = b + a * u
            else:
                beliefs[node_id] = 0.5

        return beliefs

    def compute(
        self, graph: Graph, context: ReasoningContext,
    ) -> LevelOutput:
        """Run probabilistic reasoning on the graph.

        1. Map graph nodes to random variables
        2. Set evidence from observed nodes
        3. Infer beliefs for all nodes
        4. Compute MAP configuration
        """
        if not graph.nodes:
            return LevelOutput(beliefs={}, metadata={})

        # Clear previous state
        self.clear_evidence()

        # Map graph nodes to variables
        for node_id, node in graph.nodes.items():
            var_name = self._node_to_variable_name(node)
            if var_name and var_name not in self.variables:
                # Create a simple variable for this node
                if node.opinion:
                    b, d, u, a = node.opinion
                    p_true = b + a * u
                else:
                    p_true = 0.5
                self.add_variable(
                    var_name, [],
                    {(True,): p_true, (False,): 1.0 - p_true},
                )

        # Set evidence from nodes with strong beliefs
        for node_id, node in graph.nodes.items():
            var_name = self._node_to_variable_name(node)
            if var_name and var_name in self.variables:
                if node.opinion:
                    b, d, u, a = node.opinion
                    p = b + a * u
                    if p > 0.8:
                        self.observe(var_name, True)
                    elif p < 0.2:
                        self.observe(var_name, False)

        # Compute beliefs
        beliefs = self.to_beliefs(graph)

        # Compute MAP
        map_config = self.map_inference()

        return LevelOutput(
            beliefs=beliefs,
            metadata={
                "map_configuration": {k: v for k, v in map_config.items()},
                "num_variables": len(self.variables),
                "num_evidence": len(self.evidence),
            },
        )

    # ── Private helpers ───────────────────────────────────────────

    def _topological_order(self) -> list[str]:
        """Return variables in topological order (parents before children)."""
        in_degree = {name: 0 for name in self.variables}
        children: dict[str, list[str]] = defaultdict(list)

        for name, var in self.variables.items():
            for parent in var.parents:
                if parent in self.variables:
                    in_degree[name] += 1
                    children[parent].append(name)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []
        while queue:
            name = queue.pop(0)
            order.append(name)
            for child in children[name]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Add any remaining (cycles)
        for name in self.variables:
            if name not in order:
                order.append(name)

        return order

    def _enumerate_configs(self, var_names: list[str]):
        """Enumerate all configurations of the given variables."""
        if not var_names:
            yield {}
            return

        domains = [self.variables[name].domain for name in var_names]
        for combo in product(*domains):
            yield dict(zip(var_names, combo))

    def _joint_probability(self, config: dict, var_names: list[str]) -> float:
        """Compute P(config) = Π_i P(X_i | Parents(X_i))."""
        prob = 1.0
        for name in var_names:
            if name not in self.variables:
                continue
            var = self.variables[name]
            value = config.get(name)

            # Build CPT key
            parent_values = tuple(config.get(p) for p in var.parents)
            key = parent_values + (value,)

            if key in var.cpt:
                prob *= var.cpt[key]
            elif (value,) in var.cpt:
                # No-parent CPT
                prob *= var.cpt[(value,)]
            else:
                # Uniform prior
                prob *= 1.0 / len(var.domain)

        return prob

    def _node_to_variable_name(self, node) -> Optional[str]:
        """Map a graph node to a variable name."""
        text = node.text.strip().lower() if node.text else ""
        if not text:
            return None
        return text[:50].replace(" ", "_").replace('"', "")
