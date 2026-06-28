"""Policy engine — evaluates operator policies against state.

Supports:
    - YAML policy loading (from files or dicts)
    - Builtin policies (default, scientific)
    - Policy evaluation with state feature extraction
    - Operator selection with provenance logging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from dynafx.core.state import State
from dynafx.policy.schema import OperatorPolicy, ThenAction
from dynafx.policy.builtin import BUILTIN_POLICIES

logger = logging.getLogger(__name__)


@dataclass
class PolicySelection:
    """Result of a policy evaluation."""
    operators: list[str] = field(default_factory=list)
    order: str = "sequential"
    policy_name: str = "default"
    rule_index: int = -1
    reason: str = ""
    state_metrics: dict = field(default_factory=dict)


class PolicyEngine:
    """Evaluates operator policies against current state."""

    def __init__(self, policy: Optional[OperatorPolicy] = None):
        self._policy = policy

    @property
    def policy(self) -> OperatorPolicy:
        return self._policy or BUILTIN_POLICIES["default"]

    @policy.setter
    def policy(self, p: OperatorPolicy) -> None:
        self._policy = p

    def select(
        self,
        state: State,
        cycle: int = 0,
        domain: str = "general",
    ) -> PolicySelection:
        """Select operators based on current state and policy.

        Args:
            state: Current reasoning state.
            cycle: Current inference cycle number.
            domain: Active domain name.

        Returns:
            PolicySelection with chosen operators and metadata.
        """
        metrics = self._extract_metrics(state, cycle, domain)

        for i, rule in enumerate(self.policy.rules):
            if rule.when.matches(metrics):
                return PolicySelection(
                    operators=rule.then.operators,
                    order=rule.then.order,
                    policy_name=self.policy.name,
                    rule_index=i,
                    reason=f"Rule {i}: {self._format_condition(rule.when)}",
                    state_metrics=metrics,
                )

        return PolicySelection(
            operators=self.policy.fallback.operators,
            order=self.policy.fallback.order,
            policy_name=self.policy.name,
            rule_index=-1,
            reason="Fallback rule",
            state_metrics=metrics,
        )

    def load_yaml(self, yaml_content: str) -> OperatorPolicy:
        """Load a policy from YAML content."""
        yaml = _get_yaml_parser()
        data = yaml.safe_load(yaml_content)
        return self._from_dict(data)

    def _from_dict(self, data: dict) -> OperatorPolicy:
        """Convert a dict (from YAML) to an OperatorPolicy."""
        from dynafx.policy.schema import (
            OperatorPolicy, PolicyRule, WhenCondition, ThenAction,
        )
        rules = []
        for rule_data in data.get("rules", []):
            when_data = rule_data.get("when", {})
            then_data = rule_data.get("then", {})
            rules.append(PolicyRule(
                when=WhenCondition(**when_data),
                then=ThenAction(**then_data),
            ))
        fallback_data = data.get("fallback", {})
        return OperatorPolicy(
            name=data.get("name", "custom"),
            description=data.get("description", ""),
            rules=rules,
            fallback=ThenAction(**fallback_data) if fallback_data else ThenAction(),
        )

    def _extract_metrics(
        self, state: State, cycle: int, domain: str,
    ) -> dict:
        """Extract state metrics for policy evaluation."""
        graph = state.graph
        opinions = [n.opinion for n in graph.nodes.values() if n.opinion]
        uncertainties = [o[2] for o in opinions] if opinions else [0.0]
        mean_u = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0

        contradiction_types = {"CONTRADICTS", "ATTACKS", "REBUTS"}
        contradictions = sum(
            1 for e in graph.edges.values()
            if e.type.name in contradiction_types
        )

        last_op = ""
        if hasattr(state, 'trace') and state.trace._entries:
            last_delta = state.trace._entries[-1]
            last_op = last_delta.operator

        metrics = {
            "cycle": cycle,
            "domain": domain,
            "graph_node_count": len(graph.nodes),
            "graph_has_contradictions": contradictions > 0,
            "graph_mean_uncertainty": mean_u,
            "convergence_stalled": state.metadata.get("convergence_stalled", False),
            "last_operator": last_op,
            "graph_world_model_ratio": self._world_model_ratio(graph),
            "graph_causal_edges": self._causal_edge_count(graph),
            "graph_causal_chain_depth": self._causal_chain_depth(graph),
            "graph_feedback_loops": self._feedback_loop_count(graph),
        }
        return metrics

    def _format_condition(self, when) -> str:
        parts = []
        for key, value in vars(when).items():
            if value is not None:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    @staticmethod
    def _world_model_ratio(graph) -> float:
        """Compute ratio of world-model nodes to total nodes."""
        wm_types = {"AGENT", "PROCESS", "STATE", "GOAL", "ACTION", "RESOURCE", "CONSTRAINT"}
        wm_count = sum(1 for n in graph.nodes.values() if n.type.name in wm_types)
        return wm_count / max(len(graph.nodes), 1)

    @staticmethod
    def _causal_edge_count(graph) -> int:
        """Count CAUSES edges in the graph."""
        return sum(1 for e in graph.edges.values() if e.type.name == "CAUSES")

    @staticmethod
    def _causal_chain_depth(graph) -> int:
        """Longest chain of CAUSES edges (topological depth)."""
        from dynafx.core.math import causal_chain_depth
        return causal_chain_depth(graph.nodes, graph.edges)

    @staticmethod
    def _feedback_loop_count(graph) -> int:
        """Count directed cycles in the CAUSES subgraph."""
        from dynafx.core.math import feedback_loop_count
        return feedback_loop_count(graph.nodes, graph.edges)


_yaml_parser = None


def _get_yaml_parser():
    global _yaml_parser
    if _yaml_parser is None:
        try:
            import yaml as _yaml_parser
        except ImportError:
            logger.warning("PyYAML not installed, using fallback parser")
            _yaml_parser = None
    return _yaml_parser
