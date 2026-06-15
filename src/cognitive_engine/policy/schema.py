"""Pydantic models for the declarative operator policy language.

Policy schema (YAML):
    when:
        conditions evaluated against current state
    then:
        operators to execute and their ordering
    fallback:
        what to run if no condition matches
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WhenCondition:
    """Condition evaluated against current state.

    Available conditions:
        - graph.node_count > N
        - graph.has_contradictions: true/false
        - graph.mean_uncertainty > θ
        - cycle: '>N' / '==N' / '<N'
        - domain: legal / scientific / general
        - convergence.stalled: true/false
        - last_operator: <operator_name>
    """
    cycle: Optional[str] = None
    domain: Optional[str] = None
    graph_node_count: Optional[str] = None
    graph_has_contradictions: Optional[bool] = None
    graph_mean_uncertainty: Optional[str] = None
    convergence_stalled: Optional[bool] = None
    last_operator: Optional[str] = None

    def matches(self, state_metrics: dict) -> bool:
        """Check if this condition matches the current state."""
        for key, value in vars(self).items():
            if value is None:
                continue
            actual = state_metrics.get(key)
            if actual is None:
                return False
            if isinstance(value, bool):
                if actual != value:
                    return False
            elif isinstance(value, str) and value.startswith(('>', '<', '==')):
                op = value[:2] if len(value) > 1 and value[1] == '=' else value[0]
                num = float(value.lstrip('><='))
                if op == '>' and not (actual > num):
                    return False
                elif op == '<' and not (actual < num):
                    return False
                elif op in ('=', '==') and actual != num:
                    return False
            elif isinstance(value, str) and str(actual) != value:
                return False
        return True


@dataclass
class ThenAction:
    """Operators to execute when conditions match."""
    operators: list[str] = field(default_factory=list)
    order: str = "sequential"  # sequential | parallel | priority
    priority: list[float] = field(default_factory=list)


@dataclass
class PolicyRule:
    """A single policy rule: when/then."""
    when: WhenCondition = field(default_factory=WhenCondition)
    then: ThenAction = field(default_factory=ThenAction)


@dataclass
class OperatorPolicy:
    """Full operator policy with rules and fallback."""
    name: str = "default"
    description: str = ""
    rules: list[PolicyRule] = field(default_factory=list)
    fallback: ThenAction = field(default_factory=lambda: ThenAction(
        operators=["propagate", "verify"],
        order="sequential",
    ))

    def evaluate(self, state_metrics: dict) -> ThenAction:
        """Evaluate policy rules against state metrics.

        Returns the first matching ThenAction, or fallback.
        """
        for rule in self.rules:
            if rule.when.matches(state_metrics):
                return rule.then
        return self.fallback
