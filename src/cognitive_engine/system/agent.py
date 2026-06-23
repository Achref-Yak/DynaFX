"""Agent-Based Modeling (ABM) engine.

Provides autonomous agents with numeric state, perceive-decide-act
cycles, and aggregated metrics that feed back into the shared
simulation state dict.

Agents have:
  - numeric properties (budget, inventory, satisfaction, etc.)
  - behavioral rules (when condition → effect assignments)
  - perceive-decide-act loop per timestep

Aggregated metrics (avg, sum, min, max, var) are written back to the
shared state dict so SD and DES expressions can reference agent state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cognitive_engine.system.dsl import AgentDef, AgentPropDef, AgentRuleDef


@dataclass
class AgentInstance:
    """Runtime instance of a single agent."""
    agent_def: AgentDef
    id: int
    state: dict[str, float] = field(default_factory=dict)

    def perceive(self, env: dict[str, float]) -> dict[str, float]:
        """Return merged view of self + environment for rule evaluation."""
        merged = dict(env)
        merged.update(self.state)
        # Inject agent-type prefix for disambiguation
        for k, v in self.state.items():
            merged[f"{self.agent_def.name}.{k}"] = v
        return merged

    def decide(self, perceive_state: dict[str, float]) -> list[tuple[str, float]]:
        """Evaluate rules in priority order, return list of (property, delta)."""
        effects: list[tuple[str, float]] = []
        sorted_rules = sorted(
            self.agent_def.rules,
            key=lambda r: r.priority,
            reverse=True,
        )
        for rule in sorted_rules:
            if not rule.condition:
                continue
            try:
                if not _eval_condition(rule.condition, perceive_state):
                    continue
            except Exception:
                continue
            for eff_str in rule.effects:
                prop, delta = _eval_effect(eff_str, perceive_state)
                if prop:
                    effects.append((prop, delta))
                    # Update perceive_state so subsequent effects see prior changes
                    perceive_state[prop] = perceive_state.get(prop, 0.0) + delta
        return effects

    def act(self, effects: list[tuple[str, float]]) -> None:
        """Apply effects to internal state, clamp to min/max."""
        prop_map = {p.name: p for p in self.agent_def.properties}
        for prop_name, delta in effects:
            if prop_name not in self.state:
                continue
            self.state[prop_name] += delta
            # Clamp
            pd = prop_map.get(prop_name)
            if pd:
                self.state[prop_name] = max(pd.min, min(pd.max, self.state[prop_name]))


class ABMEngine:
    """Agent-Based Modeling simulation engine.

    Manages multiple agent types, each with N instances.
    Each step: perceive → decide → act → aggregate.
    """

    def __init__(self, agents: list[AgentDef]):
        self.agent_defs = agents
        self.instances: list[AgentInstance] = []

    def initialize(self) -> None:
        """Create agent instances from definitions."""
        self.instances = []
        for ad in self.agent_defs:
            for i in range(ad.count):
                state = {p.name: p.initial for p in ad.properties}
                self.instances.append(AgentInstance(ad, i, state))

    def step(self, t: float, dt: float, env: dict[str, float]) -> dict[str, float]:
        """Run one ABM step. Returns aggregated metrics for shared state."""
        for inst in self.instances:
            perceive_state = inst.perceive(env)
            effects = inst.decide(perceive_state)
            inst.act(effects)
        return self.get_metrics()

    def get_metrics(self) -> dict[str, float]:
        """Aggregate agent properties into metrics dict.

        Naming: {AgentType}_{property}_{aggregation}
        e.g. Buyer_budget_avg, Seller_inventory_sum
        """
        metrics: dict[str, float] = {}

        # Group instances by agent type
        by_type: dict[str, list[AgentInstance]] = {}
        for inst in self.instances:
            by_type.setdefault(inst.agent_def.name, []).append(inst)

        for type_name, instances in by_type.items():
            # Collect all property names for this type
            prop_names: set[str] = set()
            for inst in instances:
                prop_names.update(inst.state.keys())

            for prop_name in prop_names:
                values = [inst.state.get(prop_name, 0.0) for inst in instances]
                if not values:
                    continue

                metrics[f"{type_name}_{prop_name}_avg"] = sum(values) / len(values)
                metrics[f"{type_name}_{prop_name}_sum"] = sum(values)
                metrics[f"{type_name}_{prop_name}_min"] = min(values)
                metrics[f"{type_name}_{prop_name}_max"] = max(values)

                if len(values) > 1:
                    mean = metrics[f"{type_name}_{prop_name}_avg"]
                    variance = sum((v - mean) ** 2 for v in values) / len(values)
                    metrics[f"{type_name}_{prop_name}_var"] = variance

            metrics[f"{type_name}_count"] = float(len(instances))

        return metrics


# ── Condition / Effect Evaluation ──────────────────────────────

def _eval_condition(condition: str, state: dict[str, float]) -> bool:
    """Evaluate a condition string against state dict.

    Supports: and, or, not, comparisons (>, <, >=, <=, ==, !=).
    Variable names resolve from state dict.
    """
    # Normalize: replace Python operators
    expr = condition.strip()
    expr = expr.replace(" and ", " and ")
    expr = expr.replace(" or ", " or ")
    expr = expr.replace(" not ", " not ")

    # Build eval namespace with state values
    ns: dict[str, Any] = {}
    for k, v in state.items():
        ns[k] = v
    # Also inject comparison builtins
    ns["True"] = True
    ns["False"] = False
    ns["always"] = True

    try:
        result = eval(expr, {"__builtins__": {}}, ns)
        return bool(result)
    except Exception:
        return False


def _eval_effect(effect_str: str, state: dict[str, float]) -> tuple[str, float]:
    """Parse and evaluate an effect string like 'budget -= Price'.

    Returns (property_name, delta).
    """
    effect_str = effect_str.strip()

    # Try: prop op= expr
    for op in ("+=", "-=", "*=", "/="):
        if op in effect_str:
            parts = effect_str.split(op, 1)
            prop_name = parts[0].strip()
            expr_str = parts[1].strip()

            # Evaluate the RHS
            ns: dict[str, Any] = {}
            for k, v in state.items():
                ns[k] = v
            try:
                value = float(eval(expr_str, {"__builtins__": {}}, ns))
            except Exception:
                value = 0.0

            if op == "+=":
                return prop_name, value
            elif op == "-=":
                return prop_name, -value
            elif op == "*=":
                current = state.get(prop_name, 0.0)
                return prop_name, current * value - current
            elif op == "/=":
                current = state.get(prop_name, 0.0)
                if abs(value) > 1e-12:
                    return prop_name, current / value - current
                return prop_name, 0.0

    # Try: prop = expr (absolute set)
    if "=" in effect_str:
        parts = effect_str.split("=", 1)
        prop_name = parts[0].strip()
        expr_str = parts[1].strip()
        ns: dict[str, Any] = {}
        for k, v in state.items():
            ns[k] = v
        try:
            value = float(eval(expr_str, {"__builtins__": {}}, ns))
        except Exception:
            value = 0.0
        current = state.get(prop_name, 0.0)
        return prop_name, value - current

    return "", 0.0
