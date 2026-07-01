"""Agent-Based Modeling (ABM) engine.

Provides autonomous agents with numeric state, perceive-decide-act
cycles, and aggregated metrics that feed back into the shared
simulation state dict.

Agents have:
  - numeric properties (budget, inventory, satisfaction, etc.)
  - behavioral rules (when condition → effect assignments)
  - perceive-decide-act loop per timestep
  - optional social network for peer influence

Aggregated metrics (avg, sum, min, max, var) are written back to the
shared state dict so SD and DES expressions can reference agent state.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

from dynafx.dynamics.dsl import AgentDef, AgentPropDef, AgentRuleDef


@dataclass
class AgentInstance:
    """Runtime instance of a single agent."""
    agent_def: AgentDef
    id: int
    state: dict[str, float] = field(default_factory=dict)
    neighbors: list[AgentInstance] = field(default_factory=list)

    def perceive(self, env: dict[str, float]) -> dict[str, float]:
        """Return merged view of self + environment + neighbor metrics for rule evaluation."""
        merged = dict(env)
        merged.update(self.state)
        # Inject agent-type prefix for disambiguation
        for k, v in self.state.items():
            merged[f"{self.agent_def.name}.{k}"] = v
        # Inject neighbor aggregate metrics
        if self.neighbors:
            n_props: dict[str, list[float]] = {}
            for n in self.neighbors:
                for k, v in n.state.items():
                    n_props.setdefault(k, []).append(v)
            for prop, vals in n_props.items():
                merged[f"neighbor_{prop}_avg"] = sum(vals) / len(vals)
                merged[f"neighbor_{prop}_min"] = min(vals)
                merged[f"neighbor_{prop}_max"] = max(vals)
            merged["neighbor_count"] = float(len(self.neighbors))
        return merged

    def decide(self, perceive_state: dict[str, float], kb_builtins: Optional[dict[str, Any]] = None) -> list[tuple[str, float]]:
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
                if not _eval_condition(rule.condition, perceive_state, extra_builtins=kb_builtins):
                    continue
            except Exception:
                continue
            for eff_str in rule.effects:
                prop, delta = _eval_effect(eff_str, perceive_state, kb_builtins=kb_builtins)
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
    Supports optional social networks for peer influence.
    """

    def __init__(self, agents: list[AgentDef], seed: int = 42, kb_builtins: Optional[dict[str, Any]] = None):
        self.agent_defs = agents
        self.instances: list[AgentInstance] = []
        self._seed = seed
        self._kb_builtins = kb_builtins or {}

    def initialize(self) -> None:
        """Create agent instances from definitions and build networks."""
        self.instances = []
        for ad in self.agent_defs:
            start_idx = len(self.instances)
            for i in range(ad.count):
                state = {p.name: p.initial for p in ad.properties}
                self.instances.append(AgentInstance(ad, i, state, []))
            # Build network for this agent type
            if ad.network_type != "none" and ad.count > 1:
                rng = random.Random(self._seed + hash(ad.name) % (2**31))
                graph = _build_network(ad.count, ad.network_type, rng)
                # Assign neighbor references
                for inst_i in range(start_idx, len(self.instances)):
                    node_idx = inst_i - start_idx
                    neighbor_indices = list(graph.neighbors(node_idx))
                    self.instances[inst_i].neighbors = [
                        self.instances[start_idx + ni] for ni in neighbor_indices
                    ]

    def step(self, t: float, dt: float, env: dict[str, float]) -> dict[str, float]:
        """Run one ABM step. Returns aggregated metrics for shared state."""
        for inst in self.instances:
            perceive_state = inst.perceive(env)
            effects = inst.decide(perceive_state, kb_builtins=self._kb_builtins)
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


def _build_network(count: int, network_type: str, rng: random.Random) -> nx.Graph:
    """Build a graph for agent connections based on network type.

    Args:
        count: Number of agents.
        network_type: one of 'complete', 'random', 'small-world', 'scale-free'.
        rng: Seeded random instance for reproducibility.

    Returns:
        networkx Graph with nodes 0..count-1.
    """
    if network_type == "complete":
        return nx.complete_graph(count)
    if network_type == "random":
        return nx.erdos_renyi_graph(count, p=0.2, seed=rng)
    if network_type == "small-world":
        k = max(2, min(4, count - 1))
        return nx.watts_strogatz_graph(count, k=k, p=0.2, seed=rng)
    if network_type == "scale-free":
        m = max(1, min(2, count - 1))
        return nx.barabasi_albert_graph(count, m=m, seed=rng)
    return nx.Graph()


# ── Condition / Effect Evaluation ──────────────────────────────

def _eval_condition(condition: str, state: dict[str, float], extra_builtins: Optional[dict[str, Any]] = None) -> bool:
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

    if extra_builtins:
        ns.update(extra_builtins)

    try:
        result = eval(expr, {"__builtins__": {}}, ns)
        return bool(result)
    except Exception:
        return False


def _eval_effect(effect_str: str, state: dict[str, float],
                 kb_builtins: Optional[dict[str, Any]] = None) -> tuple[str, float]:
    """Parse and evaluate an effect string like 'budget -= Price'.

    Returns (property_name, delta).

    Args:
        effect_str: The effect expression string.
        state: Current agent property values (used for eval namespace).
        kb_builtins: Optional dict of KB_QUERY/KB_ASSERT builtins.
    """
    effect_str = effect_str.strip()

    # Handle KB_ASSERT side effect (evaluate function call, no property change)
    if effect_str.startswith("KB_ASSERT") and "(" in effect_str:
        ns: dict[str, Any] = {}
        for k, v in state.items():
            ns[k] = v
        if kb_builtins:
            ns.update(kb_builtins)
        try:
            eval(effect_str, {"__builtins__": {}}, ns)
        except Exception:
            pass
        return "", 0.0

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
