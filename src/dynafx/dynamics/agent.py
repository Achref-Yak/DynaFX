"""Agent-Based Modeling (ABM) engine.

Provides autonomous agents with numeric state, perceive-decide-act
cycles, message passing, strategy switching, and aggregated metrics
that feed back into the shared simulation state dict.

Agents have:
  - numeric properties (budget, inventory, satisfaction, etc.)
  - behavioral rules (when condition → effect assignments)
  - topic-based message passing (SEND, mailbox, inbox perception)
  - named strategies with meta-rule controlled switching
  - perceive-decide-act loop per timestep
  - optional social network for peer influence

Aggregated metrics (avg, sum, min, max, var) are written back to the
"""

from __future__ import annotations

import copy
import logging
import math
import random
import zlib

_logger = logging.getLogger(__name__)
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

from dynafx.dynamics.dsl import AgentDef, AgentRuleDef

# Shared builtin functions for condition and effect eval namespaces
_ABM_BUILTINS: dict[str, Any] = {
    "MIN": min, "MAX": max,
    "ABS": abs, "EXP": math.exp, "LN": math.log,
    "SQRT": math.sqrt, "SIN": math.sin, "COS": math.cos,
    "PI": math.pi,
    "IF": lambda c, a, b: a if c else b,
    "always": True, "True": True, "False": False,
}


@dataclass
class Message:
    """A topic-based message sent by one agent to another."""
    sender_id: int
    sender_type: str
    target_type: str
    topic: str
    payload: dict[str, float] = field(default_factory=dict)
    ttl: int = 1


@dataclass
class AgentInstance:
    """Runtime instance of a single agent."""
    agent_def: AgentDef
    id: int
    state: dict[str, float] = field(default_factory=dict)
    neighbors: list[AgentInstance] = field(default_factory=list)
    mailbox: list[Message] = field(default_factory=list)
    _pending_outbox: list[Message] = field(default_factory=list)
    strategy: Optional[str] = None
    _strategy_locked_until: float = 0.0

    def perceive(self, env: dict[str, float], t: float = 0.0) -> dict[str, float]:
        """Return merged view of self + environment + mailbox + neighbors for rule evaluation."""
        merged = dict(env)
        merged.update(self.state)
        # Inject agent-type prefix for disambiguation
        for k, v in self.state.items():
            merged[f"{self.agent_def.name}.{k}"] = v
        # Inject mailbox metrics
        merged["inbox"] = float(len(self.mailbox))
        merged["inbox_total"] = float(len(self.mailbox))
        topic_counts: dict[str, int] = {}
        for msg in self.mailbox:
            topic_counts[msg.topic] = topic_counts.get(msg.topic, 0) + 1
        for topic, count in topic_counts.items():
            merged[f"inbox_{topic}"] = float(count)
        # Inject current strategy name
        merged["strategy"] = self.strategy or ""
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

    def decide(self, perceive_state: dict[str, float], t: float = 0.0,
               kb_builtins: Optional[dict[str, Any]] = None) -> list[tuple[str, float]]:
        """Evaluate meta-rules (always) then strategy-scoped or flat rules.

        Returns list of (property, delta). Handles SEND and SWITCH_STRATEGY
        side effects internally.
        """
        effects: list[tuple[str, float]] = []

        def _run_rules(rules: list[AgentRuleDef]) -> None:
            nonlocal effects, perceive_state
            for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
                if not rule.condition:
                    continue
                try:
                    if not _eval_condition(rule.condition, perceive_state, extra_builtins=kb_builtins):
                        continue
                except Exception as e:
                    _logger.warning("Rule condition '%s' eval failed — %s", rule.condition, e)
                    continue
                for eff_str in rule.effects:
                    if eff_str.startswith("SEND"):
                        msg = _parse_send(eff_str, self.id, self.agent_def.name, perceive_state, kb_builtins)
                        if msg:
                            self._pending_outbox.append(msg)
                    elif eff_str.startswith("SWITCH_STRATEGY"):
                        sname, cd = _parse_switch_strategy(eff_str)
                        if t >= self._strategy_locked_until and sname:
                            self.strategy = sname
                            self._strategy_locked_until = t + cd
                    else:
                        prop, delta = _eval_effect(eff_str, perceive_state, kb_builtins=kb_builtins)
                        if prop:
                            effects.append((prop, delta))
                            perceive_state[prop] = perceive_state.get(prop, 0.0) + delta

        # 1. Meta-rules (always evaluated regardless of active strategy)
        _run_rules(self.agent_def.meta_rules)

        # 2. Strategy-scoped or flat rules
        active_rules: list[AgentRuleDef] = []
        if self.agent_def.strategies and self.strategy:
            for s in self.agent_def.strategies:
                if s.name == self.strategy:
                    active_rules = s.rules
                    break
        else:
            active_rules = self.agent_def.rules
        _run_rules(active_rules)

        return effects

    def act(self, effects: list[tuple[str, float]]) -> None:
        """Apply effects to internal state, clamp to min/max."""
        prop_map = {p.name: p for p in self.agent_def.properties}
        for prop_name, delta in effects:
            if prop_name not in self.state:
                _logger.warning("Agent '%s' effect references unknown property '%s'", self.agent_def.name, prop_name)
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
                inst = AgentInstance(ad, i, state, [])
                # Set default strategy to first defined strategy if any
                if ad.strategies:
                    inst.strategy = ad.strategies[0].name
                self.instances.append(inst)
            # Build network for this agent type
            if ad.network_type != "none" and ad.count > 1:
                rng = random.Random(self._seed + zlib.adler32(ad.name.encode()) % (2**31))
                graph = _build_network(ad.count, ad.network_type, rng)
                # Assign neighbor references
                for inst_i in range(start_idx, len(self.instances)):
                    node_idx = inst_i - start_idx
                    neighbor_indices = list(graph.neighbors(node_idx))
                    self.instances[inst_i].neighbors = [
                        self.instances[start_idx + ni] for ni in neighbor_indices
                    ]

    def step(self, t: float, dt: float, env: dict[str, float]) -> dict[str, float]:
        """Run one ABM step with 4 phases. Returns aggregated metrics.

        Phases:
          1. Deliver — flush pending outboxes to recipient mailboxes
          2. Decide & Act — perceive → meta-rules + strategy rules → act
          3. Mailbox cleanup — decrement TTL, expire old messages
          4. Aggregate — compute and return metrics
        """
        # Phase 1: Deliver pending messages to recipient mailboxes
        for inst in self.instances:
            for msg in inst._pending_outbox:
                recipients = [
                    r for r in self.instances
                    if r.agent_def.name == msg.target_type
                ]
                for r in recipients:
                    r.mailbox.append(copy.copy(msg))
            inst._pending_outbox.clear()

        # Phase 2: Perceive → Decide → Act
        for inst in self.instances:
            perceive_state = inst.perceive(env, t)
            effects = inst.decide(perceive_state, t=t, kb_builtins=self._kb_builtins)
            inst.act(effects)

        # Phase 3: Mailbox cleanup (decrement TTL, expire)
        for inst in self.instances:
            alive: list[Message] = []
            for msg in inst.mailbox:
                msg.ttl -= 1
                if msg.ttl > 0:
                    alive.append(msg)
            inst.mailbox = alive

        # Phase 4: Aggregate
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
    _logger.warning("Unknown network type '%s', returning empty graph", network_type)
    return nx.Graph()


# ── Condition / Effect Evaluation ──────────────────────────────

def _eval_condition(condition: str, state: dict[str, float], extra_builtins: Optional[dict[str, Any]] = None) -> bool:
    """Evaluate a condition string against state dict.

    Supports: and, or, not, comparisons (>, <, >=, <=, ==, !=).
    Variable names resolve from state dict.
    """
    expr = condition.strip()

    # Build eval namespace with state values + shared builtins
    ns: dict[str, Any] = dict(_ABM_BUILTINS)
    ns["t"] = state.get("t", 0.0)
    for k, v in state.items():
        ns[k] = v
    if extra_builtins:
        ns.update(extra_builtins)

    try:
        result = eval(expr, {"__builtins__": {}}, ns)
        return bool(result)
    except Exception as e:
        _logger.warning("Condition eval failed: '%s' — %s", condition, e)
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
        ns: dict[str, Any] = dict(_ABM_BUILTINS)
        for k, v in state.items():
            ns[k] = v
        if kb_builtins:
            ns.update(kb_builtins)
        try:
            eval(effect_str, {"__builtins__": {}}, ns)
        except Exception as e:
            _logger.warning("KB_ASSERT effect eval failed: '%s' — %s", effect_str, e)
        return "", 0.0

    # Try: prop op= expr
    for op in ("+=", "-=", "*=", "/="):
        if op in effect_str:
            parts = effect_str.split(op, 1)
            prop_name = parts[0].strip()
            expr_str = parts[1].strip()

            # Evaluate the RHS
            ns: dict[str, Any] = dict(_ABM_BUILTINS)
            for k, v in state.items():
                ns[k] = v
            try:
                value = float(eval(expr_str, {"__builtins__": {}}, ns))
            except Exception as e:
                _logger.warning("Effect '%s' eval failed on RHS '%s' — %s", effect_str, expr_str, e)
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
        ns: dict[str, Any] = dict(_ABM_BUILTINS)
        for k, v in state.items():
            ns[k] = v
        try:
            value = float(eval(expr_str, {"__builtins__": {}}, ns))
        except Exception as e:
            _logger.warning("Effect '%s' assign eval failed on RHS '%s' — %s", effect_str, expr_str, e)
            value = 0.0
        current = state.get(prop_name, 0.0)
        return prop_name, value - current

    return "", 0.0


_SEND_RE = re.compile(r"SEND\s*\(\s*(.+?)\s*,\s*(.+?)\s*(?:,\s*(.+))?\)")
_SWITCH_RE = re.compile(r"SWITCH_STRATEGY\s*\(\s*(.+?)\s*(?:,\s*cooldown\s*=\s*(\d+(?:\.\d+)?)\s*)?\)")


def _split_top_level(s: str) -> list[str]:
    """Split string on commas not inside parentheses."""
    parts = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _parse_send(eff_str: str, sender_id: int, sender_type: str,
                state: dict[str, float],
                kb_builtins: Optional[dict[str, Any]] = None) -> Optional[Message]:
    """Parse 'SEND(Buyer, "order_qty", qty=100)' into a Message.

    Returns None if parsing fails.
    """
    m = _SEND_RE.match(eff_str.strip())
    if not m:
        return None
    target_type = m.group(1).strip().strip('"\' ')
    topic = m.group(2).strip().strip('"\' ')
    payload: dict[str, float] = {}
    kwargs_str = m.group(3)
    if kwargs_str:
        # Evaluate each kwarg=expr in the perceive state
        ns: dict[str, Any] = dict(_ABM_BUILTINS)
        for k, v in state.items():
            ns[k] = v
        if kb_builtins:
            ns.update(kb_builtins)
        for part in _split_top_level(kwargs_str):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    payload[k] = float(eval(v, {"__builtins__": {}}, ns))
                except Exception as _e:
                    _logger.warning("SEND payload eval failed for '%s=%s' — %s", k, v, _e)
                    payload[k] = 0.0
    return Message(
        sender_id=sender_id,
        sender_type=sender_type,
        target_type=target_type,
        topic=topic,
        payload=payload,
        ttl=1,
    )


def _parse_switch_strategy(eff_str: str) -> tuple[Optional[str], float]:
    """Parse 'SWITCH_STRATEGY("crisis", cooldown=10)' into (name, cooldown)."""
    m = _SWITCH_RE.match(eff_str.strip())
    if not m:
        return None, 0.0
    name = m.group(1).strip().strip('"\' ').rstrip(",")
    cd = float(m.group(2)) if m.group(2) else 0.0
    return name if name else None, cd
