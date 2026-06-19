"""Self-reflection operator — periodic graph introspection.

Supports both event-driven (Page-Hinkley) and periodic (fixed interval)
self-reflection. Uses belief tiers to assess reasoning quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from cognitive_engine.core.models import Graph, Node
from cognitive_engine.core.state import State

logger = logging.getLogger(__name__)


@dataclass
class SelfReflectionConfig:
    """Configuration for self-reflection behavior."""
    frequency: int = 5  # every N cycles
    enabled: bool = True
    ph_enabled: bool = False  # Page-Hinkley monitoring
    ph_threshold: float = 30.0  # Page-Hinkley threshold
    ph_delta: float = 0.001  # Page-Hinkley delta
    min_belief_threshold: float = 0.0  # never filter, tier-based
    max_nodes_analyzed: int = 200  # cap analysis for performance


@dataclass
class ReflectionResult:
    """Result of a self-reflection pass."""
    cycle: int
    tier_counts: dict[str, int]
    low_belief_ratio: float
    high_conflict_nodes: list[str]
    recommendations: list[str]
    ph_value: float = 0.0
    ph_triggered: bool = False


class SelfReflectOperator:
    """Self-reflection operator — assesses reasoning quality at intervals.

    Integrates with InferenceCycle via:
      - SelfReflectionConfig.frequency
      - Page-Hinkley monitoring for event-driven reflection
      - Tier-based analysis (high/medium/low/uninitialized)
    """

    def __init__(self, config: Optional[SelfReflectionConfig] = None) -> None:
        self.config = config or SelfReflectionConfig()
        self._ph_sum: float = 0.0
        self._ph_min: float = float("inf")
        self._ph_count: int = 0
        self._ph_value: float = 0.0
        self._ph_triggered: bool = False
        self._history: list[ReflectionResult] = []

    def should_reflect(self, cycle: int) -> bool:
        """Determine if reflection should occur this cycle."""
        if not self.config.enabled:
            return False
        if cycle % self.config.frequency == 0:
            return True
        if self.config.ph_enabled and self._ph_triggered:
            return True
        return False

    def reflect(self, state: State, cycle: int) -> ReflectionResult:
        """Perform self-reflection on current state."""
        graph = state.graph
        opinions = self._collect_opinions(graph)

        # Tier counts
        tier_counts = {"high": 0, "medium": 0, "low": 0, "uninitialized": 0}
        for _nid, (b, d, u, a) in opinions.items():
            tier = self._classify_tier(b)
            tier_counts[tier] += 1

        total = sum(tier_counts.values()) or 1
        low_ratio = (tier_counts["low"] + tier_counts["uninitialized"]) / total

        # High conflict nodes (have both incoming ATTACKS and outgoing SUPPORTS)
        high_conflict = self._find_conflict_nodes(graph)

        # Recommendations
        recommendations = self._generate_recommendations(
            graph, opinions, tier_counts, low_ratio, high_conflict, cycle,
        )

        # Update Page-Hinkley
        ph_triggered = False
        if self.config.ph_enabled:
            ph_triggered = self._update_ph(low_ratio)
        self._ph_triggered = ph_triggered

        result = ReflectionResult(
            cycle=cycle,
            tier_counts=tier_counts,
            low_belief_ratio=low_ratio,
            high_conflict_nodes=high_conflict,
            recommendations=recommendations,
            ph_value=self._ph_value,
            ph_triggered=ph_triggered,
        )
        self._history.append(result)
        return result

    @property
    def history(self) -> list[ReflectionResult]:
        return list(self._history)

    def _collect_opinions(self, graph: Graph) -> dict[str, tuple]:
        """Collect opinions from graph nodes (capped)."""
        opinions = {}
        for i, (nid, node) in enumerate(graph.nodes.items()):
            if i >= self.config.max_nodes_analyzed:
                break
            if node.opinion:
                opinions[str(nid)] = node.opinion
            else:
                opinions[str(nid)] = (0.0, 0.0, 1.0, 0.0)
        return opinions

    @staticmethod
    def _classify_tier(belief: float) -> str:
        """Classify belief into tier."""
        if belief >= 0.7:
            return "high"
        if belief >= 0.3:
            return "medium"
        if belief > 0.0:
            return "low"
        return "uninitialized"

    @staticmethod
    def _find_conflict_nodes(graph: Graph) -> list[str]:
        """Find nodes with conflicting evidence."""
        attack_types = {"ATTACKS", "REBUTS", "CONTRADICTS"}
        incoming_attacks: dict[str, int] = {}
        outgoing_supports: dict[str, int] = {}

        for edge in graph.edges.values():
            tid = str(edge.target_id)
            sid = str(edge.source_id)
            if edge.type.name in attack_types:
                incoming_attacks[tid] = incoming_attacks.get(tid, 0) + 1
            if edge.type.name in {"SUPPORTS", "INFERS"}:
                outgoing_supports[sid] = outgoing_supports.get(sid, 0) + 1

        conflict = []
        for nid in set(incoming_attacks) & set(outgoing_supports):
            if incoming_attacks[nid] > 1 and outgoing_supports[nid] > 1:
                conflict.append(nid)
        return sorted(conflict)[:10]

    def _generate_recommendations(
        self,
        graph: Graph,
        opinions: dict[str, tuple],
        tier_counts: dict[str, int],
        low_ratio: float,
        conflict_nodes: list[str],
        cycle: int,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs: list[str] = []
        total = sum(tier_counts.values()) or 1

        if tier_counts["uninitialized"] / total > 0.3:
            recs.append("Many nodes have uninitialized beliefs — consider evidence injection")
        if low_ratio > 0.5:
            recs.append("Over half of nodes are low/uninitialized — review extraction quality")
        if conflict_nodes:
            recs.append(f"{len(conflict_nodes)} high-conflict nodes detected — consider debate operator")
        if tier_counts["high"] / total > 0.8:
            recs.append("Most nodes have high belief — verify no overconfidence")

        # Convergence check
        if cycle > 10 and low_ratio > 0.7:
            recs.append("Stagnant reasoning with many low-belief nodes — consider abductive reasoning")

        return recs

    def _update_ph(self, value: float) -> bool:
        """Update Page-Hinkley statistic and check threshold."""
        self._ph_count += 1
        self._ph_sum += value
        mean = self._ph_sum / self._ph_count
        self._ph_min = min(self._ph_min, mean)
        self._ph_value = (mean - self._ph_min) - self.config.ph_delta
        return self._ph_value > self.config.ph_threshold
