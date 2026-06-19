"""Manager agent — lifecycle management, health assessment, reconfiguration.

The M-agent handles system initialization, termination, health monitoring,
and dynamic reconfiguration. Integrates with InferenceCycle for lifecycle hooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from cognitive_engine.core.models import Graph, Opinion

if TYPE_CHECKING:
    from cognitive_engine.kernel.inference_cycle import InferenceCycle

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """System health assessment."""
    healthy: bool = True
    convergence_rate: float = 0.0
    evidence_density: float = 0.0
    conflict_ratio: float = 0.0
    belief_variance: float = 0.0
    world_model_coverage: float = 0.0
    bottlenecks: list[str] = field(default_factory=list)


class ManagerAgent:
    """M-agent: lifecycle management, health assessment, reconfiguration.

    Responsibilities:
      - Initialize and terminate system components
      - Assess system health via graph metrics
      - Reconfigure InferenceCycle parameters based on health
      - Maintain health history for trend analysis
    """

    def __init__(self, inference_cycle: Optional[InferenceCycle] = None) -> None:
        self._cycle = inference_cycle
        self._start_time: float = 0.0
        self._health_history: list[HealthStatus] = []

    def initialize(self, context: Optional[dict] = None) -> None:
        """Initialize system with context configuration."""
        self._start_time = time.time()
        logger.info("M-agent initialized")

    def terminate(self) -> None:
        """Clean shutdown, flush pending state."""
        duration = time.time() - self._start_time
        logger.info("M-agent terminated after %.2fs", duration)

    def health_check(self, graph: Graph) -> HealthStatus:
        """Assess current system health from graph state."""
        health = self._compute_health(graph)
        self._health_history.append(health)
        if len(self._health_history) > 20:
            self._health_history.pop(0)
        return health

    def reconfigure(self, config: dict) -> None:
        """Dynamically reconfigure based on health assessment."""
        if self._cycle is None:
            return
        if "max_cycles" in config:
            self._cycle.config.max_cycles = config["max_cycles"]
        if "epsilon" in config:
            self._cycle.config.epsilon = config["epsilon"]
        logger.info("M-agent reconfigured: %s", config)

    def _compute_health(self, graph: Graph) -> HealthStatus:
        """Compute health metrics from graph state."""
        opinions = [n.opinion for n in graph.nodes.values() if n.opinion]
        uncertainties = [o[2] for o in opinions] if opinions else [0.0]
        beliefs = [o[0] for o in opinions] if opinions else [0.0]

        n = len(graph.nodes) or 1
        possible_edges = n * (n - 1) / 2
        evidence_density = len(graph.edges) / max(possible_edges, 1)

        attack_types = {"ATTACKS", "REBUTS", "CONTRADICTS"}
        attacks = sum(1 for e in graph.edges.values() if e.type.name in attack_types)
        conflict_ratio = attacks / max(len(graph.edges), 1)

        mean_belief = sum(beliefs) / max(len(beliefs), 1)
        belief_variance = sum((b - mean_belief) ** 2 for b in beliefs) / max(len(beliefs), 1)

        wm_types = {"AGENT", "PROCESS", "STATE", "GOAL", "ACTION", "RESOURCE", "CONSTRAINT"}
        wm_count = sum(1 for n in graph.nodes.values() if n.type.name in wm_types)
        world_model_coverage = wm_count / max(len(graph.nodes), 1)

        # Convergence rate from health history
        convergence_rate = 0.0
        if len(self._health_history) >= 2:
            prev = self._health_history[-1]
            curr_density = evidence_density
            convergence_rate = abs(curr_density - prev.evidence_density)

        # Identify bottlenecks
        bottlenecks: list[str] = []
        mean_u = sum(uncertainties) / max(len(uncertainties), 1)
        if mean_u > 0.8:
            bottlenecks.append("high_uncertainty")
        if conflict_ratio > 0.5:
            bottlenecks.append("excessive_conflict")
        if evidence_density < 0.01:
            bottlenecks.append("sparse_graph")
        if belief_variance < 0.01:
            bottlenecks.append("low_belief_variance")
        if world_model_coverage < 0.1:
            bottlenecks.append("low_world_model_coverage")

        healthy = len(bottlenecks) == 0

        return HealthStatus(
            healthy=healthy,
            convergence_rate=convergence_rate,
            evidence_density=evidence_density,
            conflict_ratio=conflict_ratio,
            belief_variance=belief_variance,
            world_model_coverage=world_model_coverage,
            bottlenecks=bottlenecks,
        )

    @property
    def health_history(self) -> list[HealthStatus]:
        """Return copy of health history."""
        return list(self._health_history)
