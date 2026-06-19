"""Specialist agents — domain-specific heuristic processors and hyper-heuristic selector.

O-agents are lightweight processors that handle specific operator sequences.
The HyperHeuristic selects the best O-agent based on performance history
and context similarity.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from cognitive_engine.core.state import State

logger = logging.getLogger(__name__)


class SpecialistAgent:
    """O-agent: domain-specific heuristic processor.

    Each specialist handles a specific set of operators and reports
    its confidence in handling a given context.
    """

    def __init__(self, name: str, operator_names: list[str]) -> None:
        self.name = name
        self.operator_names = operator_names
        self._performance: list[float] = []

    def can_handle(self, context: dict) -> float:
        """Return confidence that this agent can handle the context.

        Override in subclasses for domain-specific logic.
        Base implementation returns 0.5.
        """
        return 0.5

    def execute(self, state: State, operators: dict[str, Any]) -> State:
        """Execute this agent's operators on the state."""
        for op_name in self.operator_names:
            if op_name in operators:
                try:
                    state = operators[op_name](state)
                except Exception:
                    logger.warning(
                        "SpecialistAgent %s: operator %s failed",
                        self.name, op_name,
                    )
        return state

    def record_performance(self, score: float) -> None:
        """Record a performance score (0-1)."""
        self._performance.append(score)

    @property
    def avg_performance(self) -> float:
        """Average performance over last 10 evaluations."""
        if not self._performance:
            return 0.0
        recent = self._performance[-10:]
        return sum(recent) / len(recent)


class HyperHeuristic:
    """Selects the best O-agent based on performance history and context.

    Uses a combination of:
      - Agent self-assessment (can_handle)
      - Historical performance for similar contexts
      - Context similarity matching
    """

    def __init__(self, agents: Optional[list[SpecialistAgent]] = None) -> None:
        self._agents: list[SpecialistAgent] = agents or []
        self._context_scores: dict[str, dict[str, float]] = {}

    def register(self, agent: SpecialistAgent) -> None:
        """Register a specialist agent."""
        self._agents.append(agent)

    def select(self, context: dict) -> Optional[SpecialistAgent]:
        """Select the best agent for the given context."""
        if not self._agents:
            return None

        context_key = self._context_key(context)
        scores: dict[str, float] = {}

        for agent in self._agents:
            # Base confidence from agent
            base = agent.can_handle(context)
            # Historical performance for this context
            historical = self._context_scores.get(context_key, {}).get(agent.name, 0.5)
            # Combined score: 40% self-assessment, 60% historical
            scores[agent.name] = 0.4 * base + 0.6 * historical

        best_name = max(scores, key=scores.get)
        return next((a for a in self._agents if a.name == best_name), None)

    def update_performance(
        self, agent_name: str, context: dict, score: float,
    ) -> None:
        """Update performance record for an agent in a context.

        Uses exponential moving average with alpha=0.3.
        """
        context_key = self._context_key(context)
        if context_key not in self._context_scores:
            self._context_scores[context_key] = {}
        old = self._context_scores[context_key].get(agent_name, 0.5)
        self._context_scores[context_key][agent_name] = 0.7 * old + 0.3 * score

    def _context_key(self, context: dict) -> str:
        """Create a hashable key from context (bucketed)."""
        parts = [
            context.get("domain", "general"),
            str(context.get("graph_node_count", 0) // 100),
            str(context.get("cycle", 0) // 5),
        ]
        return "|".join(parts)
