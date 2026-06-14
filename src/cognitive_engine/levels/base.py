"""Abstract base class for all reasoning levels.

Every level in the formula library implements BaseLevel and returns
a LevelOutput with beliefs and metadata.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from cognitive_engine.core.models import Graph


@dataclass
class LevelOutput:
    """Standard output from any reasoning level.

    Attributes:
        beliefs: Node ID → belief score in [0, 1].
        metadata: Level-specific auxiliary data (e.g., activations, CPTs).
    """
    beliefs: dict[UUID, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningContext:
    """Context passed to levels during computation.

    Attributes:
        coefficients: Global coefficient set.
        priors: Prior opinions/warrants (from SL legacy or domain config).
        domain_config: Active DomainConfig.
        previous_outputs: Outputs from earlier levels (for composition).
        graph: The reasoning graph being processed.
    """
    coefficients: Any = None  # Coefficients dataclass (from unified/)
    priors: Any = None  # Priors dataclass (from core/config.py)
    domain_config: Any = None  # DomainConfig (from domain.py)
    previous_outputs: dict[str, LevelOutput] = field(default_factory=dict)
    graph: Optional[Graph] = None


class BaseLevel(ABC):
    """Abstract base class for all reasoning levels.

    Subclasses must implement:
        - name: Human-readable level name
        - level_number: Integer level (0-7)
        - compute(graph, context): Run the level's reasoning
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'Symbolic Logic'."""

    @property
    @abstractmethod
    def level_number(self) -> int:
        """Integer level index (0-7)."""

    @abstractmethod
    def compute(
        self, graph: Graph, context: ReasoningContext,
    ) -> LevelOutput:
        """Run this level's reasoning on the graph.

        Args:
            graph: The reasoning graph.
            context: Shared context with coefficients, priors, prior outputs.

        Returns:
            LevelOutput with computed beliefs and metadata.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} level={self.level_number}>"
