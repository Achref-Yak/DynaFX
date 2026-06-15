from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from cognitive_engine.core.trace import StateDelta, TraceBuffer

if TYPE_CHECKING:
    from cognitive_engine.core.models import Graph


@dataclass
class State:
    """S = (Graph G, ABox A, TBox T, Metadata M, Trace Σ).

    The core data structure that flows through operators.
    Every operator takes a State and returns a State.

    - ABox: individual typed assertions (facts, claims, evidence)
    - TBox: domain type hierarchy and axioms (OWL2-style)
    - graph: the live reasoning graph (nodes + edges with SL opinions)
    - trace: TraceBuffer monoid Σ — full provenance, append-only

    Backward-compatible: ``history`` and ``max_history`` fields
    remain accessible.
    """
    graph: Graph
    metadata: dict = field(default_factory=dict)
    abox: list = field(default_factory=list)
    tbox: Any = None
    history: list[StateDelta] = field(default_factory=list)
    max_history: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, '_trace', TraceBuffer(capacity=self.max_history))
        for entry in self.history:
            self._trace._entries.append(entry)

    @property
    def trace(self) -> TraceBuffer:
        return self._trace

    def delta(self, operator_name: str, description: str, effect_type: str = "deterministic", **meta) -> StateDelta:
        """Create a delta record for a state change."""
        return StateDelta(
            timestamp=time.time(),
            operator=operator_name,
            description=description,
            node_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            effect_type=effect_type,
            metadata=meta,
        )

    def record(self, operator_name: str, description: str, effect_type: str = "deterministic", **meta) -> None:
        """Record a state change in history + trace (keeps last N)."""
        d = self.delta(operator_name, description, effect_type=effect_type, **meta)
        self.history.append(d)
        self._trace.append(d)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def snapshot(self) -> dict:
        """Capture current state summary."""
        return {
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
            "entity_count": len(self.graph.entities),
            "metadata_keys": list(self.metadata.keys()),
            "history_length": len(self.history),
        }

    def fork(self) -> State:
        """Create a shallow copy (new State, same graph reference)."""
        s = State(
            graph=self.graph,
            metadata=dict(self.metadata),
            history=list(self.history),
            max_history=self.max_history,
        )
        # Preserve the trace content from the parent
        s._trace = self._trace.copy()
        return s

    def __repr__(self) -> str:
        return (
            f"State(nodes={len(self.graph.nodes)}, "
            f"edges={len(self.graph.edges)}, "
            f"history={len(self.history)})"
        )
