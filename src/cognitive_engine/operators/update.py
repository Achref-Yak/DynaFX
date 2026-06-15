"""Δ (Update) operator — Track state changes.

Records state deltas and maintains history.
"""

from __future__ import annotations

from cognitive_engine.core.state import State


class UpdateOperator:
    """Δ: Track state changes.

    Records the current state delta in history and
    computes the difference from the previous state.
    """
    name = "update"

    def __call__(
        self,
        state: State,
        description: str = "state update",
        **kwargs,
    ) -> State:
        # Compute delta from previous state if history exists
        if len(state.history) >= 2:
            prev = state.history[-2]
            curr = state.history[-1]
            state.metadata["last_delta"] = {
                "nodes_added": curr.node_count - prev.node_count,
                "edges_added": curr.edge_count - prev.edge_count,
                "time_elapsed": curr.timestamp - prev.timestamp,
            }
            delta = state.metadata["last_delta"]
            desc = (
                f"Recorded state delta: {delta['nodes_added']:+d} nodes, {delta['edges_added']:+d} edges since previous step "
                f"({delta['time_elapsed']:.3f}s elapsed). "
                f"The update operator stamps each cognitive cycle with a snapshot of graph evolution over time."
            )
        else:
            desc = (
                f"Initial state snapshot recorded. "
                f"The update operator stamps each cognitive cycle with a snapshot of graph evolution over time."
            )

        # Record current state
        state.record(self.name, desc)

        return state
