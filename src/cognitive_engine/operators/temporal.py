"""T (Temporal) operator — Time-series alignment.

Aligns graphs across time to detect drift, stability, and rupture points.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph
from cognitive_engine.core.state import State


class TemporalOperator:
    """T: Time-series graph alignment.

    Compares the current state with previous states in history
    to detect:
    - Drift: gradual changes in beliefs
    - Stability: nodes that remain consistent
    - Rupture: sudden large changes
    """
    name = "temporal"

    def __call__(
        self,
        state: State,
        drift_threshold: float = 0.1,
        rupture_threshold: float = 0.5,
        **kwargs,
    ) -> State:
        if len(state.history) < 2:
            state.metadata["temporal"] = {"status": "insufficient_history"}
            return state

        # Compare current graph with previous snapshot
        prev_snapshot = state.history[-2]
        curr_snapshot = state.history[-1]

        drift_score = self._compute_drift(
            prev_snapshot.node_count,
            curr_snapshot.node_count,
            prev_snapshot.edge_count,
            curr_snapshot.edge_count,
        )

        state.metadata["temporal"] = {
            "drift_score": drift_score,
            "is_drift": drift_score > drift_threshold,
            "is_rupture": drift_score > rupture_threshold,
            "node_count_change": curr_snapshot.node_count - prev_snapshot.node_count,
            "edge_count_change": curr_snapshot.edge_count - prev_snapshot.edge_count,
        }

        meta = state.metadata["temporal"]
        node_delta = meta.get("node_count_change", 0)
        edge_delta = meta.get("edge_count_change", 0)
        state.record(
            self.name,
            f"Measured temporal drift across consecutive graph snapshots. "
            f"Drift score: {drift_score:.3f} ({'significant change' if drift_score > 0.3 else 'minor fluctuation'}). "
            f"Nodes changed by {node_delta:+d} ({edge_delta:+d} edges). "
            f"{'RUPTURE detected' if meta.get('is_rupture') else 'No rupture'} — "
            f"{'gradual drift ongoing' if meta.get('is_drift') and not meta.get('is_rupture') else 'system relatively stable'}. "
            f"Temporal analysis tracks how the reasoning graph evolves across cognitive cycles.",
        )
        return state

    def _compute_drift(
        self,
        prev_nodes: int,
        curr_nodes: int,
        prev_edges: int,
        curr_edges: int,
    ) -> float:
        """Compute drift score between two snapshots."""
        if prev_nodes == 0 and curr_nodes == 0:
            return 0.0

        max_nodes = max(prev_nodes, curr_nodes, 1)
        max_edges = max(prev_edges, curr_edges, 1)

        node_drift = abs(curr_nodes - prev_nodes) / max_nodes
        edge_drift = abs(curr_edges - prev_edges) / max_edges

        return (node_drift + edge_drift) / 2
