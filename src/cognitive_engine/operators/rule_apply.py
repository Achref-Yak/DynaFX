"""Rule apply operator — applies rule engine to graph.

Wraps the RuleEngine as an operator that can be used in the
InferenceCycle pipeline. Evaluates rules and creates edges
for matched actions.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import uuid4

from cognitive_engine.core.models import Edge, EdgeType, Opinion
from cognitive_engine.core.state import State
from cognitive_engine.rules.engine import RuleEngine

logger = logging.getLogger(__name__)


class RuleApplyOperator:
    """Apply rule engine to graph.

    Evaluates all registered rules against the current graph state
    and creates edges for matched actions with confidence-weighted
    opinions.
    """

    name = "rule_apply"

    def __init__(self, rule_engine: Optional[RuleEngine] = None) -> None:
        self.engine = rule_engine or RuleEngine()

    def __call__(self, state: State, **kwargs) -> State:
        """Evaluate rules and apply actions to the graph."""
        actions = self.engine.evaluate(state.graph)

        created = 0
        for action, bindings, rule_confidence in actions:
            src_id = bindings.get(action.source_var)
            tgt_id = bindings.get(action.target_var)

            if src_id is None or tgt_id is None:
                continue
            if src_id == tgt_id:
                continue

            # Resolve edge type
            try:
                edge_type = EdgeType[action.edge_type.upper()]
            except KeyError:
                logger.warning("Unknown edge type: %s", action.edge_type)
                continue

            # Check for duplicate edges
            existing = any(
                e.source_id == src_id and e.target_id == tgt_id and e.type == edge_type
                for e in state.graph.edges.values()
            )
            if existing:
                continue

            # Create edge with confidence-weighted opinion
            confidence = action.confidence * rule_confidence
            edge = Edge(
                source_id=src_id,
                target_id=tgt_id,
                type=edge_type,
                weight=action.weight,
                opinion=Opinion(
                    belief=confidence * 0.5,
                    disbelief=0.0,
                    uncertainty=1.0 - confidence,
                    prior=0.5,
                ),
                metadata={
                    "rule": "rule_apply",
                    "confidence": confidence,
                    "source_var": action.source_var,
                    "target_var": action.target_var,
                },
            )
            state.graph.edges[edge.id] = edge
            created += 1

        state.record(
            self.name,
            f"Rule engine: {len(actions)} actions evaluated, {created} edges created",
        )
        return state
