"""ΠΩ (Constraint) operator — Check logical constraint violations.

Uses symbolic logic primitives from core/math.py (modus ponens,
closure, conjunction, deduction) instead of the old Level 0.
"""

from __future__ import annotations

from cognitive_engine.core.math import (
    count_violations,
)
from cognitive_engine.core.models import EDGE_BFO_CONSTRAINTS, Graph
from cognitive_engine.core.state import State


class ConstraintOperator:
    """ΠΩ: Check logical constraint violations.

    Applies symbolic logic constraints to identify
    and penalize inconsistent nodes/edges.
    """
    name = "constraint"

    def __call__(
        self,
        state: State,
        **kwargs,
    ) -> State:
        if not state.graph.nodes:
            return state

        graph = state.graph
        edges_list = list(graph.edges.values())

        # Build opinion tuples
        opin_tuples = {}
        for nid, node in graph.nodes.items():
            if node.opinion:
                opin_tuples[nid] = (node.opinion.belief, node.opinion.disbelief, node.opinion.uncertainty, node.opinion.prior)
            else:
                opin_tuples[nid] = (0.5, 0.0, 0.5, 0.5)

        # Count formal violations: opinion-invariant and edge consistency
        n_violations = count_violations(opin_tuples, edges_list, opinion_threshold=0.1)

        # Identify specific violating nodes
        violations = {}
        support_edges = [(e.source_id, e.target_id) for e in edges_list if e.type == "SUPPORTS"]
        # REBUTS is a concept-mediated correction (temporal semantics), not a
        # structural inconsistency — exclude it from conflict-based violations.
        attack_pairs = {
            (e.source_id, e.target_id) for e in edges_list
            if e.type in ("ATTACKS", "CONTRADICTS")
        }

        for a, b in support_edges:
            if (a, b) in attack_pairs or (b, a) in attack_pairs:
                violations[a] = violations.get(a, 0) + 1
                violations[b] = violations.get(b, 0) + 1

        state.metadata["constraint_beliefs"] = opin_tuples
        state.metadata["constraint_violations"] = violations

        # BFO compatibility linter
        bfo_violations = 0
        for edge in graph.edges.values():
            src = graph.nodes.get(edge.source_id)
            tgt = graph.nodes.get(edge.target_id)
            if src and tgt and src.bfo_category and tgt.bfo_category:
                constraints = EDGE_BFO_CONSTRAINTS.get(edge.type)
                if constraints:
                    src_ok, tgt_ok = constraints
                    if src.bfo_category not in src_ok or tgt.bfo_category not in tgt_ok:
                        bfo_violations += 1
                        violations[edge.source_id] = violations.get(edge.source_id, 0) + 1

        state.metadata["bfo_violations"] = bfo_violations
        violation_text = "; ".join(
            f"{k.hex[:8]}: {v}" for k, v in list(violations.items())[:5]
        ) if violations else "none detected"

        state.record(
            self.name,
            f"Checked {len(graph.nodes)} propositions for constraint violations. "
            f"Formal violations: {sum(n_violations.values())}. "
            f"Details: {violation_text}.",
        )
        return state
