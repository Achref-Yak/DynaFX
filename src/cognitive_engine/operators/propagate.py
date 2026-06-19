"""⊗ (Propagate) operator — Propagate beliefs via Master Equation.

Uses the formal math primitives from core/math.py.
Replaces the 8-level UnifiedReasoner with direct formula calls.
"""

from __future__ import annotations

from typing import Optional

from cognitive_engine.core.math import (
    master_equation_all, propagate_step, build_adjacency,
    initialize_beliefs, compute_support_sum, compute_attack_sum,
    count_violations, global_objective,
)
from cognitive_engine.core.models import Graph, Opinion
from cognitive_engine.core.state import State


class PropagateOperator:
    """⊗: Propagate beliefs via the Master Equation.

    Computes final beliefs using:
        R(x) = αP(x) + βG(x) + γL(x) - δA(x) - λ·violations(x)

    Belief propagation follows: B_i^{t+1} = σ(Σ W_ji · B_j^t + E_i)
    """
    name = "propagate"

    def __call__(
        self,
        state: State,
        coefficients=None,
        domain_config=None,
        priors=None,
        **kwargs,
    ) -> State:
        if not state.graph.nodes:
            return state

        graph = state.graph
        node_ids = set(graph.nodes)
        edges_list = list(graph.edges.values())

        # Initialize beliefs from node opinions
        def get_type_fn(nid):
            return graph.nodes[nid].type.name

        def get_opinion_fn(nid):
            n = graph.nodes[nid]
            op = n.opinion
            if op is None:
                return None
            if isinstance(op, tuple):
                return op
            return (op.belief, op.disbelief, op.uncertainty, op.prior)

        beliefs = initialize_beliefs(node_ids, get_type_fn, get_opinion_fn)

        # Build adjacency
        adjacency = build_adjacency(node_ids, edges_list)

        # Build evidence vector
        evidence = {}
        for nid in node_ids:
            opinion = get_opinion_fn(nid)
            if opinion:
                b, d, u, a = opinion
                evidence[nid] = b + a * u
            else:
                evidence[nid] = 0.5

        # Propagate until convergence
        for _ in range(50):
            new_beliefs = propagate_step(beliefs, adjacency, evidence)
            delta = sum(abs(new_beliefs.get(k, 0.5) - beliefs.get(k, 0.5)) for k in beliefs)
            beliefs = new_beliefs
            if delta < 1e-4:
                break

        # Compute master equation outputs
        probabilities = {nid: b for nid, b in beliefs.items()}
        logic_consistency = {nid: 1.0 for nid in node_ids}
        attack_strengths = {}
        for nid in node_ids:
            attack_strengths[nid] = compute_attack_sum(nid, edges_list, beliefs)

        violations = count_violations(
            {nid: (b, 0.0, 0.0, 0.5) for nid, b in beliefs.items()},
            edges_list,
            opinion_threshold=0.01,
        )

        final_beliefs = master_equation_all(
            list(node_ids), probabilities, beliefs,
            logic_consistency, attack_strengths, violations,
        )

        objective = global_objective(final_beliefs, violations)

        # Store results in metadata
        state.metadata["beliefs"] = beliefs
        state.metadata["truth_values"] = final_beliefs
        state.metadata["objective"] = objective

        # Update node opinions with final beliefs
        for nid, belief in final_beliefs.items():
            if nid in state.graph.nodes:
                node = state.graph.nodes[nid]
                existing = node.opinion
                a = existing.prior if existing else 0.5
                b = belief
                d = max(0.0, 1.0 - b - 0.05)
                u = max(0.0, 1.0 - b - d)
                node.opinion = Opinion(belief=b, disbelief=d, uncertainty=u, prior=a)

        changed = []
        for nid in list(node_ids)[:5]:
            new_b = final_beliefs.get(nid, 0.5)
            old_b = beliefs.get(nid, 0.5)
            if abs(new_b - old_b) > 0.05:
                changed.append(f"{nid.hex[:8]}: {old_b:.2f}→{new_b:.2f}")
        changed_text = "; ".join(changed) if changed else "minimal changes"

        state.record(
            self.name,
            f"Propagated beliefs across {len(node_ids)} nodes via Master Equation. "
            f"Objective: {objective:.3f}. Shifts: {changed_text}.",
        )
        return state
