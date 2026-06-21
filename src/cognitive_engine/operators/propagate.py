"""⊗ (Propagate) operator — Propagate beliefs via Trust Network Analysis.

Replaces the 50-iteration heuristic Master Equation with a single
topological pass using Jøsang's Subjective Logic operators.
Cycle handling: max-DAG extraction (extract_max_dag) drops back-edges.
"""

from __future__ import annotations

from cognitive_engine.core.math import (
    extract_max_dag, tna_propagate, projected_probability,
    check_cycle_free,
)
from cognitive_engine.core.models import Opinion
from cognitive_engine.core.state import State


class PropagateOperator:
    """⊗: Propagate beliefs via Trust Network Analysis.

    One topological pass instead of 50-iteration sigmoid loop.
    Uses conditional_deduction + cumulative_fusion (SL operators)
    with edge-appropriate warrants.

    Cycles are handled by extract_max_dag; dropped back-edges are
    logged in metadata.
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

        def get_opinion_fn(nid):
            n = graph.nodes[nid]
            op = n.opinion
            if op is None:
                return None
            if isinstance(op, tuple):
                return op
            return (op.belief, op.disbelief, op.uncertainty, op.prior)

        # Extract max DAG (dropping back-edges if cycles exist)
        if not check_cycle_free(node_ids, edges_list):
            dag_edges, dropped, topo_order = extract_max_dag(node_ids, edges_list)
            state.metadata["dropped_edges"] = len(dropped)
            state.metadata["dropped_edge_list"] = [
                f"{e.source_id.hex[:8]}→{e.target_id.hex[:8]}"
                for e in dropped[:100]
            ]
        else:
            dag_edges = edges_list
            state.metadata["dropped_edges"] = 0

        # One-pass TNA propagation
        opinions = tna_propagate(node_ids, dag_edges, get_opinion_fn)

        # Update node opinions
        for nid, op in opinions.items():
            if nid in state.graph.nodes:
                state.graph.nodes[nid].opinion = Opinion(
                    belief=op[0], disbelief=op[1], uncertainty=op[2], prior=op[3],
                )

        # Scalar beliefs for metadata compat
        beliefs = {
            nid: projected_probability(op[0], op[2], op[3])
            for nid, op in opinions.items()
        }

        state.metadata["beliefs"] = beliefs
        state.metadata["truth_values"] = dict(beliefs)
        state.metadata["objective"] = sum(beliefs.values())

        cycle_note = (
            f"Dropped {len(dropped)} back-edge(s) to break cycles."
            if state.metadata.get("dropped_edges", 0)
            else "Graph was already acyclic."
        )

        state.record(
            self.name,
            f"Propagated beliefs across {len(node_ids)} nodes via Trust Network Analysis. "
            f"Objective: {state.metadata['objective']:.3f}. {cycle_note}",
        )
        return state
