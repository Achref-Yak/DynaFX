"""Sim (Simulate) operator — What-if analysis.

Modifies the graph and re-propagates to simulate hypothetical scenarios.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph
from cognitive_engine.core.state import State


class SimulateOperator:
    """Sim: What-if analysis.

    Modifies the graph according to given modifications and
    re-propagates beliefs to see the effect.

    Modifications can be:
    - {node_id: {"belief": float}} — change node belief
    - {node_id: {"remove": true}} — remove node
    - {node_id: {"add_edge": {...}}} — add edge
    """
    name = "simulate"

    def __call__(
        self,
        state: State,
        modifications: dict = None,
        **kwargs,
    ) -> State:
        if not modifications:
            return state

        # Deep copy graph for simulation
        sim_graph = deepcopy(state.graph)

        # Apply modifications
        for target, mod in modifications.items():
            if isinstance(target, str):
                target = UUID(target)

            if "belief" in mod and target in sim_graph.nodes:
                node = sim_graph.nodes[target]
                b = mod["belief"]
                d = max(0.0, 1.0 - b - node.opinion[2])
                u = max(0.0, 1.0 - b - d)
                node.opinion = (b, d, u, node.opinion[3])

            elif mod.get("remove") and target in sim_graph.nodes:
                del sim_graph.nodes[target]
                sim_graph.edges = {
                    e.id: e for e in sim_graph.edges.values()
                    if e.source_id != target and e.target_id != target
                }

            elif "add_edge" in mod:
                from cognitive_engine.core.models import Edge
                edge_data = mod["add_edge"]
                new_edge = Edge(
                    source_id=UUID(edge_data["source"]),
                    target_id=UUID(edge_data["target"]),
                    type=edge_data.get("type", "SUPPORTS"),
                )
                sim_graph.edges[new_edge.id] = new_edge

        # Re-propagate on modified graph using core math
        from cognitive_engine.core.math import (
            master_equation_all, propagate_step, build_adjacency,
            initialize_beliefs, compute_attack_sum, count_violations, global_objective,
        )

        sim_node_ids = set(sim_graph.nodes)
        sim_edges = list(sim_graph.edges.values())

        def get_type_fn(nid):
            return sim_graph.nodes[nid].type.name

        def get_opinion_fn(nid):
            n = sim_graph.nodes[nid]
            op = n.opinion
            if op is None:
                return None
            if isinstance(op, tuple):
                return op
            return (op.belief, op.disbelief, op.uncertainty, op.prior)

        sim_beliefs = initialize_beliefs(sim_node_ids, get_type_fn, get_opinion_fn)
        adjacency = build_adjacency(sim_node_ids, sim_edges)

        for _ in range(50):
            new_beliefs = propagate_step(sim_beliefs, adjacency, {nid: 0.5 for nid in sim_node_ids})
            delta = sum(abs(new_beliefs.get(k, 0.5) - sim_beliefs.get(k, 0.5)) for k in sim_beliefs)
            sim_beliefs = new_beliefs
            if delta < 1e-4:
                break

        attack_sim = {}
        for nid in sim_node_ids:
            attack_sim[nid] = compute_attack_sum(nid, sim_edges, sim_beliefs)

        violations_sim = count_violations(
            {nid: (b, 0.0, 0.0, 0.5) for nid, b in sim_beliefs.items()},
            sim_edges, opinion_threshold=0.01,
        )

        final_sim = master_equation_all(
            list(sim_node_ids), sim_beliefs, sim_beliefs,
            {nid: 1.0 for nid in sim_node_ids}, attack_sim, violations_sim,
        )

        objective = global_objective(final_sim, violations_sim)

        result = type("SimResult", (), {
            "beliefs": sim_beliefs,
            "objective": objective,
        })()

        # Store simulation results
        state.metadata["simulation"] = {
            "original_nodes": len(state.graph.nodes),
            "simulated_nodes": len(sim_graph.nodes),
            "modifications": modifications,
            "beliefs": result.beliefs,
            "objective": result.objective,
        }

        from cognitive_engine.core.models import Opinion as OpModel

        for nid, belief in result.beliefs.items():
            if nid in state.graph.nodes:
                node = state.graph.nodes[nid]
                op = node.opinion
                prior = op[3] if op else 0.5
                b = belief
                d = max(0.0, 1.0 - b - 0.05)
                u = max(0.0, 1.0 - b - d)
                node.opinion = OpModel(belief=b, disbelief=d, uncertainty=u, prior=prior)

        mod_summary = []
        for target, mod in modifications.items():
            if "belief" in mod:
                mod_summary.append(f"set belief of node {str(target)[:8]} to {mod['belief']}")
            elif mod.get("remove"):
                mod_summary.append(f"removed node {str(target)[:8]}")
            elif "add_edge" in mod:
                e = mod["add_edge"]
                mod_summary.append(f"added edge {e['source'][:8]}→{e['target'][:8]} ({e.get('type', 'SUPPORTS')})")
        state.record(
            self.name,
            f"Ran what-if simulation: {len(modifications)} hypothetical modification(s). "
            f"{'; '.join(mod_summary)}. "
            f"After re-propagation, objective score: {result.objective:.3f} "
            f"({'improved' if result.objective < state.metadata.get('objective', 1.0) else 'worsened'} vs pre-simulation baseline). "
            f"Simulation deep-copies the graph, applies changes, re-propagates beliefs, and assesses impact on system coherence.",
        )
        return state
