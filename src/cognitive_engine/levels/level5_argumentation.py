"""Level 5: Argumentation Theory.

Implements Dung-style argumentation frameworks with support/attack
aggregation, acceptability semantics, and probabilistic argument strength.

Core formulas:
    S(v) = Σ_{u∈Supp(v)} w_{uv} * A(u)    (support aggregation)
    A(v) = Σ_{u∈Att(v)} w_{uv} * A(u)    (attack aggregation)
    Acc(v) = S(v) - A(v)                   (net acceptability)
    P(v) = σ(Acc(v))                       (probabilistic strength)
    v ∈ AF ⟺ ∀u ∈ Att(v), u ∉ AF         (Dung acceptance)

Usage:
    from cognitive_engine.levels.level5_argumentation import ArgumentationLevel
    level = ArgumentationLevel()
    output = level.compute(graph, context)
    # output.beliefs contains argument strength per node
"""
from __future__ import annotations

import logging
import math
from typing import Optional
from uuid import UUID

import networkx as nx

from cognitive_engine.core.models import Graph, EdgeType
from cognitive_engine.levels.base import BaseLevel, LevelOutput, ReasoningContext

logger = logging.getLogger(__name__)

# ── Edge type → argumentative role ────────────────────────────────
_SUPPORT_EDGES: set[EdgeType] = {
    EdgeType.SUPPORTS, EdgeType.INFERS, EdgeType.JUSTIFIES, EdgeType.DIRECT,
}
_ATTACK_EDGES: set[EdgeType] = {
    EdgeType.ATTACKS, EdgeType.CONTRADICTS, EdgeType.REBUTS,
}
_QUALIFY_EDGES: set[EdgeType] = {
    EdgeType.QUALIFIES, EdgeType.CIRCUMSTANTIAL, EdgeType.HEARSAY,
}

# ── Edge type → base weight ──────────────────────────────────────
_EDGE_WEIGHTS: dict[EdgeType, float] = {
    EdgeType.INFERS: 0.9,
    EdgeType.SUPPORTS: 0.85,
    EdgeType.DIRECT: 0.95,
    EdgeType.JUSTIFIES: 0.8,
    EdgeType.CIRCUMSTANTIAL: 0.6,
    EdgeType.QUALIFIES: 0.5,
    EdgeType.REBUTS: 0.6,
    EdgeType.HEARSAY: 0.4,
    EdgeType.CONTRADICTS: 0.85,
    EdgeType.ATTACKS: 0.8,
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


class ArgumentationLevel(BaseLevel):
    """Level 5: Argumentation Theory.

    Builds an argument graph from support/attack edges, computes
    acceptability via Dung semantics, and returns argument strengths.
    """

    @property
    def name(self) -> str:
        return "Argumentation Theory"

    @property
    def level_number(self) -> int:
        return 5

    def __init__(self, discount_factor: float = 0.9) -> None:
        self.discount_factor = discount_factor

    def compute(
        self, graph: Graph, context: ReasoningContext,
    ) -> LevelOutput:
        """Run argumentation analysis on the graph.

        1. Build argument graph (support/attack structure)
        2. Compute support and attack strengths
        3. Compute net acceptability
        4. Apply Dung acceptance semantics
        5. Return argument strengths as beliefs
        """
        if not graph.nodes:
            return LevelOutput(beliefs={}, metadata={})

        # Apply coefficient overrides
        if context.coefficients:
            self.discount_factor = context.coefficients.level5_discount_factor

        # Build argument graph
        arg_graph = self.build_argument_graph(graph)

        # Compute support and attack for each node
        support = {}
        attack = {}
        acceptability = {}

        # Iterate until convergence (Dung semantics)
        beliefs = {nid: 0.5 for nid in graph.nodes}

        for _ in range(20):  # max iterations for Dung convergence
            old_beliefs = dict(beliefs)

            for nid in graph.nodes:
                s = self.compute_support(nid, arg_graph, beliefs)
                a = self.compute_attack(nid, arg_graph, beliefs)
                support[nid] = s
                attack[nid] = a
                acceptability[nid] = s - a
                beliefs[nid] = _sigmoid(acceptability[nid])

            # Check convergence
            change = sum(
                abs(beliefs[nid] - old_beliefs[nid])
                for nid in graph.nodes
            )
            if change < 1e-6:
                break

        # Compute Dung acceptance
        accepted = self.dung_semantics(beliefs, arg_graph)

        return LevelOutput(
            beliefs=beliefs,
            metadata={
                "support": {str(k): v for k, v in support.items()},
                "attack": {str(k): v for k, v in attack.items()},
                "acceptability": {str(k): v for k, v in acceptability.items()},
                "accepted": [str(nid) for nid in accepted],
                "num_accepted": len(accepted),
            },
        )

    def build_argument_graph(self, graph: Graph) -> nx.DiGraph:
        """Build a networkx DiGraph with support/attack edge attributes."""
        arg_graph = nx.DiGraph()

        for nid in graph.nodes:
            arg_graph.add_node(nid, belief=0.5)

        for edge in graph.edges:
            if edge.source_id not in graph.nodes:
                continue
            if edge.target_id not in graph.nodes:
                continue

            weight = _EDGE_WEIGHTS.get(edge.type, 0.5)
            role = "support"
            if edge.type in _ATTACK_EDGES:
                role = "attack"
            elif edge.type in _QUALIFY_EDGES:
                role = "qualify"

            arg_graph.add_edge(
                edge.source_id, edge.target_id,
                weight=weight, role=role, edge_type=edge.type,
            )

        return arg_graph

    def compute_support(
        self, node_id: UUID, arg_graph: nx.DiGraph,
        beliefs: dict[UUID, float],
    ) -> float:
        """S(v) = Σ_{u∈Supp(v)} w_{uv} * A(u)."""
        total = 0.0
        for predecessor in arg_graph.predecessors(node_id):
            edge_data = arg_graph.edges[predecessor, node_id]
            if edge_data.get("role") == "support":
                weight = edge_data.get("weight", 0.5)
                source_belief = beliefs.get(predecessor, 0.5)
                total += weight * source_belief
        return total

    def compute_attack(
        self, node_id: UUID, arg_graph: nx.DiGraph,
        beliefs: dict[UUID, float],
    ) -> float:
        """A(v) = Σ_{u∈Att(v)} w_{uv} * A(u)."""
        total = 0.0
        for predecessor in arg_graph.predecessors(node_id):
            edge_data = arg_graph.edges[predecessor, node_id]
            if edge_data.get("role") == "attack":
                weight = edge_data.get("weight", 0.5)
                source_belief = beliefs.get(predecessor, 0.5)
                total += weight * source_belief
        return total

    def compute_acceptability(
        self, node_id: UUID, arg_graph: nx.DiGraph,
        beliefs: dict[UUID, float],
    ) -> float:
        """Acc(v) = S(v) - A(v)."""
        s = self.compute_support(node_id, arg_graph, beliefs)
        a = self.compute_attack(node_id, arg_graph, beliefs)
        return s - a

    def dung_semantics(
        self, beliefs: dict[UUID, float], arg_graph: nx.DiGraph,
    ) -> set[UUID]:
        """Find Dung's preferred extension.

        A node is accepted if all its attackers are rejected.
        Iterates until fixpoint.
        """
        accepted: set[UUID] = set()
        rejected: set[UUID] = set()

        # Initial: accept nodes with positive acceptability
        for nid in beliefs:
            if beliefs[nid] > 0.5:
                accepted.add(nid)
            else:
                rejected.add(nid)

        # Iterate until fixpoint
        for _ in range(20):
            old_accepted = set(accepted)
            old_rejected = set(rejected)

            for nid in beliefs:
                # Get all attackers
                attackers = set(arg_graph.predecessors(nid))
                attacker_roles = {
                    pred for pred in attackers
                    if arg_graph.edges[pred, nid].get("role") == "attack"
                }

                if not attacker_roles:
                    # No attackers → accept
                    accepted.add(nid)
                    rejected.discard(nid)
                elif attacker_roles.issubset(rejected):
                    # All attackers rejected → accept
                    accepted.add(nid)
                    rejected.discard(nid)
                else:
                    # Some attackers accepted → reject
                    rejected.add(nid)
                    accepted.discard(nid)

            if accepted == old_accepted and rejected == old_rejected:
                break

        return accepted

    def compute_argument_strength(self, graph: Graph) -> dict[UUID, float]:
        """Convenience: compute argument strengths for all nodes."""
        output = self.compute(graph, ReasoningContext())
        return output.beliefs
