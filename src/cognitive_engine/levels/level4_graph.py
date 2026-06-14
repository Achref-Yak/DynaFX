"""Level 4: Graph Propagation.

Implements belief propagation over graph structure using the formula:
    B_i^{t+1} = σ(Σ_j W_{ji} B_j^t + E_i)

Where:
    σ = sigmoid activation
    W = edge weight matrix (from edge types and warrants)
    B = node belief vector
    E = external evidence (prior beliefs from node types)

Also implements similarity diffusion and fixed-point convergence.

Usage:
    from cognitive_engine.levels.level4_graph import GraphLevel
    level = GraphLevel()
    output = level.compute(graph, context)
    # output.beliefs contains propagated beliefs per node
"""
from __future__ import annotations

import logging
import math
from typing import Optional
from uuid import UUID

import networkx as nx
import numpy as np

from cognitive_engine.core.models import Graph, EdgeType, NodeType
from cognitive_engine.levels.base import BaseLevel, LevelOutput, ReasoningContext

logger = logging.getLogger(__name__)

# ── Edge type → base weight mapping ──────────────────────────────
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

# ── Node type → prior belief ─────────────────────────────────────
_NODE_PRIORS: dict[NodeType, float] = {
    NodeType.AXIOM: 0.9,
    NodeType.EVIDENCE: 0.8,
    NodeType.JUSTIFICATION: 0.7,
    NodeType.CONDITION: 0.5,
    NodeType.CLAIM: 0.6,
    NodeType.COUNTERCLAIM: 0.4,
    NodeType.FALLACY: 0.2,
}


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class GraphLevel(BaseLevel):
    """Level 4: Graph Propagation.

    Propagates beliefs through the graph structure using weighted
    message passing until convergence or max iterations.
    """

    @property
    def name(self) -> str:
        return "Graph Propagation"

    @property
    def level_number(self) -> int:
        return 4

    def __init__(
        self,
        max_iterations: int = 50,
        convergence_threshold: float = 1e-4,
    ) -> None:
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def compute(
        self, graph: Graph, context: ReasoningContext,
    ) -> LevelOutput:
        """Propagate beliefs through the graph.

        1. Build adjacency structure from edges
        2. Initialize beliefs from node types (priors)
        3. Iterate B(t+1) = σ(WB(t) + E) until convergence
        4. Return final beliefs
        """
        if not graph.nodes:
            return LevelOutput(beliefs={}, metadata={"iterations": 0})

        # Apply coefficient overrides if available
        coeffs = context.coefficients
        if coeffs:
            self.max_iterations = coeffs.level4_max_iterations
            self.convergence_threshold = coeffs.level4_convergence_threshold

        # Build adjacency: node_id → list of (source_id, edge_weight)
        adjacency = self._build_adjacency(graph)

        # Initialize beliefs from node type priors
        beliefs = self._initialize_beliefs(graph)

        # Build external evidence vector
        evidence = self._build_evidence(graph)

        # Iterate until convergence
        history: list[dict[UUID, float]] = [dict(beliefs)]
        for iteration in range(self.max_iterations):
            new_beliefs = self._propagate_step(graph, beliefs, adjacency, evidence)

            # Check convergence (L2 norm of change)
            change = math.sqrt(
                sum((new_beliefs[nid] - beliefs[nid]) ** 2
                    for nid in beliefs)
            )
            beliefs = new_beliefs
            history.append(dict(beliefs))

            if change < self.convergence_threshold:
                logger.debug(
                    "Graph propagation converged at iteration %d (Δ=%.6f)",
                    iteration + 1, change,
                )
                break

        return LevelOutput(
            beliefs=beliefs,
            metadata={
                "iterations": len(history) - 1,
                "final_change": change if history else 0.0,
                "converged": change < self.convergence_threshold,
                "history_length": len(history),
            },
        )

    def find_fixed_point(
        self, graph: Graph, context: Optional[ReasoningContext] = None,
    ) -> dict[UUID, float]:
        """Iterate until B* = F(B*)."""
        if context is None:
            context = ReasoningContext()
        output = self.compute(graph, context)
        return output.beliefs

    def similarity_diffusion(self, embeddings: dict[UUID, np.ndarray]) -> dict[UUID, float]:
        """Compute similarity diffusion: S_ij = cos(v_i, v_j)."""
        node_ids = list(embeddings.keys())
        n = len(node_ids)
        if n == 0:
            return {}

        # Build matrix
        dim = len(next(iter(embeddings.values())))
        mat = np.zeros((n, dim))
        for i, nid in enumerate(node_ids):
            mat[i] = embeddings[nid]

        # Normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        mat_norm = mat / norms

        # Cosine similarity matrix
        sim = mat_norm @ mat_norm.T

        # Average similarity per node
        result = {}
        for i, nid in enumerate(node_ids):
            # Exclude self-similarity
            others = [sim[i, j] for j in range(n) if j != i]
            result[nid] = float(np.mean(others)) if others else 0.0

        return result

    # ── Private helpers ───────────────────────────────────────────

    def _build_adjacency(self, graph: Graph) -> dict[UUID, list[tuple[UUID, float]]]:
        """Build adjacency list: target → [(source, weight)]."""
        adjacency: dict[UUID, list[tuple[UUID, float]]] = {
            nid: [] for nid in graph.nodes
        }

        for edge in graph.edges:
            if edge.source_id in graph.nodes and edge.target_id in graph.nodes:
                weight = _EDGE_WEIGHTS.get(edge.type, 0.5)

                # If edge has an opinion (SL-style), use projected probability as weight
                if edge.opinion:
                    b, d, u, a = edge.opinion
                    weight = b + a * u  # projected probability

                adjacency[edge.target_id].append((edge.source_id, weight))

        return adjacency

    def _initialize_beliefs(self, graph: Graph) -> dict[UUID, float]:
        """Initialize node beliefs from type priors."""
        beliefs = {}
        for nid, node in graph.nodes.items():
            # Use existing opinion if available (SL-style)
            if node.opinion:
                b, d, u, a = node.opinion
                beliefs[nid] = b + a * u  # projected probability
            else:
                # Fall back to node type prior
                beliefs[nid] = _NODE_PRIORS.get(node.type, 0.5)
        return beliefs

    def _build_evidence(self, graph: Graph) -> dict[UUID, float]:
        """Build external evidence from node type priors."""
        evidence = {}
        for nid, node in graph.nodes.items():
            if not node.opinion:
                evidence[nid] = _NODE_PRIORS.get(node.type, 0.5)
            else:
                b, d, u, a = node.opinion
                evidence[nid] = b + a * u
        return evidence

    def _propagate_step(
        self,
        graph: Graph,
        beliefs: dict[UUID, float],
        adjacency: dict[UUID, list[tuple[UUID, float]]],
        evidence: dict[UUID, float],
    ) -> dict[UUID, float]:
        """One step of B(t+1) = σ(Σ_j W_ji * B_j + E_i)."""
        new_beliefs = {}

        for nid in graph.nodes:
            # Weighted sum of incoming beliefs
            weighted_sum = 0.0
            for source_id, weight in adjacency.get(nid, []):
                weighted_sum += weight * beliefs.get(source_id, 0.5)

            # Add external evidence
            external = evidence.get(nid, 0.5)
            raw = weighted_sum + external

            # Apply sigmoid
            new_beliefs[nid] = _sigmoid(raw)

        return new_beliefs
