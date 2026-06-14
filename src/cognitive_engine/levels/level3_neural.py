"""Level 3: Neural Reasoning (Graph Neural Network).

Implements attention-based graph neural network for learning node
embeddings and predicting beliefs via message passing.

Core formulas:
    y = Wx + b                                    (linear transformation)
    Attention(Q,K,V) = softmax(QK^T / √d) * V    (multi-head attention)
    L(θ) = E[ℓ(f_θ(x), y)]                       (loss minimization)
    θ_{t+1} = θ_t - η ∇_θ L                      (gradient descent)

Requires: torch >= 2.0

Usage:
    from cognitive_engine.levels.level3_neural import NeuralLevel
    level = NeuralLevel(embedding_dim=64, num_heads=4)
    output = level.compute(graph, context)
    # output.beliefs contains predicted beliefs per node
"""
from __future__ import annotations

import logging
import math
from typing import Optional
from uuid import UUID

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cognitive_engine.core.models import Graph, NodeType, EdgeType
from cognitive_engine.levels.base import BaseLevel, LevelOutput, ReasoningContext

logger = logging.getLogger(__name__)

# ── Node type → input feature index ──────────────────────────────
_NODE_TYPE_FEATURES: dict[NodeType, int] = {
    NodeType.AXIOM: 0,
    NodeType.EVIDENCE: 1,
    NodeType.CONDITION: 2,
    NodeType.CLAIM: 3,
    NodeType.COUNTERCLAIM: 4,
    NodeType.FALLACY: 5,
    NodeType.JUSTIFICATION: 6,
}

_NUM_NODE_TYPES = 7
_NUM_EDGE_TYPES = 10


class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention layer.

    Implements: Attention(Q,K,V) = softmax(QK^T / √d) * V
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.W_o = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_dim) — N nodes
            adj: Adjacency matrix (N, N) with edge weights

        Returns:
            Updated node features (N, out_dim)
        """
        N = x.size(0)

        # Project and reshape to (heads, N, head_dim)
        Q = self.W_q(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # Attention scores: (heads, N, N)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Mask with adjacency (set non-edges to -inf)
        adj_mask = adj.unsqueeze(0)  # (1, N, N) — broadcasts across heads
        scores = scores.masked_fill(adj_mask == 0, float('-inf'))

        # Softmax attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values: (heads, N, head_dim)
        out = torch.bmm(attn, V)
        out = out.transpose(0, 1).contiguous().view(N, -1)  # (N, out_dim)
        out = self.W_o(out)

        return out


class GNNBlock(nn.Module):
    """Single GNN block: attention + MLP + residual."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = GraphAttentionLayer(dim, dim, num_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        h = self.norm1(x + self.attention(x, adj))
        # MLP with residual
        h = self.norm2(h + self.mlp(h))
        return h


class BeliefPredictor(nn.Module):
    """MLP that maps node embeddings to belief scores."""

    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class ReasoningGNN(nn.Module):
    """Graph Neural Network for reasoning over argument graphs.

    Architecture:
        1. Node feature encoding (type one-hot + opinion)
        2. Multi-layer GNN with attention
        3. Belief prediction MLP
    """

    def __init__(
        self,
        in_dim: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_dim, embedding_dim)

        # GNN layers
        self.layers = nn.ModuleList([
            GNNBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Belief predictor
        self.predictor = BeliefPredictor(embedding_dim, hidden_dim)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_dim)
            adj: Adjacency matrix (num_nodes, num_nodes)

        Returns:
            Belief scores (num_nodes,) in [0, 1]
        """
        # Project to embedding space
        h = self.input_proj(x)

        # GNN message passing
        for layer in self.layers:
            h = layer(h, adj)

        # Predict beliefs
        beliefs = self.predictor(h)

        return beliefs


class NeuralLevel(BaseLevel):
    """Level 3: Neural Reasoning.

    Uses a graph neural network to learn node embeddings and predict
    beliefs via attention-based message passing.
    """

    @property
    def name(self) -> str:
        return "Neural Reasoning"

    @property
    def level_number(self) -> int:
        return 3

    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.model: Optional[ReasoningGNN] = None
        self._node_ids: list[UUID] = []

    def compute(
        self, graph: Graph, context: ReasoningContext,
    ) -> LevelOutput:
        """Run neural reasoning on the graph.

        1. Encode nodes as feature vectors
        2. Build adjacency matrix
        3. Forward pass through GNN
        4. Return predicted beliefs
        """
        if not graph.nodes:
            return LevelOutput(beliefs={}, metadata={})

        # Apply coefficient overrides
        if context.coefficients:
            self.embedding_dim = context.coefficients.level3_embedding_dim
            self.num_heads = context.coefficients.level3_attention_heads
            self.num_layers = context.coefficients.level3_num_layers
            self.dropout = context.coefficients.level3_dropout

        # Build node features and adjacency
        node_ids = list(graph.nodes.keys())
        self._node_ids = node_ids
        n = len(node_ids)

        # Node features: type one-hot + opinion projection
        in_dim = _NUM_NODE_TYPES + 1  # type one-hot + projected probability
        features = torch.zeros(n, in_dim)

        for i, nid in enumerate(node_ids):
            node = graph.nodes[nid]
            # Type one-hot
            type_idx = _NODE_TYPE_FEATURES.get(node.type, 0)
            features[i, type_idx] = 1.0
            # Opinion projection
            if node.opinion:
                b, d, u, a = node.opinion
                features[i, -1] = b + a * u
            else:
                features[i, -1] = 0.5

        # Adjacency matrix with edge weights
        adj = torch.zeros(n, n)
        edge_weights = {
            EdgeType.INFERS: 0.9, EdgeType.SUPPORTS: 0.85,
            EdgeType.DIRECT: 0.95, EdgeType.JUSTIFIES: 0.8,
            EdgeType.CIRCUMSTANTIAL: 0.6, EdgeType.QUALIFIES: 0.5,
            EdgeType.REBUTS: 0.6, EdgeType.HEARSAY: 0.4,
            EdgeType.CONTRADICTS: 0.85, EdgeType.ATTACKS: 0.8,
        }

        node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        for edge in graph.edges:
            if edge.source_id in node_id_to_idx and edge.target_id in node_id_to_idx:
                src_idx = node_id_to_idx[edge.source_id]
                tgt_idx = node_id_to_idx[edge.target_id]
                weight = edge_weights.get(edge.type, 0.5)
                adj[src_idx, tgt_idx] = weight

        # Add self-loops
        adj += torch.eye(n)

        # Build model
        self.model = ReasoningGNN(
            in_dim=in_dim,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        # Forward pass (no gradient for inference)
        self.model.eval()
        with torch.no_grad():
            beliefs_tensor = self.model(features, adj)

        # Convert to dict
        beliefs = {}
        for i, nid in enumerate(node_ids):
            beliefs[nid] = beliefs_tensor[i].item()

        # Compute embeddings for metadata
        embeddings = {}
        with torch.no_grad():
            h = self.model.input_proj(features)
            for layer in self.model.layers:
                h = layer(h, adj)
            for i, nid in enumerate(node_ids):
                embeddings[str(nid)] = h[i].numpy().tolist()

        return LevelOutput(
            beliefs=beliefs,
            metadata={
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "num_nodes": n,
                "embeddings": embeddings,
            },
        )

    def predict_beliefs(self, graph: Graph) -> dict[UUID, float]:
        """Convenience: predict beliefs without full context."""
        output = self.compute(graph, ReasoningContext())
        return output.beliefs

    def embed_graph(self, graph: Graph) -> dict[UUID, list[float]]:
        """Get node embeddings after GNN processing."""
        output = self.compute(graph, ReasoningContext())
        embeddings = output.metadata.get("embeddings", {})
        return {UUID(k): v for k, v in embeddings.items()}
