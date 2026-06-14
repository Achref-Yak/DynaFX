"""Tests for Level 3 — Neural Reasoning (GNN)."""
import torch
from cognitive_engine.levels.level3_neural import (
    GraphAttentionLayer, GNNBlock, BeliefPredictor, ReasoningGNN, NeuralLevel
)
from cognitive_engine.levels.base import ReasoningContext
from cognitive_engine.core.models import Graph, Node, Edge, EdgeType, NodeType
from uuid import uuid4


def _make_graph():
    n1 = Node(id=uuid4(), text="A", type=NodeType.CLAIM, opinion=(0.7, 0.2, 0.1, 0.5))
    n2 = Node(id=uuid4(), text="B", type=NodeType.EVIDENCE, opinion=(0.5, 0.3, 0.2, 0.5))
    e1 = Edge(id=uuid4(), source_id=n2.id, target_id=n1.id, type=EdgeType.SUPPORTS, warrant=((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5)))
    g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e1])
    return g


class TestGraphAttentionLayer:
    def test_output_shape(self):
        layer = GraphAttentionLayer(in_dim=16, out_dim=8, num_heads=2)
        x = torch.randn(5, 16)
        adj = torch.ones(5, 5)
        out = layer(x, adj)
        assert out.shape == (5, 8)

    def test_gradients_flow(self):
        layer = GraphAttentionLayer(in_dim=4, out_dim=4, num_heads=2)
        x = torch.randn(3, 4)
        adj = torch.ones(3, 3)
        out = layer(x, adj)
        loss = out.sum()
        loss.backward()
        assert layer.W_q.weight.grad is not None


class TestGNNBlock:
    def test_output_shape(self):
        block = GNNBlock(dim=16, num_heads=2)
        x = torch.randn(5, 16)
        adj = torch.ones(5, 5)
        out = block(x, adj)
        assert out.shape == (5, 16)


class TestBeliefPredictor:
    def test_output_shape(self):
        predictor = BeliefPredictor(dim=8)
        x = torch.randn(5, 8)
        out = predictor(x)
        assert out.shape == (5,)
        assert (out >= 0).all() and (out <= 1).all()


class TestReasoningGNN:
    def test_output_shape(self):
        gnn = ReasoningGNN(in_dim=16, hidden_dim=32, num_heads=4, num_layers=2)
        x = torch.randn(5, 16)
        adj = torch.ones(5, 5)
        out = gnn(x, adj)
        assert out.shape == (5,)


class TestNeuralLevel:
    def test_compute(self):
        g = _make_graph()
        context = ReasoningContext()
        nl = NeuralLevel(embedding_dim=16, num_heads=2, num_layers=1, hidden_dim=32)
        output = nl.compute(g, context)
        assert len(output.beliefs) == 2
        assert all(isinstance(v, float) for v in output.beliefs.values())

    def test_level_number(self):
        nl = NeuralLevel()
        assert nl.level_number == 3
