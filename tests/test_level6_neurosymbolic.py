"""Tests for Level 6 — Neuro-Symbolic Fusion."""
from cognitive_engine.levels.level6_neurosymbolic import NeuroSymbolicLevel
from cognitive_engine.levels.base import ReasoningContext
from cognitive_engine.core.models import Graph, Node, Edge, EdgeType, NodeType
from uuid import uuid4


def _make_graph():
    n1 = Node(id=uuid4(), text="A", type=NodeType.CLAIM, opinion=(0.7, 0.2, 0.1, 0.5))
    n2 = Node(id=uuid4(), text="B", type=NodeType.EVIDENCE, opinion=(0.5, 0.3, 0.2, 0.5))
    e1 = Edge(id=uuid4(), source_id=n2.id, target_id=n1.id, type=EdgeType.SUPPORTS, warrant=((0.6, 0.3, 0.1, 0.5), (0.4, 0.5, 0.1, 0.5)))
    g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e1])
    return g


class TestNeuroSymbolicLevel:
    def test_compute(self):
        g = _make_graph()
        context = ReasoningContext()
        level = NeuroSymbolicLevel()
        output = level.compute(g, context)
        assert len(output.beliefs) == 2
        assert all(isinstance(v, float) for v in output.beliefs.values())

    def test_fusion_weight(self):
        g = _make_graph()
        context = ReasoningContext()
        level = NeuroSymbolicLevel(lambda_neural=0.4)
        output = level.compute(g, context)
        assert "lambda_neural" in output.metadata

    def test_logic_penalty(self):
        g = _make_graph()
        context = ReasoningContext()
        level = NeuroSymbolicLevel(logic_penalty_weight=0.1)
        output = level.compute(g, context)
        assert "violations" in output.metadata

    def test_level_number(self):
        g = _make_graph()
        context = ReasoningContext()
        level = NeuroSymbolicLevel()
        output = level.compute(g, context)
        assert level.level_number == 6
