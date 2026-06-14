"""Tests for Level 4 — Graph Propagation."""
from cognitive_engine.levels.level4_graph import GraphLevel
from cognitive_engine.levels.base import ReasoningContext
from cognitive_engine.core.models import Graph, Node, Edge, EdgeType, NodeType
from uuid import uuid4


def _make_graph():
    n1 = Node(id=uuid4(), text="A", type=NodeType.CLAIM, opinion=(0.8, 0.1, 0.1, 0.5))
    n2 = Node(id=uuid4(), text="B", type=NodeType.EVIDENCE, opinion=(0.3, 0.5, 0.2, 0.5))
    n3 = Node(id=uuid4(), text="C", type=NodeType.EVIDENCE, opinion=(0.6, 0.2, 0.2, 0.5))
    e1 = Edge(id=uuid4(), source_id=n2.id, target_id=n1.id, type=EdgeType.SUPPORTS, warrant=((0.7, 0.2, 0.1, 0.5), (0.3, 0.6, 0.1, 0.5)))
    e2 = Edge(id=uuid4(), source_id=n3.id, target_id=n1.id, type=EdgeType.SUPPORTS, warrant=((0.5, 0.3, 0.2, 0.5), (0.4, 0.5, 0.1, 0.5)))
    g = Graph(nodes={n1.id: n1, n2.id: n2, n3.id: n3}, edges=[e1, e2])
    return g


class TestGraphLevel:
    def test_belief_propagation(self):
        g = _make_graph()
        context = ReasoningContext()
        level = GraphLevel(max_iterations=20, convergence_threshold=1e-3)
        output = level.compute(g, context)
        assert len(output.beliefs) == 3
        # Node A should have high belief since both edges support it
        nid_a = list(g.nodes.keys())[0]
        assert output.beliefs[nid_a] > 0.5

    def test_convergence(self):
        g = _make_graph()
        context = ReasoningContext()
        level = GraphLevel(max_iterations=100, convergence_threshold=1e-4)
        output = level.compute(g, context)
        assert "converged" in output.metadata
        assert output.metadata["converged"] is True

    def test_level_number(self):
        g = _make_graph()
        context = ReasoningContext()
        level = GraphLevel()
        output = level.compute(g, context)
        assert level.level_number == 4
