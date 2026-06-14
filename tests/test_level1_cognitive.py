"""Tests for Level 1 — Cognitive Architecture (ACT-R/SOAR)."""
from cognitive_engine.levels.level1_cognitive import CognitiveLevel
from cognitive_engine.levels.base import ReasoningContext
from cognitive_engine.core.models import Graph, Node, Edge, EdgeType, NodeType
from uuid import uuid4


def _make_graph():
    n1 = Node(id=uuid4(), text="A", type=NodeType.CLAIM, opinion=(0.7, 0.2, 0.1, 0.5))
    n2 = Node(id=uuid4(), text="B", type=NodeType.EVIDENCE, opinion=(0.5, 0.3, 0.2, 0.5))
    g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[])
    return g


class TestCognitiveLevel:
    def test_add_chunk_and_retrieve(self):
        cl = CognitiveLevel()
        cl.add_chunk("fact1", activation=0.9)
        cl.add_chunk("fact2", activation=0.3)
        chunk = cl.retrieve("fact1")
        assert chunk is not None
        assert chunk.content == "fact1"

    def test_empty_retrieve(self):
        cl = CognitiveLevel()
        chunk = cl.retrieve("nonexistent")
        assert chunk is None

    def test_add_production_rule(self):
        cl = CognitiveLevel()
        cl.add_chunk("A", activation=0.9)
        cl.add_chunk("B", activation=0.8)
        rule = cl.add_production_rule(
            condition=lambda ctx: True,
            action="conclude_C",
            utility=0.85,
            strength=0.85,
        )
        assert rule.action == "conclude_C"

    def test_compute(self):
        g = _make_graph()
        context = ReasoningContext()
        cl = CognitiveLevel()
        output = cl.compute(g, context)
        assert len(output.beliefs) == 2
        assert all(isinstance(v, float) for v in output.beliefs.values())

    def test_level_number(self):
        cl = CognitiveLevel()
        assert cl.level_number == 1
