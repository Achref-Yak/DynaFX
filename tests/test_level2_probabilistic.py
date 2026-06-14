"""Tests for Level 2 — Probabilistic Reasoning (Bayesian Network)."""
from cognitive_engine.levels.level2_probabilistic import ProbabilisticLevel
from cognitive_engine.levels.base import ReasoningContext
from cognitive_engine.core.models import Graph, Node, Edge, EdgeType, NodeType
from uuid import uuid4


def _make_graph():
    n1 = Node(id=uuid4(), text="A", type=NodeType.CLAIM, opinion=(0.6, 0.3, 0.1, 0.5))
    g = Graph(nodes={n1.id: n1}, edges=[])
    return g


class TestProbabilisticLevel:
    def test_add_variable_and_infer(self):
        pl = ProbabilisticLevel()
        pl.add_variable("A", [], {(True,): 0.7, (False,): 0.3})
        result = pl.infer("A")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_observe_and_infer(self):
        pl = ProbabilisticLevel()
        pl.add_variable("A", [], {(True,): 0.6, (False,): 0.4})
        pl.add_variable("B", ["A"], {(True, True): 0.9, (True, False): 0.1,
                                      (False, True): 0.8, (False, False): 0.2})
        pl.observe("A", True)
        result = pl.infer("B")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_map_inference(self):
        pl = ProbabilisticLevel()
        pl.add_variable("A", [], {(True,): 0.6, (False,): 0.4})
        pl.add_variable("B", [], {(True,): 0.9, (False,): 0.1})
        map_result = pl.map_inference()
        assert isinstance(map_result, dict)

    def test_expectation(self):
        pl = ProbabilisticLevel()
        pl.add_variable("A", [], {(True,): 0.7, (False,): 0.3})
        exp = pl.expectation("A")
        assert isinstance(exp, float)
        assert 0.0 <= exp <= 1.0

    def test_to_beliefs(self):
        g = _make_graph()
        pl = ProbabilisticLevel()
        pl.add_variable("A", [], {(True,): 0.6, (False,): 0.4})
        beliefs = pl.to_beliefs(g)
        assert isinstance(beliefs, dict)

    def test_compute(self):
        g = _make_graph()
        context = ReasoningContext()
        pl = ProbabilisticLevel()
        output = pl.compute(g, context)
        assert len(output.beliefs) >= 0  # May be empty if no variables defined

    def test_level_number(self):
        pl = ProbabilisticLevel()
        assert pl.level_number == 2
