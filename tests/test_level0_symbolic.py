"""Tests for Level 0 — Symbolic Logic (formal logic engine)."""
from cognitive_engine.levels.level0_symbolic import SymbolicLevel
from cognitive_engine.levels.base import ReasoningContext
from cognitive_engine.core.models import Graph, Node, Edge, EdgeType, NodeType
from uuid import uuid4


def _make_graph():
    n1 = Node(id=uuid4(), text="A", type=NodeType.CLAIM, opinion=(0.8, 0.1, 0.1, 0.5))
    n2 = Node(id=uuid4(), text="B", type=NodeType.EVIDENCE, opinion=(0.7, 0.2, 0.1, 0.5))
    g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[])
    return g


class TestSymbolicLevel:
    def test_add_fact_and_query(self):
        sl = SymbolicLevel()
        sl.add_fact("A", True, strength=1.0)
        sl.add_fact("B", True, strength=1.0)
        sl.add_rule(["A", "B"], "C", strength=0.9)
        sl.modus_ponens()
        assert sl.get_fact("C") is True

    def test_modus_ponens(self):
        sl = SymbolicLevel()
        sl.add_fact("A", True, strength=0.85)
        sl.add_rule(["A"], "B", strength=0.85)
        sl.modus_ponens()
        assert sl.get_fact("B") is True

    def test_missing_antecedent(self):
        sl = SymbolicLevel()
        sl.add_fact("A", True, strength=1.0)
        sl.add_rule(["A", "B"], "C", strength=0.9)
        sl.modus_ponens()
        assert sl.get_fact("C") is None  # B not true

    def test_consistency_check(self):
        sl = SymbolicLevel()
        sl.add_fact("A", True, strength=1.0)
        sl.add_fact("B", False, strength=1.0)
        sl.add_rule(["A"], "B", strength=1.0)
        sl.modus_ponens()
        contradictions = sl.check_consistency()
        assert len(contradictions) > 0

    def test_no_contradiction(self):
        sl = SymbolicLevel()
        sl.add_fact("A", True, strength=1.0)
        sl.add_fact("B", True, strength=1.0)
        sl.add_rule(["A"], "B", strength=1.0)
        sl.modus_ponens()
        contradictions = sl.check_consistency()
        assert len(contradictions) == 0

    def test_constraint_satisfaction(self):
        sl = SymbolicLevel()
        sl.add_fact("A", True, strength=1.0)
        sl.add_fact("B", False, strength=1.0)
        g = _make_graph()
        context = ReasoningContext()
        violations = sl.constraint_satisfaction(g)
        assert isinstance(violations, dict)

    def test_compute(self):
        g = _make_graph()
        context = ReasoningContext()
        sl = SymbolicLevel()
        output = sl.compute(g, context)
        assert len(output.beliefs) == 2
        assert all(isinstance(v, float) for v in output.beliefs.values())

    def test_level_number(self):
        sl = SymbolicLevel()
        assert sl.level_number == 0
