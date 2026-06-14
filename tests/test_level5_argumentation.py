"""Tests for Level 5 — Argumentation Theory."""
from cognitive_engine.levels.level5_argumentation import ArgumentationLevel
from cognitive_engine.levels.base import ReasoningContext
from cognitive_engine.core.models import Graph, Node, Edge, EdgeType, NodeType
from uuid import uuid4


def _make_argumentation_graph():
    n1 = Node(id=uuid4(), text="Claim", type=NodeType.CLAIM, opinion=(0.8, 0.1, 0.1, 0.5))
    n2 = Node(id=uuid4(), text="Evidence", type=NodeType.EVIDENCE, opinion=(0.7, 0.2, 0.1, 0.5))
    n3 = Node(id=uuid4(), text="Counter", type=NodeType.COUNTERCLAIM, opinion=(0.6, 0.3, 0.1, 0.5))
    e1 = Edge(id=uuid4(), source_id=n2.id, target_id=n1.id, type=EdgeType.SUPPORTS, warrant=((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5)))
    e2 = Edge(id=uuid4(), source_id=n3.id, target_id=n1.id, type=EdgeType.CONTRADICTS, warrant=((0.6, 0.3, 0.1, 0.5), (0.4, 0.5, 0.1, 0.5)))
    g = Graph(nodes={n1.id: n1, n2.id: n2, n3.id: n3}, edges=[e1, e2])
    return g


class TestArgumentationLevel:
    def test_compute(self):
        g = _make_argumentation_graph()
        context = ReasoningContext()
        level = ArgumentationLevel()
        output = level.compute(g, context)
        assert len(output.beliefs) == 3
        assert all(isinstance(v, float) for v in output.beliefs.values())

    def test_build_argument_graph(self):
        g = _make_argumentation_graph()
        level = ArgumentationLevel()
        arg_graph = level.build_argument_graph(g)
        assert arg_graph is not None
        assert len(arg_graph.nodes()) > 0

    def test_compute_support(self):
        g = _make_argumentation_graph()
        level = ArgumentationLevel()
        arg_graph = level.build_argument_graph(g)
        beliefs = {nid: 0.5 for nid in g.nodes}
        nid = list(g.nodes.keys())[0]
        support = level.compute_support(nid, arg_graph, beliefs)
        assert isinstance(support, float)
        assert 0.0 <= support <= 1.0

    def test_compute_acceptability(self):
        g = _make_argumentation_graph()
        level = ArgumentationLevel()
        arg_graph = level.build_argument_graph(g)
        beliefs = {nid: 0.5 for nid in g.nodes}
        nid = list(g.nodes.keys())[0]
        acceptability = level.compute_acceptability(nid, arg_graph, beliefs)
        assert isinstance(acceptability, float)
        assert -1.0 <= acceptability <= 1.0

    def test_dung_semantics(self):
        g = _make_argumentation_graph()
        level = ArgumentationLevel()
        arg_graph = level.build_argument_graph(g)
        beliefs = {nid: 0.5 for nid in g.nodes}
        result = level.dung_semantics(beliefs, arg_graph)
        assert isinstance(result, set)
        assert all(isinstance(x, uuid4().__class__) for x in result)

    def test_level_number(self):
        g = _make_argumentation_graph()
        context = ReasoningContext()
        level = ArgumentationLevel()
        output = level.compute(g, context)
        assert level.level_number == 5
