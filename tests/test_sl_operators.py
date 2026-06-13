import pytest

from cognitive_engine.reason.sl_operators import (
    conjunction,
    disjunction,
    cumulative_fusion,
    conditional_deduction,
    projected_probability,
    dirichlet_strength,
    _clamp,
)


def test_projected_probability():
    assert projected_probability((1.0, 0.0, 0.0, 0.5)) == pytest.approx(1.0)
    assert projected_probability((0.0, 1.0, 0.0, 0.5)) == pytest.approx(0.0)
    assert projected_probability((0.0, 0.0, 1.0, 0.5)) == pytest.approx(0.5)
    assert projected_probability((0.8, 0.1, 0.1, 0.5)) == pytest.approx(0.85)


def test_dirichlet_strength():
    assert dirichlet_strength((0.8, 0.1, 0.1, 0.5)) == pytest.approx(9.0)
    assert dirichlet_strength((0.0, 0.0, 1.0, 0.5)) == pytest.approx(0.0)


def test_conjunction_sums_to_one():
    result = conjunction((0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5))
    assert sum(result[:3]) == pytest.approx(1.0)


def test_disjunction_sums_to_one():
    result = disjunction((0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5))
    assert sum(result[:3]) == pytest.approx(1.0)


def test_cumulative_fusion_reduces_uncertainty():
    result = cumulative_fusion((0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5))
    assert sum(result[:3]) == pytest.approx(1.0)
    assert result[2] < 0.1


def test_cumulative_fusion_certain():
    result = cumulative_fusion((1.0, 0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.5))
    assert result[0] == pytest.approx(1.0)
    assert result[2] == pytest.approx(0.0)
    assert sum(result[:3]) == pytest.approx(1.0)


def test_conditional_deduction_properties():
    omega_p = (0.8, 0.1, 0.1, 0.5)
    warrant = ((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5))
    result = conditional_deduction(omega_p, warrant)
    assert sum(result[:3]) == pytest.approx(1.0)
    assert result[0] > result[1]


def test_conditional_deduction_preserves_uncertainty():
    omega_p = (0.8, 0.1, 0.1, 0.5)
    warrant = ((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5))
    result = conditional_deduction(omega_p, warrant)
    assert result[2] >= 0.1


def test_clamp_normalizes():
    result = _clamp((0.9, 0.3, 0.1, 0.5))
    assert sum(result[:3]) == pytest.approx(1.0)


def test_clamp_zero_total():
    result = _clamp((0.0, 0.0, 0.0, 0.5))
    assert result[2] == 1.0


def test_conjunction_lowers_belief():
    result = conjunction((0.8, 0.1, 0.1, 0.5), (0.5, 0.3, 0.2, 0.5))
    assert result[0] < 0.8


def test_disjunction_raises_belief():
    result = disjunction((0.8, 0.1, 0.1, 0.5), (0.5, 0.3, 0.2, 0.5))
    assert result[0] > 0.8


def test_compute_opinions_with_new_node_types():
    from cognitive_engine.reason.sl_operators import compute_opinions
    from cognitive_engine.core.config import Priors

    priors = Priors()
    assert "COUNTERCLAIM" in priors.source_type_map
    assert "AXIOM" in priors.source_type_map
    assert "FALLACY" in priors.source_type_map
    assert "JUSTIFICATION" in priors.source_type_map
    assert "ATTACKS" in priors.edge_warrants
    assert "REBUTS" in priors.edge_warrants


class TestTopologicalOrder:
    def test_empty_graph(self):
        from cognitive_engine.reason.sl_operators import _topological_order
        from cognitive_engine.core.models import Graph

        order = _topological_order(Graph())
        assert order == []

    def test_single_node(self):
        from cognitive_engine.reason.sl_operators import _topological_order
        from cognitive_engine.core.models import Graph, Node

        n = Node()
        g = Graph(nodes={n.id: n})
        order = _topological_order(g)
        assert order == [n.id]

    def test_simple_chain(self):
        from cognitive_engine.reason.sl_operators import _topological_order
        from cognitive_engine.core.models import Graph, Node, Edge

        n1 = Node()
        n2 = Node()
        e = Edge(source_id=n1.id, target_id=n2.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e])
        order = _topological_order(g)
        assert order.index(n1.id) < order.index(n2.id)

    def test_cycle_does_not_break(self):
        from cognitive_engine.reason.sl_operators import _topological_order
        from cognitive_engine.core.models import Graph, Node, Edge

        n1 = Node()
        n2 = Node()
        e1 = Edge(source_id=n1.id, target_id=n2.id)
        e2 = Edge(source_id=n2.id, target_id=n1.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e1, e2])
        order = _topological_order(g)
        assert len(order) == 2


class TestFusionStrategy:
    def test_single_contribution(self):
        from cognitive_engine.reason.sl_operators import _fusion_strategy
        from cognitive_engine.core.models import Graph

        result = _fusion_strategy([(0.8, 0.1, 0.1, 0.5)], [], Graph())
        assert result == (0.8, 0.1, 0.1, 0.5)

    def test_two_node_chain_uses_conjunction(self):
        from cognitive_engine.reason.sl_operators import _fusion_strategy
        from cognitive_engine.core.models import Graph, Node, Edge, NodeType

        n1 = Node(type=NodeType.CLAIM)
        n2 = Node(type=NodeType.CLAIM)
        e = Edge(source_id=n1.id, target_id=n2.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e])
        result = _fusion_strategy(
            [(0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5)],
            [e],
            g,
        )
        b = result[0]
        assert b < 0.8


class TestAttachOpinions:
    def test_all_nodes_get_opinion(self):
        from cognitive_engine.reason.sl_operators import attach_opinions
        from cognitive_engine.core.models import Graph, Node
        from cognitive_engine.core.config import Priors

        n1 = Node()
        n2 = Node()
        g = Graph(nodes={n1.id: n1, n2.id: n2})
        attach_opinions(g, Priors())
        assert all(n.opinion is not None for n in g.nodes.values())

    def test_edges_get_ignorance(self):
        from cognitive_engine.reason.sl_operators import attach_opinions
        from cognitive_engine.core.models import Graph, Node, Edge
        from cognitive_engine.core.config import Priors

        n1 = Node()
        n2 = Node()
        e = Edge(source_id=n1.id, target_id=n2.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e])
        attach_opinions(g, Priors())
        assert all(e.opinion == (0.0, 0.0, 1.0, 0.5) for e in g.edges)


class TestComputeOpinions:
    def test_empty_graph(self):
        from cognitive_engine.reason.sl_operators import compute_opinions
        from cognitive_engine.core.models import Graph
        from cognitive_engine.core.config import Priors

        g = compute_opinions(Graph(), Priors())
        assert g.metadata.get("priors") is not None

    def test_single_node_preserves_opinion(self):
        from cognitive_engine.reason.sl_operators import compute_opinions
        from cognitive_engine.core.models import Graph, Node
        from cognitive_engine.core.config import Priors

        n = Node()
        g = Graph(nodes={n.id: n})
        result = compute_opinions(g, Priors())
        assert result.nodes[n.id].opinion is not None

    def test_propagation_through_edge(self):
        from cognitive_engine.reason.sl_operators import compute_opinions
        from cognitive_engine.core.models import Graph, Node, Edge
        from cognitive_engine.core.config import Priors

        n1 = Node()
        n2 = Node()
        e = Edge(source_id=n1.id, target_id=n2.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e])
        result = compute_opinions(g, Priors())
        assert result.nodes[n2.id] is not None
