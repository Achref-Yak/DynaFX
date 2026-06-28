"""Tests for cross-domain complexity metrics, context similarity, and risk attitude."""

import pytest
from uuid import uuid4
from dynafx.core.models import Graph, Node, NodeType, EdgeType, Edge, Opinion
from dynafx.core.math import (
    cross_domain_edge_density,
    domain_entanglement,
    causal_chain_depth,
    feedback_loop_count,
    context_similarity,
    risk_adjusted_belief,
    stakeholder_utility,
    aggregate_stakeholder_beliefs,
)


def _make_node(ntype: NodeType = NodeType.CLAIM) -> Node:
    return Node(id=uuid4(), type=ntype, opinion=Opinion.from_tuple((0.6, 0.2, 0.2, 0.5)))


def _make_edge(src, tgt, etype: EdgeType = EdgeType.SUPPORTS) -> Edge:
    return Edge(source_id=src, target_id=tgt, type=etype, weight=1.0)


class TestCrossDomainEdgeDensity:
    def test_empty(self):
        assert cross_domain_edge_density({}, {}, {}) == 0.0

    def test_all_intra(self):
        n1, n2 = uuid4(), uuid4()
        nodes = {n1: _make_node(), n2: _make_node()}
        edges = {uuid4(): _make_edge(n1, n2)}
        comms = {n1: 0, n2: 0}
        assert cross_domain_edge_density(nodes, edges, comms) == 0.0

    def test_all_inter(self):
        n1, n2 = uuid4(), uuid4()
        nodes = {n1: _make_node(), n2: _make_node()}
        edges = {uuid4(): _make_edge(n1, n2)}
        comms = {n1: 0, n2: 1}
        assert cross_domain_edge_density(nodes, edges, comms) == 1.0

    def test_mixed(self):
        n1, n2, n3 = uuid4(), uuid4(), uuid4()
        nodes = {n1: _make_node(), n2: _make_node(), n3: _make_node()}
        edges = {
            uuid4(): _make_edge(n1, n2),  # intra (comm 0)
            uuid4(): _make_edge(n1, n3),  # inter (comm 0→1)
        }
        comms = {n1: 0, n2: 0, n3: 1}
        assert cross_domain_edge_density(nodes, edges, comms) == pytest.approx(0.5)


class TestDomainEntanglement:
    def test_empty(self):
        assert domain_entanglement({}, {}, {}) == 0.0

    def test_segregated(self):
        """All CLAIMs in comm 0, all EVIDENCE in comm 1 → low entropy."""
        n1, n2 = uuid4(), uuid4()
        nodes = {
            n1: Node(id=n1, type=NodeType.CLAIM),
            n2: Node(id=n2, type=NodeType.EVIDENCE),
        }
        comms = {n1: 0, n2: 1}
        e = domain_entanglement(nodes, {}, comms)
        assert e >= 0.0  # low entropy, low entanglement

    def test_entangled(self):
        """CLAIMs and EVIDENCE mixed in both communities → higher entropy."""
        n1, n2, n3, n4 = uuid4(), uuid4(), uuid4(), uuid4()
        nodes = {
            n1: Node(id=n1, type=NodeType.CLAIM),
            n2: Node(id=n2, type=NodeType.EVIDENCE),
            n3: Node(id=n3, type=NodeType.CLAIM),
            n4: Node(id=n4, type=NodeType.EVIDENCE),
        }
        comms = {n1: 0, n2: 0, n3: 1, n4: 1}
        e = domain_entanglement(nodes, {}, comms)
        assert e > 0.0  # higher entropy


class TestCausalChainDepth:
    def test_no_causes(self):
        n1 = uuid4()
        nodes = {n1: _make_node()}
        edges = {uuid4(): _make_edge(n1, uuid4())}
        assert causal_chain_depth(nodes, edges) == 0

    def test_linear_chain(self):
        """A→B→C→D chain has depth 3."""
        n1, n2, n3, n4 = uuid4(), uuid4(), uuid4(), uuid4()
        nodes = {nid: _make_node(NodeType.PROCESS) for nid in [n1, n2, n3, n4]}
        edges = {
            uuid4(): _make_edge(n1, n2, EdgeType.CAUSES),
            uuid4(): _make_edge(n2, n3, EdgeType.CAUSES),
            uuid4(): _make_edge(n3, n4, EdgeType.CAUSES),
        }
        assert causal_chain_depth(nodes, edges) == 3

    def test_parallel_chains(self):
        """A→B→C and A→D, depth = 2 (longest path)."""
        n1, n2, n3, n4 = uuid4(), uuid4(), uuid4(), uuid4()
        nodes = {nid: _make_node(NodeType.PROCESS) for nid in [n1, n2, n3, n4]}
        edges = {
            uuid4(): _make_edge(n1, n2, EdgeType.CAUSES),
            uuid4(): _make_edge(n2, n3, EdgeType.CAUSES),
            uuid4(): _make_edge(n1, n4, EdgeType.CAUSES),
        }
        assert causal_chain_depth(nodes, edges) == 2


class TestFeedbackLoopCount:
    def test_no_loops(self):
        n1, n2 = uuid4(), uuid4()
        nodes = {n1: _make_node(), n2: _make_node()}
        edges = {uuid4(): _make_edge(n1, n2, EdgeType.CAUSES)}
        assert feedback_loop_count(nodes, edges) == 0

    def test_one_loop(self):
        """A→B→A is one cycle."""
        n1, n2 = uuid4(), uuid4()
        nodes = {n1: _make_node(), n2: _make_node()}
        edges = {
            uuid4(): _make_edge(n1, n2, EdgeType.CAUSES),
            uuid4(): _make_edge(n2, n1, EdgeType.CAUSES),
        }
        assert feedback_loop_count(nodes, edges) == 1


class TestContextSimilarity:
    def test_same_type_same_community(self):
        n1, n2 = uuid4(), uuid4()
        nodes = {n1: _make_node(NodeType.CLAIM), n2: _make_node(NodeType.CLAIM)}
        comms = {n1: 0, n2: 0}
        sim = context_similarity(n1, n2, nodes, {}, comms)
        # type=1.0, community=1.0, both isolated=0.5 → avg=0.833
        assert sim == pytest.approx(0.833, abs=0.01)

    def test_different_type_different_community(self):
        n1, n2 = uuid4(), uuid4()
        nodes = {n1: _make_node(NodeType.CLAIM), n2: _make_node(NodeType.EVIDENCE)}
        comms = {n1: 0, n2: 1}
        sim = context_similarity(n1, n2, nodes, {}, comms)
        # type=0.0, community=0.0, both isolated=0.5 → avg=0.167
        assert sim == pytest.approx(0.167, abs=0.01)

    def test_shared_neighbors(self):
        n1, n2, n3 = uuid4(), uuid4(), uuid4()
        nodes = {
            n1: _make_node(NodeType.CLAIM),
            n2: _make_node(NodeType.CLAIM),
            n3: _make_node(),
        }
        edges = {
            uuid4(): _make_edge(n1, n3, EdgeType.SUPPORTS),
            uuid4(): _make_edge(n2, n3, EdgeType.SUPPORTS),
        }
        sim = context_similarity(n1, n2, nodes, edges, {n1: 0, n2: 0})
        # type=1.0, community=1.0, neighbors: {n3}∩{n3}/{n3}∪{n3}=1.0 → avg=1.0
        assert sim == pytest.approx(1.0)

    def test_missing_node(self):
        n1 = uuid4()
        assert context_similarity(n1, uuid4(), {}, {}) == 0.0


class TestRiskAdjustedBelief:
    def test_neutral(self):
        assert risk_adjusted_belief(0.8, alpha=1.0) == 0.8

    def test_risk_averse(self):
        # alpha > 1 → concave utility → lowers high beliefs
        assert risk_adjusted_belief(0.8, alpha=2.0) < 0.8

    def test_risk_seeking(self):
        # alpha < 1 → convex utility → amplifies high beliefs
        assert risk_adjusted_belief(0.8, alpha=0.5) > 0.8

    def test_zero_belief(self):
        assert risk_adjusted_belief(0.0, alpha=2.0) == 0.0

    def test_cara(self):
        val = risk_adjusted_belief(0.5, alpha=2.0, risk_measure="cara")
        assert 0.0 <= val <= 1.0


class TestStakeholderUtility:
    def test_neutral_at_reference(self):
        assert stakeholder_utility(0.5, alpha=1.0) == pytest.approx(0.0)

    def test_gain_frame(self):
        assert stakeholder_utility(0.8, alpha=1.0) > 0

    def test_loss_frame(self):
        assert stakeholder_utility(0.2, alpha=1.0) < 0

    def test_loss_aversion_amplifies(self):
        loss_utility = abs(stakeholder_utility(0.2, alpha=1.0))
        gain_utility = stakeholder_utility(0.8, alpha=1.0)
        # loss should be ~2.25x the gain
        assert loss_utility > gain_utility


class TestAggregateStakeholderBeliefs:
    def test_empty(self):
        assert aggregate_stakeholder_beliefs([]) == 0.0

    def test_equal_weights(self):
        beliefs = [(0.8, 1.0), (0.4, 1.0)]
        assert aggregate_stakeholder_beliefs(beliefs) == pytest.approx(0.6)

    def test_weighted(self):
        beliefs = [(0.8, 3.0), (0.4, 1.0)]
        expected = (0.8 * 3 + 0.4 * 1) / 4
        assert aggregate_stakeholder_beliefs(beliefs) == pytest.approx(expected)

    def test_zero_weight(self):
        beliefs = [(0.8, 0.0), (0.4, 0.0)]
        assert aggregate_stakeholder_beliefs(beliefs) == 0.0
