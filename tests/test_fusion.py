import pytest

from cognitive_engine.core.models import (
    Edge,
    FusionSituation,
    Graph,
    Node,
    NodeType,
    Opinion,
)
from cognitive_engine.reason.fusion import (
    classify_fusion_situation,
    consensus_compromise,
    cumulative_fusion,
    trust_transfer,
    weighted_belief_fusion,
)
from cognitive_engine.reason.sl_operators import conjunction, disjunction


class TestConsensusCompromise:
    def test_agreement_preserves_belief(self):
        a = (0.8, 0.1, 0.1, 0.5)
        b = (0.7, 0.2, 0.1, 0.5)
        result = consensus_compromise(a, b)
        assert sum(result[:3]) == pytest.approx(1.0)
        assert result[0] > 0.5

    def test_conflict_equalizes_belief(self):
        a = (0.9, 0.0, 0.1, 0.5)
        b = (0.0, 0.9, 0.1, 0.5)
        result = consensus_compromise(a, b)
        assert sum(result[:3]) == pytest.approx(1.0)
        assert result[0] == pytest.approx(result[1], rel=1e-3)

    def test_complete_conflict_yields_ignorance(self):
        a = (1.0, 0.0, 0.0, 0.5)
        b = (0.0, 1.0, 0.0, 0.5)
        result = consensus_compromise(a, b)
        assert sum(result[:3]) == pytest.approx(1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_partial_conflict(self):
        a = (0.6, 0.2, 0.2, 0.5)
        b = (0.2, 0.6, 0.2, 0.5)
        result = consensus_compromise(a, b)
        assert sum(result[:3]) == pytest.approx(1.0)
        assert result[0] < 0.6
        assert result[1] < 0.6

    def test_invariant(self):
        a = (0.8, 0.1, 0.1, 0.5)
        b = (0.4, 0.3, 0.3, 0.5)
        result = consensus_compromise(a, b)
        assert sum(result[:3]) == pytest.approx(1.0)

    def test_sums_to_one_under_random_opinions(self):
        for b_a, d_a, b_b, d_b in [
            (0.7, 0.1, 0.5, 0.3),
            (0.9, 0.0, 0.1, 0.8),
            (0.2, 0.7, 0.6, 0.2),
            (0.3, 0.3, 0.3, 0.3),
            (0.5, 0.4, 0.4, 0.5),
        ]:
            u_a = 1.0 - b_a - d_a
            u_b = 1.0 - b_b - d_b
            a_op = (b_a, d_a, u_a, 0.5)
            b_op = (b_b, d_b, u_b, 0.5)
            result = consensus_compromise(a_op, b_op)
            assert sum(result[:3]) == pytest.approx(1.0), f"Failed for {a_op}, {b_op}"


class TestWeightedBeliefFusion:
    def test_equal_weights_average(self):
        a = (0.8, 0.1, 0.1, 0.5)
        b = (0.4, 0.3, 0.3, 0.5)
        result = weighted_belief_fusion(a, b, 0.5, 0.5)
        assert result[0] == pytest.approx(0.6)
        assert result[1] == pytest.approx(0.2)
        assert result[2] == pytest.approx(0.2)

    def test_biased_weights(self):
        a = (0.9, 0.0, 0.1, 0.5)
        b = (0.1, 0.8, 0.1, 0.5)
        result = weighted_belief_fusion(a, b, 0.9, 0.1)
        assert result[0] > 0.7
        assert result[1] < 0.1

    def test_sums_to_one(self):
        a = (0.8, 0.1, 0.1, 0.5)
        b = (0.2, 0.6, 0.2, 0.5)
        for wa, wb in [(0.3, 0.7), (0.5, 0.5), (0.8, 0.2)]:
            result = weighted_belief_fusion(a, b, wa, wb)
            assert sum(result[:3]) == pytest.approx(1.0)

    def test_zero_total_weight_falls_back(self):
        a = (0.8, 0.1, 0.1, 0.5)
        b = (0.2, 0.6, 0.2, 0.5)
        result = weighted_belief_fusion(a, b, 0.0, 0.0)
        assert result == a


class TestTrustTransfer:
    def test_trusted_source_high_belief(self):
        omega_s = (0.9, 0.0, 0.1, 0.5)
        omega_r = (0.8, 0.1, 0.1, 0.5)
        result = trust_transfer(omega_s, omega_r)
        assert result[0] == pytest.approx(0.72)
        assert result[1] == pytest.approx(0.09)
        assert result[2] == pytest.approx(0.19)

    def test_untrusted_source_high_uncertainty(self):
        omega_s = (0.0, 0.0, 1.0, 0.5)
        omega_r = (0.9, 0.05, 0.05, 0.5)
        result = trust_transfer(omega_s, omega_r)
        assert result[2] == pytest.approx(1.0)
        assert result[0] == pytest.approx(0.0)

    def test_invariant(self):
        omega_s = (0.7, 0.1, 0.2, 0.5)
        omega_r = (0.6, 0.2, 0.2, 0.5)
        result = trust_transfer(omega_s, omega_r)
        assert sum(result[:3]) == pytest.approx(1.0)


class TestClassifyFusionSituation:
    def test_single_contribution_is_independent(self):
        result = classify_fusion_situation(
            [(0.8, 0.1, 0.1, 0.5)], [], Graph(),
        )
        assert result == FusionSituation.INDEPENDENT_SOURCES

    def test_conflicting_opinions_detected(self):
        n1 = Node()
        n2 = Node()
        e = Edge(source_id=n1.id, target_id=n2.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e])
        result = classify_fusion_situation(
            [(0.9, 0.0, 0.1, 0.5), (0.0, 0.9, 0.1, 0.5)],
            [e, e],
            g,
        )
        assert result == FusionSituation.CONFLICTING_VIEWS

    def test_non_conflicting_is_independent(self):
        n1 = Node()
        n2 = Node()
        n3 = Node()
        e1 = Edge(source_id=n1.id, target_id=n3.id)
        e2 = Edge(source_id=n2.id, target_id=n3.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2, n3.id: n3}, edges=[e1, e2])
        result = classify_fusion_situation(
            [(0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5)],
            [e1, e2],
            g,
        )
        assert result == FusionSituation.INDEPENDENT_SOURCES

    def test_same_source_edge_detected(self):
        n1 = Node()
        n2 = Node()
        e1 = Edge(source_id=n1.id, target_id=n2.id)
        e2 = Edge(source_id=n1.id, target_id=n2.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e1, e2])
        result = classify_fusion_situation(
            [(0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5)],
            [e1, e2],
            g,
        )
        assert result == FusionSituation.SAME_SOURCE

    def test_dependent_sources_shared_ancestor(self):
        root = Node()
        a = Node()
        b = Node()
        e1 = Edge(source_id=root.id, target_id=a.id)
        e2 = Edge(source_id=root.id, target_id=b.id)
        e3 = Edge(source_id=a.id, target_id=b.id)
        g = Graph(
            nodes={root.id: root, a.id: a, b.id: b},
            edges=[e1, e2, e3],
        )
        result = classify_fusion_situation(
            [(0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5)],
            [e2, e3],
            g,
        )
        assert result == FusionSituation.DEPENDENT_SOURCES


class TestFusionEdgeToEdge:
    def test_non_conflicting_claims_use_cumulative(self):
        from cognitive_engine.reason.sl_operators import _fusion_strategy
        n1 = Node(type=NodeType.CLAIM)
        n2 = Node(type=NodeType.CLAIM)
        e = Edge(source_id=n1.id, target_id=n2.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e])
        result = _fusion_strategy(
            [(0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5)],
            [e],
            g,
        )
        expected = cumulative_fusion(
            (0.8, 0.1, 0.1, 0.5),
            (0.7, 0.2, 0.1, 0.5),
        )
        assert result[0] == pytest.approx(expected[0])
        assert sum(result[:3]) == pytest.approx(1.0)

    def test_evidence_to_evidence_uses_cumulative(self):
        from cognitive_engine.reason.sl_operators import _fusion_strategy
        n1 = Node(type=NodeType.EVIDENCE)
        n2 = Node(type=NodeType.EVIDENCE)
        e = Edge(source_id=n1.id, target_id=n2.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e])
        result = _fusion_strategy(
            [(0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5)],
            [e],
            g,
        )
        expected = cumulative_fusion(
            (0.8, 0.1, 0.1, 0.5),
            (0.7, 0.2, 0.1, 0.5),
        )
        assert result[0] == pytest.approx(expected[0])
        assert sum(result[:3]) == pytest.approx(1.0)

    def test_conflicting_evidence_uses_consensus_compromise(self):
        from cognitive_engine.reason.sl_operators import _fusion_strategy
        n1 = Node(type=NodeType.EVIDENCE)
        n2 = Node(type=NodeType.EVIDENCE)
        e1 = Edge(source_id=n1.id, target_id=n2.id)
        e2 = Edge(source_id=n2.id, target_id=n1.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e1, e2])
        result = _fusion_strategy(
            [(0.9, 0.0, 0.1, 0.5), (0.0, 0.9, 0.1, 0.5)],
            [e1, e2],
            g,
        )
        assert sum(result[:3]) == pytest.approx(1.0)
        assert abs(result[0] - result[1]) < 0.1

    def test_non_conflicting_mixed_types_now_uses_cumulative(self):
        from cognitive_engine.reason.sl_operators import _fusion_strategy
        n1 = Node(type=NodeType.AXIOM)
        n2 = Node(type=NodeType.CLAIM)
        e = Edge(source_id=n1.id, target_id=n2.id)
        g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e])
        result = _fusion_strategy(
            [(0.8, 0.1, 0.1, 0.5), (0.5, 0.3, 0.2, 0.5)],
            [e],
            g,
        )
        assert sum(result[:3]) == pytest.approx(1.0)
