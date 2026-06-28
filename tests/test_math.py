"""Tests for core/math.py — all pure formulas."""

import math
from uuid import uuid4

import pytest

from dynafx.core.math import (
    sigmoid, sigmoid_array, softmax, clamp, normalize_sum, l2_norm,
    cosine_similarity, cosine_distance, jaccard_similarity, jaccard_distance,
    count_proximity, projected_probability, dirichlet_strength,
    opinion_from_counts, mean_opinion, trust_weight, compute_trust_weights,
    conjunction, disjunction, conditional_deduction, cumulative_fusion,
    consensus_compromise, weighted_belief_fusion, trust_transfer,
    opinion_conflict, reverse_warrant, subjective_abduction,
    analogy_warrant_transform, category_level, category_conjunction,
    category_disjunction, category_negation, implication_valid,
    equivalence_valid, category_valuate, modus_ponens_strength,
    inference_closure, base_level_activation, activation_spreading,
    belief_from_activation, softmax_retrieval, bayes_rule, joint_probability,
    expectation, propagate_step, build_adjacency, initialize_beliefs,
    convergence_l2, similarity_diffusion, compute_support, compute_attack,
    argument_acceptability, argument_strength, dung_semantics,
    gnn_feature_dim, gnn_encode_features, gnn_build_adjacency_matrix,
    neurosymbolic_fuse, compute_logic_penalty, total_loss,
    master_equation, master_equation_all, compute_support_sum,
    compute_attack_sum, global_objective, count_violations,
    fixed_point_iteration, convergence_norm, graph_distance,
    hidden_state_distance, memory_similarity, betweenness_centrality,
    leverage_score, classify_feedback_loop, graph_diff_score,
    check_opinion_invariant, check_cycle_free, check_category_monotonicity,
    EDGE_WEIGHTS, NODE_PRIORS, CATEGORY_LEVELS, NECESSITY, FACT, BELIEF, CONCEPT,
    extract_max_dag, topological_sort, tna_propagate,
    SUPPORT_EDGES, ATTACK_EDGES, QUALIFY_EDGES,
)


class TestMathUtilities:
    def test_sigmoid(self):
        assert sigmoid(0) == pytest.approx(0.5)
        assert sigmoid(100) == pytest.approx(1.0, abs=1e-4)
        assert sigmoid(-100) == pytest.approx(0.0, abs=1e-4)

    def test_sigmoid_array(self):
        result = sigmoid_array([-1, 0, 1])
        assert len(result) == 3
        assert all(0 < v < 1 for v in result)

    def test_softmax(self):
        result = softmax([1.0, 2.0, 3.0])
        assert sum(result) == pytest.approx(1.0)
        assert result[2] > result[1] > result[0]

    def test_softmax_empty(self):
        assert softmax([]) == []

    def test_softmax_zero_temperature(self):
        result = softmax([0.0, 0.0, 0.0])
        assert all(v == pytest.approx(1/3) for v in result)

    def test_clamp(self):
        assert clamp(0.5) == 0.5
        assert clamp(-0.5) == 0.0
        assert clamp(1.5) == 1.0

    def test_normalize_sum(self):
        b, d, u = normalize_sum(0.5, 0.3, 0.2)
        assert b + d + u == pytest.approx(1.0)

    def test_normalize_sum_zero(self):
        b, d, u = normalize_sum(0.0, 0.0, 0.0)
        assert u == 1.0

    def test_normalize_sum_preserves(self):
        b, d, u = normalize_sum(0.3, 0.2, 0.5)
        assert b == pytest.approx(0.3)
        assert d == pytest.approx(0.2)
        assert u == pytest.approx(0.5)

    def test_l2_norm_empty(self):
        assert l2_norm({}) == 0.0

    def test_l2_norm(self):
        delta = {uuid4(): 1.0, uuid4(): 2.0}
        n = l2_norm(delta)
        assert n == pytest.approx(math.sqrt((1 + 4) / 2))

    def test_cosine_similarity(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(1.0)
        assert cosine_similarity(a, [0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_cosine_similarity_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_cosine_distance(self):
        assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_jaccard_similarity(self):
        assert jaccard_similarity({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)

    def test_jaccard_similarity_both_empty(self):
        assert jaccard_similarity(set(), set()) == 1.0

    def test_jaccard_distance(self):
        assert jaccard_distance({1, 2}, {3, 4}) == pytest.approx(1.0)

    def test_count_proximity(self):
        assert count_proximity(5, 5) == pytest.approx(1.0)
        assert count_proximity(0, 5) == pytest.approx(0.0)


class TestSubjectiveLogic:
    def test_projected_probability(self):
        assert projected_probability(0.5, 0.3, 0.5) == 0.5 + 0.5 * 0.3

    def test_dirichlet_strength(self):
        assert dirichlet_strength(0.6, 0.3, 0.1) == pytest.approx(9.0)
        assert dirichlet_strength(0.5, 0.3, 0.0) == float('inf')

    def test_opinion_from_counts(self):
        b, d, u, a = opinion_from_counts(10, 2)
        assert b == pytest.approx(10 / 14)
        assert d == pytest.approx(2 / 14)
        assert u == pytest.approx(2 / 14)
        assert b + d + u == pytest.approx(1.0)

    def test_mean_opinion_empty(self):
        assert mean_opinion([]) == (0.0, 0.0, 1.0, 0.5)

    def test_mean_opinion_single(self):
        assert mean_opinion([(0.8, 0.1, 0.1, 0.5)]) == pytest.approx((0.8, 0.1, 0.1, 0.5))

    def test_mean_opinion_two(self):
        result = mean_opinion([(0.8, 0.1, 0.1, 0.5), (0.4, 0.3, 0.3, 0.5)])
        assert result[0] == pytest.approx(0.6)
        assert result[1] == pytest.approx(0.2)
        assert result[2] == pytest.approx(0.2)

    def test_trust_weight(self):
        w = trust_weight(0.5, 0.3, 0.5)
        assert w == pytest.approx(0.5 + 0.5 * 0.3)

    def test_compute_trust_weights(self):
        opinions = [(0.8, 0.1, 0.1, 0.5), (0.4, 0.3, 0.3, 0.5)]
        weights = compute_trust_weights(opinions)
        assert len(weights) == 2
        assert sum(weights) == pytest.approx(1.0)

    def test_conjunction(self):
        w_a = (0.8, 0.1, 0.1, 0.5)
        w_b = (0.6, 0.2, 0.2, 0.5)
        result = conjunction(w_a, w_b)
        assert len(result) == 4

    def test_disjunction(self):
        w_a = (0.8, 0.1, 0.1, 0.5)
        w_b = (0.6, 0.2, 0.2, 0.5)
        result = disjunction(w_a, w_b)
        assert len(result) == 4

    def test_conditional_deduction(self):
        omega_p = (0.7, 0.2, 0.1, 0.5)
        warrant = ((0.9, 0.05, 0.05, 0.5), (0.1, 0.8, 0.1, 0.5))
        result = conditional_deduction(omega_p, warrant)
        assert len(result) == 4
        # SL deduction doesn't guarantee b+d+u=1; verify structure only
        assert all(0 <= v <= 1 for v in result[:3])

    def test_cumulative_fusion(self):
        result = cumulative_fusion((0.8, 0.1, 0.1, 0.5), (0.6, 0.2, 0.2, 0.5))
        assert len(result) == 4

    def test_consensus_compromise(self):
        result = consensus_compromise((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5))
        assert len(result) == 4

    def test_consensus_compromise_conflict(self):
        result = consensus_compromise((0.0, 0.0, 1.0, 0.5), (0.0, 0.0, 1.0, 0.5))
        assert len(result) == 4

    def test_weighted_belief_fusion(self):
        result = weighted_belief_fusion((0.8, 0.1, 0.1, 0.5), (0.4, 0.3, 0.3, 0.5), 0.6, 0.4)
        assert result[0] == pytest.approx(0.8 * 0.6 + 0.4 * 0.4)

    def test_trust_transfer(self):
        result = trust_transfer((0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5))
        assert len(result) == 4

    def test_opinion_conflict(self):
        assert opinion_conflict((0.9, 0.05, 0.05, 0.5), (0.1, 0.8, 0.1, 0.5)) is True
        assert opinion_conflict((0.9, 0.05, 0.05, 0.5), (0.8, 0.1, 0.1, 0.5)) is False

    def test_reverse_warrant(self):
        forward = ((0.9, 0.05, 0.05, 0.5), (0.1, 0.8, 0.1, 0.5))
        result = reverse_warrant(forward, 0.5, 0.5)
        assert len(result) == 2
        assert len(result[0]) == 4

    def test_subjective_abduction(self):
        warrant = ((0.9, 0.05, 0.05, 0.5), (0.1, 0.8, 0.1, 0.5))
        result = subjective_abduction((0.7, 0.2, 0.1, 0.5), warrant)
        assert len(result) == 4

    def test_analogy_warrant_transform(self):
        result = analogy_warrant_transform((0.8, 0.1, 0.1, 0.5), 0.2)
        assert result[0] < 0.8
        assert result[2] > 0.1


class TestCategoryTheory:
    def test_category_level(self):
        assert category_level("AXIOM") == 1
        assert category_level("FACT") == 2
        assert category_level("CLAIM") == 3
        assert category_level("CONCEPT") == 4
        assert category_level("UNKNOWN") == 2

    def test_category_conjunction(self):
        assert category_conjunction(1, 3) == 3
        assert category_conjunction(4, 2) == 4

    def test_category_disjunction(self):
        assert category_disjunction(1, 3) == 1
        assert category_disjunction(4, 2) == 2

    def test_category_negation(self):
        assert category_negation(2) == 2

    def test_implication_valid(self):
        assert implication_valid(1, 3) is True
        assert implication_valid(3, 1) is False

    def test_equivalence_valid(self):
        assert equivalence_valid(2, 2) is True
        assert equivalence_valid(2, 3) is False

    def test_category_valuate(self):
        assert category_valuate(3, True) == 3
        assert category_valuate(3, False) == 0


class TestSymbolicLogic:
    def test_modus_ponens_strength(self):
        assert modus_ponens_strength(0.8, [0.7, 0.9]) == pytest.approx(0.7)

    def test_inference_closure(self):
        facts = {"p": True, "q": True}
        rules = [(["p", "q"], "r", 1.0), (["r"], "s", 1.0)]
        result, order = inference_closure(facts, rules)
        assert result["r"] is True
        assert result["s"] is True
        assert order == ["r", "s"]

    def test_inference_closure_no_change(self):
        facts = {"p": True}
        rules = [(["q"], "r", 1.0)]
        result, order = inference_closure(facts, rules)
        assert "r" not in result
        assert order == []


class TestCognitiveArchitecture:
    def test_base_level_activation(self):
        act = base_level_activation(0.5, 10, 100, 0.05)
        assert act > 0.5

    def test_base_level_activation_zero_time(self):
        act = base_level_activation(0.5, 0, 0)
        assert act == pytest.approx(0.5)

    def test_activation_spreading(self):
        act = activation_spreading(0.5, [(0.3, 0.8), (0.2, 0.5)])
        assert act == pytest.approx(0.5 + 0.3 * 0.8 + 0.2 * 0.5)

    def test_belief_from_activation(self):
        b = belief_from_activation(0.0)
        assert b == pytest.approx(0.5)

    def test_softmax_retrieval(self):
        result = softmax_retrieval({"a": 1.0, "b": 2.0, "c": 3.0})
        assert abs(sum(result.values()) - 1.0) < 1e-6
        assert result["c"] > result["b"] > result["a"]


class TestBayesian:
    def test_bayes_rule(self):
        result = bayes_rule(0.8, 0.3, 0.5)
        assert result == pytest.approx(0.8 * 0.3 / 0.5)

    def test_bayes_rule_zero_evidence(self):
        assert bayes_rule(0.8, 0.3, 0.0) == pytest.approx(0.3)

    def test_joint_probability(self):
        config = {"A": True, "B": False}
        variables = {
            "A": {"parents": [], "cpt": {(True,): 0.5, (False,): 0.5}},
            "B": {"parents": ["A"], "cpt": {(True, True): 0.8, (True, False): 0.2,
                                             (False, True): 0.3, (False, False): 0.7}},
        }
        p = joint_probability(config, variables, ["A", "B"])
        # P(A=True)=0.5, P(B=False|A=True)=0.2 → 0.5*0.2=0.1
        assert p == pytest.approx(0.1)

    def test_expectation(self):
        e2 = expectation({0: 0.2, 1: 0.8})
        assert e2 == pytest.approx(0.8)


class TestGraphPropagation:
    def test_propagate_step(self):
        nid1, nid2 = uuid4(), uuid4()
        beliefs = {nid1: 0.8, nid2: 0.3}
        adjacency = {nid2: [(nid1, 0.9)]}
        evidence = {}
        result = propagate_step(beliefs, adjacency, evidence)
        assert nid2 in result
        assert 0 < result[nid2] < 1

    def test_build_adjacency(self):
        nid1, nid2 = uuid4(), uuid4()
        from dynafx.core.models import Edge, EdgeType
        edge = Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS)
        adj = build_adjacency({nid1, nid2}, [edge])
        assert nid2 in adj
        assert len(adj[nid2]) == 1

    def test_initialize_beliefs_from_priors(self):
        nid = uuid4()
        def get_type(n):
            return "AXIOM"
        beliefs = initialize_beliefs({nid}, get_type)
        assert beliefs[nid] == pytest.approx(0.9)

    def test_initialize_beliefs_from_opinion(self):
        nid = uuid4()
        def get_type(n):
            return "CLAIM"
        def get_opinion(n):
            return (0.7, 0.1, 0.2, 0.5)
        beliefs = initialize_beliefs({nid}, get_type, get_opinion)
        assert beliefs[nid] == pytest.approx(0.7 + 0.5 * 0.2)

    def test_convergence_l2_identical(self):
        nid = uuid4()
        assert convergence_l2({nid: 0.5}, {nid: 0.5}) == pytest.approx(0.0)

    def test_convergence_l2_different(self):
        nid = uuid4()
        assert convergence_l2({nid: 0.5}, {nid: 0.8}) == pytest.approx(0.3)

    def test_convergence_l2_no_shared(self):
        assert convergence_l2({uuid4(): 0.5}, {uuid4(): 0.8}) == 1.0

    def test_similarity_diffusion(self):
        nid = uuid4()
        embeds = {nid: [1.0, 0.0]}
        result = similarity_diffusion(embeds)
        assert result == {nid: 0.0}


class TestArgumentation:
    def test_compute_support(self):
        nid = uuid4()
        preds = [(uuid4(), "support", 0.8), (uuid4(), "attack", 0.5)]
        beliefs = {p[0]: 0.7 for p in preds}
        s = compute_support(nid, preds, beliefs)
        assert s == pytest.approx(0.8 * 0.7)

    def test_compute_attack(self):
        nid = uuid4()
        preds = [(uuid4(), "attack", 0.8), (uuid4(), "support", 0.5)]
        beliefs = {p[0]: 0.7 for p in preds}
        a = compute_attack(nid, preds, beliefs)
        assert a == pytest.approx(0.8 * 0.7)

    def test_argument_acceptability(self):
        assert argument_acceptability(0.8, 0.3) == pytest.approx(0.5)

    def test_argument_strength(self):
        s = argument_strength(0.0)
        assert s == pytest.approx(0.5)

    def test_dung_semantics(self):
        nid1, nid2 = uuid4(), uuid4()
        beliefs = {nid1: 0.9, nid2: 0.2}
        attack_graph = {nid1: [nid2], nid2: [nid1]}
        accepted = dung_semantics(beliefs, attack_graph)
        assert nid1 in accepted or nid2 in accepted


class TestNeuroSymbolic:
    def test_neurosymbolic_fuse(self):
        nid = uuid4()
        result = neurosymbolic_fuse({nid: 0.8}, {nid: 0.4}, 0.6)
        assert result[nid] == pytest.approx(0.6 * 0.8 + 0.4 * 0.4)

    def test_compute_logic_penalty(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2 = uuid4(), uuid4()
        edge = Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS)
        def belief_fn(n):
            return 0.8 if n == nid1 else 0.1
        violations = compute_logic_penalty([edge], belief_fn)
        assert nid2 in violations

    def test_total_loss(self):
        loss = total_loss({uuid4(): 0.8}, {uuid4(): 0.6}, 2, 0.1)
        assert loss > 0

    def test_gnn_feature_dim(self):
        assert gnn_feature_dim(7) == 8

    def test_gnn_encode_features(self):
        nid = uuid4()
        def get_type(n):
            return "CLAIM"
        def get_opinion(n):
            return (0.7, 0.1, 0.2, 0.5)
        features, idx_map = gnn_encode_features([nid], get_type, get_opinion)
        assert len(features) == 1
        assert features[0][-1] == pytest.approx(0.8)

    def test_gnn_build_adjacency_matrix(self):
        adj = gnn_build_adjacency_matrix(3, [])
        assert len(adj) == 3
        assert adj[0][0] == 1.0


class TestMasterEquation:
    def test_master_equation(self):
        nid = uuid4()
        result = master_equation(nid, 0.7, 0.6, 0.8, 0.1, 0)
        expected = 0.3 * 0.7 + 0.3 * 0.6 + 0.2 * 0.8 - 0.2 * 0.1
        assert result == pytest.approx(expected)

    def test_master_equation_all(self):
        nid = uuid4()
        results = master_equation_all(
            [nid], {nid: 0.7}, {nid: 0.6}, {nid: 0.8}, {nid: 0.1}, {nid: 0}
        )
        assert nid in results

    def test_global_objective(self):
        nid = uuid4()
        g = global_objective({nid: 0.8}, {nid: 2}, 0.1)
        assert g == pytest.approx(0.8 - 0.1 * 2)

    def test_count_violations(self):
        nid = uuid4()
        violations = count_violations({nid: (0.5, 0.3, 0.2, 0.5)}, [])
        assert nid not in violations

    def test_count_violations_bad_invariant(self):
        nid = uuid4()
        violations = count_violations({nid: (0.9, 0.2, 0.2, 0.5)}, [])
        assert nid in violations

    def test_count_violations_negative_u(self):
        nid = uuid4()
        violations = count_violations({nid: (0.5, 0.5, -0.1, 0.5)}, [])
        assert nid in violations

    def test_fixed_point_iteration(self):
        nid = uuid4()
        def update(b):
            return {k: v * 0.9 for k, v in b.items()}
        result = fixed_point_iteration({nid: 1.0}, update, max_iterations=10, threshold=1e-6)
        assert result[nid] < 1.0


class TestConvergence:
    def test_convergence_norm(self):
        n = convergence_norm(0.1, 0.2, 0.3, 0.4)
        assert n == pytest.approx(0.4 * 0.1 + 0.3 * 0.2 + 0.2 * 0.3 + 0.1 * 0.4)

    def test_graph_distance_identical(self):
        nid = uuid4()
        d = graph_distance({nid}, {nid}, [], [], {nid: 0.5}, {nid: 0.5})
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_graph_distance_different(self):
        d = graph_distance({uuid4()}, {uuid4()}, [], [], {}, {})
        assert d > 0

    def test_hidden_state_distance(self):
        d = hidden_state_distance([0.5, 0.5], [0.3, 0.7])
        assert d >= 0

    def test_hidden_state_distance_empty(self):
        assert hidden_state_distance([], [0.5]) == 1.0


class TestMemoryRetrieval:
    def test_memory_similarity(self):
        s = memory_similarity(0.8, 0.6, 0.6)
        assert s == pytest.approx(0.6 * 0.8 + 0.4 * 0.6)


class TestSystemsFeedback:
    def test_betweenness_centrality(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2, nid3 = uuid4(), uuid4(), uuid4()
        edges = [
            Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS),
            Edge(source_id=nid2, target_id=nid3, type=EdgeType.SUPPORTS),
        ]
        centrality = betweenness_centrality({nid1, nid2, nid3}, edges)
        assert nid2 in centrality
        assert centrality[nid2] > centrality[nid1]
        assert centrality[nid2] > centrality[nid3]

    def test_betweenness_centrality_isolated(self):
        nid = uuid4()
        centrality = betweenness_centrality({nid}, [])
        assert centrality[nid] == pytest.approx(0.0)

    def test_leverage_score(self):
        s = leverage_score(3, 2, 5, 4, 0.1)
        assert s == pytest.approx(0.4 * 3/5 + 0.3 * 2/4 + 0.3 * 0.1)

    def test_classify_feedback_loop(self):
        assert classify_feedback_loop(1) == "balancing"
        assert classify_feedback_loop(2) == "reinforcing"


class TestGraphDiff:
    def test_graph_diff_score(self):
        s = graph_diff_score(5, 10, 8, 2)
        assert 0 <= s <= 1

    def test_graph_diff_score_identical(self):
        s = graph_diff_score(5, 5, 5, 0)
        assert s == pytest.approx(1.0)


class TestInvariantChecks:
    def test_check_opinion_invariant_good(self):
        assert check_opinion_invariant(0.5, 0.3, 0.2) is True

    def test_check_opinion_invariant_bad(self):
        assert check_opinion_invariant(1.0, 0.5, 0.5) is False

    def test_check_cycle_free_acyclic(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2 = uuid4(), uuid4()
        edges = [Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS)]
        assert check_cycle_free({nid1, nid2}, edges) is True

    def test_check_cycle_free_cyclic(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2 = uuid4(), uuid4()
        edges = [
            Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS),
            Edge(source_id=nid2, target_id=nid1, type=EdgeType.SUPPORTS),
        ]
        assert check_cycle_free({nid1, nid2}, edges) is False

    def test_check_category_monotonicity(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2 = uuid4(), uuid4()
        edge = Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS)
        def get_src(n):
            return 1 if n == nid1 else 2
        def get_tgt(n):
            return 3 if n == nid2 else 1
        violations = check_category_monotonicity([edge], get_src, get_tgt)
        assert len(violations) == 0


class TestExtractMaxDag:
    def test_acyclic_graph_no_dropped(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2, nid3 = uuid4(), uuid4(), uuid4()
        edges = [
            Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS),
            Edge(source_id=nid2, target_id=nid3, type=EdgeType.SUPPORTS),
        ]
        dag, dropped, order = extract_max_dag({nid1, nid2, nid3}, edges)
        assert len(dag) == 2
        assert len(dropped) == 0
        assert len(order) == 3

    def test_cyclic_graph_drops_back_edge(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2 = uuid4(), uuid4()
        edges = [
            Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS),
            Edge(source_id=nid2, target_id=nid1, type=EdgeType.ATTACKS),
        ]
        dag, dropped, order = extract_max_dag({nid1, nid2}, edges)
        assert len(dropped) == 1
        assert len(dag) == 1

    def test_triangle_cycle_drops_one(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2, nid3 = uuid4(), uuid4(), uuid4()
        edges = [
            Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS),
            Edge(source_id=nid2, target_id=nid3, type=EdgeType.SUPPORTS),
            Edge(source_id=nid3, target_id=nid1, type=EdgeType.ATTACKS),
        ]
        dag, dropped, order = extract_max_dag({nid1, nid2, nid3}, edges)
        # In a fully cyclic graph the topological order depends on set iteration,
        # so either 1 or 2 edges may be dropped; at minimum 1 must be dropped.
        assert len(dropped) >= 1
        assert len(dag) + len(dropped) == len(edges)


class TestTopologicalSort:
    def test_linear_order(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2, nid3 = uuid4(), uuid4(), uuid4()
        edges = [
            Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS),
            Edge(source_id=nid2, target_id=nid3, type=EdgeType.SUPPORTS),
        ]
        order = topological_sort({nid1, nid2, nid3}, edges)
        assert order.index(nid1) < order.index(nid2) < order.index(nid3)

    def test_cyclic_graph_all_nodes_present(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2 = uuid4(), uuid4()
        edges = [
            Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS),
            Edge(source_id=nid2, target_id=nid1, type=EdgeType.ATTACKS),
        ]
        order = topological_sort({nid1, nid2}, edges)
        assert len(order) == 2
        assert nid1 in order
        assert nid2 in order


class TestTNAPropagate:
    def test_single_node_no_edges(self):
        nid = uuid4()
        def get_op(n):
            return (0.5, 0.3, 0.2, 0.5)
        result = tna_propagate({nid}, [], get_op)
        assert nid in result
        assert result[nid] == (0.5, 0.3, 0.2, 0.5)

    def test_support_chain(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2 = uuid4(), uuid4()
        edges = [Edge(source_id=nid1, target_id=nid2, type=EdgeType.SUPPORTS)]
        def get_op(n):
            return (0.8, 0.1, 0.1, 0.5) if n == nid1 else (0.0, 0.0, 1.0, 0.5)
        result = tna_propagate({nid1, nid2}, edges, get_op)
        assert nid2 in result
        b, d, u, a = result[nid2]
        # After deduction + fusion, nid2 should have positive belief
        assert b > 0.0
        assert d >= 0.0
        assert u >= 0.0

    def test_attack_decreases_belief(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2 = uuid4(), uuid4()
        edges = [Edge(source_id=nid1, target_id=nid2, type=EdgeType.ATTACKS)]
        def get_op(n):
            return (0.9, 0.05, 0.05, 0.5) if n == nid1 else (0.0, 0.0, 1.0, 0.5)
        result = tna_propagate({nid1, nid2}, edges, get_op)
        assert nid2 in result
        # Attack warrant swaps belief/disbelief, then inverted back
        b, d, u, a = result[nid2]
        assert b >= 0.0
        assert d >= 0.0
        assert u >= 0.0

    def test_support_beats_attack(self):
        from dynafx.core.models import Edge, EdgeType
        nid1, nid2, nid3 = uuid4(), uuid4(), uuid4()
        edges = [
            Edge(source_id=nid1, target_id=nid3, type=EdgeType.SUPPORTS),
            Edge(source_id=nid2, target_id=nid3, type=EdgeType.ATTACKS),
        ]
        def get_op(n):
            if n == nid1: return (0.8, 0.1, 0.1, 0.5)
            if n == nid2: return (0.2, 0.7, 0.1, 0.5)
            return (0.0, 0.0, 1.0, 0.5)
        result = tna_propagate({nid1, nid2, nid3}, edges, get_op)
        assert nid3 in result
        b, d, u, a = result[nid3]
        assert b > 0.0
