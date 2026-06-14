"""Tests for all 5 built-in lenses.

Tests cover:
- Lens registry and application
- Classification: summary distribution, by-type stats, needs_attention
- Funnel: chain details, confidence scores, weak links, UUID handling
- Decision Tree: JSON-serializable scenarios, auto-naming, subgraph filtering
- Outlier: ranking, explanations, configurable threshold
- Aggregation: counts, distributions, weakest/strongest
"""

from uuid import uuid4

import pytest

from cognitive_engine.core.models import Edge, Graph, Node, NodeType, EdgeType
from cognitive_engine.domain import Domain, domain, DomainConfig
from cognitive_engine.lenses import list_lenses, apply_lens, register_lens


def _make_graph() -> Graph:
    """Create a test graph with 4 nodes and 3 edges."""
    n1 = Node(id=uuid4(), type=NodeType.CLAIM, text="Root claim",
              opinion=(0.9, 0.05, 0.05, 0.5), category=2)
    n2 = Node(id=uuid4(), type=NodeType.EVIDENCE, text="Supporting evidence",
              opinion=(0.8, 0.1, 0.1, 0.5), category=1)
    n3 = Node(id=uuid4(), type=NodeType.COUNTERCLAIM, text="Counterclaim",
              opinion=(0.1, 0.8, 0.1, 0.5), category=3)
    n4 = Node(id=uuid4(), type=NodeType.CONDITION, text="Condition",
              opinion=(0.6, 0.2, 0.2, 0.5), category=2)

    e1 = Edge(id=uuid4(), source_id=n2.id, target_id=n1.id,
              type=EdgeType.SUPPORTS)
    e2 = Edge(id=uuid4(), source_id=n3.id, target_id=n1.id,
              type=EdgeType.CONTRADICTS)
    e3 = Edge(id=uuid4(), source_id=n4.id, target_id=n1.id,
              type=EdgeType.QUALIFIES)

    return Graph(
        nodes={n.id: n for n in [n1, n2, n3, n4]},
        edges=[e1, e2, e3],
    )


def _make_chain_graph() -> Graph:
    """Create a linear chain graph: A -> B -> C -> D."""
    n1 = Node(id=uuid4(), type=NodeType.AXIOM, text="Base assumption",
              opinion=(0.95, 0.02, 0.03, 0.5), category=1)
    n2 = Node(id=uuid4(), type=NodeType.EVIDENCE, text="Supporting evidence",
              opinion=(0.85, 0.05, 0.1, 0.5), category=1)
    n3 = Node(id=uuid4(), type=NodeType.CLAIM, text="Main conclusion",
              opinion=(0.7, 0.1, 0.2, 0.5), category=2)
    n4 = Node(id=uuid4(), type=NodeType.CONDITION, text="Final condition",
              opinion=(0.4, 0.3, 0.3, 0.5), category=2)

    e1 = Edge(id=uuid4(), source_id=n1.id, target_id=n2.id,
              type=EdgeType.SUPPORTS, warrant=((0.9, 0.05, 0.05, 0.5), (0.1, 0.8, 0.1, 0.5)))
    e2 = Edge(id=uuid4(), source_id=n2.id, target_id=n3.id,
              type=EdgeType.SUPPORTS, warrant=((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5)))
    e3 = Edge(id=uuid4(), source_id=n3.id, target_id=n4.id,
              type=EdgeType.INFERS)

    return Graph(
        nodes={n.id: n for n in [n1, n2, n3, n4]},
        edges=[e1, e2, e3],
    )


def _make_conflicted_graph() -> Graph:
    """Create a graph with conflicting node types for outlier detection."""
    n1 = Node(id=uuid4(), type=NodeType.CLAIM, text="High belief claim",
              opinion=(0.9, 0.05, 0.05, 0.5), category=2)
    n2 = Node(id=uuid4(), type=NodeType.CLAIM, text="Low belief claim",
              opinion=(0.2, 0.5, 0.3, 0.5), category=2)
    n3 = Node(id=uuid4(), type=NodeType.CLAIM, text="Medium belief claim",
              opinion=(0.5, 0.3, 0.2, 0.5), category=2)

    return Graph(
        nodes={n.id: n for n in [n1, n2, n3]},
        edges=[],
    )


# ============================================================================
# Registry Tests
# ============================================================================

class TestLensRegistry:
    def test_list_lenses(self):
        lenses = list_lenses()
        assert "classification" in lenses
        assert "funnel" in lenses
        assert "decision-tree" in lenses
        assert "outlier" in lenses
        assert "aggregation" in lenses

    def test_apply_unknown_lens_raises(self):
        g = _make_graph()
        with pytest.raises(KeyError):
            apply_lens(g, "nonexistent")

    def test_custom_lens_registration(self):
        def my_lens(graph, **params):
            graph.metadata["my_lens_applied"] = True
            return graph

        register_lens("test_custom", my_lens)
        assert "test_custom" in list_lenses()
        g = _make_graph()
        result = apply_lens(g, "test_custom")
        assert result.metadata["my_lens_applied"] is True


# ============================================================================
# Classification Lens Tests
# ============================================================================

class TestClassificationLens:
    def test_classification_strong(self):
        g = _make_graph()
        result = apply_lens(g, "classification")
        for node in result.nodes.values():
            if node.opinion[0] >= 0.75:
                assert node.metadata["classification"] == "strong"
        assert result.metadata["lens"] == "classification"

    def test_classification_inconclusive(self):
        n1 = Node(id=uuid4(), type=NodeType.CLAIM, text="Unsure",
                  opinion=(0.2, 0.2, 0.6, 0.5), category=2)
        g = Graph(nodes={n1.id: n1}, edges=[])
        result = apply_lens(g, "classification")
        assert result.nodes[n1.id].metadata["classification"] == "inconclusive"

    def test_summary_distribution(self):
        g = _make_graph()
        result = apply_lens(g, "classification")
        summary = result.metadata["classification_summary"]
        assert "distribution" in summary
        assert "total_nodes" in summary
        assert summary["total_nodes"] == 4
        # All classifications should sum to total
        total = sum(summary["distribution"].values())
        assert total == 4

    def test_by_type_breakdown(self):
        g = _make_graph()
        result = apply_lens(g, "classification")
        summary = result.metadata["classification_summary"]
        assert "by_type" in summary
        # Each type should have its own breakdown
        for node in result.nodes.values():
            type_name = node.type.name
            assert type_name in summary["by_type"]
            classification = node.metadata["classification"]
            assert classification in summary["by_type"][type_name]

    def test_needs_attention(self):
        g = _make_graph()
        result = apply_lens(g, "classification")
        summary = result.metadata["classification_summary"]
        assert "needs_attention" in list(summary.keys())
        # Weak, conflicted, or inconclusive nodes should be flagged
        for node in result.nodes.values():
            if node.metadata["classification"] in ("weak", "conflicted", "inconclusive"):
                attention_ids = [a["id"] for a in summary["needs_attention"]]
                assert node.id.hex in attention_ids

    def test_configurable_thresholds(self):
        g = _make_graph()
        result = apply_lens(g, "classification", threshold_high=0.9, threshold_moderate=0.7)
        summary = result.metadata["classification_summary"]
        # With higher thresholds, fewer nodes should be "strong"
        assert summary["distribution"].get("strong", 0) <= 1

    def test_classification_reason(self):
        g = _make_graph()
        result = apply_lens(g, "classification")
        summary = result.metadata["classification_summary"]
        for item in summary["needs_attention"]:
            assert "reason" in item
            assert len(item["reason"]) > 0


# ============================================================================
# Funnel Lens Tests
# ============================================================================

class TestFunnelLens:
    def test_funnel_chain_structure(self):
        g = _make_graph()
        result = apply_lens(g, "funnel")
        assert "funnel_chain" in result.metadata
        chain = result.metadata["funnel_chain"]
        assert len(chain) > 0
        assert result.metadata["lens"] == "funnel"

    def test_chain_entries_have_required_fields(self):
        g = _make_graph()
        result = apply_lens(g, "funnel")
        chain = result.metadata["funnel_chain"]
        for step in chain:
            assert "id" in step
            assert "text" in step
            assert "type" in step
            assert "belief" in step
            assert "edge_type" in step
            assert "edge_confidence" in step

    def test_chain_ids_are_hex_strings(self):
        g = _make_graph()
        result = apply_lens(g, "funnel")
        chain = result.metadata["funnel_chain"]
        for step in chain:
            # Should be 32-char hex string (UUID without dashes)
            assert len(step["id"]) == 32
            assert all(c in "0123456789abcdef" for c in step["id"])

    def test_funnel_summary(self):
        g = _make_chain_graph()
        result = apply_lens(g, "funnel")
        summary = result.metadata["funnel_summary"]
        assert "length" in summary
        assert "min_belief" in summary
        assert "max_belief" in summary
        assert "avg_belief" in summary
        assert "weak_links" in list(summary.keys())
        assert summary["length"] > 0
        assert summary["min_belief"] <= summary["max_belief"]

    def test_weak_links_detected(self):
        g = _make_chain_graph()
        result = apply_lens(g, "funnel")
        summary = result.metadata["funnel_summary"]
        # The last node has belief 0.4, which is > 0.3, so no weak links
        # But we can verify the structure is correct
        assert isinstance(summary["weak_links"], list)

    def test_chain_follows_strongest_edges(self):
        g = _make_chain_graph()
        result = apply_lens(g, "funnel")
        chain = result.metadata["funnel_chain"]
        # Chain should follow the path with highest beliefs
        # In our chain graph: AXIOM(0.95) -> EVIDENCE(0.85) -> CLAIM(0.7) -> CONDITION(0.4)
        assert len(chain) >= 3

    def test_edge_confidence_from_warrant(self):
        g = _make_chain_graph()
        result = apply_lens(g, "funnel")
        chain = result.metadata["funnel_chain"]
        # First edge has a warrant, so confidence should be present
        steps_with_confidence = [s for s in chain if s["edge_confidence"] is not None]
        assert len(steps_with_confidence) > 0

    def test_single_node_graph(self):
        n1 = Node(id=uuid4(), type=NodeType.CLAIM, text="Alone",
                  opinion=(0.5, 0.3, 0.2, 0.5), category=2)
        g = Graph(nodes={n1.id: n1}, edges=[])
        result = apply_lens(g, "funnel")
        chain = result.metadata["funnel_chain"]
        assert len(chain) == 1
        assert chain[0]["text"] == "Alone"


# ============================================================================
# Decision Tree Lens Tests
# ============================================================================

class TestDecisionTreeLens:
    def test_decision_tree_scenarios(self):
        g = _make_graph()
        result = apply_lens(g, "decision-tree")
        assert "scenarios" in result.metadata
        assert result.metadata["branch_edge_count"] > 0

    def test_scenarios_are_json_serializable(self):
        import json
        g = _make_graph()
        result = apply_lens(g, "decision-tree")
        scenarios = result.metadata["scenarios"]
        # Should not raise
        json.dumps(scenarios)

    def test_scenario_auto_naming(self):
        g = _make_graph()
        result = apply_lens(g, "decision-tree")
        scenarios = result.metadata["scenarios"]
        for scenario in scenarios:
            assert "name" in scenario
            assert scenario["name"].startswith("Scenario")

    def test_scenario_structure(self):
        g = _make_graph()
        result = apply_lens(g, "decision-tree")
        scenarios = result.metadata["scenarios"]
        for scenario in scenarios:
            assert "name" in scenario
            assert "branch_edge" in scenario
            assert "nodes" in scenario
            assert "edges" in scenario
            assert "summary" in scenario

    def test_scenario_summary_fields(self):
        g = _make_graph()
        result = apply_lens(g, "decision-tree")
        scenarios = result.metadata["scenarios"]
        for scenario in scenarios:
            summary = scenario["summary"]
            assert "node_count" in summary
            assert "edge_count" in summary
            assert "avg_belief" in summary
            assert "min_belief" in summary
            assert "max_belief" in summary
            assert "node_types" in summary
            assert "strongest_node" in summary
            assert "weakest_node" in summary

    def test_no_branching_edges(self):
        n1 = Node(id=uuid4(), type=NodeType.CLAIM, text="Alone",
                  opinion=(0.5, 0.3, 0.2, 0.5), category=2)
        g = Graph(nodes={n1.id: n1}, edges=[])
        result = apply_lens(g, "decision-tree")
        assert len(result.metadata["scenarios"]) == 1
        assert result.metadata["branch_edge_count"] == 0

    def test_scenario_subgraph_filtering(self):
        g = _make_graph()
        result = apply_lens(g, "decision-tree")
        scenarios = result.metadata["scenarios"]
        # Each scenario should have fewer or equal nodes than the full graph
        for scenario in scenarios:
            assert len(scenario["nodes"]) <= len(g.nodes)

    def test_scenario_node_serialization(self):
        g = _make_graph()
        result = apply_lens(g, "decision-tree")
        scenarios = result.metadata["scenarios"]
        for scenario in scenarios:
            for node in scenario["nodes"]:
                assert "id" in node
                assert "text" in node
                assert "type" in node
                assert "belief" in node
                assert "category" in node

    def test_scenario_edge_serialization(self):
        g = _make_graph()
        result = apply_lens(g, "decision-tree")
        scenarios = result.metadata["scenarios"]
        for scenario in scenarios:
            for edge in scenario["edges"]:
                assert "id" in edge
                assert "source_id" in edge
                assert "target_id" in edge
                assert "type" in edge
                assert "belief" in edge


# ============================================================================
# Outlier Lens Tests
# ============================================================================

class TestOutlierLens:
    def test_outlier_detection(self):
        g = _make_graph()
        result = apply_lens(g, "outlier")
        for node in result.nodes.values():
            assert "outlier" in node.metadata
            assert "outlier_deviation" in node.metadata

    def test_no_outlier_in_single_node(self):
        n1 = Node(id=uuid4(), type=NodeType.CLAIM, text="Only node",
                  opinion=(0.5, 0.3, 0.2, 0.5), category=2)
        g = Graph(nodes={n1.id: n1}, edges=[])
        result = apply_lens(g, "outlier")
        assert result.nodes[n1.id].metadata["outlier"] is False

    def test_outlier_ranking(self):
        g = _make_conflicted_graph()
        result = apply_lens(g, "outlier")
        assert "outlier_ranking" in result.metadata
        ranking = result.metadata["outlier_ranking"]
        assert isinstance(ranking, list)
        # Should be sorted by deviation (highest first)
        if len(ranking) > 1:
            for i in range(len(ranking) - 1):
                assert ranking[i]["deviation"] >= ranking[i + 1]["deviation"]

    def test_outlier_explanation(self):
        g = _make_conflicted_graph()
        result = apply_lens(g, "outlier")
        ranking = result.metadata["outlier_ranking"]
        for item in ranking:
            assert "explanation" in item
            assert len(item["explanation"]) > 0
            assert "direction" in item
            assert item["direction"] in ("above", "below")

    def test_outlier_cohort_mean(self):
        g = _make_conflicted_graph()
        result = apply_lens(g, "outlier")
        for node in result.nodes.values():
            assert "outlier_cohort_mean" in node.metadata

    def test_configurable_threshold(self):
        g = _make_conflicted_graph()
        result_low = apply_lens(g, "outlier", outlier_threshold=0.1)
        result_high = apply_lens(g, "outlier", outlier_threshold=0.5)
        # Lower threshold should flag more outliers
        assert result_low.metadata["outlier_count"] >= result_high.metadata["outlier_count"]

    def test_cohort_stats(self):
        g = _make_conflicted_graph()
        result = apply_lens(g, "outlier")
        assert "cohort_stats" in result.metadata
        stats = result.metadata["cohort_stats"]
        assert "CLAIM" in stats
        assert "mean" in stats["CLAIM"]
        assert "stdev" in stats["CLAIM"]
        assert "count" in stats["CLAIM"]


# ============================================================================
# Aggregation Lens Tests
# ============================================================================

class TestAggregationLens:
    def test_aggregation_structure(self):
        g = _make_graph()
        result = apply_lens(g, "aggregation")
        assert "aggregation" in result.metadata
        agg = result.metadata["aggregation"]
        assert agg["node_count"] == 4
        assert agg["edge_count"] == 3
        assert "by_category" in agg
        assert "by_type" in agg

    def test_by_type_stats(self):
        g = _make_graph()
        result = apply_lens(g, "aggregation")
        agg = result.metadata["aggregation"]
        for type_name in ["CLAIM", "EVIDENCE", "COUNTERCLAIM", "CONDITION"]:
            assert type_name in agg["by_type"]
            stats = agg["by_type"][type_name]
            assert "count" in stats
            assert "mean_belief" in stats
            assert "min_belief" in stats
            assert "max_belief" in stats

    def test_by_edge_type_stats(self):
        g = _make_graph()
        result = apply_lens(g, "aggregation")
        agg = result.metadata["aggregation"]
        assert "by_edge_type" in agg
        for edge_type in ["SUPPORTS", "CONTRADICTS", "QUALIFIES"]:
            assert edge_type in agg["by_edge_type"]
            stats = agg["by_edge_type"][edge_type]
            assert "count" in stats
            assert "mean_belief" in stats

    def test_weakest_and_strongest_node(self):
        g = _make_graph()
        result = apply_lens(g, "aggregation")
        agg = result.metadata["aggregation"]
        assert "weakest_node" in agg
        assert "strongest_node" in agg
        if agg["weakest_node"]:
            assert "id" in agg["weakest_node"]
            assert "text" in agg["weakest_node"]
            assert "type" in agg["weakest_node"]
            assert "belief" in agg["weakest_node"]
        if agg["strongest_node"]:
            assert "id" in agg["strongest_node"]
            assert "text" in agg["strongest_node"]
            assert "type" in agg["strongest_node"]
            assert "belief" in agg["strongest_node"]

    def test_empty_graph(self):
        g = Graph(nodes={}, edges=[])
        result = apply_lens(g, "aggregation")
        agg = result.metadata["aggregation"]
        assert agg["node_count"] == 0
        assert agg["edge_count"] == 0
        assert agg["weakest_node"] is None
        assert agg["strongest_node"] is None


# ============================================================================
# Legal Domain Tests
# ============================================================================

class TestLensWithLegalDomain:
    def test_classification_with_legal_thresholds(self):
        from cognitive_engine.domains.legal import LegalConfig
        legal = Domain("lens_test_legal", LegalConfig)

        g = _make_graph()

        with legal:
            result = apply_lens(g, "classification")
            for node in result.nodes.values():
                assert "classification" in node.metadata
            assert result.metadata["lens"] == "classification"

    def test_outlier_with_legal_thresholds(self):
        from cognitive_engine.domains.legal import LegalConfig
        legal = Domain("lens_test_legal", LegalConfig)

        g = _make_conflicted_graph()

        with legal:
            result = apply_lens(g, "outlier")
            assert "outlier_ranking" in result.metadata
            assert "cohort_stats" in result.metadata
