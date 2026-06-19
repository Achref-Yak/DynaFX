"""Tests for Systems Thinking Operators."""

from __future__ import annotations

import pytest
from cognitive_engine.core.state import State
from cognitive_engine.core.models import Graph, Node, NodeType, Edge, EdgeType
from cognitive_engine.operators.systems import (
    FeedbackLoopDetector,
    LeveragePointScorer,
    SystemArchetypeClassifier,
    CausalSCM,
)


def _make_causal_graph() -> Graph:
    """Create a causal graph with feedback loops for testing."""
    graph = Graph()

    nodes = {
        "rain": Node(id="rain", type=NodeType.CLAIM, text="It rains"),
        "wet": Node(id="wet", type=NodeType.CLAIM, text="Ground is wet"),
        "plants": Node(id="plants", type=NodeType.CLAIM, text="Plants grow"),
        "transpiration": Node(id="transpiration", type=NodeType.CLAIM, text="Transpiration increases"),
        "clouds": Node(id="clouds", type=NodeType.CLAIM, text="Clouds form"),
    }
    graph.nodes = nodes

    edges = [
        Edge(source_id="rain", target_id="wet", type=EdgeType.CAUSES),
        Edge(source_id="wet", target_id="plants", type=EdgeType.CAUSES),
        Edge(source_id="plants", target_id="transpiration", type=EdgeType.CAUSES),
        Edge(source_id="transpiration", target_id="clouds", type=EdgeType.CAUSES),
        Edge(source_id="clouds", target_id="rain", type=EdgeType.CAUSES),
    ]
    graph.edges = {e.id: e for e in edges}

    return graph


def _make_fixes_that_fail_graph() -> Graph:
    """Create a graph with Fixes that Fail pattern (CAUSES + ATTACKS loop)."""
    graph = Graph()

    nodes = {
        "problem": Node(id="problem", type=NodeType.CLAIM, text="Problem exists"),
        "fix": Node(id="fix", type=NodeType.CLAIM, text="Apply quick fix"),
        "side_effect": Node(id="side_effect", type=NodeType.CLAIM, text="Side effect"),
    }
    graph.nodes = nodes

    edges = [
        Edge(source_id="problem", target_id="fix", type=EdgeType.CAUSES),
        Edge(source_id="fix", target_id="side_effect", type=EdgeType.CAUSES),
        Edge(source_id="side_effect", target_id="problem", type=EdgeType.ATTACKS),
    ]
    graph.edges = {e.id: e for e in edges}

    return graph


class TestFeedbackLoopDetector:
    """Tests for FeedbackLoopDetector."""

    def test_detects_feedback_loops(self):
        """Should detect feedback loops in causal graph."""
        graph = _make_causal_graph()
        op = FeedbackLoopDetector()
        state = State(graph=graph)
        result = op(state)
        assert result.metadata["feedback_loops"]["total_loops"] > 0

    def test_classifies_reinforcing(self):
        """Should classify all-positive loops as reinforcing."""
        graph = _make_causal_graph()
        op = FeedbackLoopDetector()
        state = State(graph=graph)
        result = op(state)
        loops = result.metadata["feedback_loops"]["loops"]
        assert any(l["loop_type"] == "reinforcing" for l in loops)

    def test_empty_graph(self):
        """Empty graph should have no loops."""
        op = FeedbackLoopDetector()
        state = State(graph=Graph())
        result = op(state)
        assert result.metadata["feedback_loops"]["total_loops"] == 0


class TestLeveragePointScorer:
    """Tests for LeveragePointScorer."""

    def test_identifies_leverage_points(self):
        """Should identify high-degree nodes as leverage points."""
        graph = _make_causal_graph()
        op = LeveragePointScorer()
        state = State(graph=graph)
        result = op(state)
        assert result.metadata["leverage_points"]["total_points"] > 0

    def test_scores_by_degree(self):
        """Higher degree nodes should have higher scores."""
        graph = _make_causal_graph()
        op = LeveragePointScorer()
        state = State(graph=graph)
        result = op(state)
        points = result.metadata["leverage_points"]["points"]
        if len(points) > 1:
            assert points[0]["score"] >= points[-1]["score"]

    def test_empty_graph(self):
        """Empty graph should have no leverage points."""
        op = LeveragePointScorer()
        state = State(graph=Graph())
        result = op(state)
        assert result.metadata["leverage_points"]["total_points"] == 0


class TestSystemArchetypeClassifier:
    """Tests for SystemArchetypeClassifier."""

    def test_classifies_archetypes(self):
        """Should classify archetypes from graph structure."""
        graph = _make_causal_graph()
        op = SystemArchetypeClassifier()
        state = State(graph=graph)
        result = op(state)
        assert result.metadata["system_archetypes"]["total_archetypes"] > 0

    def test_detects_fixes_that_fail(self):
        """Graph with CAUSES+PREVENTS loop should be classified as Fixes that Fail."""
        graph = _make_fixes_that_fail_graph()
        op = SystemArchetypeClassifier()
        state = State(graph=graph)
        result = op(state)
        archetypes = result.metadata["system_archetypes"]["archetypes"]
        assert any(a["name"] == "Fixes that Fail" for a in archetypes)

    def test_empty_graph(self):
        """Empty graph should have no archetypes."""
        op = SystemArchetypeClassifier()
        state = State(graph=Graph())
        result = op(state)
        assert result.metadata["system_archetypes"]["total_archetypes"] == 0


class TestCausalSCM:
    """Tests for CausalSCM."""

    def test_estimates_causal_effects(self):
        """Should estimate causal effects from edges."""
        graph = _make_causal_graph()
        op = CausalSCM()
        state = State(graph=graph)
        result = op(state)
        assert "causal_effects" in result.metadata["causal_scm"]

    def test_do_intervention(self):
        """Should simulate do-operator intervention."""
        graph = _make_causal_graph()
        op = CausalSCM()
        state = State(graph=graph)
        result = op(state, intervention={"rain": 1.0})
        assert "intervention" in result.metadata["causal_scm"]

    def test_counterfactual(self):
        """Should reason about counterfactuals."""
        graph = _make_causal_graph()
        op = CausalSCM()
        state = State(graph=graph)
        result = op(state, counterfactual={"rain": 0.0})
        assert "counterfactual" in result.metadata["causal_scm"]

    def test_finds_backdoor_paths(self):
        """Should find backdoor paths (confounders)."""
        graph = _make_causal_graph()
        op = CausalSCM()
        state = State(graph=graph)
        result = op(state)
        assert "backdoor_paths" in result.metadata["causal_scm"]
