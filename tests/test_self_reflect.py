"""Tests for the self-reflection operator."""

import pytest
import uuid
from cognitive_engine.core.models import Graph, Node, NodeType, EdgeType, Edge
from cognitive_engine.core.state import State
from cognitive_engine.kernel.self_reflect import (
    SelfReflectOperator, SelfReflectionConfig, ReflectionResult,
)


def _make_state_with_beliefs(beliefs: list[float] = None) -> State:
    """Create a test state with nodes having specific beliefs."""
    g = Graph(source_text="test")
    if beliefs is None:
        beliefs = [0.8, 0.5, 0.2, 0.0]
    for i, b in enumerate(beliefs):
        nid = uuid.uuid4()
        g.nodes[nid] = Node(
            id=nid,
            text=f"node_{i}",
            type=NodeType.CLAIM,
            opinion=(b, max(0.0, 1.0 - b - 0.1), 0.1, 0.5),
        )
    return State(graph=g)


class TestSelfReflectionConfig:
    def test_defaults(self):
        cfg = SelfReflectionConfig()
        assert cfg.frequency == 5
        assert cfg.enabled is True
        assert cfg.ph_enabled is False
        assert cfg.min_belief_threshold == 0.0

    def test_custom(self):
        cfg = SelfReflectionConfig(frequency=10, enabled=False)
        assert cfg.frequency == 10
        assert cfg.enabled is False


class TestSelfReflectOperator:
    def test_should_reflect_disabled(self):
        op = SelfReflectOperator(SelfReflectionConfig(enabled=False))
        assert op.should_reflect(5) is False

    def test_should_reflect_frequency(self):
        op = SelfReflectOperator(SelfReflectionConfig(frequency=3))
        assert op.should_reflect(3) is True
        assert op.should_reflect(6) is True
        assert op.should_reflect(4) is False

    def test_reflect_returns_result(self):
        op = SelfReflectOperator(SelfReflectionConfig(frequency=1))
        state = _make_state_with_beliefs([0.8, 0.5, 0.2, 0.0])
        result = op.reflect(state, cycle=1)
        assert isinstance(result, ReflectionResult)
        assert result.cycle == 1
        assert "high" in result.tier_counts
        assert "medium" in result.tier_counts
        assert "low" in result.tier_counts
        assert "uninitialized" in result.tier_counts

    def test_reflect_tier_counts(self):
        op = SelfReflectOperator(SelfReflectionConfig(frequency=1))
        state = _make_state_with_beliefs([0.9, 0.5, 0.1, 0.0])
        result = op.reflect(state, cycle=1)
        assert result.tier_counts["high"] >= 1  # 0.9
        assert result.tier_counts["medium"] >= 1  # 0.5
        assert result.tier_counts["low"] >= 1  # 0.1
        assert result.tier_counts["uninitialized"] >= 1  # 0.0

    def test_reflect_history(self):
        op = SelfReflectOperator(SelfReflectionConfig(frequency=1))
        state = _make_state_with_beliefs()
        op.reflect(state, cycle=1)
        op.reflect(state, cycle=2)
        assert len(op.history) == 2

    def test_reflect_recommendations(self):
        op = SelfReflectOperator(SelfReflectionConfig(frequency=1))
        # Graph with many uninitialized nodes
        state = _make_state_with_beliefs([0.0, 0.0, 0.0, 0.0, 0.0])
        result = op.reflect(state, cycle=1)
        assert len(result.recommendations) > 0

    def test_page_hinkley(self):
        cfg = SelfReflectionConfig(frequency=1, ph_enabled=True, ph_threshold=30.0)
        op = SelfReflectOperator(cfg)
        state = _make_state_with_beliefs()
        # Run multiple reflections to trigger PH
        for i in range(20):
            op.reflect(state, cycle=i + 1)
        # PH should update (value may be negative if mean is constant)
        assert op._ph_count == 20

    def test_reflect_with_conflict_edges(self):
        op = SelfReflectOperator(SelfReflectionConfig(frequency=1))
        g = Graph(source_text="test")
        n1 = uuid.uuid4()
        n2 = uuid.uuid4()
        n3 = uuid.uuid4()
        g.nodes[n1] = Node(id=n1, text="source", type=NodeType.AGENT, opinion=(0.8, 0.1, 0.1, 0.5))
        g.nodes[n2] = Node(id=n2, text="target", type=NodeType.CLAIM, opinion=(0.5, 0.3, 0.2, 0.5))
        g.nodes[n3] = Node(id=n3, text="attacker", type=NodeType.CLAIM, opinion=(0.3, 0.5, 0.2, 0.5))
        g.edges[uuid.uuid4()] = Edge(source_id=n1, target_id=n2, type=EdgeType.SUPPORTS, weight=1.0)
        g.edges[uuid.uuid4()] = Edge(source_id=n3, target_id=n2, type=EdgeType.ATTACKS, weight=1.0)
        state = State(graph=g)
        result = op.reflect(state, cycle=1)
        assert isinstance(result, ReflectionResult)


class TestTierClassification:
    def test_high_tier(self):
        assert SelfReflectOperator._classify_tier(0.8) == "high"
        assert SelfReflectOperator._classify_tier(0.7) == "high"

    def test_medium_tier(self):
        assert SelfReflectOperator._classify_tier(0.5) == "medium"
        assert SelfReflectOperator._classify_tier(0.3) == "medium"

    def test_low_tier(self):
        assert SelfReflectOperator._classify_tier(0.2) == "low"
        assert SelfReflectOperator._classify_tier(0.01) == "low"

    def test_uninitialized_tier(self):
        assert SelfReflectOperator._classify_tier(0.0) == "uninitialized"
