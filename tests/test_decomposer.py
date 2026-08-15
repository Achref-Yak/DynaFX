"""Tests for SystemDecomposer (core/decomposer.py)."""

from dynafx.core.decomposer import SystemDecomposer
from dynafx.core.models import EmergentProperty


def _make_simple() -> SystemDecomposer:
    d = SystemDecomposer(name="Test")
    d.add_node("a", type="ENTITY")
    d.add_node("b", type="ENTITY")
    d.add_node("c", type="ENTITY")
    return d


def test_acyclic_graph_no_emergence():
    d = _make_simple()
    d.add_node("x", type="ENTITY")
    d.add_edge("a", "b")
    d.add_edge("b", "x")
    assert d.detect() == []


def test_single_feedback_loop_detected():
    d = _make_simple()
    d.add_edge("a", "b", "CAUSES", polarity=1)
    d.add_edge("b", "a", "CAUSES", polarity=-1)
    eps = d.detect()
    assert len(eps) == 1
    ep = eps[0]
    assert isinstance(ep, EmergentProperty)
    assert "a" in ep.condition.split(" ")[0]
    assert "b" in ep.condition
    assert "balancing" in ep.condition
    assert len(ep.involved_ids) == 2


def test_reinforcing_loop_sign():
    d = _make_simple()
    d.add_edge("a", "b", "CAUSES", polarity=1)
    d.add_edge("b", "c", "CAUSES", polarity=1)
    d.add_edge("c", "a", "CAUSES", polarity=1)
    eps = d.detect()
    assert len(eps) == 1
    assert "reinforcing" in eps[0].condition


def test_multiple_distinct_loops():
    d = _make_simple()
    d.add_node("d", type="ENTITY")
    d.add_edge("a", "b", "CAUSES")
    d.add_edge("b", "a", "CAUSES")
    d.add_edge("b", "c", "CAUSES")
    d.add_edge("c", "a", "CAUSES")
    eps = d.detect()
    assert len(eps) >= 2


def test_detect_does_not_self_link():
    d = _make_simple()
    d.add_edge("a", "b", "CAUSES")
    assert d.detect() == []


def test_assigned_partition_preserved():
    d = _make_simple()
    d.assign_partition("a", "technical")
    assert d.graph.nodes[next(iter(d.graph.nodes))].orthogonal_partition == "technical"
