"""Tests for kernel/assertion_gate.py."""

from uuid import uuid4

import pytest

from cognitive_engine.kernel.assertion_gate import (
    Assertion, AssertionGate, GateResult,
)
from cognitive_engine.core.models import Node, Edge, NodeType, EdgeType


class TestAssertion:
    def test_create_assertion(self):
        a = Assertion(source="test", text="hello", node_type="CLAIM")
        assert a.source == "test"
        assert a.text == "hello"
        assert a.node_type == "CLAIM"
        assert a.opinion is None
        assert a.id is not None

    def test_assertion_with_opinion(self):
        a = Assertion(
            source="test", text="hello", node_type="EVIDENCE",
            opinion=(0.8, 0.1, 0.1, 0.5),
        )
        assert a.opinion == (0.8, 0.1, 0.1, 0.5)

    def test_assertion_default_fields(self):
        a = Assertion()
        assert a.source == ""
        assert a.text == ""
        assert a.node_type is None


class TestGateResult:
    def test_empty_result(self):
        r = GateResult()
        assert r.passed == []
        assert r.quarantined == []
        assert r.errors == []

    def test_with_items(self):
        a1 = Assertion(source="a")
        a2 = Assertion(source="b")
        r = GateResult(passed=[a1], quarantined=[a2], errors=["err"])
        assert len(r.passed) == 1
        assert len(r.quarantined) == 1
        assert r.errors == ["err"]


class TestAssertionGate:
    def test_gate_default_epsilon(self):
        gate = AssertionGate()
        assert gate.clamp_epsilon == pytest.approx(1e-9)

    def test_gate_custom_epsilon(self):
        gate = AssertionGate(clamp_epsilon=0.01)
        assert gate.clamp_epsilon == 0.01

    def test_process_empty(self):
        gate = AssertionGate()
        result = gate.process([])
        assert len(result.passed) == 0
        assert len(result.quarantined) == 0

    def test_process_valid_assertion(self):
        gate = AssertionGate()
        a = Assertion(source="test", node_type="CLAIM", text="test claim")
        result = gate.process([a])
        assert len(result.passed) == 1
        assert len(result.quarantined) == 0
        assert len(result.errors) == 0
        assert result.passed[0].opinion is not None

    def test_process_invalid_type(self):
        gate = AssertionGate()
        a = Assertion(source="test", node_type="INVALID_TYPE", text="bad")
        result = gate.process([a])
        assert len(result.passed) == 0
        assert len(result.quarantined) == 1
        assert len(result.errors) == 1

    def test_process_quarantines_invalid_type(self):
        gate = AssertionGate()
        a = Assertion(
            source="test", node_type="__INVALID__", text="bad type",
        )
        result = gate.process([a])
        assert len(result.passed) == 0
        assert len(result.quarantined) == 1

    def test_process_mixed(self):
        gate = AssertionGate()
        good = Assertion(source="a", node_type="EVIDENCE", text="good")
        bad = Assertion(source="b", node_type="INVALID", text="bad")
        result = gate.process([good, bad])
        assert len(result.passed) == 1
        assert len(result.quarantined) == 1

    def test_assign_opinion_high_confidence(self):
        gate = AssertionGate()
        a = Assertion(source="test", node_type="CLAIM", text="high",
                      metadata={"confidence": 0.9})
        result = gate.process([a])
        b, d, u, a_rate = result.passed[0].opinion
        assert b == pytest.approx(0.8)
        assert d == pytest.approx(0.1)
        assert u == pytest.approx(0.1)

    def test_assign_opinion_medium_confidence(self):
        gate = AssertionGate()
        a = Assertion(source="test", node_type="CLAIM", text="medium",
                      metadata={"confidence": 0.6})
        result = gate.process([a])
        b, d, u, a_rate = result.passed[0].opinion
        assert b == pytest.approx(0.4)

    def test_assign_opinion_low_confidence(self):
        gate = AssertionGate()
        a = Assertion(source="test", node_type="CLAIM", text="low",
                      metadata={"confidence": 0.2})
        result = gate.process([a])
        b, d, u, a_rate = result.passed[0].opinion
        assert b == pytest.approx(0.5)

    def test_assign_opinion_zero_confidence(self):
        gate = AssertionGate()
        a = Assertion(source="test", node_type="CLAIM", text="zero",
                      metadata={"confidence": 0.0})
        result = gate.process([a])
        b, d, u, a_rate = result.passed[0].opinion
        assert u == pytest.approx(1.0)

    def test_to_node(self):
        gate = AssertionGate()
        a = Assertion(
            source="test", node_type="CLAIM", text="hello",
            opinion=(0.8, 0.1, 0.1, 0.5),
        )
        node = gate.to_node(a)
        assert isinstance(node, Node)
        assert node.text == "hello"
        assert node.type == NodeType.CLAIM
        assert node.opinion is not None
        assert node.opinion.belief == pytest.approx(0.8)

    def test_to_node_none_type(self):
        gate = AssertionGate()
        a = Assertion(source="test", node_type=None, text="no type")
        node = gate.to_node(a)
        assert node.type == NodeType.CLAIM

    def test_to_edge(self):
        gate = AssertionGate()
        src = uuid4()
        tgt = uuid4()
        edge = gate.to_edge(src, tgt, "SUPPORTS", (0.8, 0.1, 0.1, 0.5))
        assert isinstance(edge, Edge)
        assert edge.source_id == src
        assert edge.target_id == tgt
        assert edge.type == EdgeType.SUPPORTS
        assert edge.opinion.belief == pytest.approx(0.8)

    def test_to_edge_no_opinion(self):
        gate = AssertionGate()
        edge = gate.to_edge(uuid4(), uuid4(), "ATTACKS")
        assert edge.opinion.belief == 0.0
