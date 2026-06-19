"""Tests for policy/engine.py."""

from uuid import uuid4

import pytest

from cognitive_engine.core.models import Graph, Node, NodeType, Opinion, Edge, EdgeType
from cognitive_engine.core.state import State
from cognitive_engine.policy.engine import PolicyEngine, PolicySelection
from cognitive_engine.policy.schema import (
    OperatorPolicy, PolicyRule, WhenCondition, ThenAction,
)
from cognitive_engine.policy.builtin import (
    DEFAULT_POLICY, SCIENTIFIC_POLICY, BUILTIN_POLICIES,
)


class TestPolicySelection:
    def test_defaults(self):
        s = PolicySelection()
        assert s.operators == []
        assert s.order == "sequential"
        assert s.policy_name == "default"
        assert s.rule_index == -1

    def test_with_operators(self):
        s = PolicySelection(operators=["propagate", "verify"], policy_name="custom", rule_index=0)
        assert s.operators == ["propagate", "verify"]
        assert s.rule_index == 0


class TestPolicyEngine:
    def _make_state(self, node_count=0, contradictions=0, uncertainty=0.5):
        graph = Graph()
        for i in range(node_count):
            nid = uuid4()
            graph.nodes[nid] = Node(text=f"node_{i}", type=NodeType.CLAIM,
                                     opinion=Opinion(0.5, 0.3, uncertainty, 0.5))
        for i in range(contradictions):
            nid1 = uuid4()
            nid2 = uuid4()
            graph.nodes[nid1] = Node(text=f"c_src_{i}", type=NodeType.CLAIM)
            graph.nodes[nid2] = Node(text=f"c_tgt_{i}", type=NodeType.CLAIM)
            eid = uuid4()
            graph.edges[eid] = Edge(
                id=eid, source_id=nid1, target_id=nid2, type=EdgeType.CONTRADICTS,
            )
        state = State(graph=graph)
        return state

    def test_init_default_policy(self):
        engine = PolicyEngine()
        assert engine.policy.name == "default"

    def test_init_custom_policy(self):
        policy = OperatorPolicy(name="test")
        engine = PolicyEngine(policy=policy)
        assert engine.policy.name == "test"

    def test_policy_setter(self):
        engine = PolicyEngine()
        new_policy = OperatorPolicy(name="custom")
        engine.policy = new_policy
        assert engine.policy.name == "custom"

    def test_select_empty_state(self):
        engine = PolicyEngine()
        state = self._make_state()
        selection = engine.select(state)
        assert selection.policy_name == "default"
        assert isinstance(selection.operators, list)

    def test_select_cycle_one(self):
        engine = PolicyEngine()
        state = self._make_state()
        selection = engine.select(state, cycle=1)
        assert selection.policy_name == "default"
        assert "extract" in selection.operators or selection.rule_index >= 0

    def test_select_with_contradictions(self):
        engine = PolicyEngine()
        state = self._make_state(contradictions=3)
        selection = engine.select(state, cycle=5)
        assert "constraint" in selection.operators or "debate" in selection.operators or "verify" in selection.operators

    def test_select_high_uncertainty(self):
        engine = PolicyEngine()
        state = self._make_state(node_count=5, uncertainty=0.6)
        selection = engine.select(state, cycle=2)
        assert selection.rule_index >= 0

    def test_select_scientific_policy(self):
        engine = PolicyEngine(policy=SCIENTIFIC_POLICY)
        state = self._make_state()
        selection = engine.select(state, cycle=1, domain="scientific")
        assert selection.policy_name == "scientific"

    def test_fallback(self):
        policy = OperatorPolicy(
            name="strict",
            rules=[
                PolicyRule(
                    when=WhenCondition(graph_node_count=">100"),
                    then=ThenAction(operators=["propagate"]),
                ),
            ],
            fallback=ThenAction(operators=["verify"], order="sequential"),
        )
        engine = PolicyEngine(policy=policy)
        state = self._make_state(node_count=3)
        selection = engine.select(state)
        assert selection.rule_index == -1
        assert selection.reason == "Fallback rule"
        assert selection.operators == ["verify"]

    def test_load_yaml(self):
        yaml_content = """
name: custom
description: Test policy
rules:
  - when:
      cycle: "==1"
    then:
      operators: [extract, schema]
      order: sequential
fallback:
  operators: [propagate]
  order: sequential
"""
        engine = PolicyEngine()
        policy = engine.load_yaml(yaml_content)
        assert policy.name == "custom"
        assert len(policy.rules) == 1
        assert policy.rules[0].when.cycle == "==1"
        assert policy.rules[0].then.operators == ["extract", "schema"]
        assert policy.fallback.operators == ["propagate"]

    def test_load_yaml_empty_fallback(self):
        yaml_content = """
name: minimal
rules: []
fallback:
  operators: [verify]
"""
        engine = PolicyEngine()
        policy = engine.load_yaml(yaml_content)
        assert policy.name == "minimal"
        assert len(policy.rules) == 0
        assert policy.fallback.operators == ["verify"]

    def test_extract_metrics(self):
        engine = PolicyEngine()
        state = self._make_state(node_count=3, contradictions=1, uncertainty=0.4)
        metrics = engine._extract_metrics(state, 2, "general")
        # node_count=3 → 3 nodes, contradictions=1 → 2 more nodes = 5 total
        assert metrics["graph_node_count"] == 5
        assert metrics["graph_has_contradictions"] is True
        assert metrics["cycle"] == 2
        assert metrics["domain"] == "general"

    def test_format_condition(self):
        engine = PolicyEngine()
        when = WhenCondition(cycle="==1", domain="general")
        formatted = engine._format_condition(when)
        assert "cycle===1" in formatted
        assert "domain=general" in formatted


class TestBuiltinPolicies:
    def test_default_policy_structure(self):
        assert DEFAULT_POLICY.name == "default"
        assert len(DEFAULT_POLICY.rules) >= 1
        assert DEFAULT_POLICY.fallback.operators == ["propagate", "verify"]

    def test_scientific_policy_structure(self):
        assert SCIENTIFIC_POLICY.name == "scientific"
        assert len(SCIENTIFIC_POLICY.rules) >= 1

    def test_builtin_policies_registry(self):
        assert "default" in BUILTIN_POLICIES
        assert "scientific" in BUILTIN_POLICIES
