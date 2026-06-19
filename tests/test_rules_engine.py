"""Tests for the rule engine."""

import pytest
from cognitive_engine.rules.engine import RuleEngine, Pattern, Action, Rule
from cognitive_engine.rules.parser import parse_rules
from cognitive_engine.core.models import Graph, Node, NodeType, EdgeType, Edge, Opinion
from cognitive_engine.core.state import State


def _make_state_with_nodes(node_texts: list[str] = None) -> State:
    """Create a test state with nodes."""
    import uuid
    g = Graph(source_text="test")
    if node_texts is None:
        node_texts = ["claim_a", "claim_b", "evidence_c"]
    for text in node_texts:
        nid = uuid.uuid4()
        g.nodes[nid] = Node(
            id=nid,
            text=text,
            type=NodeType.CLAIM,
            opinion=Opinion.from_tuple((0.6, 0.2, 0.2, 0.5)),
        )
    return State(graph=g)


class TestPattern:
    def test_pattern_creation(self):
        p = Pattern(source_type="CLAIM", min_belief=0.5)
        assert p.source_type == "CLAIM"
        assert p.min_belief == 0.5

    def test_pattern_defaults(self):
        p = Pattern()
        assert p.source_type is None
        assert p.min_belief is None
        assert p.edge_type is None
        assert p.negated is False


class TestAction:
    def test_action_creation(self):
        a = Action(source_var="?a", target_var="?b", edge_type="SUPPORTS", confidence=0.8)
        assert a.source_var == "?a"
        assert a.target_var == "?b"
        assert a.edge_type == "SUPPORTS"
        assert a.confidence == 0.8

    def test_action_defaults(self):
        a = Action(source_var="?a", target_var="?b", edge_type="SUPPORTS")
        assert a.weight == 0.5
        assert a.confidence == 0.5


class TestRule:
    def test_rule_creation(self):
        r = Rule(
            name="test_rule",
            when=[Pattern(source_type="CLAIM")],
            then=[Action(source_var="?a", target_var="?b", edge_type="SUPPORTS")],
        )
        assert r.name == "test_rule"
        assert len(r.when) == 1
        assert len(r.then) == 1
        assert r.confidence == 0.5
        assert r.enabled is True

    def test_rule_disabled(self):
        r = Rule(name="off", enabled=False)
        assert r.enabled is False


class TestRuleEngine:
    def test_empty_engine(self):
        engine = RuleEngine()
        state = _make_state_with_nodes()
        results = engine.evaluate(state.graph)
        assert results == []  # no rules, no results

    def test_engine_no_match(self):
        rule = Rule(
            name="no_match",
            when=[Pattern(source_type="AGENT")],  # no AGENT nodes
            then=[Action(source_var="?a", target_var="?b", edge_type="SUPPORTS")],
        )
        engine = RuleEngine()
        engine.add_rule(rule)
        state = _make_state_with_nodes()
        results = engine.evaluate(state.graph)
        assert results == []

    def test_engine_with_match(self):
        import uuid
        g = Graph(source_text="test")
        n1 = uuid.uuid4()
        n2 = uuid.uuid4()
        g.nodes[n1] = Node(id=n1, text="a", type=NodeType.CLAIM, opinion=Opinion.from_tuple((0.8, 0.1, 0.1, 0.5)))
        g.nodes[n2] = Node(id=n2, text="b", type=NodeType.CLAIM, opinion=Opinion.from_tuple((0.6, 0.2, 0.2, 0.5)))
        g.edges[uuid.uuid4()] = Edge(
            source_id=n1, target_id=n2, type=EdgeType.SUPPORTS, weight=1.0,
        )
        rule = Rule(
            name="transitivity",
            when=[
                Pattern(source_var="?a", edge_type="SUPPORTS", target_var="?b"),
            ],
            then=[Action(source_var="?a", target_var="?b", edge_type="SUPPORTS")],
        )
        engine = RuleEngine()
        engine.add_rule(rule)
        results = engine.evaluate(g)
        assert len(results) == 1
        action, bindings, conf = results[0]
        assert action.edge_type == "SUPPORTS"
        assert "?a" in bindings
        assert "?b" in bindings

    def test_engine_negation(self):
        """Test negation-as-failure: pattern with negated=True matches when NOT present."""
        rule = Rule(
            name="no_evidence_rule",
            when=[Pattern(source_type="EVIDENCE", negated=True)],
            then=[Action(source_var="?a", target_var="?b", edge_type="SUPPORTS")],
        )
        engine = RuleEngine()
        engine.add_rule(rule)
        # State with no EVIDENCE nodes — negated pattern should match
        state = _make_state_with_nodes(["claim_a", "claim_b"])
        results = engine.evaluate(state.graph)
        assert len(results) == 1  # negation succeeds → action fires

    def test_engine_negation_fails_when_present(self):
        import uuid
        g = Graph(source_text="test")
        n1 = uuid.uuid4()
        n2 = uuid.uuid4()
        g.nodes[n1] = Node(id=n1, text="ev", type=NodeType.EVIDENCE, opinion=(0.6, 0.2, 0.2, 0.5))
        g.nodes[n2] = Node(id=n2, text="cl", type=NodeType.CLAIM, opinion=(0.6, 0.2, 0.2, 0.5))
        g.edges[uuid.uuid4()] = Edge(source_id=n1, target_id=n2, type=EdgeType.SUPPORTS, weight=1.0)
        rule = Rule(
            name="no_evidence_rule",
            when=[Pattern(source_type="EVIDENCE", edge_type="SUPPORTS", target_type="CLAIM", negated=True)],
            then=[Action(source_var="?a", target_var="?b", edge_type="SUPPORTS")],
        )
        engine = RuleEngine()
        engine.add_rule(rule)
        results = engine.evaluate(g)
        assert results == []  # EVIDENCE→SUPPORTS→CLAIM exists → negation fails

    def test_engine_disabled_rule(self):
        rule = Rule(
            name="disabled",
            when=[Pattern(source_type="CLAIM")],
            then=[Action(source_var="?a", target_var="?b", edge_type="SUPPORTS")],
            enabled=False,
        )
        engine = RuleEngine()
        engine.add_rule(rule)
        state = _make_state_with_nodes()
        results = engine.evaluate(state.graph)
        assert results == []

    def test_engine_belief_filter(self):
        import uuid
        g = Graph(source_text="test")
        n1 = uuid.uuid4()
        n2 = uuid.uuid4()
        g.nodes[n1] = Node(id=n1, text="high", type=NodeType.CLAIM, opinion=Opinion.from_tuple((0.9, 0.05, 0.05, 0.5)))
        g.nodes[n2] = Node(id=n2, text="low", type=NodeType.CLAIM, opinion=Opinion.from_tuple((0.1, 0.05, 0.85, 0.5)))
        g.edges[uuid.uuid4()] = Edge(source_id=n1, target_id=n2, type=EdgeType.SUPPORTS, weight=1.0)
        rule = Rule(
            name="high_belief_only",
            when=[Pattern(source_type="CLAIM", min_belief=0.7)],
            then=[Action(source_var="?a", target_var="?b", edge_type="SUPPORTS")],
        )
        engine = RuleEngine()
        engine.add_rule(rule)
        results = engine.evaluate(g)
        # Only n1 matches min_belief=0.7
        assert len(results) == 1


class TestParser:
    def test_parse_empty(self):
        rules = parse_rules({"rules": []})
        assert rules == []

    def test_parse_dict(self):
        data = {
            "rules": [
                {
                    "name": "test_rule",
                    "when": [
                        {"source_type": "CLAIM", "min_belief": 0.5},
                    ],
                    "then": [
                        {"source": "?a", "target": "?b", "edge": "SUPPORTS", "weight": 0.8},
                    ],
                    "confidence": 0.7,
                }
            ]
        }
        rules = parse_rules(data)
        assert len(rules) == 1
        assert rules[0].name == "test_rule"
        assert rules[0].when[0].source_type == "CLAIM"
        assert rules[0].then[0].edge_type == "SUPPORTS"
        assert rules[0].confidence == 0.7

    def test_parse_with_negation(self):
        data = {
            "rules": [
                {
                    "name": "negated_rule",
                    "when": [
                        {"source_type": "EVIDENCE", "negated": True},
                    ],
                    "then": [
                        {"source": "?a", "target": "?b", "edge": "SUPPORTS"},
                    ],
                }
            ]
        }
        rules = parse_rules(data)
        assert rules[0].when[0].negated is True

    def test_parse_transitivity(self):
        data = {
            "rules": [
                {
                    "name": "transitivity",
                    "when": [
                        {"source": "?a", "edge": "SUPPORTS", "target": "?b"},
                        {"source": "?b", "edge": "SUPPORTS", "target": "?c"},
                    ],
                    "then": [
                        {"source": "?a", "target": "?c", "edge": "SUPPORTS", "weight": 0.7},
                    ],
                    "confidence": 0.8,
                }
            ]
        }
        rules = parse_rules(data)
        assert len(rules) == 1
        assert len(rules[0].when) == 2
        assert len(rules[0].then) == 1
