from uuid import uuid4

from cognitive_engine.core.models import Graph, Node, NodeType, Edge, EdgeType, Severity
from cognitive_engine.reason.validators import validate_all, product_logic_check, level_mapping_check


def _make_node(text: str, type_: NodeType = NodeType.CLAIM, category: int = 2) -> Node:
    return Node(type=type_, text=text, category=category)


def _make_edge(
    source: Node, target: Node, type_: EdgeType = EdgeType.SUPPORTS
) -> Edge:
    return Edge(source_id=source.id, target_id=target.id, type=type_)


class TestProductLogic:
    def test_valid_same_category(self):
        a = _make_node("fact a", category=2)
        b = _make_node("fact b", category=2)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[_make_edge(a, b)])
        assert len(product_logic_check(g)) == 0

    def test_valid_higher_to_lower_category(self):
        a = _make_node("necessity", category=1)
        b = _make_node("fact", category=2)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[_make_edge(a, b)])
        assert len(product_logic_check(g)) == 0

    def test_invalid_concept_to_fact(self):
        a = _make_node("concept", category=4)
        b = _make_node("fact", category=2)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[_make_edge(a, b)])
        violations = product_logic_check(g)
        assert len(violations) == 1
        assert violations[0].type == "CATEGORY_ERROR"
        assert violations[0].severity == Severity.ERROR

    def test_invalid_belief_to_necessity(self):
        a = _make_node("opinion", category=3)
        b = _make_node("law", category=1)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[_make_edge(a, b)])
        violations = product_logic_check(g)
        assert len(violations) == 1

    def test_multiple_edges_one_bad(self):
        a = _make_node("concept", category=4)
        b = _make_node("fact", category=2)
        c = _make_node("fact", category=2)
        e1 = _make_edge(a, b)
        e2 = _make_edge(c, b)
        g = Graph(nodes={a.id: a, b.id: b, c.id: c}, edges=[e1, e2])
        violations = product_logic_check(g)
        assert len(violations) == 1

    def test_missing_node_reference(self):
        a = _make_node("exists")
        orphan = Edge(source_id=a.id, target_id=uuid4())
        g = Graph(nodes={a.id: a}, edges=[orphan])
        violations = product_logic_check(g)
        assert any(v.type == "MISSING_NODE" for v in violations)


class TestLevelMapping:
    def test_acyclic_chain_passes(self):
        a = _make_node("a")
        b = _make_node("b")
        c = _make_node("c")
        g = Graph(
            nodes={a.id: a, b.id: b, c.id: c},
            edges=[_make_edge(a, b), _make_edge(b, c)],
        )
        assert len(level_mapping_check(g)) == 0

    def test_simple_cycle_detected(self):
        a = _make_node("a")
        b = _make_node("b")
        c = _make_node("c")
        g = Graph(
            nodes={a.id: a, b.id: b, c.id: c},
            edges=[
                _make_edge(a, b),
                _make_edge(b, c),
                _make_edge(c, a),
            ],
        )
        violations = level_mapping_check(g)
        assert len(violations) >= 1
        assert any(v.type == "CYCLE_DETECTED" for v in violations)

    def test_self_loop_detected(self):
        a = _make_node("a")
        e = Edge(source_id=a.id, target_id=a.id)
        g = Graph(nodes={a.id: a}, edges=[e])
        violations = level_mapping_check(g)
        assert any(v.type == "CYCLE_DETECTED" for v in violations)

    def test_empty_graph_no_violations(self):
        assert len(level_mapping_check(Graph())) == 0


class TestValidateAll:
    def test_valid_graph_no_violations(self):
        a = _make_node("evidence", type_=NodeType.EVIDENCE, category=1)
        b = _make_node("claim", type_=NodeType.CLAIM, category=2)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[_make_edge(a, b)])
        assert len(validate_all(g)) == 0

    def test_graph_with_multiple_issues(self):
        a = _make_node("concept", category=4)
        b = _make_node("fact", category=2)
        c = _make_node("fact", category=2)
        g = Graph(
            nodes={a.id: a, b.id: b, c.id: c},
            edges=[
                _make_edge(a, b),
                _make_edge(c, a),
                _make_edge(b, c),
            ],
        )
        violations = validate_all(g)
        cat_errors = [v for v in violations if v.type == "CATEGORY_ERROR"]
        cycle_errors = [v for v in violations if v.type == "CYCLE_DETECTED"]
        assert len(cat_errors) >= 1
        assert len(cycle_errors) >= 1
