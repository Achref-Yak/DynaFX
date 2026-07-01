from uuid import uuid4

from dynafx.core.models import Graph, Node, NodeType, Edge, EdgeType, Severity
from dynafx.epistemics.product_logic import (
    validate_categories,
    valuate,
    negation_cat,
    conjunction_cat,
    disjunction_cat,
    implication_valid,
    equivalence_valid,
    Category,
)


class TestProductLogicOperators:
    def test_negation_idempotent(self):
        for c in Category:
            assert negation_cat(c) == c

    def test_conjunction_takes_higher(self):
        assert conjunction_cat(1, 4) == 4
        assert conjunction_cat(3, 2) == 3
        assert conjunction_cat(2, 2) == 2

    def test_disjunction_takes_lower(self):
        assert disjunction_cat(1, 4) == 1
        assert disjunction_cat(3, 2) == 2
        assert disjunction_cat(2, 2) == 2

    def test_implication_valid(self):
        assert implication_valid(1, 2)
        assert implication_valid(2, 2)
        assert implication_valid(3, 4)
        assert not implication_valid(4, 2)
        assert not implication_valid(3, 1)

    def test_equivalence_valid(self):
        assert equivalence_valid(2, 2)
        assert equivalence_valid(4, 4)
        assert not equivalence_valid(1, 2)
        assert not equivalence_valid(3, 4)

    def test_valuate(self):
        assert valuate(2, True) == 2
        assert valuate(2, False) == 0
        assert valuate(4, True) == 4
        assert valuate(1, False) == 0


class TestValidateCategories:
    def test_valid_same_category(self):
        a = Node(text="fact a", category=2)
        b = Node(text="fact b", category=2)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[Edge(source_id=a.id, target_id=b.id)])
        assert len(validate_categories(g)) == 0

    def test_valid_higher_to_lower_category(self):
        a = Node(text="necessity", category=1)
        b = Node(text="fact", category=2)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[Edge(source_id=a.id, target_id=b.id)])
        assert len(validate_categories(g)) == 0

    def test_invalid_concept_to_fact(self):
        a = Node(text="concept", category=4)
        b = Node(text="fact", category=2)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[Edge(source_id=a.id, target_id=b.id)])
        violations = validate_categories(g)
        assert len(violations) == 1
        assert violations[0].type == "CATEGORY_ERROR"
        assert violations[0].severity == Severity.ERROR

    def test_invalid_belief_to_necessity(self):
        a = Node(text="opinion", category=3)
        b = Node(text="law", category=1)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[Edge(source_id=a.id, target_id=b.id)])
        violations = validate_categories(g)
        assert len(violations) == 1

    def test_new_node_types_accepted(self):
        for nt in (NodeType.COUNTERCLAIM, NodeType.AXIOM, NodeType.FALLACY, NodeType.JUSTIFICATION):
            a = Node(type=nt, text="test node", category=2)
            b = Node(type=NodeType.CLAIM, text="claim", category=2)
            g = Graph(nodes={a.id: a, b.id: b}, edges=[Edge(source_id=a.id, target_id=b.id)])
            assert len(validate_categories(g)) == 0

    def test_new_edge_types_accepted(self):
        for et in (EdgeType.ATTACKS, EdgeType.REBUTS):
            a = Node(text="source", category=2)
            b = Node(text="target", category=2)
            g = Graph(nodes={a.id: a, b.id: b}, edges=[Edge(source_id=a.id, target_id=b.id, type=et)])
            assert len(validate_categories(g)) == 0

    def test_concept_contradicts_warning(self):
        a = Node(text="abstract ideal", category=4)
        b = Node(text="pure concept", category=4)
        g = Graph(nodes={a.id: a, b.id: b}, edges=[Edge(source_id=a.id, target_id=b.id, type=EdgeType.CONTRADICTS)])
        violations = validate_categories(g)
        assert any(v.type == "CATEGORY_ERROR" and v.severity == Severity.WARNING for v in violations)

    def test_multiple_edges_one_bad(self):
        a = Node(text="concept", category=4)
        b = Node(text="fact", category=2)
        c = Node(text="fact", category=2)
        g = Graph(
            nodes={a.id: a, b.id: b, c.id: c},
            edges=[
                Edge(source_id=a.id, target_id=b.id),
                Edge(source_id=c.id, target_id=b.id),
            ],
        )
        violations = validate_categories(g)
        assert len(violations) == 1

    def test_missing_node_reference(self):
        a = Node(text="exists")
        orphan = Edge(source_id=a.id, target_id=uuid4())
        g = Graph(nodes={a.id: a}, edges=[orphan])
        violations = validate_categories(g)
        assert any(v.type == "MISSING_NODE" for v in violations)
