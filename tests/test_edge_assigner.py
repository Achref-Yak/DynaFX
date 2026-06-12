from uuid import UUID, uuid4

from cognitive_engine.chunker import PropSpan
from cognitive_engine.edge_assigner import assign_edges
from cognitive_engine.models import EdgeType, Node, NodeType
from cognitive_engine.type_mapper import Relation


def _span(start: int, end: int, text: str = "") -> PropSpan:
    return PropSpan(start_char=start, end_char=end, text=text)


def _key(text: str, start: int, end: int) -> tuple:
    return (start, end)


def _typed(text: str, start: int, end: int, type_: NodeType) -> tuple:
    return (_span(start, end, text), type_)


class TestSupportRelations:
    def test_evidence_supports_claim(self):
        spans = [_typed("Traffic is high.", 0, 16, NodeType.EVIDENCE),
                 _typed("System is overloaded.", 17, 38, NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {_key("Traffic is high.", 0, 16): uuid4(), _key("System is overloaded.", 17, 38): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.SUPPORTS
        assert edges[0].source_id == nmap[(0, 16)]

    def test_axiom_infers_claim(self):
        spans = [_typed("Must handle 10k.", 0, 16, NodeType.AXIOM),
                 _typed("System scales.", 17, 31, NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {_key("Must handle 10k.", 0, 16): uuid4(), _key("System scales.", 17, 31): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.INFERS

    def test_claim_justifies_evidence(self):
        spans = [_typed("I reason that.", 0, 14, NodeType.CLAIM),
                 _typed("Observed data shows.", 15, 35, NodeType.EVIDENCE)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {_key("I reason that.", 0, 14): uuid4(), _key("Observed data shows.", 15, 35): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.JUSTIFIES

    def test_condition_qualifies_claim(self):
        spans = [_typed("If traffic spikes.", 0, 18, NodeType.CONDITION),
                 (_span(19, 30, "Scale out."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {_key("If traffic spikes.", 0, 18): uuid4(), (19, 30): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.QUALIFIES

    def test_justification_justifies_claim(self):
        spans = [_typed("Because it works.", 0, 17, NodeType.JUSTIFICATION),
                 (_span(18, 33, "System is good."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {_key("Because it works.", 0, 17): uuid4(), (18, 33): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.JUSTIFIES


class TestAttackRelations:
    def test_evidence_attacks_claim(self):
        spans = [_typed("Contradicting data.", 0, 18, NodeType.EVIDENCE),
                 (_span(19, 34, "Claim is false."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Attack")]
        nmap = {_key("Contradicting data.", 0, 18): uuid4(), (19, 34): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.ATTACKS

    def test_claim_contradicts_claim(self):
        spans = [_typed("A is true.", 0, 10, NodeType.CLAIM),
                 (_span(11, 22, "A is false."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Attack")]
        nmap = {_key("A is true.", 0, 10): uuid4(), (11, 22): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.CONTRADICTS

    def test_counterclaim_rebuts_claim(self):
        spans = [_typed("However, not true.", 0, 18, NodeType.COUNTERCLAIM),
                 (_span(19, 32, "Claim is X."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Attack")]
        nmap = {_key("However, not true.", 0, 18): uuid4(), (19, 32): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.REBUTS

    def test_axiom_contradicts_claim(self):
        spans = [_typed("Must never fail.", 0, 16, NodeType.AXIOM),
                 (_span(17, 31, "Failure is ok."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Attack")]
        nmap = {_key("Must never fail.", 0, 16): uuid4(), (17, 31): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.CONTRADICTS

    def test_fallacy_attacks_claim(self):
        spans = [_typed("Invalid reasoning.", 0, 18, NodeType.FALLACY),
                 (_span(19, 35, "Claim is valid."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Attack")]
        nmap = {_key("Invalid reasoning.", 0, 18): uuid4(), (19, 35): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.ATTACKS


class TestNoneRelations:
    def test_support_from_claim_to_condition_is_none(self):
        spans = [_typed("Scale out.", 0, 10, NodeType.CLAIM),
                 (_span(11, 35, "If traffic exceeds 10k."), NodeType.CONDITION)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {_key("Scale out.", 0, 10): uuid4(), (11, 35): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 0

    def test_attack_claim_to_counterclaim_is_rebuts(self):
        spans = [_typed("Claim is X.", 0, 12, NodeType.CLAIM),
                 (_span(13, 34, "However, claim is Y."), NodeType.COUNTERCLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Attack")]
        nmap = {_key("Claim is X.", 0, 12): uuid4(), (13, 34): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.REBUTS


class TestEdgeDeduplication:
    def test_duplicate_pair_not_added_twice(self):
        spans = [_typed("Traffic is high.", 0, 16, NodeType.EVIDENCE),
                 (_span(17, 38, "System overloaded."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support"),
                Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {_key("Traffic is high.", 0, 16): uuid4(), (17, 38): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1


class TestSelfEdge:
    def test_self_relation_not_added(self):
        span = _span(0, 10, "Same span.")
        spans = [(span, NodeType.CLAIM)]
        rels = [Relation(source_span=span, target_span=span, label="Support")]
        nmap = {(0, 10): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 0


class TestUnmatchedSpan:
    def test_span_not_in_nmap_dropped(self):
        spans = [_typed("Ev 1.", 0, 5, NodeType.EVIDENCE),
                 (_span(6, 18, "Claim here."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {_key("Ev 1.", 0, 5): uuid4()}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 0


class TestFallbackDefaults:
    def test_unknown_support_falls_back_to_supports(self):
        uid_a, uid_b = uuid4(), uuid4()
        spans = [_typed("Some fallacy.", 0, 12, NodeType.FALLACY),
                 (_span(13, 35, "Some condition here."), NodeType.CONDITION)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {(0, 12): uid_a, (13, 35): uid_b}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.SUPPORTS

    def test_unknown_attack_falls_back_to_contradicts(self):
        uid_a, uid_b = uuid4(), uuid4()
        spans = [_typed("Justification attack.", 0, 20, NodeType.JUSTIFICATION),
                 (_span(21, 40, "Counterclaim text."), NodeType.COUNTERCLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Attack")]
        nmap = {(0, 20): uid_a, (21, 40): uid_b}
        edges = assign_edges(spans, rels, nmap, {})
        assert len(edges) == 1
        assert edges[0].type == EdgeType.CONTRADICTS


class TestDemarcationRefinement:
    def test_institutional_source_with_epistemic_target_qualifies(self):
        uid_a, uid_b = uuid4(), uuid4()
        src_node = Node(
            id=uid_a,
            type=NodeType.EVIDENCE,
            metadata={"demarcation": {"epistemic_vs_institutional": "INSTITUTIONAL"}},
        )
        tgt_node = Node(
            id=uid_b,
            type=NodeType.CLAIM,
            metadata={"demarcation": {"epistemic_vs_institutional": "EPISTEMIC"}},
        )
        spans = [_typed("Must handle 10k.", 0, 16, NodeType.EVIDENCE),
                 (_span(17, 31, "System scales."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {(0, 16): uid_a, (17, 31): uid_b}
        existing = {uid_a: src_node, uid_b: tgt_node}
        edges = assign_edges(spans, rels, nmap, existing)
        assert len(edges) == 1
        assert edges[0].type == EdgeType.QUALIFIES

    def test_no_demarcation_no_refinement(self):
        uid_a, uid_b = uuid4(), uuid4()
        spans = [_typed("Traffic is high.", 0, 16, NodeType.EVIDENCE),
                 (_span(17, 38, "System overloaded."), NodeType.CLAIM)]
        rels = [Relation(source_span=spans[0][0], target_span=spans[1][0], label="Support")]
        nmap = {(0, 16): uid_a, (17, 38): uid_b}
        existing = {uid_a: Node(id=uid_a, type=NodeType.EVIDENCE),
                    uid_b: Node(id=uid_b, type=NodeType.CLAIM)}
        edges = assign_edges(spans, rels, nmap, existing)
        assert len(edges) == 1
        assert edges[0].type == EdgeType.SUPPORTS
