import pytest
from cognitive_engine.nlp.chunker import PropSpan
from cognitive_engine.core.models import NodeType
from cognitive_engine.extract.types import assign_type, map_types, Relation, _char_span_relaxed

pytest.importorskip("spacy")
from cognitive_engine.nlp.preprocessor import load_spacy_pipeline

nlp = load_spacy_pipeline()


def _make_doc(text: str, chunk_start: int = 0) -> "spacy.tokens.Doc":
    doc = nlp(text)
    doc.user_data["chunk_start_char"] = chunk_start
    doc.user_data["chunk_end_char"] = chunk_start + len(text)
    return doc


def _make_span(start: int, end: int, text: str) -> PropSpan:
    return PropSpan(start_char=start, end_char=end, text=text, chunk_offsets=[0])


class TestAssignType:
    def test_axiom_with_modal(self):
        text = "The system must handle 10k requests."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.AXIOM

    def test_axiom_with_shall(self):
        text = "The system shall process all data."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.AXIOM

    def test_condition_with_if(self):
        text = "If load exceeds 10k"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.CONDITION

    def test_condition_with_unless(self):
        text = "Unless we scale"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.CONDITION

    def test_condition_requires_verb(self):
        text = "If it is okay with you"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.CONDITION

    def test_justification_because(self):
        text = "because the query timed out"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.JUSTIFICATION

    def test_justification_since(self):
        text = "since traffic increased"
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.JUSTIFICATION

    def test_fallacy_by_keyword(self):
        text = "That argument is misleading."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.FALLACY

    def test_fallacy_flawed(self):
        text = "This reasoning is flawed."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.FALLACY

    def test_counterclaim_adversative_however(self):
        text = "However, the latency increased."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.COUNTERCLAIM

    def test_counterclaim_adversative_but(self):
        text = "The system works but latency is high."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.COUNTERCLAIM

    def test_counterclaim_requires_lexical_marker(self):
        doc = _make_doc("Some claim.")
        span = _make_span(0, 10, "Some claim")
        attack_source = _make_span(20, 30, "attack text")
        rel = Relation(source_span=attack_source, target_span=span, label="Attack")
        assert assign_type(span, doc, relations=[rel]) == NodeType.CLAIM

    def test_claim_root_proposition(self):
        text = "The system provides a robust framework."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) == NodeType.CLAIM

    def test_evidence_non_root(self):
        text = "The system scales because traffic is high."
        doc = _make_doc(text)
        claim_span = _make_span(0, 17, "The system scales")
        assert assign_type(claim_span, doc) == NodeType.CLAIM

        ev_span = _make_span(18, len(text), "because traffic is high.")
        ev_type = assign_type(ev_span, doc)
        assert ev_type in (NodeType.EVIDENCE, NodeType.JUSTIFICATION)

    def test_no_modal_no_axiom(self):
        text = "The system handles requests."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        assert assign_type(span, doc) != NodeType.AXIOM

    def test_empty_span_defaults_to_evidence(self):
        doc = _make_doc("")
        span = _make_span(0, 0, "")
        assert assign_type(span, doc) == NodeType.EVIDENCE

    def test_char_span_out_of_bounds(self):
        doc = _make_doc("Short text.")
        span = _make_span(100, 200, "out of bounds")
        assert assign_type(span, doc) == NodeType.EVIDENCE


class TestMapTypes:
    def test_map_types_basic(self):
        text = "The system must handle 10k requests."
        doc = _make_doc(text)
        span = _make_span(0, len(text), text)
        results = map_types([span], [doc])
        assert len(results) == 1
        assert results[0][1] == NodeType.AXIOM

    def test_map_types_multiple_spans(self):
        doc = _make_doc("If load exceeds 10k, scale out.")
        if_span = _make_span(0, 21, "If load exceeds 10k")
        scale_span = _make_span(23, 33, "scale out")
        results = map_types([if_span, scale_span], [doc])
        types = {r[1] for r in results}
        assert NodeType.CONDITION in types

    def test_map_types_no_matching_doc(self):
        doc = _make_doc("Some text.")
        span = _make_span(999, 1005, "nowhere")
        results = map_types([span], [doc])
        assert results[0][1] == NodeType.EVIDENCE

    def test_map_types_empty_spans(self):
        doc = _make_doc("Some text.")
        results = map_types([], [doc])
        assert results == []

    def test_map_types_with_relations(self):
        text_a = "First claim."
        text_b = "But it fails."
        doc_a = _make_doc(text_a, chunk_start=0)
        doc_b = _make_doc(text_b, chunk_start=20)
        span_a = _make_span(0, 12, text_a)
        span_b = _make_span(20, 33, text_b)
        rel = Relation(source_span=span_b, target_span=span_a, label="Attack")
        results = map_types([span_a, span_b], [doc_a, doc_b], relations=[rel])
        types_by_span = {id(s): t for s, t in results}
        assert types_by_span[id(span_a)] == NodeType.CLAIM
