import pytest
from typing import Optional
from uuid import uuid4

pytest.importorskip("spacy")
from cognitive_engine.preprocessor import load_spacy_pipeline

from cognitive_engine.models import Graph, Node, NodeType, Span
from cognitive_engine.demarcation_rules import assign_demarcations

nlp = load_spacy_pipeline()


def _make_doc(text: str, chunk_start: int = 0):
    doc = nlp(text)
    doc.user_data["chunk_start_char"] = chunk_start
    doc.user_data["chunk_end_char"] = chunk_start + len(text)
    return doc


def _make_node(text: str, start: int = 0, end: Optional[int] = None, type_: NodeType = NodeType.CLAIM) -> Node:
    if end is None:
        end = start + len(text)
    return Node(
        id=uuid4(),
        type=type_,
        text=text,
        span=Span(start=start, end=end, text=text),
    )


class TestCognitiveVsEpistemic:
    def test_condition_is_cognitive(self):
        doc = _make_doc("If load exceeds 10k, scale out.")
        node = _make_node("If load exceeds 10k", type_=NodeType.CONDITION)
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["cognitive_vs_epistemic"] == "COGNITIVE"

    def test_evidence_is_epistemic(self):
        doc = _make_doc("Traffic is high.")
        node = _make_node("Traffic is high.", type_=NodeType.EVIDENCE)
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["cognitive_vs_epistemic"] == "EPISTEMIC"

    def test_claim_is_epistemic(self):
        doc = _make_doc("The system is reliable.")
        node = _make_node("The system is reliable.", type_=NodeType.CLAIM)
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["cognitive_vs_epistemic"] == "EPISTEMIC"

    def test_axiom_is_na(self):
        doc = _make_doc("The system must handle 10k.")
        node = _make_node("The system must handle 10k.", type_=NodeType.AXIOM)
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["cognitive_vs_epistemic"] == "NA"


class TestEpistemicVsInstitutional:
    def test_modal_on_action_verb_is_institutional(self):
        doc = _make_doc("The system must handle 10k requests.")
        node = _make_node("The system must handle 10k requests.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["epistemic_vs_institutional"] == "INSTITUTIONAL"

    def test_modal_on_stative_verb_is_epistemic(self):
        doc = _make_doc("He must be tired.")
        node = _make_node("He must be tired.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["epistemic_vs_institutional"] == "EPISTEMIC"

    def test_no_modal_is_na(self):
        doc = _make_doc("The system handles requests.")
        node = _make_node("The system handles requests.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["epistemic_vs_institutional"] == "NA"


class TestAffectVsCognition:
    def test_sentiment_adjective_is_affect(self):
        doc = _make_doc("An effective solution.")
        node = _make_node("An effective solution.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["affect_vs_cognition"] == "AFFECT"

    def test_no_sentiment_is_cognition(self):
        doc = _make_doc("The blue sky.")
        node = _make_node("The blue sky.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["affect_vs_cognition"] == "COGNITION"

    def test_no_adjective_is_cognition(self):
        doc = _make_doc("The system scales.")
        node = _make_node("The system scales.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["affect_vs_cognition"] == "COGNITION"


class TestConstraintVsEnablement:
    def test_enablement_verb(self):
        doc = _make_doc("The system can scale horizontally.")
        node = _make_node("The system can scale horizontally.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["constraint_vs_enablement"] == "ENABLEMENT"

    def test_constraint_verb(self):
        doc = _make_doc("The system cannot handle the load.")
        node = _make_node("The system cannot handle the load.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["constraint_vs_enablement"] == "CONSTRAINT"

    def test_negation_is_constraint(self):
        doc = _make_doc("The system does not scale.")
        node = _make_node("The system does not scale.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["constraint_vs_enablement"] == "CONSTRAINT"

    def test_no_constraint_or_enablement_is_na(self):
        doc = _make_doc("The system scales.")
        node = _make_node("The system scales.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["constraint_vs_enablement"] == "NA"


class TestSynchronicVsDiachronic:
    def test_present_tense_is_synchronic(self):
        doc = _make_doc("The system handles requests.")
        node = _make_node("The system handles requests.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["synchronic_vs_diachronic"] == "SYNCHRONIC"

    def test_past_tense_is_diachronic(self):
        doc = _make_doc("The system handled requests.")
        node = _make_node("The system handled requests.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["synchronic_vs_diachronic"] == "DIACHRONIC"

    def test_no_verb_is_na(self):
        doc = _make_doc("In summary.")
        node = _make_node("In summary.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        assert g.nodes[node.id].metadata["demarcation"]["synchronic_vs_diachronic"] == "NA"


class TestAssignDemarcations:
    def test_multiple_nodes_all_assigned(self):
        doc = _make_doc("The system must handle 10k requests. Traffic is high.")
        n1 = _make_node("The system must handle 10k requests.", type_=NodeType.AXIOM)
        n2 = _make_node("Traffic is high.", start=37, type_=NodeType.EVIDENCE)
        g = Graph(nodes={n1.id: n1, n2.id: n2})
        assign_demarcations(g, [doc])
        for nid in g.nodes:
            d = g.nodes[nid].metadata["demarcation"]
            assert all(k in d for k in [
                "cognitive_vs_epistemic", "epistemic_vs_institutional",
                "affect_vs_cognition", "constraint_vs_enablement",
                "synchronic_vs_diachronic",
            ])

    def test_node_without_span_defaults_to_na(self):
        node = Node(id=uuid4(), type=NodeType.CLAIM, text="Some text", span=None)
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [])
        d = g.nodes[node.id].metadata["demarcation"]
        assert all(v == "NA" for v in d.values())

    def test_no_matching_doc_uses_fallback(self):
        doc = _make_doc("Main text here.", chunk_start=100)
        node = _make_node("Not in doc xyz")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc])
        d = g.nodes[node.id].metadata["demarcation"]
        assert all(v == "NA" for v in d.values())

    def test_docs_matched_by_chunk_range(self):
        doc_a = _make_doc("First chunk.", chunk_start=0)
        doc_b = _make_doc("Second chunk.", chunk_start=20)
        node = _make_node("First chunk.")
        g = Graph(nodes={node.id: node})
        assign_demarcations(g, [doc_a, doc_b])
        assert g.nodes[node.id].metadata["demarcation"]["cognitive_vs_epistemic"] != "NA"
