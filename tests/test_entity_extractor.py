# tests/test_entity_extractor.py

import pytest
from uuid import uuid4

from cognitive_engine.core.models import (
    BfoCategory,
    Edge, EdgeType, Node, NodeType,
    ReasoningMode, Span, Opinion, Graph, Entity, WorldRelation
)
from cognitive_engine.core.state import State
from cognitive_engine.operators.extract import ExtractOperator
from cognitive_engine.extract.linker import resolve_entities
from cognitive_engine.extract.types import assign_bfo


pytest.importorskip("spacy")
from spacy.tokens import Doc


import spacy

def _make_doc(text: str, chunk_start: int = 0) -> Doc:
    # Mocking spaCy Doc and its user_data for testing purposes
    nlp = spacy.blank("en")
    doc = nlp(text)
    doc.user_data["chunk_start_char"] = chunk_start
    doc.user_data["chunk_end_char"] = chunk_start + len(text)
    return doc


def _make_span(start: int, end: int, text: str) -> Span:
    return Span(start=start, end=end, text=text)


class TestEntityExtractor:

    def test_basic_entity_extraction(self):
        operator = ExtractOperator()
        text = "Alice lives in London."
        state = State(graph=Graph(source_text=text, mode=ReasoningMode.ARGUMENT))
        state.graph.nodes = {}
        state.graph.entities = {}
        state.graph.relations = []
        state.graph.world_relations = []

        # Mocking spaCy Doc for the operator
        mock_doc = _make_doc(text)
        mock_spacy_docs = [mock_doc]

        # Manually populate graph.entities which would normally be done by extract_entities()
        # This mock simulates the output of extract_entities()
        entity_alice = Entity(
            id=uuid4(),
            kind="PERSON",
            name="Alice",
            spans=[_make_span(0, 5, "Alice")]
        )
        entity_london = Entity(
            id=uuid4(),
            kind="GPE",
            name="London",
            spans=[_make_span(11, 17, "London")]
        )
        state.graph.entities[entity_alice.id] = entity_alice
        state.graph.entities[entity_london.id] = entity_london

        # Mocking the result of extract_relations (no relations in this simple sentence)
        state.graph.world_relations = []

        # Run the operator to trigger the bridging logic
        operator(state, text=text, use_deposition_parser=False, use_heuristic_classifier=False)

        # Assertions for the bridged nodes
        assert len(state.graph.nodes) >= 2
        
        alice_node = next(n for n in state.graph.nodes.values() if n.text == "Alice")
        london_node = next(n for n in state.graph.nodes.values() if n.text == "London")

        assert alice_node.type == NodeType.ENTITY
        assert alice_node.metadata["entity_kind"] == "PERSON"
        assert alice_node.bfo_category == BfoCategory.QUALITY

        assert london_node.type == NodeType.ENTITY
        assert london_node.metadata["entity_kind"] == "GPE"
        assert london_node.bfo_category == BfoCategory.MATERIAL_ENTITY

    def test_attribute_extraction(self):
        operator = ExtractOperator(compute_embeddings=False)
        text = "Alice's email is alice@example.com"
        state = State(graph=Graph(source_text=text, mode=ReasoningMode.ARGUMENT))
        
        entity_alice = Entity(
            id=uuid4(),
            kind="PERSON",
            name="Alice",
            spans=[_make_span(0, 5, "Alice")]
        )
        state.graph.entities[entity_alice.id] = entity_alice
        
        operator(state, text=text, use_deposition_parser=False, use_heuristic_classifier=True)
        
        edges = list(state.graph.edges.values())
        contact_edges = [e for e in edges if e.type == EdgeType.CONTACT_OF and e.metadata.get("attribute") == "email"]
        assert len(contact_edges) == 1
        assert contact_edges[0].metadata["value"] == "alice@example.com"

    def test_world_relation_conversion(self):
        operator = ExtractOperator(compute_embeddings=False)
        text = "Alice caused the event."
        state = State(graph=Graph(source_text=text, mode=ReasoningMode.ARGUMENT))
        
        entity_alice = Entity(
            id=uuid4(),
            kind="PERSON",
            name="Alice",
            spans=[_make_span(0, 5, "Alice")]
        )
        entity_event = Entity(
            id=uuid4(),
            kind="EVENT",
            name="event",
            spans=[_make_span(17, 22, "event")]
        )
        state.graph.entities[entity_alice.id] = entity_alice
        state.graph.entities[entity_event.id] = entity_event
        
        wr = WorldRelation(
            id=uuid4(),
            source_id=entity_alice.id,
            target_id=entity_event.id,
            kind="CAUSES",
            metadata={"relation_phrase": "caused"}
        )
        state.graph.world_relations.append(wr)
        
        operator(state, text=text, use_deposition_parser=False, use_heuristic_classifier=True)
        
        edges = list(state.graph.edges.values())
        cause_edges = [e for e in edges if e.type == EdgeType.CAUSES]
        assert len(cause_edges) == 1
        assert cause_edges[0].metadata.get("relation_kind") == "CAUSES"

    def test_soft_wordlist_linking(self):
        from cognitive_engine.domain import Domain, DomainConfig
        # Unit test the resolve_entities linker directly
        graph = Graph()
        entity_db = Entity(
            id=uuid4(),
            kind="Entity",
            name="MySQL DB",
        )
        graph.entities[entity_db.id] = entity_db
        
        with Domain("test", DomainConfig(entity_linking_threshold=0.5)):
            resolve_entities(graph)
        
        # It should have linked to "Database" due to the canonical concept embedding
        assert entity_db.kind == "Database"
        assert entity_db.metadata.get("linked_concept") == "Database"
        assert "linking_score" in entity_db.metadata
