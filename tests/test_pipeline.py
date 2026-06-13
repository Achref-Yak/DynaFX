from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from cognitive_engine.nlp.chunker import Chunk, PropSpan
from cognitive_engine.core.models import EdgeType, Graph, NodeType, ReasoningMode
from cognitive_engine.pipeline.pipeline import run
from cognitive_engine.nlp.tagger import PropositionTagger, RelationClassifier


class FakeTagger:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    def tag_chunk(self, chunk):
        return ["B-Prop"] * len(chunk.offsets)


class FakeClassifier:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    def classify(self, a, b):
        return "Support"


class TestPipelineIntegration:
    def test_pipeline_returns_graph(self):
        tagger = FakeTagger()
        classifier = FakeClassifier()
        text = "The system is reliable. Traffic is high."
        result = run(text, tagger=tagger, classifier=classifier)  # type: ignore
        assert isinstance(result, Graph)
        assert len(result.nodes) > 0

    def test_empty_text_returns_empty_graph(self):
        result = run("", tagger=FakeTagger(), classifier=FakeClassifier())  # type: ignore
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    def test_graph_has_source_text(self):
        text = "Test text."
        result = run(text, tagger=FakeTagger(), classifier=FakeClassifier())  # type: ignore
        assert result.source_text == text

    def test_graph_has_default_mode(self):
        text = "Test."
        result = run(text, tagger=FakeTagger(), classifier=FakeClassifier())  # type: ignore
        assert result.mode == ReasoningMode.ARGUMENT

    def test_custom_mode_is_respected(self):
        text = "Test."
        result = run(text, tagger=FakeTagger(), classifier=FakeClassifier(),
                     mode=ReasoningMode.CAUSAL)  # type: ignore
        assert result.mode == ReasoningMode.CAUSAL

    def test_nodes_have_demarcations(self):
        text = "The system must handle 10k. Traffic is high."
        tagger = FakeTagger()
        classifier = FakeClassifier()
        result = run(text, tagger=tagger, classifier=classifier)  # type: ignore
        for node in result.nodes.values():
            assert "demarcation" in node.metadata
            d = node.metadata["demarcation"]
            assert all(k in d for k in [
                "cognitive_vs_epistemic", "epistemic_vs_institutional",
                "affect_vs_cognition", "constraint_vs_enablement",
                "synchronic_vs_diachronic",
            ])

    def test_edge_types_are_valid(self):
        text = "Must handle 10k. The system scales."
        tagger = FakeTagger()
        classifier = FakeClassifier()
        result = run(text, tagger=tagger, classifier=classifier)  # type: ignore
        valid = {e.value for e in EdgeType}
        for edge in result.edges:
            assert edge.type.value in valid
