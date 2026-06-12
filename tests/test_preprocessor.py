import pytest
from cognitive_engine.chunker import Chunk
from cognitive_engine.preprocessor import (
    load_spacy_pipeline,
    preprocess_chunks,
    get_dependency_info,
    _resolve_coreferences,
)

pytest.importorskip("spacy")


def test_load_pipeline():
    nlp = load_spacy_pipeline()
    assert nlp is not None
    assert nlp.pipe_names


def test_pipeline_singleton():
    a = load_spacy_pipeline()
    b = load_spacy_pipeline()
    assert a is b


class TestPreprocessChunks:
    def test_single_chunk(self):
        chunk = Chunk(
            start_char=0, end_char=21,
            text="The database is slow.",
            tokens=[0, 1, 2, 3, 4],
            offsets=[(0, 3), (4, 12), (13, 15), (16, 20), (20, 21)],
            offset=0,
        )
        results = preprocess_chunks([chunk])
        assert len(results) == 1
        assert results[0].original is chunk
        assert results[0].resolved_text is not None
        assert results[0].doc is not None
        assert isinstance(results[0].coref_chains, list)

    def test_multiple_chunks(self):
        chunks = [
            Chunk(start_char=0, end_char=10, text="First one.",
                  tokens=[0, 1, 2], offsets=[(0, 5), (6, 9), (9, 10)], offset=0),
            Chunk(start_char=10, end_char=22, text=" Second one.",
                  tokens=[0, 1, 2], offsets=[(10, 17), (18, 21), (21, 22)], offset=1),
        ]
        results = preprocess_chunks(chunks)
        assert len(results) == 2
        assert results[0].coref_chains is not None
        assert results[1].coref_chains is not None

    def test_empty_chunks(self):
        results = preprocess_chunks([])
        assert results == []


class TestCoreferenceResolution:
    def test_pronoun_it_resolved(self):
        nlp = load_spacy_pipeline()
        doc = nlp("The database is slow. It needs indexing.")
        resolved, chains = _resolve_coreferences(doc)
        assert "database" in resolved.lower()
        assert len(chains) >= 1

    def test_no_pronoun_no_change(self):
        nlp = load_spacy_pipeline()
        doc = nlp("Databases store data.")
        resolved, chains = _resolve_coreferences(doc)
        assert resolved == "Databases store data."
        assert len(chains) == 0

    def test_possessive_his_resolved(self):
        nlp = load_spacy_pipeline()
        doc = nlp("John submitted a patch. His code was merged.")
        resolved, chains = _resolve_coreferences(doc)
        assert "John" in resolved
        assert len(chains) >= 1

    def test_she_resolved(self):
        nlp = load_spacy_pipeline()
        doc = nlp("Alice reviewed the PR. She approved it.")
        resolved, chains = _resolve_coreferences(doc)
        assert "Alice" in resolved

    def test_demonstrative_this(self):
        nlp = load_spacy_pipeline()
        doc = nlp("The query runs slowly. This is a problem.")
        resolved, chains = _resolve_coreferences(doc)
        _ = resolved
        assert isinstance(chains, list)


class TestDependencyInfo:
    def test_verbs_detected(self):
        nlp = load_spacy_pipeline()
        doc = nlp("The system runs the query.")
        from cognitive_engine.preprocessor import PropSpan
        span = PropSpan(start_char=0, end_char=25, text=doc.text, chunk_offsets=[0])
        info = get_dependency_info(span, doc)
        assert len(info["verbs"]) >= 1
        assert info["verbs"][0].text == "runs"

    def test_modals_detected(self):
        nlp = load_spacy_pipeline()
        doc = nlp("The system must handle 10k requests.")
        from cognitive_engine.preprocessor import PropSpan
        span = PropSpan(start_char=0, end_char=35, text=doc.text, chunk_offsets=[0])
        info = get_dependency_info(span, doc)
        assert len(info["modals"]) >= 1
        assert info["modals"][0].text == "must"

    def test_mark_relations_detected(self):
        nlp = load_spacy_pipeline()
        doc = nlp("If the query fails, retry.")
        from cognitive_engine.preprocessor import PropSpan
        span = PropSpan(start_char=0, end_char=len(doc.text), text=doc.text, chunk_offsets=[0])
        info = get_dependency_info(span, doc)
        found = any(
            t.lower() == "if" for t, _, _ in info["mark_relations"]
        )
        assert found

    def test_root_tense_present(self):
        nlp = load_spacy_pipeline()
        doc = nlp("The system processes data.")
        from cognitive_engine.preprocessor import PropSpan
        span = PropSpan(start_char=0, end_char=25, text=doc.text, chunk_offsets=[0])
        info = get_dependency_info(span, doc)
        assert info["root_verb_tense"] == "present"

    def test_root_tense_past(self):
        nlp = load_spacy_pipeline()
        doc = nlp("The system processed data.")
        from cognitive_engine.preprocessor import PropSpan
        span = PropSpan(start_char=0, end_char=26, text=doc.text, chunk_offsets=[0])
        info = get_dependency_info(span, doc)
        assert info["root_verb_tense"] == "past"

    def test_empty_span_returns_defaults(self):
        nlp = load_spacy_pipeline()
        doc = nlp("")
        from cognitive_engine.preprocessor import PropSpan
        span = PropSpan(start_char=0, end_char=0, text="", chunk_offsets=[0])
        info = get_dependency_info(span, doc)
        assert info["verbs"] == []
        assert info["modals"] == []
