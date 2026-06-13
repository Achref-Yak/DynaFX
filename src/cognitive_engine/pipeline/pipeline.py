from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Set
from uuid import UUID, uuid4

from cognitive_engine.nlp.chunker import chunk_text, merge_propositions, PropSpan
from cognitive_engine.pipeline.argumentation import run_argumentation
from cognitive_engine.extract.entities import extract_entities
from cognitive_engine.core.models import Graph, Node, NodeType, ReasoningMode, Span as ModelSpan
from cognitive_engine.pipeline.registry import ModuleDef, ModuleRegistry
from cognitive_engine.nlp.preprocessor import preprocess_chunks
from cognitive_engine.extract.relations import extract_relations
from cognitive_engine.nlp.tagger import (
    PropositionTagger,
    RelationClassifier,
    SentenceTagger,
)

logger = logging.getLogger(__name__)


def _ensure_tagger(
    tagger: Optional[PropositionTagger | SentenceTagger],
) -> PropositionTagger | SentenceTagger:
    return tagger if tagger is not None else SentenceTagger()


def _ensure_classifier(
    classifier: Optional[RelationClassifier],
) -> RelationClassifier:
    return classifier if classifier is not None else RelationClassifier()


def _extract_spans(
    text: str,
    chunks: List,
    preprocessed: List,
    tagger: PropositionTagger | SentenceTagger,
    merge_margin: int,
) -> Optional[List[PropSpan]]:
    if isinstance(tagger, SentenceTagger):
        return tagger.extract_spans(preprocessed, text)

    tags_per_chunk = [tagger.tag_chunk(chunk) for chunk in chunks]
    return merge_propositions(chunks, tags_per_chunk, text, merge_margin_chars=merge_margin)


def run(
    text: str,
    tagger: Optional[PropositionTagger | SentenceTagger] = None,
    classifier: Optional[RelationClassifier] = None,
    max_tokens: int = 512,
    overlap: int = 128,
    merge_margin: int = 20,
    mode: ReasoningMode = ReasoningMode.ARGUMENT,
) -> Graph:
    tagger = _ensure_tagger(tagger)
    classifier = _ensure_classifier(classifier)

    chunks = chunk_text(text, tokenizer=classifier.tokenizer, max_tokens=max_tokens, overlap=overlap)
    if not chunks:
        return Graph(source_text=text, mode=mode)

    preprocessed = preprocess_chunks(chunks)

    spans = _extract_spans(text, chunks, preprocessed, tagger, merge_margin)
    if not spans:
        return Graph(source_text=text, mode=mode)

    spacy_docs = [pp.doc for pp in preprocessed if pp.doc is not None]

    graph = Graph(source_text=text, mode=mode)
    seen_spans = {(s.start_char, s.end_char) for s in spans}

    registry = ModuleRegistry()
    registry.register(ModuleDef("entity", [], extract_entities))
    registry.register(ModuleDef("entity_relation", ["entity"], extract_relations))
    registry.register(ModuleDef("argumentation", ["entity"], run_argumentation))

    registry.run(
        ["entity", "entity_relation", "argumentation"],
        graph,
        docs=spacy_docs,
        source_text=text,
        spans=spans,
        seen_spans=seen_spans,
        classifier=classifier,
        tagger=tagger,
    )

    return graph
