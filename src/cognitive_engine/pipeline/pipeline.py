from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Set
from uuid import UUID, uuid4

from cognitive_engine.chunker import chunk_text, merge_propositions, PropSpan
from cognitive_engine.argumentation_module import run_argumentation
from cognitive_engine.entity_module import extract_entities
from cognitive_engine.models import Graph, Node, NodeType, ReasoningMode, Span as ModelSpan
from cognitive_engine.module_registry import ModuleDef, ModuleRegistry
from cognitive_engine.preprocessor import preprocess_chunks
from cognitive_engine.relation_extractor import extract_relations
from cognitive_engine.tagger import (
    PropositionTagger,
    RelationClassifier,
    SentenceTagger,
)

logger = logging.getLogger(__name__)


def run(
    text: str,
    tagger: Optional[PropositionTagger | SentenceTagger] = None,
    classifier: Optional[RelationClassifier] = None,
    max_tokens: int = 512,
    overlap: int = 128,
    merge_margin: int = 20,
    mode: ReasoningMode = ReasoningMode.ARGUMENT,
) -> Graph:
    if tagger is None:
        tagger = SentenceTagger()
    if classifier is None:
        classifier = RelationClassifier()

    chunks = chunk_text(text, tokenizer=classifier.tokenizer, max_tokens=max_tokens, overlap=overlap)
    if not chunks:
        return Graph(source_text=text, mode=mode)

    preprocessed = preprocess_chunks(chunks)

    if isinstance(tagger, SentenceTagger):
        spans = tagger.extract_spans(preprocessed, text)
    else:
        tags_per_chunk: List[List[str]] = []
        for chunk in chunks:
            tags = tagger.tag_chunk(chunk)
            tags_per_chunk.append(tags)
        spans = merge_propositions(chunks, tags_per_chunk, text, merge_margin_chars=merge_margin)

    if not spans:
        return Graph(source_text=text, mode=mode)

    spacy_docs = [pp.doc for pp in preprocessed if pp.doc is not None]

    graph = Graph(source_text=text, mode=mode)

    seen_spans: Set[Tuple[int, int]] = set()
    for s in spans:
        seen_spans.add((s.start_char, s.end_char))

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
