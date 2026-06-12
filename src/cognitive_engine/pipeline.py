from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from cognitive_engine.chunker import chunk_text, merge_propositions, PropSpan
from cognitive_engine.edge_assigner import SpanKey, assign_edges
from cognitive_engine.demarcation_rules import assign_demarcations
from cognitive_engine.models import Graph, Node, NodeType, ReasoningMode, Span as ModelSpan
from cognitive_engine.preprocessor import preprocess_chunks
from cognitive_engine.tagger import (
    PropositionTagger,
    RelationClassifier,
    SentenceTagger,
)
from cognitive_engine.type_mapper import map_types, Relation

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

    span_to_node: Dict[SpanKey, UUID] = {}
    nodes: Dict[UUID, Node] = {}
    for s in spans:
        uid = uuid4()
        span_to_node[(s.start_char, s.end_char)] = uid
        nodes[uid] = Node(
            id=uid,
            type=NodeType.CLAIM,
            text=s.text,
            span=ModelSpan(start=s.start_char, end=s.end_char, text=s.text),
        )

    relations: List[Relation] = []
    for i, sa in enumerate(spans):
        for j, sb in enumerate(spans):
            if i >= j:
                continue
            label = classifier.classify(sa.text, sb.text)
            if label != "None":
                relations.append(Relation(source_span=sa, target_span=sb, label=label))

    spacy_docs = [pp.doc for pp in preprocessed if pp.doc is not None]

    typed = map_types(spans, spacy_docs, relations)

    typed_spans: List[Tuple[PropSpan, NodeType]] = typed
    for s, nt in typed_spans:
        key = (s.start_char, s.end_char)
        if key in span_to_node:
            nodes[span_to_node[key]].type = nt

    graph = Graph(
        nodes=nodes,
        source_text=text,
        mode=mode,
    )

    assign_demarcations(graph, spacy_docs)

    edges = assign_edges(typed_spans, relations, span_to_node, nodes)
    graph.edges = edges

    return graph
