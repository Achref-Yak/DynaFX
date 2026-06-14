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
from cognitive_engine.nlp.heuristic_classifier import HeuristicClassifier
from cognitive_engine.nlp.deposition_parser import parse_deposition
from cognitive_engine.domain import domain

logger = logging.getLogger(__name__)


def _ensure_tagger(
    tagger: Optional[PropositionTagger | SentenceTagger],
) -> PropositionTagger | SentenceTagger:
    if tagger is not None:
        return tagger
    try:
        return SentenceTagger()
    except Exception as e:
        logger.warning("SentenceTagger unavailable, falling back to PropositionTagger: %s", e)
        return PropositionTagger()


def _ensure_classifier(
    classifier: Optional[RelationClassifier],
) -> RelationClassifier:
    return classifier if classifier is not None else RelationClassifier()


def _get_fallback_tokenizer():
    """Get a tokenizer for chunking when neither classifier nor tagger provides one."""
    from transformers import AutoTokenizer
    from pathlib import Path
    model_dir = Path(__file__).resolve().parents[3] / "models" / "roberta-relation-classifier"
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


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
    use_deposition_parser: bool = True,
    use_heuristic_classifier: bool = True,
) -> Graph:
    # Parse deposition structure if applicable
    parsed_deposition = None
    if use_deposition_parser:
        parsed_deposition = parse_deposition(text)

    if parsed_deposition is not None and parsed_deposition.is_deposition:
        return _run_deposition(text, parsed_deposition, max_tokens, overlap, merge_margin, mode, use_heuristic_classifier)

    # Non-deposition path
    tagger = _ensure_tagger(tagger)
    classifier = _ensure_classifier(classifier)

    base_chunks = chunk_text(text, tokenizer=classifier.tokenizer, max_tokens=max_tokens, overlap=overlap)
    if not base_chunks:
        return Graph(source_text=text, mode=mode)

    preprocessed = preprocess_chunks(base_chunks)

    spans = _extract_spans(text, base_chunks, preprocessed, tagger, merge_margin)
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
        classifier=classifier if not use_heuristic_classifier else HeuristicClassifier(),
        tagger=tagger,
        use_heuristic_classifier=use_heuristic_classifier,
    )

    return graph


def _run_deposition(
    text: str,
    parsed,
    max_tokens: int,
    overlap: int,
    merge_margin: int,
    mode: ReasoningMode,
    use_heuristic_classifier: bool,
) -> Graph:
    """Run extraction on a deposition — skip tagger, use Q/A answers as propositions."""
    from cognitive_engine.nlp.deposition_parser import ParsedDeposition
    from cognitive_engine.core.models import Edge, EdgeType

    # Create PropSpan objects directly from Q/A answers (skip tagger entirely)
    spans: List[PropSpan] = []
    span_sections: Dict[int, str] = {}  # index -> section type (direct/cross/etc)
    idx = 0
    for section in parsed.sections:
        section_type = section.exam_type.name.lower()
        for qa in section.qa_pairs:
            answer_start = text.find(qa.answer, qa.char_start)
            if answer_start == -1:
                answer_start = qa.char_start
            answer_end = answer_start + len(qa.answer)
            spans.append(PropSpan(
                start_char=answer_start,
                end_char=answer_end,
                text=qa.answer,
            ))
            span_sections[idx] = section_type
            idx += 1

    if not spans:
        return Graph(source_text=text, mode=mode)

    # Create minimal chunks for preprocessing
    from cognitive_engine.nlp.chunker import Chunk
    from cognitive_engine.nlp.preprocessor import preprocess_chunks

    base_chunks = []
    for i, section in enumerate(parsed.sections):
        for j, qa in enumerate(section.qa_pairs):
            base_chunks.append(Chunk(
                start_char=qa.char_start,
                end_char=qa.char_end,
                text=text[qa.char_start:qa.char_end],
                tokens=[],
                offsets=[],
                offset=len(base_chunks),
            ))

    preprocessed = preprocess_chunks(base_chunks)
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
        classifier=HeuristicClassifier() if use_heuristic_classifier else _ensure_classifier(None),
        tagger=None,
        use_heuristic_classifier=use_heuristic_classifier,
    )

    # Create structural edges: consecutive answers in same section support each other
    from uuid import uuid4
    node_list = list(graph.nodes.values())
    config = domain.active()

    for i in range(len(node_list) - 1):
        n1, n2 = node_list[i], node_list[i + 1]
        s1_type = span_sections.get(i, "")
        s2_type = span_sections.get(i + 1, "")

        if s1_type == s2_type and s1_type:
            # Same section — consecutive answers support each other
            edge = Edge(
                source_id=n1.id,
                target_id=n2.id,
                type=EdgeType.SUPPORTS,
            )
            if "SUPPORTS" in config.edge_warrants:
                edge.warrant = config.edge_warrants["SUPPORTS"]
            graph.edges.append(edge)

    logger.info(
        "Deposition extraction: %d nodes, %d edges, %d spans",
        len(graph.nodes), len(graph.edges), len(spans),
    )
    return graph
