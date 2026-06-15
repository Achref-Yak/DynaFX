"""Ξ (Extract) operator — Text -> Graph.

Converts raw text into a structured Graph by chunking text into
proposition spans, extracting entities and relations, assigning
node types and edge types, and building argumentation structure.

Inlines the extraction logic directly rather than routing through
the old pipeline module (which has been removed).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from cognitive_engine.nlp.chunker import (
    Chunk, PropSpan, chunk_text, merge_propositions,
)
from cognitive_engine.nlp.preprocessor import preprocess_chunks
from cognitive_engine.nlp.tagger import (
    PropositionTagger, RelationClassifier, SentenceTagger,
)
from cognitive_engine.nlp.heuristic_classifier import HeuristicClassifier
from cognitive_engine.nlp.deposition_parser import (
    parse_deposition, ParsedDeposition,
)
from cognitive_engine.extract.entities import extract_entities
from cognitive_engine.extract.relations import extract_relations
from cognitive_engine.extract.edges import assign_edges, infer_causal_edges
from cognitive_engine.extract.types import map_types, Relation
from cognitive_engine.extract.demarcation import assign_demarcations
from cognitive_engine.extract.edges import SpanKey
from cognitive_engine.core.models import (
    Edge, Entity, Graph, Interpretation, Node, NodeType,
    ReasoningMode, Span as ModelSpan, TypedEdge,
)
from cognitive_engine.core.state import State
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


def _find_entity_for_span(
    graph: Graph, start: int, end: int,
) -> Optional[UUID]:
    for eid, entity in graph.entities.items():
        for s in entity.spans:
            if s.start <= start <= s.end or s.start <= end <= s.end:
                return eid
    return None


def _classify_relations(
    spans: List[PropSpan], classifier,
) -> List[Relation]:
    relations: List[Relation] = []
    for i, sa in enumerate(spans):
        for j, sb in enumerate(spans):
            if i >= j:
                continue
            label = classifier.classify(sa.text, sb.text)
            if label != "None":
                relations.append(Relation(source_span=sa, target_span=sb, label=label))
    return relations


def _build_nodes(
    typed_spans: List[Tuple[PropSpan, NodeType]],
) -> Tuple[Dict[SpanKey, UUID], Dict[UUID, Node]]:
    span_to_node: Dict[SpanKey, UUID] = {}
    nodes: Dict[UUID, Node] = {}
    for s, nt in typed_spans:
        uid = uuid4()
        span_to_node[(s.start_char, s.end_char)] = uid
        nodes[uid] = Node(
            id=uid,
            type=nt,
            text=s.text,
            span=ModelSpan(start=s.start_char, end=s.end_char, text=s.text),
        )
    return span_to_node, nodes


def _register_interpretation(
    graph: Graph, typed_spans: List[Tuple[PropSpan, NodeType]],
) -> Interpretation:
    interp = Interpretation(name="argumentation")
    for s, nt in typed_spans:
        eid = _find_entity_for_span(graph, s.start_char, s.end_char)
        if eid is not None:
            interp.roles[eid] = nt.name
        elif s.text.strip():
            eid2 = uuid4()
            graph.entities[eid2] = Entity(
                id=eid2,
                kind="Proposition",
                name=s.text.strip(),
                spans=[ModelSpan(start=s.start_char, end=s.end_char, text=s.text)],
            )
            interp.roles[eid2] = nt.name

    for edge in graph.edges.values():
        interp.edges.append(TypedEdge(
            id=uuid4(),
            source_id=edge.source_id,
            target_id=edge.target_id,
            type=edge.type.name,
        ))
    return interp


def run_argumentation(
    graph: Graph,
    spans: List[PropSpan],
    docs: List["spacy.tokens.Doc"],
    source_text: str,
    classifier=None,
    tagger: Optional[SentenceTagger] = None,
    use_heuristic_classifier: bool = True,
    **kwargs,
) -> None:
    if classifier is None:
        if use_heuristic_classifier:
            classifier = HeuristicClassifier()
        else:
            classifier = RelationClassifier()
    if tagger is None:
        tagger = SentenceTagger()

    relations = _classify_relations(spans, classifier)
    typed = map_types(spans, docs, relations)

    span_to_node, nodes = _build_nodes(typed)
    graph.nodes = nodes
    graph.edges = assign_edges(typed, relations, span_to_node, nodes)
    graph.edges.update(infer_causal_edges(nodes))

    config = domain.active()
    for edge in graph.edges.values():
        edge_type_name = edge.type.name
        if edge_type_name in config.edge_warrants:
            edge.warrant = config.edge_warrants[edge_type_name]

    assign_demarcations(graph, docs)

    graph.interpretations["argumentation"] = _register_interpretation(graph, typed)


def _extract_graph(
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
    """Core text→graph extraction. Inlines the old pipeline.run()."""
    parsed_deposition = None
    if use_deposition_parser:
        parsed_deposition = parse_deposition(text)

    if parsed_deposition is not None and parsed_deposition.is_deposition:
        return _run_deposition(text, parsed_deposition, max_tokens, overlap, merge_margin, mode, use_heuristic_classifier)

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

    extract_entities(graph, docs=spacy_docs, source_text=text, spans=spans, seen_spans=seen_spans)
    extract_relations(graph, docs=spacy_docs, source_text=text, spans=spans, seen_spans=seen_spans)
    run_argumentation(
        graph, spans=spans, docs=spacy_docs, source_text=text,
        classifier=classifier if not use_heuristic_classifier else HeuristicClassifier(),
        tagger=tagger, use_heuristic_classifier=use_heuristic_classifier,
    )

    return graph


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


def _run_deposition(
    text: str,
    parsed: ParsedDeposition,
    max_tokens: int,
    overlap: int,
    merge_margin: int,
    mode: ReasoningMode,
    use_heuristic_classifier: bool,
) -> Graph:
    from cognitive_engine.core.models import Edge as EdgeModel

    spans: List[PropSpan] = []
    span_sections: Dict[int, str] = {}
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

    base_chunks = []
    for i, section in enumerate(parsed.sections):
        for j, qa in enumerate(section.qa_pairs):
            base_chunks.append(Chunk(
                start_char=qa.char_start,
                end_char=qa.char_end,
                text=text[qa.char_start:qa.char_end],
                tokens=[], offsets=[], offset=len(base_chunks),
            ))

    preprocessed = preprocess_chunks(base_chunks)
    spacy_docs = [pp.doc for pp in preprocessed if pp.doc is not None]

    graph = Graph(source_text=text, mode=mode)
    seen_spans = {(s.start_char, s.end_char) for s in spans}

    extract_entities(graph, docs=spacy_docs, source_text=text, spans=spans, seen_spans=seen_spans)
    extract_relations(graph, docs=spacy_docs, source_text=text, spans=spans, seen_spans=seen_spans)
    run_argumentation(
        graph, spans=spans, docs=spacy_docs, source_text=text,
        classifier=HeuristicClassifier() if use_heuristic_classifier else _ensure_classifier(None),
        tagger=None, use_heuristic_classifier=use_heuristic_classifier,
    )

    node_list = list(graph.nodes.values())
    config = domain.active()

    for i in range(len(node_list) - 1):
        n1, n2 = node_list[i], node_list[i + 1]
        s1_type = span_sections.get(i, "")
        s2_type = span_sections.get(i + 1, "")

        if s1_type == s2_type and s1_type:
            edge = EdgeModel(
                source_id=n1.id,
                target_id=n2.id,
                type="SUPPORTS",
            )
            if "SUPPORTS" in config.edge_warrants:
                edge.warrant = config.edge_warrants["SUPPORTS"]
            graph.edges[edge.id] = edge

    logger.info(
        "Deposition extraction: %d nodes, %d edges, %d spans",
        len(graph.nodes), len(graph.edges), len(spans),
    )
    return graph


class ExtractOperator:
    """Ξ: Text → Graph

    Converts raw text into a structured graph by:
    1. Chunking text into proposition spans
    2. Extracting entities and relations
    3. Assigning node types and edge types
    4. Building argumentation structure
    5. Optionally computing embeddings for each node
    """
    name = "extract"

    def __init__(self, compute_embeddings: bool = True):
        self._compute_embeddings_flag = compute_embeddings

    def __call__(
        self,
        state: State,
        text: str = None,
        max_tokens: int = 512,
        overlap: int = 128,
        merge_margin: int = 20,
        use_deposition_parser: bool = True,
        use_heuristic_classifier: bool = True,
        compute_embeddings: bool = None,
        **kwargs,
    ) -> State:
        if state.metadata.get("extracted"):
            return state

        input_text = text or state.metadata.get("text", "") or state.graph.source_text
        if not input_text:
            return state

        graph = _extract_graph(
            input_text,
            max_tokens=max_tokens,
            overlap=overlap,
            merge_margin=merge_margin,
            use_deposition_parser=use_deposition_parser,
            use_heuristic_classifier=use_heuristic_classifier,
        )

        should_compute = compute_embeddings if compute_embeddings is not None else self._compute_embeddings_flag
        if should_compute and graph.nodes:
            self._compute_embeddings(graph)

        state.graph = graph
        state.metadata["extracted"] = True
        state.metadata["text"] = input_text

        node_texts = [n.text[:60] for n in list(graph.nodes.values())[:5]]
        entity_names = [e.name for e in list(graph.entities.values())[:5]]
        type_counts = {}
        for n in graph.nodes.values():
            type_counts[n.type.name] = type_counts.get(n.type.name, 0) + 1
        type_summary = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))
        state.record(
            self.name,
            f"Extracted {len(graph.nodes)} propositions and {len(graph.edges)} edges from the input text. "
            f"Node type distribution: {type_summary}. "
            f"Sample propositions: {'; '.join(node_texts)}. "
            f"Entities identified: {', '.join(entity_names) if entity_names else 'none detected'}. "
            f"The resulting graph structure captures the key claims, beliefs, entities, and their causal or inferential relationships.",
        )
        return state

    def _compute_embeddings(self, graph: Graph) -> None:
        """Compute embeddings for all nodes in the graph."""
        from cognitive_engine.core.embeddings import EmbeddingModel
        model = EmbeddingModel.get_instance()
        texts = [node.text for node in graph.nodes.values()]
        embeddings = model.encode_batch(texts)
        for node, embedding in zip(graph.nodes.values(), embeddings):
            node.embedding = embedding
