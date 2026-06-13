from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from cognitive_engine.nlp.chunker import PropSpan
from cognitive_engine.extract.demarcation import assign_demarcations
from cognitive_engine.extract.edges import assign_edges, SpanKey
from cognitive_engine.core.models import (
    Edge,
    EdgeType,
    Entity,
    Graph,
    Interpretation,
    Node,
    NodeType,
    Span as ModelSpan,
    TypedEdge,
)
from cognitive_engine.nlp.tagger import RelationClassifier, SentenceTagger
from cognitive_engine.extract.types import map_types, Relation

logger = logging.getLogger(__name__)


def _find_entity_for_span(
    graph: Graph, start: int, end: int,
) -> Optional[UUID]:
    for eid, entity in graph.entities.items():
        for s in entity.spans:
            if s.start <= start <= s.end or s.start <= end <= s.end:
                return eid
    return None


def _classify_relations(
    spans: List[PropSpan], classifier: RelationClassifier,
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

    for edge in graph.edges:
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
    classifier: Optional[RelationClassifier] = None,
    tagger: Optional[SentenceTagger] = None,
    **kwargs,
) -> None:
    if classifier is None:
        classifier = RelationClassifier()
    if tagger is None:
        tagger = SentenceTagger()

    relations = _classify_relations(spans, classifier)
    typed = map_types(spans, docs, relations)

    span_to_node, nodes = _build_nodes(typed)
    graph.nodes = nodes
    graph.edges = assign_edges(typed, relations, span_to_node, nodes)
    assign_demarcations(graph, docs)

    graph.interpretations["argumentation"] = _register_interpretation(graph, typed)
