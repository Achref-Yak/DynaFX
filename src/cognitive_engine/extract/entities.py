from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from cognitive_engine.core.models import Entity, Graph, Span

logger = logging.getLogger(__name__)



def _ner_kind_for_span(
    doc: "spacy.tokens.Doc", chunk_start: int, chunk_abs_start: int, chunk_abs_end: int,
) -> Optional[str]:
    for ent in doc.ents:
        ent_abs_start = chunk_start + ent.start_char
        ent_abs_end = chunk_start + ent.end_char
        if chunk_abs_start < ent_abs_end and ent_abs_start < chunk_abs_end:
            return ent.label_
    return None


def _noun_kind(root: "spacy.tokens.Token") -> str:
    if root.pos_ == "PROPN":
        return root.text
    if root.pos_ == "NOUN":
        return root.lemma_.title()
    return "Entity"


def extract_entities(
    graph: Graph,
    docs: List["spacy.tokens.Doc"],
    source_text: str,
    seen_spans: Set[Tuple[int, int]] | None = None,
    **kwargs,
) -> None:
    if seen_spans is None:
        seen_spans = set()

    for doc in docs:
        chunk_start = doc.user_data.get("chunk_start_char", 0)
        for chunk in doc.noun_chunks:
            start = chunk_start + chunk.start_char
            end = chunk_start + chunk.end_char
            key = (start, end)
            if key in seen_spans:
                continue
            seen_spans.add(key)

            text = source_text[start:end]
            if not text.strip():
                continue

            ner_label = _ner_kind_for_span(doc, chunk_start, start, end)
            kind = ner_label if ner_label is not None else _noun_kind(chunk.root)

            entity = Entity(
                id=uuid4(),
                kind=kind,
                name=text.strip(),
                spans=[Span(start=start, end=end, text=text)],
            )
            graph.entities[entity.id] = entity


def deduplicate_entities(graph: Graph) -> None:
    """Merge entities with identical normalized names, keeping the one with most spans."""
    name_groups: dict[str, list[UUID]] = defaultdict(list)
    for eid, entity in graph.entities.items():
        norm_name = entity.name.lower().strip()
        name_groups[norm_name].append(eid)

    for norm_name, eids in name_groups.items():
        if len(eids) <= 1:
            continue
        best_eid = max(eids, key=lambda eid: len(graph.entities[eid].spans))
        best_entity = graph.entities[best_eid]

        for eid in eids:
            if eid == best_eid:
                continue
            other = graph.entities[eid]
            existing_spans = {(s.start, s.end) for s in best_entity.spans}
            for span in other.spans:
                if (span.start, span.end) not in existing_spans:
                    best_entity.spans.append(span)
                    existing_spans.add((span.start, span.end))
            for edge in list(graph.edges.values()):
                if edge.source_id == eid:
                    edge.source_id = best_eid
                if edge.target_id == eid:
                    edge.target_id = best_eid
            del graph.entities[eid]
