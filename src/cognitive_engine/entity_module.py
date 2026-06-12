from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple
from uuid import uuid4

from cognitive_engine.models import Entity, Graph, Span

logger = logging.getLogger(__name__)


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

            root = chunk.root
            kind = root.text if root.pos_ in ("PROPN",) else root.lemma_.title() if root.pos_ == "NOUN" else "Entity"

            entity = Entity(
                id=uuid4(),
                kind=kind,
                name=text.strip(),
                spans=[Span(start=start, end=end, text=text)],
            )
            graph.entities[entity.id] = entity
