from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from cognitive_engine.models import Graph, Span, WorldRelation

logger = logging.getLogger(__name__)

_VERB_TO_RELATION: Dict[str, str] = {
    "consist": "CONSISTS_OF",
    "contain": "CONSISTS_OF",
    "include": "CONSISTS_OF",
    "comprise": "CONSISTS_OF",
    "run": "RUNS",
    "manage": "RUNS",
    "control": "RUNS",
    "operate": "RUNS",
    "depend": "DEPENDS_ON",
    "require": "DEPENDS_ON",
    "rely": "DEPENDS_ON",
    "cause": "CAUSES",
    "trigger": "CAUSES",
    "lead": "CAUSES",
    "produce": "CAUSES",
    "prevent": "CONSTRAINS",
    "block": "CONSTRAINS",
    "constrain": "CONSTRAINS",
    "limit": "CONSTRAINS",
    "enable": "ENABLES",
    "allow": "ENABLES",
    "facilitate": "ENABLES",
    "support": "ENABLES",
    "part": "PART_OF",
    "belong": "PART_OF",
    "define": "DEFINES",
    "represent": "DEFINES",
    "denote": "DEFINES",
    "expose": "EXPOSES",
    "route": "ROUTES_TO",
    "forward": "ROUTES_TO",
    "govern": "GOVERNS",
    "regulate": "GOVERNS",
    "specify": "SPECIFIES",
    "describe": "SPECIFIES",
    "provide": "PROVIDES",
    "offer": "PROVIDES",
    "connect": "CONNECTS_TO",
    "link": "CONNECTS_TO",
    "attach": "CONNECTS_TO",
}



def _entity_for_token(
    graph: Graph, token: "spacy.tokens.Token", chunk_start: int,
) -> Optional[UUID]:
    tok_start = chunk_start + token.idx
    tok_end = tok_start + len(token.text)
    best_eid = None
    best_gap = 6
    for eid, entity in graph.entities.items():
        for s in entity.spans:
            if s.start <= tok_start and s.end >= tok_end:
                return eid
            gap = min(abs(s.start - tok_start), abs(s.end - tok_end))
            if gap < best_gap:
                best_gap = gap
                best_eid = eid
    return best_eid


def extract_relations(
    graph: Graph,
    docs: List["spacy.tokens.Doc"],
    **kwargs,
) -> None:
    seen: Set[Tuple[str, str, str]] = set()

    for doc in docs:
        chunk_start = doc.user_data.get("chunk_start_char", 0)

        for sent in doc.sents:
            for token in sent:
                if token.pos_ != "VERB":
                    continue
                verb_lemma = token.lemma_.lower()
                rel_kind = _VERB_TO_RELATION.get(verb_lemma)
                if rel_kind is None:
                    continue

                subj = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                        subj = child
                        break

                obj = None
                for child in token.children:
                    if child.dep_ in ("dobj", "pobj", "attr", "ccomp"):
                        obj = child
                        break

                if subj is not None and obj is not None:
                    src_id = _entity_for_token(graph, subj, chunk_start)
                    tgt_id = _entity_for_token(graph, obj, chunk_start)
                    if src_id is not None and tgt_id is not None:
                        key = (src_id.hex, tgt_id.hex, rel_kind)
                        if key not in seen:
                            seen.add(key)
                            graph.world_relations.append(WorldRelation(
                                id=uuid4(),
                                source_id=src_id,
                                target_id=tgt_id,
                                kind=rel_kind,
                            ))
