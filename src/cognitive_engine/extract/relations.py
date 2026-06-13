from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from cognitive_engine.core.models import Graph, WorldRelation

logger = logging.getLogger(__name__)

_PHRASE_MAP: Dict[str, str] = {
    "consists of": "CONSISTS_OF",
    "consist of": "CONSISTS_OF",
    "depends on": "DEPENDS_ON",
    "depend on": "DEPENDS_ON",
    "depends upon": "DEPENDS_ON",
    "depend upon": "DEPENDS_ON",
    "part of": "PART_OF",
    "leads to": "CAUSES",
    "lead to": "CAUSES",
    "results in": "CAUSES",
    "result in": "CAUSES",
    "gives rise to": "CAUSES",
    "give rise to": "CAUSES",
    "controlled by": "GOVERNS",
    "governed by": "GOVERNS",
    "based on": "DEPENDS_ON",
    "base on": "DEPENDS_ON",
    "connected to": "CONNECTS_TO",
    "connect to": "CONNECTS_TO",
    "originates from": "ORIGINATES_FROM",
    "originate from": "ORIGINATES_FROM",
    "derived from": "ORIGINATES_FROM",
    "derive from": "ORIGINATES_FROM",
    "transitions to": "TRANSITIONS",
    "transition to": "TRANSITIONS",
    "migrates to": "TRANSITIONS",
    "migrate to": "TRANSITIONS",
    "transforms into": "TRANSITIONS",
    "transform into": "TRANSITIONS",
    "evolves into": "TRANSITIONS",
    "evolve into": "TRANSITIONS",
}

_VERB_LEMMA_MAP: Dict[str, str] = {
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
    "authenticate": "AUTHENTICATES",
    "encrypt": "ENCRYPTS",
    "decrypt": "ENCRYPTS",
    "deploy": "DEPLOYS",
    "configure": "CONFIGURES",
    "set": "CONFIGURES",
    "balance": "TRADE_OFF",
    "evolve": "TRANSITIONS",
    "migrate": "TRANSITIONS",
    "transition": "TRANSITIONS",
    "experience": "AFFECTS",
    "affect": "AFFECTS",
    "impact": "AFFECTS",
    "guarantee": "ENSURES",
    "ensure": "ENSURES",
    "ignore": "NEGLECTS",
    "neglect": "NEGLECTS",
    "introduce": "INTRODUCES",
    "maintain": "MAINTAINS",
    "preserve": "MAINTAINS",
    "originate": "ORIGINATES_FROM",
    "derive": "ORIGINATES_FROM",
    "prioritize": "PRIORITIZES",
    "recommend": "RECOMMENDS",
    "advise": "RECOMMENDS",
    "scale": "SCALES",
    "show": "DEMONSTRATES",
    "demonstrate": "DEMONSTRATES",
    "indicate": "DEMONSTRATES",
    "satisfy": "SATISFIES",
    "meet": "SATISFIES",
}


def _expand_to_noun_chunk(doc, token):
    for chunk in doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return chunk
    return None


def _collect_auxiliary_parts(verb):
    parts = []
    order_map = {}
    for child in verb.children:
        if child.dep_ == "aux":
            order_map[child.i] = child.text.lower()
        elif child.dep_ == "neg":
            order_map[child.i] = child.text.lower()
    for i in sorted(order_map):
        parts.append(order_map[i])
    return parts


def _build_relation_phrase(verb):
    parts = _collect_auxiliary_parts(verb)
    parts.append(verb.lemma_)
    for child in verb.children:
        if child.dep_ == "prt":
            parts.append(child.text.lower())
    for child in verb.children:
        if child.dep_ == "prep":
            parts.append(child.text.lower())
    return " ".join(parts)


def _find_subject(verb):
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child
    for child in verb.children:
        if child.dep_ == "csubj":
            return child
    if verb.dep_ == "conj":
        for child in verb.head.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return child
    return None


def _find_object(verb):
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "ccomp"):
            return child
    for child in verb.children:
        if child.dep_ != "prep":
            continue
        for grandchild in child.children:
            if grandchild.dep_ == "pobj":
                return grandchild
    return None


def _extract_triple(doc, verb):
    subj = _find_subject(verb)
    if subj is None:
        return None

    obj = _find_object(verb)
    if obj is None:
        return None

    subj_chunk = _expand_to_noun_chunk(doc, subj)
    obj_chunk = _expand_to_noun_chunk(doc, obj)

    subj_text = subj_chunk.text if subj_chunk else subj.text
    obj_text = obj_chunk.text if obj_chunk else obj.text
    rel_text = _build_relation_phrase(verb)

    if subj_chunk:
        subj_start = subj_chunk.start_char
        subj_end = subj_chunk.end_char
    else:
        subj_start = subj.idx
        subj_end = subj.idx + len(subj.text)

    if obj_chunk:
        obj_start = obj_chunk.start_char
        obj_end = obj_chunk.end_char
    else:
        obj_start = obj.idx
        obj_end = obj.idx + len(obj.text)

    return (
        subj_text, obj_text, rel_text,
        subj_start, subj_end, obj_start, obj_end,
        verb.lemma_,
    )


def _classify_relation(rel_text: str, verb_lemma: str) -> str:
    key = rel_text.lower().strip()
    mapped = _PHRASE_MAP.get(key)
    if mapped is not None:
        return mapped
    mapped = _VERB_LEMMA_MAP.get(verb_lemma.lower())
    if mapped is not None:
        return mapped
    return "RELATED_TO"


def _entity_for_span(graph: Graph, start: int, end: int) -> Optional[UUID]:
    for eid, entity in graph.entities.items():
        for s in entity.spans:
            if s.start <= start <= s.end or s.start <= end <= s.end:
                return eid
    return None


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
                if token.dep_ not in ("ROOT", "conj"):
                    continue

                triple = _extract_triple(doc, token)
                if triple is None:
                    logger.debug("Skipping verb '%s' at %d: no subject or object found",
                                 token.text, token.i)
                    continue

                (subj_text, obj_text, rel_text,
                 subj_start, subj_end, obj_start, obj_end,
                 verb_lemma) = triple

                rel_kind = _classify_relation(rel_text, verb_lemma)

                abs_subj_start = chunk_start + subj_start
                abs_subj_end = chunk_start + subj_end
                abs_obj_start = chunk_start + obj_start
                abs_obj_end = chunk_start + obj_end

                src_id = _entity_for_span(graph, abs_subj_start, abs_subj_end)
                tgt_id = _entity_for_span(graph, abs_obj_start, abs_obj_end)

                if src_id is not None and tgt_id is not None:
                    key = (src_id.hex, tgt_id.hex, rel_kind)
                    if key not in seen:
                        seen.add(key)
                        graph.world_relations.append(WorldRelation(
                            id=uuid4(),
                            source_id=src_id,
                            target_id=tgt_id,
                            kind=rel_kind,
                            metadata={
                                "relation_phrase": rel_text,
                                "subject_text": subj_text,
                                "object_text": obj_text,
                                "confidence": 1.0,
                            },
                        ))
