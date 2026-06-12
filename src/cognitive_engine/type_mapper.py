from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cognitive_engine.chunker import PropSpan
from cognitive_engine.models import NodeType

logger = logging.getLogger(__name__)

_FALLACY_KEYWORDS = {
    "fallacy", "mistake", "confuses", "erroneous",
    "misleading", "flawed", "invalid",
}

_ADVERSATIVES = {
    "however", "but", "although", "nevertheless",
    "conversely", "yet", "whereas", "nonetheless",
}


@dataclass
class Relation:
    source_span: PropSpan
    target_span: PropSpan
    label: str


def _global_to_local(span: PropSpan, doc: "spacy.tokens.Doc") -> Tuple[int, int]:
    chunk_start = doc.user_data.get("chunk_start_char", 0)
    return span.start_char - chunk_start, span.end_char - chunk_start


def _char_span_relaxed(
    doc: "spacy.tokens.Doc", local_start: int, local_end: int,
) -> Optional["spacy.tokens.Span"]:
    span = doc.char_span(local_start, local_end, alignment_mode="contract")
    if span is not None:
        return span
    span = doc.char_span(local_start, local_end, alignment_mode="expand")
    return span


def assign_type(
    span: PropSpan,
    doc: "spacy.tokens.Doc",
    relations: Optional[List[Relation]] = None,
) -> NodeType:
    import spacy

    deps = _extract_deps(span, doc)

    if _is_condition(deps):
        return NodeType.CONDITION

    if _is_fallacy(span):
        return NodeType.FALLACY

    if _is_justification(deps):
        return NodeType.JUSTIFICATION

    if _is_axiom(deps):
        return NodeType.AXIOM

    if _is_counterclaim(span, doc, relations or []):
        return NodeType.COUNTERCLAIM

    if _is_root_proposition(span, doc):
        return NodeType.CLAIM

    return NodeType.EVIDENCE


def _extract_deps(span: PropSpan, doc: "spacy.tokens.Doc") -> Dict:
    local_start, local_end = _global_to_local(span, doc)
    char_span = _char_span_relaxed(doc, local_start, local_end)
    verbs: list = []
    modals: list = []
    mark_relations: list = []

    if char_span is not None:
        verbs = [t for t in char_span if t.pos_ == "VERB"]
        modals = [t for t in char_span if t.tag_ == "MD"]
        mark_relations = [
            (t.text, t.head.text, t.head.pos_)
            for t in char_span
            if t.dep_ == "mark"
        ]

    return {
        "verbs": verbs,
        "modals": modals,
        "mark_relations": mark_relations,
    }


def _is_axiom(deps: Dict) -> bool:
    return any(
        t.tag_ == "MD" and t.dep_ in ("aux", "ROOT")
        for t in deps.get("modals", [])
    )


def _is_counterclaim(
    span: PropSpan,
    doc: "spacy.tokens.Doc",
    relations: List[Relation],
) -> bool:
    local_start, local_end = _global_to_local(span, doc)
    char_span = _char_span_relaxed(doc, local_start, local_end)
    if char_span is not None:
        for t in char_span:
            if t.dep_ in ("cc", "mark", "advmod") and t.text.lower() in _ADVERSATIVES:
                return True
    return False


def _is_condition(deps: Dict) -> bool:
    for mark_text, _, head_pos in deps.get("mark_relations", []):
        if mark_text.lower() in ("if", "unless", "provided") and head_pos in ("VERB", "AUX"):
            return True
    return False


def _is_justification(deps: Dict) -> bool:
    for mark_text, _, head_pos in deps.get("mark_relations", []):
        if mark_text.lower() in ("because", "since", "for") and head_pos in ("VERB", "AUX"):
            return True
    return False


def _is_fallacy(span: PropSpan) -> bool:
    text_lower = span.text.lower()
    return any(kw in text_lower for kw in _FALLACY_KEYWORDS)


def _is_root_proposition(
    span: PropSpan,
    doc: "spacy.tokens.Doc",
) -> bool:
    local_start, local_end = _global_to_local(span, doc)
    char_span = _char_span_relaxed(doc, local_start, local_end)
    if char_span is None:
        return False
    return any(t.dep_ == "ROOT" for t in char_span)


def _spans_overlap(a: PropSpan, b: PropSpan) -> bool:
    return a.start_char < b.end_char and b.start_char < a.end_char


def map_types(
    spans: List[PropSpan],
    docs: List["spacy.tokens.Doc"],
    relations: Optional[List[Relation]] = None,
) -> List[Tuple[PropSpan, NodeType]]:
    if relations is None:
        relations = []

    results: List[Tuple[PropSpan, NodeType]] = []
    for span in spans:
        doc = _find_doc_for_span(span, docs)
        if doc is None:
            logger.warning(
                "No doc found for span (%d, %d), defaulting to EVIDENCE",
                span.start_char, span.end_char,
            )
            results.append((span, NodeType.EVIDENCE))
            continue
        node_type = assign_type(span, doc, relations)
        results.append((span, node_type))
    return results


def _find_doc_for_span(
    span: PropSpan,
    docs: List["spacy.tokens.Doc"],
) -> Optional["spacy.tokens.Doc"]:
    for doc in docs:
        doc_start = doc.user_data.get("chunk_start_char", 0)
        doc_end = doc.user_data.get("chunk_end_char", len(doc.text))
        if span.start_char >= doc_start and span.end_char <= doc_end:
            return doc

    for doc in docs:
        doc_start = doc.user_data.get("chunk_start_char", 0)
        doc_end = doc.user_data.get("chunk_end_char", len(doc.text))
        if span.start_char < doc_end and span.end_char > doc_start:
            return doc

    return None
