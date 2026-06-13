from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from cognitive_engine.core.models import Graph, NodeType

logger = logging.getLogger(__name__)

_STATIVE_VERBS: Set[str] = {
    "be", "seem", "appear", "look", "feel", "become",
    "remain", "stay", "sound", "taste", "smell", "constitute",
}

_POSITIVE_SENTIMENT: Set[str] = {
    "good", "great", "excellent", "beneficial", "effective",
    "efficient", "reliable", "secure", "fast", "scalable",
    "robust", "superior", "optimal", "safe", "improved",
}

_NEGATIVE_SENTIMENT: Set[str] = {
    "bad", "poor", "slow", "unreliable", "insecure",
    "expensive", "difficult", "problematic", "dangerous",
    "costly", "wasteful", "fragile", "risky", "obsolete",
}

_ALL_SENTIMENT = _POSITIVE_SENTIMENT | _NEGATIVE_SENTIMENT

_CONSTRAINT_VERBS: Set[str] = {
    "cannot", "unable", "prevent", "block", "restrict", "limit",
    "prohibit", "forbid", "constrain",
}

_ENABLEMENT_VERBS: Set[str] = {
    "can", "enable", "allow", "capable", "permit", "support",
    "facilitate", "empower", "authorize",
}


def _global_to_local(
    start_char: int, end_char: int, doc: "spacy.tokens.Doc",
) -> Tuple[int, int]:
    chunk_start = doc.user_data.get("chunk_start_char", 0)
    return start_char - chunk_start, end_char - chunk_start


def _char_span_relaxed(
    doc: "spacy.tokens.Doc", local_start: int, local_end: int,
) -> Optional["spacy.tokens.Span"]:
    span = doc.char_span(local_start, local_end, alignment_mode="contract")
    if span is not None:
        return span
    return doc.char_span(local_start, local_end, alignment_mode="expand")


def assign_demarcations(graph: Graph, docs: List["spacy.tokens.Doc"]) -> None:
    for node_id, node in graph.nodes.items():
        doc = _find_doc_for_node(node, docs)
        if doc is None:
            node.metadata["demarcation"] = {
                "cognitive_vs_epistemic": "NA",
                "epistemic_vs_institutional": "NA",
                "affect_vs_cognition": "NA",
                "constraint_vs_enablement": "NA",
                "synchronic_vs_diachronic": "NA",
            }
            continue

        modal_info = _get_modal_info(node, doc)
        tense = _get_tense(node, doc)

        node.metadata["demarcation"] = {
            "cognitive_vs_epistemic": _cog_epi(node, doc),
            "epistemic_vs_institutional": _epi_inst(node, doc, modal_info),
            "affect_vs_cognition": _affect_cog(node, doc),
            "constraint_vs_enablement": _constraint_enable(node, doc, modal_info),
            "synchronic_vs_diachronic": _synch_dia(node, doc, tense),
        }


def _find_doc_for_node(
    node,
    docs: List["spacy.tokens.Doc"],
) -> Optional["spacy.tokens.Doc"]:
    if node.span is None:
        for doc in docs:
            if node.text in doc.text:
                return doc
        return None

    start = node.span.start
    end = node.span.end

    for doc in docs:
        doc_start = doc.user_data.get("chunk_start_char", 0)
        doc_end = doc.user_data.get("chunk_end_char", len(doc.text))
        if start >= doc_start and end <= doc_end:
            return doc

    for doc in docs:
        doc_start = doc.user_data.get("chunk_start_char", 0)
        doc_end = doc.user_data.get("chunk_end_char", len(doc.text))
        if start < doc_end and end > doc_start:
            return doc

    return None


def _get_tokens(
    node, doc: "spacy.tokens.Doc",
) -> Optional[List]:
    if node.span is None:
        return None
    local_start, local_end = _global_to_local(node.span.start, node.span.end, doc)
    char_span = _char_span_relaxed(doc, local_start, local_end)
    if char_span is None:
        return None
    return list(char_span)


def _get_modal_info(
    node, doc: "spacy.tokens.Doc",
) -> Dict:
    tokens = _get_tokens(node, doc)
    if not tokens:
        return {"has_modal": False, "verb_is_stative": False}

    for t in tokens:
        if t.tag_ == "MD":
            head = t.head
            verb_is_stative = head.lemma_ in _STATIVE_VERBS
            return {
                "has_modal": True,
                "modal_text": t.text.lower(),
                "head_verb": head.lemma_,
                "head_pos": head.pos_,
                "verb_is_stative": verb_is_stative,
            }

    return {"has_modal": False}


def _get_tense(
    node, doc: "spacy.tokens.Doc",
) -> Optional[str]:
    tokens = _get_tokens(node, doc)
    if not tokens:
        return None

    for t in tokens:
        if t.dep_ == "ROOT":
            morph_tense = t.morph.get("Tense")
            if morph_tense:
                return morph_tense[0]
            if t.tag_ == "MD":
                return "Future"
            if t.pos_ in ("VERB", "AUX"):
                return "Present"
            break

    for t in tokens:
        if t.pos_ == "VERB":
            morph_tense = t.morph.get("Tense")
            if morph_tense:
                return morph_tense[0]

    return None


def _cog_epi(
    node, doc: "spacy.tokens.Doc",
) -> str:
    if node.type in (NodeType.EVIDENCE, NodeType.JUSTIFICATION):
        return "EPISTEMIC"
    if node.type == NodeType.CONDITION:
        return "COGNITIVE"
    if node.type == NodeType.AXIOM:
        return "NA"
    return "EPISTEMIC"


def _epi_inst(
    node, doc: "spacy.tokens.Doc",
    modal_info: Dict,
) -> str:
    if not modal_info.get("has_modal"):
        return "NA"

    if modal_info["head_pos"] in ("VERB", "AUX"):
        if modal_info["verb_is_stative"]:
            return "EPISTEMIC"
        return "INSTITUTIONAL"

    return "NA"


def _affect_cog(
    node, doc: "spacy.tokens.Doc",
) -> str:
    tokens = _get_tokens(node, doc)
    if not tokens:
        return "NA"

    for t in tokens:
        if t.pos_ == "ADJ" and t.lemma_.lower() in _ALL_SENTIMENT:
            if t.dep_ in ("amod", "acomp", "attr", "ROOT"):
                return "AFFECT"

    return "COGNITION"


def _constraint_enable(
    node, doc: "spacy.tokens.Doc",
    modal_info: Dict,
) -> str:
    tokens = _get_tokens(node, doc)
    if not tokens:
        return "NA"

    has_negation = False
    for t in tokens:
        if t.dep_ == "neg" and t.head.tag_ in ("VB", "MD", "VBZ", "VBP"):
            has_negation = True
            break

    for t in tokens:
        text = t.text.lower()
        lemma = t.lemma_.lower()
        if text in _CONSTRAINT_VERBS:
            return "CONSTRAINT"
        if lemma in _ENABLEMENT_VERBS and t.dep_ == "aux":
            if has_negation:
                return "CONSTRAINT"
            return "ENABLEMENT"
        if lemma in _ENABLEMENT_VERBS and t.dep_ == "ROOT":
            if has_negation:
                return "CONSTRAINT"
            return "ENABLEMENT"

    if has_negation:
        return "CONSTRAINT"

    return "NA"


def _synch_dia(
    node, doc: "spacy.tokens.Doc",
    tense: Optional[str],
) -> str:
    if tense is None:
        return "NA"
    if tense in ("Pres", "Present"):
        return "SYNCHRONIC"
    return "DIACHRONIC"
