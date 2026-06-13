from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cognitive_engine.nlp.chunker import Chunk, PropSpan

logger = logging.getLogger(__name__)

_SPACY_PIPELINE = None


@dataclass
class PreprocessedChunk:
    original: Chunk
    resolved_text: str
    doc: "spacy.tokens.Doc"
    coref_chains: List[List[Tuple[int, int, str]]] = field(default_factory=list)


def load_spacy_pipeline() -> "spacy.language.Language":
    import spacy

    global _SPACY_PIPELINE
    if _SPACY_PIPELINE is None:
        logger.info("Loading en_core_web_trf (first call, may take a moment)...")
        _SPACY_PIPELINE = spacy.load("en_core_web_trf")
    return _SPACY_PIPELINE


def preprocess_chunks(chunks: List[Chunk]) -> List[PreprocessedChunk]:
    import spacy

    nlp = load_spacy_pipeline()

    results: List[PreprocessedChunk] = []
    for chunk in chunks:
        doc = nlp(chunk.text)
        doc.user_data["chunk_start_char"] = chunk.start_char
        doc.user_data["chunk_end_char"] = chunk.end_char
        resolved, chains = _resolve_coreferences(doc)
        results.append(PreprocessedChunk(
            original=chunk,
            resolved_text=resolved,
            doc=doc,
            coref_chains=chains,
        ))
    return results


PRONOUN_MAP = {
    "he": "masculine", "him": "masculine", "his": "masculine",
    "she": "feminine", "her": "feminine", "hers": "feminine",
    "it": "neuter", "its": "neuter",
    "they": "plural", "them": "plural", "their": "plural", "theirs": "plural",
    "this": "demonstrative_sg", "that": "demonstrative_sg",
    "these": "demonstrative_pl", "those": "demonstrative_pl",
}

_SUBJECT_PRONOUNS = {"he", "she", "it", "they"}
_OBJECT_PRONOUNS = {"him", "her", "it", "them"}
_POSSESSIVE_PRONOUNS = {"his", "her", "its", "their"}
_DEMONSTRATIVES = {"this", "that", "these", "those"}
_ALL_PRONOUNS = set(PRONOUN_MAP)


def _find_antecedent(
    pronoun_index: int,
    doc: "spacy.tokens.Doc",
    pronoun_gender: str,
) -> Optional[Tuple[int, int, str]]:
    candidates: List[Tuple[int, int, str]] = []
    for i in range(0, pronoun_index):
        token = doc[i]
        if token.pos_ in ("NOUN", "PROPN"):
            if pronoun_gender == "neuter" and token.pos_ == "NOUN":
                candidates.append((i, i + 1, token.text))
            elif pronoun_gender == "plural" and token.tag_ in ("NNS", "NNPS"):
                candidates.append((i, i + 1, token.text))
            elif pronoun_gender in ("masculine", "feminine") and token.pos_ == "PROPN":
                candidates.append((i, i + 1, token.text))
            elif token.pos_ == "NOUN":
                candidates.append((i, i + 1, token.text))
        elif token.pos_ == "PRON" and token.text.lower() in _SUBJECT_PRONOUNS:
            candidates.append((i, i + 1, token.text))

    if pronoun_gender in ("neuter", "plural") and not candidates:
        for i in range(0, pronoun_index):
            token = doc[i]
            if token.pos_ in ("NOUN", "PROPN"):
                candidates.append((i, i + 1, token.text))

    if pronoun_gender in ("demonstrative_sg", "demonstrative_pl"):
        for i in range(0, pronoun_index):
            token = doc[i]
            if token.pos_ in ("NOUN", "PROPN", "VERB") and token.dep_ == "ROOT":
                np_start, np_end = _expand_noun_phrase(i, doc)
                if np_end > np_start:
                    candidates.append((np_start, np_end, doc[np_start:np_end].text))
                break

    if not candidates:
        for i in range(0, pronoun_index):
            token = doc[i]
            if token.pos_ in ("NOUN", "PROPN"):
                candidates.append((i, i + 1, token.text))
                break

    if not candidates:
        return None

    best = candidates[-1]
    return best


def _expand_noun_phrase(token_index: int, doc: "spacy.tokens.Doc") -> Tuple[int, int]:
    start = token_index
    while start > 0 and doc[start - 1].dep_ in ("det", "amod", "nummod", "poss"):
        start -= 1
    end = token_index + 1
    while end < len(doc) and doc[end].dep_ in ("prep", "pobj", "appos"):
        np_start = end + 1
        while np_start < len(doc) and doc[np_start].dep_ == "pobj":
            np_start += 1
        if doc[end].dep_ == "appos":
            end = np_start + 1
        else:
            end = np_start
    return start, end


def _resolve_coreferences(
    doc: "spacy.tokens.Doc",
) -> Tuple[str, List[List[Tuple[int, int, str]]]]:
    tokens = list(doc)
    resolved = [t.text for t in tokens]
    chains: List[List[Tuple[int, int, str]]] = []

    for i, token in enumerate(tokens):
        text_lower = token.text.lower()
        if text_lower not in _ALL_PRONOUNS:
            continue

        gender = PRONOUN_MAP.get(text_lower, "neuter")
        antecedent = _find_antecedent(i, doc, gender)
        if antecedent is not None:
            resolved[i] = antecedent[2]
            chain = [(antecedent[0], antecedent[1], antecedent[2]), (i, i + 1, text_lower)]
            chains.append(chain)

    result = ""
    for j, token in enumerate(tokens):
        result += resolved[j]
        if j < len(tokens) - 1:
            result += doc.text[token.idx + len(token.text):tokens[j + 1].idx]

    return result, chains


def get_dependency_info(span: PropSpan, doc: "spacy.tokens.Doc") -> Dict:
    import spacy

    char_span = doc.char_span(span.start_char, span.end_char)
    if char_span is None:
        return {
            "verbs": [],
            "modals": [],
            "mark_relations": [],
            "root_verb_tense": None,
        }

    verbs = [t for t in char_span if t.pos_ == "VERB"]
    modals = [t for t in char_span if t.tag_ == "MD"]
    mark_relations = [
        (t.text, t.head.text, t.head.pos_)
        for t in char_span
        if t.dep_ == "mark"
    ]
    root_verb_tense = _get_root_tense(char_span)

    return {
        "verbs": verbs,
        "modals": modals,
        "mark_relations": mark_relations,
        "root_verb_tense": root_verb_tense,
    }


def _get_root_tense(tokens: "spacy.tokens.Doc | spacy.tokens.Span") -> Optional[str]:
    for t in tokens:
        if t.dep_ == "ROOT" and t.pos_ == "VERB":
            if t.tag_ == "VBD":
                return "past"
            elif t.tag_ == "VBG":
                return "present_participle"
            elif t.tag_ in ("VBZ", "VBP"):
                return "present"
            elif t.tag_ == "VBN":
                return "past_participle"
            elif t.tag_ == "VB":
                return "base"
    for t in tokens:
        if t.dep_ == "ROOT" and t.pos_ == "AUX":
            if t.tag_ == "MD":
                return "future"
            return "present"
    return None
