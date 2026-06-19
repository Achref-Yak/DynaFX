"""Lightweight SRL-based relation extraction.

Uses spaCy dependency parse + VerbNet syntactic frames to extract
predicate-argument structures (ARG0/ARG1/ARGM) without requiring
a heavy neural SRL model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph, NodeType, WorldRelation
from cognitive_engine.extract.verbnet_roles import (
    vn_classes_for_lemma, vn_themroles, vn_roles_for_span,
)

logger = logging.getLogger(__name__)


@dataclass
class SRLArgument:
    """A single argument in an SRL frame."""
    role: str       # ARG0, ARG1, ARG2, ARGM-LOC, etc.
    text: str
    start: int
    end: int


@dataclass
class SRLFrame:
    """Predicate-argument structure for a single verb."""
    verb: str
    verb_lemma: str
    start: int
    end: int
    arguments: list[SRLArgument]
    vn_class: str = ""
    vn_roles: dict[str, str] = None

    def __post_init__(self) -> None:
        if self.vn_roles is None:
            self.vn_roles = {}


def predict_srl_lightweight(doc: object) -> list[SRLFrame]:
    """Extract SRL frames using spaCy deps + VerbNet.

    Maps spaCy dependency labels to PropBank argument roles using
    VerbNet thematic role information.

    Args:
        doc: spaCy Doc object.

    Returns:
        List of SRLFrame objects, one per verb.
    """
    frames: list[SRLFrame] = []
    chunk_start = doc.user_data.get("chunk_start_char", 0)

    for sent in doc.sents:
        for token in sent:
            if token.pos_ != "VERB":
                continue
            if token.dep_ not in ("ROOT", "conj", "ccomp", "xcomp"):
                continue

            # Get VerbNet class for this verb
            lemma = token.lemma_.lower()
            vn_classes = vn_classes_for_lemma(lemma)
            vn_class = vn_classes[0] if vn_classes else ""
            vn_roles = vn_roles_for_span(lemma) if vn_class else {}

            # Extract arguments based on dependency structure
            arguments = _extract_arguments(token, vn_roles, chunk_start)

            frame = SRLFrame(
                verb=token.text,
                verb_lemma=lemma,
                start=token.idx + chunk_start,
                end=token.idx + chunk_start + len(token.text),
                arguments=arguments,
                vn_class=vn_class,
                vn_roles=vn_roles,
            )
            frames.append(frame)

    return frames


def _extract_arguments(
    verb_token: object,
    vn_roles: dict[str, str],
    chunk_start: int,
) -> list[SRLArgument]:
    """Extract arguments from a verb's dependency children.

    Maps spaCy dependency labels to PropBank argument roles:
      - nsubj → ARG0 (Agent)
      - dobj → ARG1 (Theme/Patient)
      - pobj (via prep) → ARG2 (Goal/Instrument/Benefactive)
      - iobj → ARG2 (Goal/Recipient)
      - advmod → ARGM (modifiers)
      - prep + pobj → ARGM-LOC, ARGM-MNR, etc.
    """
    args: list[SRLArgument] = []

    for child in verb_token.children:
        dep = child.dep_
        text = child.text
        start = child.idx + chunk_start
        end = start + len(child.text)

        # Expand to full noun chunk for nominal arguments
        if dep in ("nsubj", "nsubjpass", "dobj", "iobj", "attr"):
            expanded = _expand_to_subtree(child, chunk_start)
            text = expanded["text"]
            start = expanded["start"]
            end = expanded["end"]

        # Map dependency to PropBank role
        role = _dep_to_argrole(dep, child, vn_roles)
        if role:
            args.append(SRLArgument(role=role, text=text, start=start, end=end))

    return args


def _dep_to_argrole(
    dep: str,
    token: object,
    vn_roles: dict[str, str],
) -> Optional[str]:
    """Map a spaCy dependency label to a PropBank argument role.

    Uses VerbNet thematic roles when available for more precise mapping.
    """
    # Core arguments
    if dep in ("nsubj", "nsubjpass"):
        return "ARG0"  # Agent/Patient
    if dep == "dobj":
        return "ARG1"  # Theme/Patient
    if dep == "iobj":
        return "ARG2"  # Goal/Recipient
    if dep == "attr":
        return "ARG1"  # Attribute (often Theme-equivalent)

    # Prepositional arguments
    if dep == "prep":
        return "ARGM-LOC"  # Default; could be refined by preposition
    if dep == "pobj":
        return "ARG2"  # Object of preposition

    # Modifiers
    if dep == "advmod":
        return "ARGM-ADV"
    if dep == "npadvmod":
        return "ARGM-ADV"
    if dep in ("tmod", "nummod"):
        return "ARGM-TMP"
    if dep == "acomp":
        return "ARG1"
    if dep == "oprd":
        return "ARG1"

    return None


def _expand_to_subtree(token: object, chunk_start: int) -> dict:
    """Expand a token to its full subtree text and span."""
    subtree = list(token.subtree)
    if not subtree:
        return {
            "text": token.text,
            "start": token.idx + chunk_start,
            "end": token.idx + chunk_start + len(token.text),
        }
    start = subtree[0].idx + chunk_start
    end = subtree[-1].idx + chunk_start + len(subtree[-1].text)
    text = " ".join(t.text for t in subtree)
    return {"text": text, "start": start, "end": end}


def extract_srl_relations(
    graph: Graph,
    docs: list,
    source_text: str = "",
) -> list[WorldRelation]:
    """Extract relations using lightweight SRL (spaCy + VerbNet).

    For each SRL frame with ARG0 and ARG1, creates a WorldRelation
    between the corresponding entities in the graph.

    Args:
        graph: The graph to add relations to.
        docs: List of spaCy Doc objects.
        source_text: Original source text.

    Returns:
        List of created WorldRelation objects.
    """
    from cognitive_engine.extract.types import _char_span_relaxed, _global_to_local

    relations: list[WorldRelation] = []

    for doc in docs:
        frames = predict_srl_lightweight(doc)

        for frame in frames:
            if len(frame.arguments) < 2:
                continue

            # Find ARG0 and ARG1 spans
            arg0 = next((a for a in frame.arguments if a.role == "ARG0"), None)
            arg1 = next((a for a in frame.arguments if a.role == "ARG1"), None)

            if not arg0 or not arg1:
                continue

            # Find entities in graph that overlap with ARG0 and ARG1
            source_entity = _find_entity_for_span(
                graph, arg0.start, arg0.end, source_text,
            )
            target_entity = _find_entity_for_span(
                graph, arg1.start, arg1.end, source_text,
            )

            if source_entity and target_entity:
                # Classify relation using VerbNet
                from cognitive_engine.extract.verbnet_roles import classify_relation_by_vn
                edge_name = classify_relation_by_vn(
                    frame.verb_lemma,
                    NodeType.ENTITY,  # entities
                    NodeType.ENTITY,
                )
                if edge_name:
                    relation = WorldRelation(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        kind=edge_name,
                    )
                    relations.append(relation)

    return relations


def _find_entity_for_span(
    graph: Graph,
    start: int,
    end: int,
    source_text: str,
):
    """Find an entity in the graph whose span overlaps [start, end)."""
    from cognitive_engine.extract.types import _char_span_relaxed

    for entity in graph.entities.values():
        e_start = entity.span.start
        e_end = entity.span.end
        # Check overlap
        if e_start < end and e_end > start:
            return entity
    return None
