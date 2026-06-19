from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from cognitive_engine.nlp.chunker import PropSpan
from cognitive_engine.core.models import BfoCategory, NodeType
from cognitive_engine.extract.demarcation import _match_span_to_doc

logger = logging.getLogger(__name__)

# ── Pattern sets ─────────────────────────────────────────────────

_FALLACY_KEYWORDS = {
    "fallacy", "mistake", "confuses", "erroneous",
    "misleading", "flawed", "invalid",
}

_ADVERSATIVES = {
    "however", "but", "although", "nevertheless",
    "conversely", "yet", "whereas", "nonetheless",
}

_AGENT_PATTERNS = {"i am", "my name is", "i work as", "my role is", "i represent", "on behalf of"}
_PROCESS_PATTERNS = {"the process", "the procedure", "the method", "the workflow", "the pipeline", "the operation"}
_STATE_PATTERNS = {"state of", "condition of", "in a situation", "current status"}
_GOAL_PATTERNS = {"want to", "need to", "aim to", "goal is", "my objective", "intend to", "plan to"}
_RESOURCE_PATTERNS = {"the resource", "the material", "the supply", "the budget", "the funding"}
_CONSTRAINT_PATTERNS = {"required", "mandatory", "prohibited", "forbidden", "restricted", "limited"}

_CLAUSE_SPLIT_RE = re.compile(
    r'(?:'
    r',\s*(?!(?:and|but|or)\s+)(?=(?:my|your|his|her|its|our|their|i|you|he|she|it|we|they)\s+)|'
    r'\s+(?:and|but|or)\s+(?=(?:my|your|his|her|its|our|their|i|you|he|she|it|we|they)\s+\w+\s+)'
    r')',
    re.IGNORECASE,
)


# ── Step 1: Strongly-typed dataclasses ───────────────────────────

@dataclass(frozen=True)
class DependencyFeatures:
    """Strongly-typed dependency features extracted from spaCy."""
    verbs: tuple       # tuple of spacy.Token
    modals: tuple      # tuple of spacy.Token
    mark_relations: tuple  # tuple of (mark_text, head_text, head_pos)


@dataclass(frozen=True)
class ClassificationContext:
    """Pre-computed context for type classification rules."""
    span: PropSpan
    doc: object  # spacy.tokens.Doc — avoiding forward reference issues
    text_lower: str
    deps: DependencyFeatures


# ── Step 2: TypeRule + rule list ─────────────────────────────────

@dataclass
class TypeRule:
    """A single type-classification rule with priority ordering."""
    priority: int
    name: str
    node_type: NodeType
    matcher: Callable[[ClassificationContext], bool]


def _match_agent(ctx: ClassificationContext) -> bool:
    return any(p in ctx.text_lower for p in _AGENT_PATTERNS)


def _match_process(ctx: ClassificationContext) -> bool:
    return any(p in ctx.text_lower for p in _PROCESS_PATTERNS)


def _match_state(ctx: ClassificationContext) -> bool:
    return any(p in ctx.text_lower for p in _STATE_PATTERNS)


def _match_goal(ctx: ClassificationContext) -> bool:
    return any(p in ctx.text_lower for p in _GOAL_PATTERNS)


def _match_resource(ctx: ClassificationContext) -> bool:
    return any(p in ctx.text_lower for p in _RESOURCE_PATTERNS)


def _match_constraint(ctx: ClassificationContext) -> bool:
    return any(p in ctx.text_lower for p in _CONSTRAINT_PATTERNS)


def _match_condition(ctx: ClassificationContext) -> bool:
    for mark_text, _, head_pos in ctx.deps.mark_relations:
        if mark_text.lower() in ("if", "unless", "provided") and head_pos in ("VERB", "AUX"):
            return True
    return False


def _match_fallacy(ctx: ClassificationContext) -> bool:
    return any(re.search(r'\b' + re.escape(kw) + r'\b', ctx.text_lower) for kw in _FALLACY_KEYWORDS)


def _match_justification(ctx: ClassificationContext) -> bool:
    for mark_text, _, head_pos in ctx.deps.mark_relations:
        if mark_text.lower() in ("because", "since", "for") and head_pos in ("VERB", "AUX"):
            return True
    return False


def _match_axiom(ctx: ClassificationContext) -> bool:
    return any(
        t.tag_ == "MD" and t.dep_ in ("aux", "ROOT")
        for t in ctx.deps.modals
    )


def _match_counterclaim(ctx: ClassificationContext) -> bool:
    local_start, local_end = _global_to_local(ctx.span, ctx.doc)
    char_span = _char_span_relaxed(ctx.doc, local_start, local_end)
    if char_span is not None:
        for t in char_span:
            if t.dep_ in ("cc", "mark", "advmod") and t.text.lower() in _ADVERSATIVES:
                return True
    return False


def _match_claim(ctx: ClassificationContext) -> bool:
    local_start, local_end = _global_to_local(ctx.span, ctx.doc)
    char_span = _char_span_relaxed(ctx.doc, local_start, local_end)
    if char_span is None:
        return False
    return any(t.dep_ == "ROOT" for t in char_span)


# ── FrameNet frame-based matchers ────────────────────────────────

def _make_frame_matcher(*target_frames: str):
    """Create a matcher that checks for specific FrameNet frames."""
    def matcher(ctx: ClassificationContext) -> bool:
        from cognitive_engine.nlp.semantic_resources import SemanticResources
        res = SemanticResources.instance()
        for token in ctx.deps.verbs:
            frames = res.frames_for_lemma(token.lemma_.lower())
            if any(f in target_frames for f in frames):
                return True
        return False
    return matcher


# Frame groupings derived from FRAMENodeType_MAP (single source of truth)
from cognitive_engine.extract.frame_rules import FRAMENodeType_MAP

def _frames_for_type(*types: NodeType) -> tuple[str, ...]:
    """Extract frame names from FRAMENodeType_MAP for the given NodeTypes."""
    return tuple(f for f, t in FRAMENodeType_MAP.items() if t in types)

_AGENT_FRAMES = _frames_for_type(NodeType.AGENT)
_PROCESS_FRAMES = _frames_for_type(NodeType.PROCESS)
_STATE_FRAMES = _frames_for_type(NodeType.STATE)
_GOAL_FRAMES = _frames_for_type(NodeType.GOAL)
_RESOURCE_FRAMES = _frames_for_type(NodeType.RESOURCE)
_CONSTRAINT_FRAMES = _frames_for_type(NodeType.CONSTRAINT)
_COUNTERCLAIM_FRAMES = _frames_for_type(NodeType.COUNTERCLAIM)
_FALLACY_FRAMES = _frames_for_type(NodeType.FALLACY)
_JUSTIFICATION_FRAMES = _frames_for_type(NodeType.JUSTIFICATION)


# ── TYPE_RULES: Frame-based (higher priority) + keyword fallback ─

TYPE_RULES: list[TypeRule] = [
    # Keyword rules for high-specificity patterns (highest priority)
    TypeRule(priority=200, name="agent",         node_type=NodeType.AGENT,         matcher=_match_agent),
    TypeRule(priority=195, name="condition",     node_type=NodeType.CONDITION,     matcher=_match_condition),
    TypeRule(priority=190, name="fallacy",       node_type=NodeType.FALLACY,       matcher=_match_fallacy),
    TypeRule(priority=185, name="justification", node_type=NodeType.JUSTIFICATION, matcher=_match_justification),
    TypeRule(priority=180, name="axiom",         node_type=NodeType.AXIOM,         matcher=_match_axiom),
    TypeRule(priority=175, name="counterclaim",  node_type=NodeType.COUNTERCLAIM,  matcher=_match_counterclaim),
    TypeRule(priority=170, name="claim",         node_type=NodeType.CLAIM,         matcher=_match_claim),

    # Frame-based rules (semantic precision for world-model types)
    TypeRule(priority=150, name="agent_frame",         node_type=NodeType.AGENT,         matcher=_make_frame_matcher(*_AGENT_FRAMES)),
    TypeRule(priority=140, name="process_frame",       node_type=NodeType.PROCESS,       matcher=_make_frame_matcher(*_PROCESS_FRAMES)),
    TypeRule(priority=130, name="state_frame",         node_type=NodeType.STATE,         matcher=_make_frame_matcher(*_STATE_FRAMES)),
    TypeRule(priority=120, name="goal_frame",          node_type=NodeType.GOAL,          matcher=_make_frame_matcher(*_GOAL_FRAMES)),
    TypeRule(priority=110, name="resource_frame",      node_type=NodeType.RESOURCE,      matcher=_make_frame_matcher(*_RESOURCE_FRAMES)),
    TypeRule(priority=100, name="constraint_frame",    node_type=NodeType.CONSTRAINT,    matcher=_make_frame_matcher(*_CONSTRAINT_FRAMES)),

    # Keyword fallback for world-model types (broader coverage)
    TypeRule(priority=80,  name="process",       node_type=NodeType.PROCESS,       matcher=_match_process),
    TypeRule(priority=70,  name="state",         node_type=NodeType.STATE,         matcher=_match_state),
    TypeRule(priority=60,  name="goal",          node_type=NodeType.GOAL,          matcher=_match_goal),
    TypeRule(priority=50,  name="resource",      node_type=NodeType.RESOURCE,      matcher=_match_resource),
    TypeRule(priority=40,  name="constraint",    node_type=NodeType.CONSTRAINT,    matcher=_match_constraint),
]


# ── Public API ───────────────────────────────────────────────────

def assign_type(
    span: PropSpan,
    doc: object,
    relations: Optional[List[Relation]] = None,
) -> NodeType:
    ctx = ClassificationContext(
        span=span,
        doc=doc,
        text_lower=span.text.lower(),
        deps=_extract_deps(span, doc),
    )
    for rule in TYPE_RULES:
        if rule.matcher(ctx):
            return rule.node_type
    return NodeType.OBSERVATION


def _extract_deps(span: PropSpan, doc: object) -> DependencyFeatures:
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

    return DependencyFeatures(
        verbs=tuple(verbs),
        modals=tuple(modals),
        mark_relations=tuple(mark_relations),
    )


# ── Step 3: Declarative BFO mapping ─────────────────────────────

_BFO_NODETYPE_MAP: dict[NodeType, BfoCategory] = {
    NodeType.EVENT: BfoCategory.PROCESS,
    NodeType.ACTION: BfoCategory.PROCESS,
}

_BFO_CONCEPT_MAP: dict[str, BfoCategory] = {
    # Identity
    "PERSON_NAME": BfoCategory.QUALITY,
    "EMAIL": BfoCategory.QUALITY,
    "PHONE": BfoCategory.QUALITY,
    # Measurements
    "TEMPERATURE": BfoCategory.QUALITY,
    "BUDGET": BfoCategory.QUALITY,
    "DATE": BfoCategory.QUALITY,
    # Preferences
    "PREFERENCE": BfoCategory.QUALITY,
    "STYLE": BfoCategory.QUALITY,
    # Location
    "LOCATION": BfoCategory.MATERIAL_ENTITY,
    # Realizable
    "CONDITION": BfoCategory.REALIZABLE_ENTITY,
    "RULE": BfoCategory.REALIZABLE_ENTITY,
    "STATUTE": BfoCategory.REALIZABLE_ENTITY,
    "CONTRACT_TERM": BfoCategory.REALIZABLE_ENTITY,
    "JURISDICTION": BfoCategory.REALIZABLE_ENTITY,
    "PRECEDENT": BfoCategory.REALIZABLE_ENTITY,
}

_BFO_NER_MAP: dict[str, BfoCategory] = {
    "PERSON": BfoCategory.QUALITY,
    "ORG": BfoCategory.IMMATERIAL_ENTITY,
    "GPE": BfoCategory.MATERIAL_ENTITY,
    "LOC": BfoCategory.MATERIAL_ENTITY,
    "PRODUCT": BfoCategory.QUALITY,
    "LAW": BfoCategory.REALIZABLE_ENTITY,
    "EVENT": BfoCategory.PROCESS,
    "DATE": BfoCategory.QUALITY,
    "TIME": BfoCategory.TEMPORAL_REGION,
    "MONEY": BfoCategory.QUALITY,
    "NORP": BfoCategory.IMMATERIAL_ENTITY,
}


def assign_bfo(node_type: NodeType, concept: str = "", text: str = "", entity_kind: str = "") -> BfoCategory:
    if node_type == NodeType.ENTITY and entity_kind:
        return _BFO_NER_MAP.get(entity_kind, BfoCategory.INFORMATION_CONTENT_ENTITY)
    if node_type in _BFO_NODETYPE_MAP:
        return _BFO_NODETYPE_MAP[node_type]
    if concept in _BFO_CONCEPT_MAP:
        # LOCATION has a special case: ENTITY → MATERIAL, non-ENTITY → IMMATERIAL
        if concept == "LOCATION" and node_type != NodeType.ENTITY:
            return BfoCategory.IMMATERIAL_ENTITY
        return _BFO_CONCEPT_MAP[concept]
    return BfoCategory.INFORMATION_CONTENT_ENTITY


# ── Step 4: Concept assignment (delegates to domain modules) ─────

def assign_concept(span: PropSpan, node_type: NodeType, vn_class: str = "") -> str:
    """Heuristically assign a concept label based on text + node type.

    Returns the canonical concept name (must be in the ConceptRegistry).
    Falls back to the NodeType name as a concept.
    """
    from cognitive_engine.extract.concepts import dispatch_concept
    return dispatch_concept(span.text.lower(), node_type, vn_class=vn_class)


# ── Helpers ──────────────────────────────────────────────────────

def split_parallel_clauses(spans: list[PropSpan]) -> list[PropSpan]:
    """Split PropSpans containing comma-separated or conjoined parallel clauses."""
    result: list[PropSpan] = []
    for span in spans:
        text = span.text
        splits = list(_CLAUSE_SPLIT_RE.finditer(text))
        if not splits:
            result.append(span)
            continue
        cursor = 0
        for m in splits:
            part = text[cursor : m.start()].strip(" ,.!?;:")
            if part:
                result.append(PropSpan(
                    start_char=span.start_char + cursor,
                    end_char=span.start_char + cursor + len(part),
                    text=part,
                    chunk_offsets=span.chunk_offsets,
                ))
            cursor = m.end()
        remaining = text[cursor:].strip(" ,.!?;:")
        if remaining:
            result.append(PropSpan(
                start_char=span.start_char + cursor,
                end_char=span.end_char,
                text=remaining,
                chunk_offsets=span.chunk_offsets,
            ))
    return result


@dataclass
class Relation:
    source_span: PropSpan
    target_span: PropSpan
    label: str


def _global_to_local(span: PropSpan, doc: object) -> Tuple[int, int]:
    chunk_start = doc.user_data.get("chunk_start_char", 0)
    return span.start_char - chunk_start, span.end_char - chunk_start


def _char_span_relaxed(
    doc: object, local_start: int, local_end: int,
) -> Optional[object]:
    span = doc.char_span(local_start, local_end, alignment_mode="contract")
    if span is not None:
        return span
    span = doc.char_span(local_start, local_end, alignment_mode="expand")
    return span


def _match_any(text: str, patterns: set[str]) -> bool:
    return any(p in text for p in patterns)


def map_types(
    spans: List[PropSpan],
    docs: List[object],
    relations: Optional[List[Relation]] = None,
) -> List[Tuple[PropSpan, NodeType, str]]:
    """Returns (span, node_type, concept_name) triples."""
    if relations is None:
        relations = []

    results: List[Tuple[PropSpan, NodeType, str]] = []
    for span in spans:
        doc = _find_doc_for_span(span, docs)
        if doc is None:
            logger.warning(
                "No doc found for span (%d, %d), defaulting to EVIDENCE",
                span.start_char, span.end_char,
            )
            results.append((span, NodeType.EVIDENCE, "EVIDENCE"))
            continue
        node_type = assign_type(span, doc, relations)
        concept = assign_concept(span, node_type)
        results.append((span, node_type, concept))
    return results


def _find_doc_for_span(
    span: PropSpan,
    docs: List[object],
) -> Optional[object]:
    return _match_span_to_doc(span.start_char, span.end_char, docs)
