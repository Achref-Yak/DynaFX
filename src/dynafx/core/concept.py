"""Controlled vocabulary layer — concept registry, temporal semantics, conflict resolution.

The ConceptRegistry sits between the assertion gate and the ABox,
defining for each concept type what it IS so the engine can decide
whether two facts about it should coexist or replace each other.

This is the Library Science / Ontology Engineering layer the essay
argues is missing from current agent memory architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Provenance(Enum):
    USER_STATED = 1.0
    TOOL_RETURNED = 0.85
    AGENT_OBSERVED = 0.8
    AGENT_INFERRED = 0.6
    COMPUTED = 0.4


class TemporalSemantics(Enum):
    MUTATE_IN_PLACE = "mutate_in_place"
    SUPERSEEDE_WITH_HISTORY = "supersede"
    APPEND_ONLY = "append_only"


class ConflictType(Enum):
    HISTORICAL_CORRECTION = "correction"
    COMPETING_CLAIMS = "competing"
    CATEGORY_ERROR = "category_error"
    NO_CONFLICT = "none"


@dataclass(frozen=True)
class ConceptDef:
    name: str
    parent: Optional[str] = None
    temporal_semantics: TemporalSemantics = TemporalSemantics.APPEND_ONLY
    cardinality: str = "multiple"
    provenance_weight: float = 0.5
    description: str = ""
    synonyms: frozenset[str] = field(default_factory=frozenset)


@dataclass
class Appraisal:
    uniqueness: float = 0.5
    replaceability: float = 0.5
    actionability: float = 0.5
    stability: float = 0.5
    sensitivity: float = 0.0


@dataclass
class FactEnvelope:
    concept: str
    value: str
    original: str = ""
    provenance: Provenance = Provenance.AGENT_OBSERVED
    appraisal: Optional[Appraisal] = None
    timestamp: float = 0.0
    confidence: float = 0.5


DEFAULT_CONCEPTS: dict[str, ConceptDef] = {
    # ── Parent concepts (for inheritance) ─────────────────────
    "IDENTITY": ConceptDef(
        "IDENTITY", parent=None,
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.5,
        description="Abstract parent for identity-bound facts",
    ),
    "MEASUREMENT": ConceptDef(
        "MEASUREMENT", parent=None,
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.5,
        description="Abstract parent for measurement facts",
    ),
    "TASTE": ConceptDef(
        "TASTE", parent=None,
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.5,
        description="Abstract parent for preference/style facts",
    ),
    "SPATIAL": ConceptDef(
        "SPATIAL", parent=None,
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.5,
        description="Abstract parent for location/spatial facts",
    ),
    "STATEMENT": ConceptDef(
        "STATEMENT", parent=None,
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.5,
        description="Abstract parent for statement/claim facts",
    ),
    "RULE": ConceptDef(
        "RULE", parent=None,
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.5,
        description="Abstract parent for rule/precedent facts",
    ),
    "EVIDENCE": ConceptDef(
        "EVIDENCE", parent=None,
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.5,
        description="Abstract parent for evidence/testimony facts",
    ),
    "DECISION": ConceptDef(
        "DECISION", parent=None,
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.5,
        description="Abstract parent for decision/ruling facts",
    ),
    "PROOF": ConceptDef(
        "PROOF", parent=None,
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.5,
        description="Abstract parent for proof/evidence facts",
    ),
    "IDEA": ConceptDef(
        "IDEA", parent=None,
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.5,
        description="Abstract parent for idea/hypothesis facts",
    ),
    "ACTION": ConceptDef(
        "ACTION", parent=None,
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.5,
        description="Abstract parent for action/decision facts",
    ),
    "DEPENDENCY": ConceptDef(
        "DEPENDENCY", parent=None,
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.5,
        description="Abstract parent for dependency/condition facts",
    ),
    "FACT": ConceptDef(
        "FACT", parent=None,
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.5,
        description="Abstract parent for fact/observation facts",
    ),
    "ARGUMENTATION": ConceptDef(
        "ARGUMENTATION", parent=None,
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.5,
        description="Abstract parent for argumentation facts",
    ),

    # ── Identity ──────────────────────────────────────────────
    "PERSON_NAME": ConceptDef(
        "PERSON_NAME", parent="IDENTITY",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.95,
        description="A personal name — only one current value per identity",
        synonyms=frozenset({"name", "called", "known as", "goes by", "my name is"}),
    ),
    "EMAIL": ConceptDef(
        "EMAIL", parent="IDENTITY",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.9,
        synonyms=frozenset({"email", "e-mail", "mail address"}),
    ),
    "PHONE": ConceptDef(
        "PHONE", parent="IDENTITY",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.85,
    ),

    # ── Measurements — append only, multiple values coexist ──
    "TEMPERATURE": ConceptDef(
        "TEMPERATURE", parent="MEASUREMENT",
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.7,
        synonyms=frozenset({"°C", "°F", "temperature", "measured"}),
    ),
    "BUDGET": ConceptDef(
        "BUDGET", parent="MEASUREMENT",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.85,
        synonyms=frozenset({"cost", "price", "budget", "spend", "costs", "$", "dollars"}),
    ),
    "DATE": ConceptDef(
        "DATE", parent="MEASUREMENT",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.85,
        synonyms=frozenset({"deadline", "due", "scheduled", "on"}),
    ),

    # ── Preferences — latest is current ───────────────────────
    "PREFERENCE": ConceptDef(
        "PREFERENCE", parent="TASTE",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.7,
        synonyms=frozenset({"prefer", "like", "want", "would like", "rather"}),
    ),
    "STYLE": ConceptDef(
        "STYLE", parent="TASTE",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.6,
        synonyms=frozenset({"style", "theme", "look", "dark mode", "night mode"}),
    ),

    # ── Location — both coexist ───────────────────────────────
    "LOCATION": ConceptDef(
        "LOCATION", parent="SPATIAL",
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.75,
        synonyms=frozenset({"at", "in", "located", "address", "place"}),
    ),

    # ── Legal domain ──────────────────────────────────────────
    "CLAIM": ConceptDef(
        "CLAIM", parent="STATEMENT",
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.7,
        synonyms=frozenset({"allegation", "assertion", "contends"}),
    ),
    "PRECEDENT": ConceptDef(
        "PRECEDENT", parent="RULE",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.9,
        synonyms=frozenset({"precedent", "case law", "holding"}),
    ),
    "TESTIMONY": ConceptDef(
        "TESTIMONY", parent="EVIDENCE",
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.6,
        synonyms=frozenset({"testified", "stated under oath", "deposed"}),
    ),
    "RULING": ConceptDef(
        "RULING", parent="DECISION",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=1.0,
        synonyms=frozenset({"ruling", "held", "decided", "ordered"}),
    ),
    "EVIDENCE": ConceptDef(
        "EVIDENCE", parent="PROOF",
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.75,
        synonyms=frozenset({"exhibit", "evidence", "proof", "document"}),
    ),

    # ── General reasoning ─────────────────────────────────────
    "HYPOTHESIS": ConceptDef(
        "HYPOTHESIS", parent="IDEA",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.5,
        synonyms=frozenset({"hypothesis", "theory", "guess", "suspect"}),
    ),
    "DECISION": ConceptDef(
        "DECISION", parent="ACTION",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.9,
        synonyms=frozenset({"decided", "chose", "selected", "commit"}),
    ),
    "ACTION": ConceptDef(
        "ACTION", parent="ACTION",
        temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
        cardinality="single", provenance_weight=0.7,
        synonyms=frozenset({"action", "do", "perform", "execute", "carry out"}),
    ),
    "CONDITION": ConceptDef(
        "CONDITION", parent="DEPENDENCY",
        temporal_semantics=TemporalSemantics.APPEND_ONLY,
        cardinality="multiple", provenance_weight=0.5,
        synonyms=frozenset({"if", "unless", "provided", "requires", "must"}),
    ),
    "OBSERVATION": ConceptDef(
        "OBSERVATION", parent="FACT",
        temporal_semantics=TemporalSemantics.MUTATE_IN_PLACE,
        cardinality="multiple", provenance_weight=0.5,
        synonyms=frozenset({"observed", "saw", "noted", "found"}),
    ),
}


class ConceptRegistry:
    """Controlled vocabulary registry — the spine of the knowledge layer.

    Every user-facing fact belongs to a concept. The concept's temporal
    semantics and cardinality determine how conflicting facts are resolved.
    """

    def __init__(self, concepts: Optional[dict[str, ConceptDef]] = None) -> None:
        self._concepts: dict[str, ConceptDef] = dict(concepts or DEFAULT_CONCEPTS)
        self._synonym_map: dict[str, str] = {}
        self._rebuild_synonym_map()

    def _rebuild_synonym_map(self) -> None:
        self._synonym_map.clear()
        for name, cd in self._concepts.items():
            key = name.lower()
            self._synonym_map[key] = name
            for syn in cd.synonyms:
                self._synonym_map[syn.lower()] = name

    def get(self, name: str) -> ConceptDef:
        raw = self._concepts.get(name)
        if raw is not None:
            return raw
        resolved = self._synonym_map.get(name.lower())
        if resolved is not None:
            return self._concepts[resolved]
        raise KeyError(f"Unknown concept: {name}")

    def has(self, name: str) -> bool:
        return name in self._concepts or name.lower() in self._synonym_map

    def normalize(self, surface: str) -> Optional[str]:
        """Resolve a surface form to its canonical concept name.

        >>> reg = ConceptRegistry()
        >>> reg.normalize("dark mode")
        'STYLE'
        >>> reg.normalize("my name is")
        'PERSON_NAME'
        """
        surface_lower = surface.lower().strip()
        if surface_lower in self._synonym_map:
            return self._synonym_map[surface_lower]
        for name, cd in self._concepts.items():
            if name.lower() == surface_lower:
                return name
        return None

    def resolve_temporal_semantics(self, concept_name: str) -> TemporalSemantics:
        """Walk parent chain for fallback temporal semantics.

        Returns the concept's own semantics if explicitly set (not APPEND_ONLY).
        Otherwise walks up the parent chain. Ultimate fallback is APPEND_ONLY.
        """
        try:
            cd = self.get(concept_name)
        except KeyError:
            return TemporalSemantics.APPEND_ONLY
        if cd.temporal_semantics != TemporalSemantics.APPEND_ONLY:
            return cd.temporal_semantics
        if cd.parent is not None:
            try:
                parent_cd = self.get(cd.parent)
                if parent_cd.temporal_semantics != TemporalSemantics.APPEND_ONLY:
                    return parent_cd.temporal_semantics
            except KeyError:
                pass
        return TemporalSemantics.APPEND_ONLY

    def resolve_conflict(self, concept_name: str) -> ConflictType:
        """Determine what kind of conflict two facts on this concept produce."""
        try:
            cd = self.get(concept_name)
        except KeyError:
            return ConflictType.NO_CONFLICT
        semantics = self.resolve_temporal_semantics(concept_name)
        if semantics in (TemporalSemantics.APPEND_ONLY, TemporalSemantics.MUTATE_IN_PLACE):
            return ConflictType.NO_CONFLICT
        if cd.cardinality == "single":
            return ConflictType.HISTORICAL_CORRECTION
        return ConflictType.COMPETING_CLAIMS

    def register(self, concept: ConceptDef) -> None:
        self._concepts[concept.name] = concept
        self._rebuild_synonym_map()

    def register_batch(self, concepts: list[ConceptDef]) -> None:
        for c in concepts:
            self._concepts[c.name] = c
        self._rebuild_synonym_map()

    @property
    def concept_names(self) -> list[str]:
        return list(self._concepts.keys())

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __repr__(self) -> str:
        return f"ConceptRegistry({len(self._concepts)} concepts)"

    def __copy__(self) -> ConceptRegistry:
        return ConceptRegistry(dict(self._concepts))


_default_registry: Optional[ConceptRegistry] = None


def default_registry() -> ConceptRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ConceptRegistry()
    return _default_registry


def reset_default_registry() -> None:
    global _default_registry
    _default_registry = None
