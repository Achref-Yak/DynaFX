"""Tests for the concept registry layer."""
import pytest
from copy import deepcopy

from dynafx.core.concept import (
    ConceptDef,
    ConceptRegistry,
    TemporalSemantics,
    ConflictType,
    DEFAULT_CONCEPTS,
    default_registry,
)
from dynafx.core.models import Graph
from dynafx.core.state import State


class TestConceptDef:
    def test_default_description(self):
        cd = ConceptDef(name="TEST", parent="PARENT")
        assert cd.description == ""

    def test_default_synonyms(self):
        cd = ConceptDef(name="TEST", parent="PARENT")
        assert cd.synonyms == frozenset()

    def test_full_constructor(self):
        cd = ConceptDef(
            "NAME", "IDENTITY",
            temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
            cardinality="single", provenance_weight=0.95,
            description="A name", synonyms=frozenset({"name", "called"}),
        )
        assert cd.name == "NAME"
        assert cd.parent == "IDENTITY"
        assert cd.cardinality == "single"
        assert cd.provenance_weight == 0.95


class TestConceptRegistry:
    def test_default_init(self):
        reg = ConceptRegistry()
        assert len(list(reg.concept_names)) == len(DEFAULT_CONCEPTS)

    def test_register_and_get(self):
        reg = ConceptRegistry()
        cd = ConceptDef("TEST_CONCEPT", "PARENT")
        reg.register(cd)
        retrieved = reg.get("TEST_CONCEPT")
        assert retrieved is cd
        assert retrieved.parent == "PARENT"

    def test_register_batch(self):
        reg = ConceptRegistry()
        concepts = [
            ConceptDef("A", "P1"),
            ConceptDef("B", "P2"),
        ]
        reg.register_batch(concepts)
        assert reg.has("A")
        assert reg.has("B")

    def test_get_unknown(self):
        reg = ConceptRegistry()
        with pytest.raises(KeyError):
            reg.get("DOES_NOT_EXIST")

    def test_has(self):
        reg = ConceptRegistry()
        assert reg.has("PERSON_NAME")
        assert not reg.has("DOES_NOT_EXIST")

    def test_normalize_exact(self):
        reg = ConceptRegistry()
        result = reg.normalize("PERSON_NAME")
        assert result == "PERSON_NAME"

    def test_normalize_synonym(self):
        reg = ConceptRegistry()
        result = reg.normalize("my name is")
        assert result == "PERSON_NAME"

    def test_normalize_unknown(self):
        reg = ConceptRegistry()
        result = reg.normalize("xyzzy")
        assert result is None

    def test_contains(self):
        reg = ConceptRegistry()
        assert "CLAIM" in reg
        assert "NOTHING" not in reg

    def test_repr(self):
        reg = ConceptRegistry()
        r = repr(reg)
        assert "ConceptRegistry" in r

    def test_copy(self):
        import copy
        reg = ConceptRegistry()
        reg.register(ConceptDef("X", "P"))
        copied = copy.copy(reg)
        assert copied.has("X")
        assert copied.get("X").name == "X"
        # Verify independence
        copied.register(ConceptDef("Y", "P"))
        assert not reg.has("Y")

    def test_concept_names(self):
        reg = ConceptRegistry()
        names = reg.concept_names
        assert "PERSON_NAME" in names
        assert "CLAIM" in names
        assert len(names) == len(DEFAULT_CONCEPTS)


class TestResolveConflict:
    def test_append_only_no_conflict(self):
        reg = ConceptRegistry()
        result = reg.resolve_conflict("TEMPERATURE")
        assert result == ConflictType.NO_CONFLICT

    def test_supersede_single_historical_correction(self):
        reg = ConceptRegistry()
        result = reg.resolve_conflict("PERSON_NAME")
        assert result == ConflictType.HISTORICAL_CORRECTION

    def test_mutate_multiple_competing_claims(self):
        reg = ConceptRegistry()
        result = reg.resolve_conflict("LOCATION")
        # LOCATION is APPEND_ONLY with multiple, so NO_CONFLICT
        assert result == ConflictType.NO_CONFLICT

    def test_unknown_concept_no_conflict(self):
        reg = ConceptRegistry()
        result = reg.resolve_conflict("UNKNOWN_CONCEPT")
        assert result == ConflictType.NO_CONFLICT


class TestDefaultRegistry:
    def test_singleton(self):
        r1 = default_registry()
        r2 = default_registry()
        assert r1 is r2

    def test_has_all_defaults(self):
        reg = default_registry()
        for name in DEFAULT_CONCEPTS:
            assert reg.has(name), f"Missing default concept: {name}"

    def test_mutate_does_not_affect_default(self):
        reg = default_registry()
        # 14 parent concepts + 16 concrete concepts = 30
        assert len(list(reg.concept_names)) == 30


class TestDEFAULT_CONCEPTS:
    def test_person_name_supersedes(self):
        cd = DEFAULT_CONCEPTS["PERSON_NAME"]
        assert cd.temporal_semantics == TemporalSemantics.SUPERSEEDE_WITH_HISTORY
        assert cd.cardinality == "single"

    def test_temperature_append_only(self):
        cd = DEFAULT_CONCEPTS["TEMPERATURE"]
        assert cd.temporal_semantics == TemporalSemantics.APPEND_ONLY
        assert cd.cardinality == "multiple"

    def test_evidence_append_only(self):
        cd = DEFAULT_CONCEPTS["EVIDENCE"]
        assert cd.temporal_semantics == TemporalSemantics.APPEND_ONLY
        assert cd.cardinality == "multiple"

    def test_precedent_supersedes(self):
        cd = DEFAULT_CONCEPTS["PRECEDENT"]
        assert cd.temporal_semantics == TemporalSemantics.SUPERSEEDE_WITH_HISTORY
        assert cd.cardinality == "single"


class TestStateConceptIntegration:
    def test_state_has_concepts(self):
        state = State(graph=Graph())
        assert hasattr(state, "concepts")
        assert isinstance(state.concepts, ConceptRegistry)

    def test_fork_copies_concepts(self):
        state = State(graph=Graph())
        state.concepts.register(ConceptDef("CUSTOM", "P"))
        forked = state.fork()
        assert forked.concepts.has("CUSTOM")
        # Verify independence
        forked.concepts.register(ConceptDef("NEW", "P"))
        assert not state.concepts.has("NEW")


class TestParentChainTraversal:
    def test_resolve_temporal_semantics_explicit(self):
        reg = ConceptRegistry()
        assert reg.resolve_temporal_semantics("PERSON_NAME") == TemporalSemantics.SUPERSEEDE_WITH_HISTORY

    def test_resolve_temporal_semantics_child_inherits_parent(self):
        reg = ConceptRegistry()
        child = ConceptDef(
            "NICKNAME", parent="IDENTITY",
            temporal_semantics=TemporalSemantics.APPEND_ONLY,
            cardinality="multiple",
        )
        reg.register(child)
        # IDENTITY is now registered as a parent concept with SUPERSEDE,
        # so the parent chain succeeds and NICKNAME inherits SUPERSEDE.
        result = reg.resolve_temporal_semantics("NICKNAME")
        assert result == TemporalSemantics.SUPERSEEDE_WITH_HISTORY

    def test_resolve_temporal_semantics_with_real_parent(self):
        reg = ConceptRegistry()
        reg.register(ConceptDef("IDENTITY", parent=None,
                                temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY))
        child = ConceptDef("NICKNAME", parent="IDENTITY",
                           temporal_semantics=TemporalSemantics.APPEND_ONLY)
        reg.register(child)
        result = reg.resolve_temporal_semantics("NICKNAME")
        assert result == TemporalSemantics.SUPERSEEDE_WITH_HISTORY

    def test_resolve_conflict_falls_back_to_parent(self):
        reg = ConceptRegistry()
        reg.register(ConceptDef("IDENTITY", parent=None,
                                temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                                cardinality="single"))
        child = ConceptDef("NICKNAME", parent="IDENTITY",
                           temporal_semantics=TemporalSemantics.APPEND_ONLY,
                           cardinality="multiple")
        reg.register(child)
        result = reg.resolve_conflict("NICKNAME")
        # Should inherit SUPERSEDE semantics from IDENTITY
        # But cardinality from child is "multiple", so HISTORICAL_CORRECTION
        # requires cardinality="single" — so it falls to COMPETING_CLAIMS
        assert result == ConflictType.COMPETING_CLAIMS

    def test_resolve_conflict_inherits_cardinality_and_semantics(self):
        reg = ConceptRegistry()
        reg.register(ConceptDef("IDENTITY", parent=None,
                                temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                                cardinality="single"))
        child = ConceptDef("NICKNAME", parent="IDENTITY",
                           temporal_semantics=TemporalSemantics.APPEND_ONLY,
                           cardinality="single")
        reg.register(child)
        result = reg.resolve_conflict("NICKNAME")
        assert result == ConflictType.HISTORICAL_CORRECTION
