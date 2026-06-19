"""Tests for BFO-inspired ontology layer."""

import pytest
from uuid import uuid4

from cognitive_engine.core.models import (
    BfoCategory, EDGE_BFO_CONSTRAINTS, EdgeType, Node, NodeType,
)
from cognitive_engine.core.diff import _snapshot_node
from cognitive_engine.extract.types import assign_bfo


class TestBfoCategoryEnum:
    def test_all_categories_present(self):
        names = {c.name for c in BfoCategory}
        expected = {
            "MATERIAL_ENTITY", "IMMATERIAL_ENTITY", "QUALITY",
            "REALIZABLE_ENTITY", "PROCESS", "TEMPORAL_REGION",
            "INFORMATION_CONTENT_ENTITY",
        }
        assert names == expected

    def test_values_are_auto(self):
        assert len(BfoCategory) == 7
        assert all(isinstance(c.value, int) for c in BfoCategory)


class TestAssignBfo:
    def test_identity_concepts_are_quality(self):
        for concept in ("PERSON_NAME", "EMAIL", "PHONE"):
            assert assign_bfo(NodeType.CLAIM, concept) == BfoCategory.QUALITY

    def test_measurement_concepts_are_quality(self):
        for concept in ("TEMPERATURE", "BUDGET", "DATE"):
            assert assign_bfo(NodeType.CLAIM, concept) == BfoCategory.QUALITY

    def test_taste_concepts_are_quality(self):
        for concept in ("PREFERENCE", "STYLE"):
            assert assign_bfo(NodeType.CLAIM, concept) == BfoCategory.QUALITY

    def test_location_entity_is_material(self):
        assert assign_bfo(NodeType.ENTITY, "LOCATION") == BfoCategory.MATERIAL_ENTITY

    def test_location_nonentity_is_immaterial(self):
        assert assign_bfo(NodeType.CLAIM, "LOCATION") == BfoCategory.IMMATERIAL_ENTITY

    def test_event_and_action_are_process(self):
        assert assign_bfo(NodeType.EVENT, "") == BfoCategory.PROCESS
        assert assign_bfo(NodeType.ACTION, "") == BfoCategory.PROCESS

    def test_realizable_concepts(self):
        for concept in ("CONDITION", "RULE", "STATUTE", "CONTRACT_TERM"):
            assert assign_bfo(NodeType.CLAIM, concept) == BfoCategory.REALIZABLE_ENTITY

    def test_node_types_are_ice(self):
        ice_types = {
            NodeType.CLAIM, NodeType.EVIDENCE,
            NodeType.HYPOTHESIS, NodeType.OBSERVATION, NodeType.DECISION,
            NodeType.AXIOM, NodeType.COUNTERCLAIM, NodeType.JUSTIFICATION,
            NodeType.FALLACY, NodeType.DOCUMENT, NodeType.CONCEPT,
        }
        for nt in ice_types:
            assert assign_bfo(nt, "") == BfoCategory.INFORMATION_CONTENT_ENTITY

    def test_fallback_is_ice(self):
        assert assign_bfo(NodeType.ENTITY, "UNKNOWN") == BfoCategory.INFORMATION_CONTENT_ENTITY


class TestEdgeBfoConstraints:
    def test_ice_edges_require_ice_both_sides(self):
        for etype in (EdgeType.INFERS, EdgeType.SUPPORTS, EdgeType.REBUTS,
                      EdgeType.ATTACKS, EdgeType.CONTRADICTS):
            src_allowed, tgt_allowed = EDGE_BFO_CONSTRAINTS[etype]
            assert src_allowed == frozenset({BfoCategory.INFORMATION_CONTENT_ENTITY})
            assert tgt_allowed == frozenset({BfoCategory.INFORMATION_CONTENT_ENTITY})

    def test_process_edges_allow_process_or_ice(self):
        for etype in (EdgeType.CAUSES, EdgeType.TEMPORAL):
            src_allowed, tgt_allowed = EDGE_BFO_CONSTRAINTS[etype]
            assert BfoCategory.PROCESS in src_allowed
            assert BfoCategory.INFORMATION_CONTENT_ENTITY in src_allowed

    def test_part_of_requires_material_or_immaterial(self):
        src_allowed, tgt_allowed = EDGE_BFO_CONSTRAINTS[EdgeType.PART_OF]
        assert BfoCategory.MATERIAL_ENTITY in src_allowed
        assert BfoCategory.IMMATERIAL_ENTITY in src_allowed
        assert BfoCategory.INFORMATION_CONTENT_ENTITY not in src_allowed

    def test_qualifies_domain(self):
        src_allowed, tgt_allowed = EDGE_BFO_CONSTRAINTS[EdgeType.QUALIFIES]
        assert BfoCategory.REALIZABLE_ENTITY in src_allowed
        assert BfoCategory.INFORMATION_CONTENT_ENTITY in tgt_allowed
        assert BfoCategory.INFORMATION_CONTENT_ENTITY not in src_allowed or len(src_allowed) > 1


class TestBfoOnNode:
    def test_node_defaults_to_none(self):
        n = Node(text="test")
        assert n.bfo_category is None

    def test_node_bfo_round_trip(self):
        n = Node(text="test", bfo_category=BfoCategory.QUALITY)
        assert n.bfo_category == BfoCategory.QUALITY


class TestBfoInSnapshot:
    def test_snapshot_includes_bfo(self):
        n = Node(text="hello", bfo_category=BfoCategory.QUALITY)
        snap = _snapshot_node(n)
        assert snap.bfo_category == "QUALITY"

    def test_snapshot_none_bfo(self):
        n = Node(text="hello")
        snap = _snapshot_node(n)
        assert snap.bfo_category is None
