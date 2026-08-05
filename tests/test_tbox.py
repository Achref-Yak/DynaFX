"""Tests for tbox/loader.py."""

import json
from pathlib import Path

import pytest

from dynafx.knowledge.loader import (
    TBox, load_tbox, validate_against_tbox,
    GENERAL_TBOX, BUILTIN_TBOXES,
)
from dynafx.knowledge.loader import CATEGORY_LEVELS


class TestTBox:
    def test_defaults(self):
        t = TBox()
        assert t.name == "general"
        assert t.node_types == {}
        assert t.edge_types == {}
        assert t.axioms == []
        assert t.valid_edges == []

    def test_fields(self):
        t = TBox(
            name="test",
            node_types={"CLAIM": 3, "EVIDENCE": 2},
            edge_types={"SUPPORTS": 0.85},
            axioms=[{"antecedents": ["a"], "consequent": "b"}],
        )
        assert t.name == "test"
        assert t.node_types["CLAIM"] == 3
        assert t.edge_types["SUPPORTS"] == 0.85
        assert len(t.axioms) == 1


class TestGeneralTBox:
    def test_name(self):
        assert GENERAL_TBOX.name == "general"

    def test_node_types_match_category_levels(self):
        for k, v in GENERAL_TBOX.node_types.items():
            assert k in CATEGORY_LEVELS
            assert v == CATEGORY_LEVELS[k]

    def test_has_edges(self):
        assert "SUPPORTS" in GENERAL_TBOX.edge_types
        assert "INFERS" in GENERAL_TBOX.edge_types
        assert "ATTACKS" in GENERAL_TBOX.edge_types
        assert GENERAL_TBOX.edge_types["SUPPORTS"] == 0.85
        assert GENERAL_TBOX.edge_types["INFERS"] == 0.9

    def test_has_axioms(self):
        assert len(GENERAL_TBOX.axioms) >= 2
        assert GENERAL_TBOX.axioms[0]["antecedents"] == ["type_EVIDENCE", "edge_SUPPORTS"]


class TestBuiltinTBoxes:
    def test_general_in_builtins(self):
        assert "general" in BUILTIN_TBOXES
        assert BUILTIN_TBOXES["general"] is GENERAL_TBOX


class TestLoadTBox:
    def test_load_general(self):
        t = load_tbox("general")
        assert t.name == "general"
        assert t is GENERAL_TBOX

    def test_load_unknown_falls_back(self):
        t = load_tbox("nonexistent_tbox")
        assert t.name == "general"
        assert t is GENERAL_TBOX

    def test_load_from_json_file(self, tmp_path: Path):
        data = {
            "name": "custom",
            "node_types": {"CUSTOM_TYPE": 3},
            "edge_types": {"CUSTOM_EDGE": 0.5},
            "axioms": [],
            "valid_edges": [],
        }
        path = tmp_path / "custom_tbox.json"
        path.write_text(json.dumps(data))
        t = load_tbox(str(path))
        assert t.name == "custom"
        assert t.node_types["CUSTOM_TYPE"] == 3

    def test_load_from_json_file_with_edges(self, tmp_path: Path):
        data = {
            "name": "domain_test",
            "node_types": {"A": 1, "B": 2},
            "edge_types": {"FLOW": 0.7},
            "axioms": [],
            "valid_edges": [("A", "FLOW", "B")],
        }
        path = tmp_path / "domain.json"
        path.write_text(json.dumps(data))
        t = load_tbox(str(path))
        assert t.name == "domain_test"
        assert len(t.valid_edges) == 1


class TestValidateAgainstTBox:
    def test_valid(self):
        assert validate_against_tbox("CLAIM", "SUPPORTS", GENERAL_TBOX) is True

    def test_invalid_node_type(self):
        assert validate_against_tbox("INVALID", "SUPPORTS", GENERAL_TBOX) is False

    def test_invalid_edge_type(self):
        assert validate_against_tbox("CLAIM", "INVALID_EDGE", GENERAL_TBOX) is False

    def test_both_invalid(self):
        assert validate_against_tbox("INVALID", "INVALID", GENERAL_TBOX) is False

    def test_case_insensitive(self):
        assert validate_against_tbox("claim", "supports", GENERAL_TBOX) is True

    def test_with_custom_tbox(self):
        t = TBox(
            name="custom",
            node_types={"SPECIAL": 1},
            edge_types={"SPECIAL_EDGE": 1.0},
        )
        assert validate_against_tbox("SPECIAL", "SPECIAL_EDGE", t) is True
        assert validate_against_tbox("CLAIM", "SUPPORTS", t) is False
