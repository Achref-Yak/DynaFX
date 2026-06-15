"""Tests for memory consolidation — STM → LTM pattern extraction."""

import pytest
from uuid import uuid4

from cognitive_engine.core.models import Graph, Node
from cognitive_engine.core.state import State
from cognitive_engine.memory.consolidate import build_pattern, _label_cluster


class TestConsolidate:
    def test_empty_stm(self):
        pattern = build_pattern([])
        assert pattern is not None
        assert len(pattern.operator_trace) == 0

    def test_single_state_pattern(self):
        g = Graph()
        g.nodes[uuid4()] = Node(text="A")
        state = State(graph=g)
        state.record("extract", "extracted")
        state.record("propagate", "propagated")

        pattern = build_pattern([state])
        assert len(pattern.operator_trace) == 2
        assert "extract" in pattern.operator_trace

    def test_pattern_captures_belief_signature(self):
        g = Graph()
        nid = uuid4()
        node = Node(text="A")
        # opinion: (b, d, u, a)
        node.opinion = (0.7, 0.1, 0.2, 0.5)
        g.nodes[nid] = node

        state = State(graph=g)
        state.record("propagate", "done")

        pattern = build_pattern([state])
        assert str(nid) in pattern.belief_signature
        assert pattern.belief_signature[str(nid)] == 0.7

    def test_cluster_labels_from_metadata(self):
        g = Graph()
        g.nodes[uuid4()] = Node(text="A")

        state = State(graph=g)
        state.metadata["emergence"] = {
            "clusters": [
                {"node_texts": ["A", "B", "C"], "cluster_type": "Consensus Block"},
            ]
        }
        state.record("emergence", "done")

        pattern = build_pattern([state])
        assert len(pattern.cluster_labels) >= 1
        assert "Consensus Block" in pattern.cluster_labels[0]


class TestLabelCluster:
    def test_label_with_texts(self):
        label = _label_cluster({
            "node_texts": ["R&D", "Quality", "Satisfaction", "Revenue"],
            "cluster_type": "Consensus Block",
        })
        assert "Consensus Block" in label
        assert "R&D" in label
        assert "+1 more" in label

    def test_label_no_texts(self):
        label = _label_cluster({
            "node_texts": [],
            "cluster_type": "Hub Interface",
        })
        assert "Hub Interface" in label

    def test_label_single_item(self):
        label = _label_cluster({
            "node_texts": ["Just one"],
            "cluster_type": "Ambiguous Region",
        })
        assert "Just one" in label
        assert "+" not in label
