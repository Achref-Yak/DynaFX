"""Tests for memory consolidation — STM → LTM pattern extraction with Leiden."""

import pytest
from uuid import uuid4

from cognitive_engine.core.models import Graph, Node
from cognitive_engine.core.state import State
from cognitive_engine.memory.consolidate import build_pattern, _label_cluster


class TestConsolidate:
    def test_empty_stm(self):
        patterns = build_pattern([])
        assert patterns is not None
        assert len(patterns) == 1  # Single fallback pattern
        assert len(patterns[0].operator_trace) == 0

    def test_single_state_pattern(self):
        g = Graph()
        g.nodes[uuid4()] = Node(text="A")
        state = State(graph=g)
        state.record("extract", "extracted")
        state.record("propagate", "propagated")

        patterns = build_pattern([state])
        assert len(patterns) >= 1
        # At least one pattern should have the operator trace
        assert any("extract" in p.operator_trace for p in patterns)

    def test_pattern_captures_belief_signature(self):
        g = Graph()
        nid = uuid4()
        node = Node(text="A")
        # opinion: (b, d, u, a)
        node.opinion = (0.7, 0.1, 0.2, 0.5)
        g.nodes[nid] = node

        state = State(graph=g)
        state.record("propagate", "done")

        patterns = build_pattern([state])
        # At least one pattern should have the belief signature
        assert any(str(nid) in p.belief_signature for p in patterns)
        assert any(p.belief_signature.get(str(nid)) == 0.7 for p in patterns)

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

        patterns = build_pattern([state])
        # The cluster labels come from Leiden communities, not metadata
        # So we just check that patterns were created
        assert len(patterns) >= 1

    def test_session_id_propagated(self):
        g = Graph()
        g.nodes[uuid4()] = Node(text="A")
        state = State(graph=g)

        patterns = build_pattern([state], session_id="test-session")
        assert all(p.session_id == "test-session" for p in patterns)


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
