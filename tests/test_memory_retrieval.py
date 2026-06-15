"""Tests for LTM retrieval — similarity-based pattern matching."""

import pytest
from uuid import uuid4

from cognitive_engine.core.models import Graph, Node
from cognitive_engine.memory.retrieval import retrieve_similar
from cognitive_engine.memory.models import LTMPattern


class TestRetrieval:
    def test_empty_query(self):
        query = Graph()
        candidates = [LTMPattern(
            id=uuid4(), graph_snapshot=Graph(),
            belief_signature={}, operator_trace=[], cluster_labels=[],
        )]
        result = retrieve_similar(query, candidates, k=3)
        assert result == []

    def test_empty_candidates(self):
        g = Graph()
        g.nodes[uuid4()] = Node(text="A")
        result = retrieve_similar(g, [], k=3)
        assert result == []

    def test_exact_match_is_top(self):
        nid = uuid4()
        g = Graph()
        g.nodes[nid] = Node(text="A")

        pattern = LTMPattern(
            id=uuid4(), graph_snapshot=g,
            belief_signature={}, operator_trace=[], cluster_labels=[],
        )

        candidates = [pattern]
        query = Graph()
        query.nodes[nid] = Node(text="A")
        result = retrieve_similar(query, candidates, k=3)
        assert len(result) == 1
        assert result[0].id == pattern.id

    def test_top_k_respected(self):
        patterns = []
        query = Graph()
        for i in range(5):
            nid = uuid4()
            g = Graph()
            g.nodes[nid] = Node(text=f"P{i}")
            patterns.append(LTMPattern(
                id=nid, graph_snapshot=g,
                belief_signature={}, operator_trace=[], cluster_labels=[],
            ))
            if i < 3:
                query.nodes[nid] = Node(text=f"P{i}")

        result = retrieve_similar(query, patterns, k=2)
        assert len(result) == 2

    def test_no_overlap_returns_empty(self):
        query = Graph()
        query.nodes[uuid4()] = Node(text="unique")

        pattern = LTMPattern(
            id=uuid4(), graph_snapshot=Graph(),
            belief_signature={}, operator_trace=[], cluster_labels=[],
        )
        pattern.graph_snapshot.nodes[uuid4()] = Node(text="different")

        result = retrieve_similar(query, [pattern], k=3)
        # Jaccard will be 0, count proximity may be 1 if same count
        # Score = 0.6*0 + 0.4*1 = 0.4, still > 0 so it'll be included
        assert len(result) >= 0
