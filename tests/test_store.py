from pathlib import Path

import pytest

from cognitive_engine.core.models import (
    Graph, Node, Edge, NodeType, EdgeType, EvidenceCounts,
)
from cognitive_engine.reason.store import CorpusStore


@pytest.fixture
def store(tmp_path: Path) -> CorpusStore:
    db = tmp_path / "test_evidence.db"
    s = CorpusStore(db)
    yield s
    s.close()


@pytest.fixture
def sample_graph() -> Graph:
    a = Node(type=NodeType.EVIDENCE, text="data", opinion=(0.8, 0.1, 0.1, 0.5))
    b = Node(type=NodeType.CLAIM, text="conclusion", opinion=(0.6, 0.2, 0.2, 0.5))
    g = Graph(
        nodes={a.id: a, b.id: b},
        edges=[
            Edge(
                source_id=a.id, target_id=b.id, type=EdgeType.INFERS,
                warrant=((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5)),
            ),
        ],
    )
    return g


class TestCorpusStore:
    def test_empty_store(self, store: CorpusStore):
        assert store.source_count == 0
        assert store.list_sources() == []
        assert store.accumulate_node_counts() == {}
        assert store.accumulate_edge_warrants() == {}

    def test_store_and_count_sources(self, store: CorpusStore, sample_graph: Graph):
        store.store_graph(sample_graph, "doc1", filename="doc1.txt")
        assert store.source_count == 1
        sources = store.list_sources()
        assert len(sources) == 1
        assert sources[0]["id"] == "doc1"
        assert sources[0]["filename"] == "doc1.txt"

    def test_accumulate_node_counts(self, store: CorpusStore, sample_graph: Graph):
        store.store_graph(sample_graph, "doc1", filename="doc1.txt")
        counts = store.accumulate_node_counts()
        assert "EVIDENCE" in counts
        assert "CLAIM" in counts
        assert counts["EVIDENCE"].positive >= 1
        assert counts["CLAIM"].positive >= 1

    def test_accumulate_edge_warrants(self, store: CorpusStore, sample_graph: Graph):
        store.store_graph(sample_graph, "doc1", filename="doc1.txt")
        warrants = store.accumulate_edge_warrants()
        assert "INFERS" in warrants
        assert len(warrants["INFERS"]) == 1
        (b1, d1, u1, a1), (b2, d2, u2, a2) = warrants["INFERS"][0]
        assert b1 == 0.9
        assert b2 == 0.0

    def test_multiple_sources_accumulate(self, store: CorpusStore, sample_graph: Graph):
        store.store_graph(sample_graph, "doc1", filename="doc1.txt")
        store.store_graph(sample_graph, "doc2", filename="doc2.txt")
        assert store.source_count == 2
        counts = store.accumulate_node_counts()
        assert counts["EVIDENCE"].positive >= 2
        warrants = store.accumulate_edge_warrants()
        assert len(warrants["INFERS"]) == 2

    def test_remove_source(self, store: CorpusStore, sample_graph: Graph):
        store.store_graph(sample_graph, "doc1")
        store.store_graph(sample_graph, "doc2")
        assert store.source_count == 2
        store.remove_source("doc1")
        assert store.source_count == 1
        assert store.list_sources()[0]["id"] == "doc2"

    def test_clear(self, store: CorpusStore, sample_graph: Graph):
        store.store_graph(sample_graph, "doc1")
        store.store_graph(sample_graph, "doc2")
        store.clear()
        assert store.source_count == 0
        assert store.list_sources() == []
        assert store.accumulate_node_counts() == {}

    def test_duplicate_source_replaced(self, store: CorpusStore, sample_graph: Graph):
        store.store_graph(sample_graph, "doc1", filename="first.txt")
        store.store_graph(sample_graph, "doc1", filename="second.txt")
        assert store.source_count == 1
        assert store.list_sources()[0]["filename"] == "second.txt"

    def test_to_priors(self, store: CorpusStore, sample_graph: Graph):
        store.store_graph(sample_graph, "doc1", filename="doc1.txt")
        priors = store.to_priors()
        opinions = priors.default_opinions
        assert "empirical_pattern" in opinions
        assert "consensus_principle" in opinions
        assert "total_ignorance" in opinions
        b, d, u, a = opinions["empirical_pattern"]
        assert abs(b + d + u - 1.0) < 1e-9

    def test_to_priors_empty_store(self, store: CorpusStore):
        priors = store.to_priors()
        opinions = priors.default_opinions
        assert opinions["total_ignorance"] == (0.0, 0.0, 1.0, 0.5)
        assert "empirical_pattern" in opinions

    def test_store_without_warrant(self, tmp_path: Path):
        db = tmp_path / "no_warrant.db"
        s = CorpusStore(db)
        a = Node(type=NodeType.EVIDENCE, text="x", opinion=(0.5, 0.2, 0.3, 0.5))
        b = Node(type=NodeType.CLAIM, text="y", opinion=(0.4, 0.3, 0.3, 0.5))
        g = Graph(
            nodes={a.id: a, b.id: b},
            edges=[Edge(source_id=a.id, target_id=b.id, type=EdgeType.SUPPORTS)],
        )
        s.store_graph(g, "nowarrant")
        assert s.source_count == 1
        assert s.accumulate_edge_warrants() == {}
        s.close()

    def test_context_manager(self, tmp_path: Path):
        db = tmp_path / "ctx.db"
        with CorpusStore(db) as s:
            assert s.source_count == 0
        import sqlite3
        with pytest.raises(sqlite3.ProgrammingError):
            s.source_count  # conn should be closed

    def test_database_persistence(self, tmp_path: Path):
        db = tmp_path / "persist.db"
        a = Node(type=NodeType.EVIDENCE, text="x", opinion=(0.9, 0.05, 0.05, 0.5))
        g = Graph(nodes={a.id: a}, edges=[])
        with CorpusStore(db) as s:
            s.store_graph(g, "persist_doc")
            assert s.source_count == 1

        with CorpusStore(db) as s:
            assert s.source_count == 1
            assert s.list_sources()[0]["id"] == "persist_doc"

    def test_to_priors_includes_edge_warrants(self, store: CorpusStore, sample_graph: Graph):
        store.store_graph(sample_graph, "doc1")
        priors = store.to_priors()
        warrants = priors.edge_warrants
        assert "INFERS" in warrants
        (b1, d1, u1, a1), (b2, d2, u2, a2) = warrants["INFERS"]
        assert b1 == 0.9
