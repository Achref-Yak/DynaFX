"""Tests for MemoryStore (LTM ⊣ STM adjunction)."""

import pytest
from uuid import uuid4

from cognitive_engine.core.models import Graph, Node
from cognitive_engine.core.state import State
from cognitive_engine.memory.store import MemoryStore
from cognitive_engine.memory.models import LTMPattern


class TestMemoryStore:
    def test_create_store(self):
        store = MemoryStore(ltm_path=":memory:")
        assert store is not None
        assert len(store.stm) == 0
        store.close()

    def test_store_state_into_stm(self):
        store = MemoryStore(ltm_path=":memory:", stm_capacity=5)
        state = State(graph=Graph())
        store.store(state)
        assert len(store.stm) == 1
        store.close()

    def test_stm_eviction_to_ltm(self):
        store = MemoryStore(ltm_path=":memory:", stm_capacity=2)
        for i in range(3):
            g = Graph()
            g.nodes[uuid4()] = Node(text=f"node_{i}")
            state = State(graph=g)
            store.store(state)

        # STM has at most 2 entries (evicted oldest to LTM)
        assert len(store.stm) <= 2
        store.close()

    def test_consolidate_creates_ltm_pattern(self):
        store = MemoryStore(ltm_path=":memory:", stm_capacity=3)
        for i in range(4):
            g = Graph()
            g.nodes[uuid4()] = Node(text=f"node_{i}")
            state = State(graph=g)
            store.store(state)

        # At least 1 consolidation should have happened
        rows = store._conn.execute("SELECT COUNT(*) FROM ltm_patterns").fetchone()
        assert rows[0] >= 1
        store.close()

    def test_retrieve_empty_ltm_no_change(self):
        store = MemoryStore(ltm_path=":memory:")
        g = Graph()
        g.nodes[uuid4()] = Node(text="test")
        state = State(graph=g)
        result = store.retrieve(state, k=3)
        assert result.metadata.get("ltm_retrieved", 0) == 0
        assert len(result.graph.nodes) == 1
        store.close()

    def test_retrieve_with_matching_pattern(self):
        store = MemoryStore(ltm_path=":memory:", stm_capacity=3)

        # Seed LTM with a pattern
        g1 = Graph()
        nid = uuid4()
        g1.nodes[nid] = Node(text="shared_node")
        pattern = LTMPattern(
            id=uuid4(),
            graph_snapshot=g1,
            belief_signature={},
            operator_trace=["extract"],
            cluster_labels=[],
        )
        store._insert_pattern(pattern)

        # Retrieve with a similar graph
        g2 = Graph()
        g2.nodes[nid] = Node(text="shared_node")
        state = State(graph=g2)
        result = store.retrieve(state, k=3)
        assert result.metadata.get("ltm_retrieved", 0) >= 1
        store.close()


class TestLTMPattern:
    def test_create_pattern(self):
        g = Graph()
        g.nodes[uuid4()] = Node(text="A")
        pattern = LTMPattern(
            id=uuid4(),
            graph_snapshot=g,
            belief_signature={"node1": 0.8},
            operator_trace=["extract", "propagate"],
            cluster_labels=["Consensus Block: A"],
        )
        assert len(pattern.operator_trace) == 2
        assert "Consensus" in pattern.cluster_labels[0]

    def test_pattern_frequency(self):
        pattern = LTMPattern(
            id=uuid4(),
            graph_snapshot=Graph(),
            belief_signature={},
            operator_trace=[],
            cluster_labels=[],
            frequency=1,
        )
        assert pattern.frequency == 1
        pattern.frequency += 1
        assert pattern.frequency == 2
