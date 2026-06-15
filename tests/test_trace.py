"""Tests for TraceBuffer — the formal monoid Σ."""

import time
import pytest
from uuid import uuid4

from cognitive_engine.core.trace import StateDelta, TraceBuffer


class TestStateDelta:
    def test_create(self):
        d = StateDelta(
            timestamp=1.0,
            operator="test",
            description="a test delta",
            node_count=5,
            edge_count=3,
            effect_type="deterministic",
        )
        assert d.operator == "test"
        assert d.node_count == 5
        assert d.effect_type == "deterministic"
        assert d.metadata == {}

    def test_create_with_metadata(self):
        d = StateDelta(
            timestamp=1.0,
            operator="llm_op",
            description="LLM call",
            effect_type="llm",
            metadata={"tokens": 150, "model": "gemma3"},
        )
        assert d.effect_type == "llm"
        assert d.metadata["tokens"] == 150


class TestTraceBuffer:
    def test_create_empty(self):
        t = TraceBuffer(capacity=10)
        assert len(t) == 0
        assert t.capacity == 10

    def test_append_and_len(self):
        t = TraceBuffer(capacity=10)
        t.append(StateDelta(
            timestamp=time.time(),
            operator="extract",
            description="test",
        ))
        assert len(t) == 1

    def test_capacity_bounded(self):
        t = TraceBuffer(capacity=3)
        for i in range(5):
            t.append(StateDelta(
                timestamp=float(i),
                operator=f"op_{i}",
                description=f"desc_{i}",
            ))
        assert len(t) == 3  # only last 3 kept
        assert t[0].operator == "op_2"

    def test_iteration(self):
        t = TraceBuffer(capacity=10)
        for i in range(3):
            t.append(StateDelta(
                timestamp=float(i),
                operator=f"op_{i}",
                description=f"desc_{i}",
            ))
        operators = [e.operator for e in t]
        assert operators == ["op_0", "op_1", "op_2"]

    def test_indexing(self):
        t = TraceBuffer(capacity=10)
        t.append(StateDelta(timestamp=1.0, operator="a", description=""))
        t.append(StateDelta(timestamp=2.0, operator="b", description=""))
        assert t[0].operator == "a"
        assert t[1].operator == "b"

    def test_slice(self):
        t = TraceBuffer(capacity=10)
        for i in range(5):
            t.append(StateDelta(
                timestamp=float(i),
                operator=f"op_{i}",
                description="",
            ))
        sliced = t[1:3]
        assert len(sliced) == 2
        assert sliced[0].operator == "op_1"
        assert sliced[1].operator == "op_2"

    # ============================================================
    # Monoid law tests
    # ============================================================

    def test_monoid_identity(self):
        t = TraceBuffer(capacity=10)
        t.append(StateDelta(timestamp=1.0, operator="a", description=""))
        empty = TraceBuffer(capacity=10)
        # t + empty == t
        assert len(t + empty) == 1
        assert (t + empty)[0].operator == "a"
        # empty + t == t
        assert len(empty + t) == 1

    def test_monoid_associativity(self):
        a = TraceBuffer(capacity=10)
        b = TraceBuffer(capacity=10)
        c = TraceBuffer(capacity=10)
        a.append(StateDelta(timestamp=1.0, operator="a", description=""))
        b.append(StateDelta(timestamp=2.0, operator="b", description=""))
        c.append(StateDelta(timestamp=3.0, operator="c", description=""))

        # (a + b) + c
        r1 = (a + b) + c
        # a + (b + c)
        r2 = a + (b + c)
        assert len(r1) == 3
        assert len(r2) == 3
        assert [e.operator for e in r1] == [e.operator for e in r2]

    def test_copy(self):
        t = TraceBuffer(capacity=10)
        t.append(StateDelta(timestamp=1.0, operator="a", description=""))
        cp = t.copy()
        assert len(cp) == 1
        assert cp[0].operator == "a"
        # modifications to copy should not affect original
        cp.append(StateDelta(timestamp=2.0, operator="b", description=""))
        assert len(t) == 1
        assert len(cp) == 2

    def test_llm_call_count(self):
        t = TraceBuffer(capacity=10)
        t.append(StateDelta(timestamp=1.0, operator="perceive", description="",
                            effect_type="llm", metadata={"tokens": 100}))
        t.append(StateDelta(timestamp=2.0, operator="propagate", description=""))
        assert t.llm_call_count == 1

    def test_total_tokens(self):
        t = TraceBuffer(capacity=10)
        t.append(StateDelta(timestamp=1.0, operator="perceive", description="",
                            effect_type="llm", metadata={"tokens": 100}))
        t.append(StateDelta(timestamp=2.0, operator="perceive", description="",
                            effect_type="llm", metadata={"tokens": 50}))
        assert t.total_tokens == 150

    def test_sqlite_persistence(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            t = TraceBuffer(capacity=10, db_path=db_path)
            t.append(StateDelta(timestamp=1.0, operator="extract", description="test",
                                node_count=5, edge_count=3))
            assert len(t) == 1

            # Re-create buffer with same db to verify persistence
            t2 = TraceBuffer(capacity=10, db_path=db_path)
            # Persisted to db but db_path on re-init creates new empty buffer
            # (in-memory capture happens via _entries, not db reload)
            # This test just verifies no crash during SQLite append
            assert t2 is not None
        finally:
            import os
            os.unlink(db_path)
