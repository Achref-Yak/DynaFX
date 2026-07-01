"""Tests for Transaction Layer (kb/transactions.py)."""

import time
from dataclasses import dataclass

from dynafx.knowledge.execution import ExecutionRecord, ExecutionStore
from dynafx.knowledge.model import (
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
)
from dynafx.knowledge.store import TripleStore
from dynafx.knowledge.transactions import Transaction, TransactionStore


def make_store() -> TripleStore:
    return TripleStore()


# ── TransactionStore tests ──────────────────────────────────────────


class TestTransactionStore:
    def test_record_transaction(self):
        st = make_store()
        txs = TransactionStore(st)
        tx = txs.record("TestEvent", {"key": "value"}, source="test")
        assert tx.event_type == "TestEvent"
        assert tx.payload == {"key": "value"}
        assert tx.source == "test"
        assert tx.id is not None
        assert tx.timestamp > 0
        assert txs.total_count == 1

    def test_record_stores_rdf_triples(self):
        st = make_store()
        txs = TransactionStore(st)
        tx = txs.record("ContainerDelayed", {"container_id": "C-123"}, source="ERP")
        # Should have 4 base triples + 1 payload triple
        all_triples = list(st.all_triples())
        assert len(all_triples) >= 5

        # Should have a type triple
        found_type = False
        for t in all_triples:
            if "type" in str(t.predicate):
                found_type = True
                break
        assert found_type

    def test_query_by_type(self):
        st = make_store()
        txs = TransactionStore(st)
        txs.record("TypeA", {"v": 1})
        txs.record("TypeB", {"v": 2})
        txs.record("TypeA", {"v": 3})
        results = txs.query(event_type="TypeA")
        assert len(results) == 2
        assert all(t.event_type == "TypeA" for t in results)

    def test_query_by_source(self):
        st = make_store()
        txs = TransactionStore(st)
        txs.record("A", {"v": 1}, source="erp")
        txs.record("B", {"v": 2}, source="iot")
        txs.record("C", {"v": 3}, source="erp")
        results = txs.query(source="erp")
        assert len(results) == 2

    def test_query_by_time_range(self):
        st = make_store()
        txs = TransactionStore(st)
        now = time.time()
        t1 = txs.record("Early", {"v": 1}, timestamp=now - 100)
        t2 = txs.record("Middle", {"v": 2}, timestamp=now - 50)
        t3 = txs.record("Late", {"v": 3}, timestamp=now - 1)
        results = txs.query(t_start=now - 60, t_end=now - 0.5)
        assert len(results) == 2  # "Middle" at now-50, "Late" at now-1
        assert results[0].event_type == "Late"  # newest first

    def test_query_combined_filters(self):
        st = make_store()
        txs = TransactionStore(st)
        now = time.time()
        txs.record("A", {"v": 1}, source="erp", timestamp=now - 100)
        txs.record("A", {"v": 2}, source="iot", timestamp=now - 50)
        txs.record("B", {"v": 3}, source="erp", timestamp=now - 1)
        results = txs.query(event_type="A", source="erp")
        assert len(results) == 1

    def test_recent_returns_n_most_recent(self):
        st = make_store()
        txs = TransactionStore(st)
        for i in range(10):
            txs.record(f"Event{i}", {"i": i})
        results = txs.recent(n=3)
        assert len(results) == 3
        # Most recent first
        for i in range(len(results) - 1):
            assert results[i].timestamp >= results[i + 1].timestamp

    def test_recent_with_n_above_count(self):
        st = make_store()
        txs = TransactionStore(st)
        txs.record("A", {"v": 1})
        results = txs.recent(n=100)
        assert len(results) == 1

    def test_count_by_type(self):
        st = make_store()
        txs = TransactionStore(st)
        now = time.time()
        txs.record("A", {"v": 1}, timestamp=now - 100)
        txs.record("A", {"v": 2}, timestamp=now - 50)
        txs.record("B", {"v": 3}, timestamp=now - 1)
        assert txs.count_by_type("A") == 2
        assert txs.count_by_type("B") == 1
        assert txs.count_by_type("C") == 0

    def test_count_by_type_since(self):
        st = make_store()
        txs = TransactionStore(st)
        now = time.time()
        txs.record("A", {"v": 1}, timestamp=now - 100)
        txs.record("A", {"v": 2}, timestamp=now - 50)
        txs.record("A", {"v": 3}, timestamp=now - 1)
        assert txs.count_by_type("A", since=now - 60) == 2
        assert txs.count_by_type("A", since=now - 10) == 1

    def test_count_by_source(self):
        st = make_store()
        txs = TransactionStore(st)
        txs.record("A", {"v": 1}, source="erp")
        txs.record("B", {"v": 2}, source="iot")
        assert txs.count_by_source("erp") == 1
        assert txs.count_by_source("iot") == 1
        assert txs.count_by_source("manual") == 0

    def test_query_empty_store(self):
        st = make_store()
        txs = TransactionStore(st)
        assert txs.query() == []
        assert txs.recent() == []

    def test_query_newest_first(self):
        st = make_store()
        txs = TransactionStore(st)
        now = time.time()
        t1 = txs.record("A", {"v": 1}, timestamp=now - 100)
        t2 = txs.record("A", {"v": 2}, timestamp=now - 50)
        t3 = txs.record("A", {"v": 3}, timestamp=now - 1)
        results = txs.query(event_type="A")
        assert results[0].timestamp >= results[1].timestamp >= results[2].timestamp

    def test_transaction_suppresses_callbacks(self):
        st = make_store()
        txs = TransactionStore(st)
        callback_fired = []

        def on_add(triple, graph):
            callback_fired.append((triple, graph))

        st.on_add(on_add)
        tx = txs.record("TestEvent", {"k": "v"})
        assert len(callback_fired) == 0  # callbacks suppressed during tx recording
        # Triples ARE in the store despite suppressed callbacks
        assert tx.id is not None
        assert txs.total_count == 1


# ── ExecutionStore tests ────────────────────────────────────────────


class TestExecutionStore:
    def test_record_execution(self):
        st = make_store()
        es = ExecutionStore(st)
        rec = es.record("test-rule", "log", {"var": "val"}, {"result": "ok"})
        assert rec.rule_name == "test-rule"
        assert rec.action_type == "log"
        assert rec.bindings == {"var": "val"}
        assert rec.output == {"result": "ok"}
        assert rec.status == "executed"
        assert es.total_count == 1

    def test_record_stores_rdf(self):
        st = make_store()
        es = ExecutionStore(st)
        es.record("test-rule", "simulate", {}, {"cash": 100.0})
        all_triples = list(st.all_triples())
        assert len(all_triples) >= 5

    def test_get_by_id(self):
        st = make_store()
        es = ExecutionStore(st)
        rec = es.record("rule1", "log", {}, {})
        fetched = es.get(rec.action_id)
        assert fetched is not None
        assert fetched.action_id == rec.action_id

    def test_get_nonexistent(self):
        st = make_store()
        es = ExecutionStore(st)
        assert es.get("nonexistent") is None

    def test_by_rule(self):
        st = make_store()
        es = ExecutionStore(st)
        es.record("rule-a", "log", {}, {})
        es.record("rule-b", "log", {}, {})
        es.record("rule-a", "simulate", {}, {})
        rule_a = es.by_rule("rule-a")
        assert len(rule_a) == 2
        rule_b = es.by_rule("rule-b")
        assert len(rule_b) == 1

    def test_by_type(self):
        st = make_store()
        es = ExecutionStore(st)
        es.record("r1", "log", {}, {})
        es.record("r2", "simulate", {}, {})
        es.record("r3", "log", {}, {})
        logs = es.by_type("log")
        assert len(logs) == 2
        sims = es.by_type("simulate")
        assert len(sims) == 1

    def test_recent(self):
        st = make_store()
        es = ExecutionStore(st)
        for i in range(10):
            es.record(f"rule-{i}", "log", {}, {"i": i})
        recent = es.recent(n=3)
        assert len(recent) == 3

    def test_last_execution(self):
        st = make_store()
        es = ExecutionStore(st)
        rec1 = es.record("rule-x", "log", {"v": 1}, {})
        rec2 = es.record("rule-x", "simulate", {"v": 2}, {})
        last = es.last_execution("rule-x")
        assert last is not None
        assert last.action_type == "simulate"

    def test_last_execution_no_executions(self):
        st = make_store()
        es = ExecutionStore(st)
        assert es.last_execution("nonexistent") is None

    def test_failed_status(self):
        st = make_store()
        es = ExecutionStore(st)
        rec = es.record("rule", "simulate", {}, {}, status="failed", message="Error")
        assert rec.status == "failed"
        assert rec.message == "Error"

    def test_empty_store(self):
        st = make_store()
        es = ExecutionStore(st)
        assert es.total_count == 0
        assert es.recent() == []
        assert es.by_rule("any") == []
        assert es.by_type("any") == []
