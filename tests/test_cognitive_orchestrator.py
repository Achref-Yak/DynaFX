"""Integration tests for CognitiveOrchestrator (event → rule → action loop)."""

import pytest

from dynafx.bridge import CognitiveOrchestrator
from dynafx.knowledge.inference import InferencePattern
from dynafx.knowledge.model import (
    NamedNode,
    Triple,
)
from dynafx.knowledge.production import (
    LogAction,
    ProductionRule,
    TripleCondition,
)
from dynafx.knowledge.store import TripleStore

S = NamedNode("http://example.org/s")
P = NamedNode("http://example.org/p")
O = NamedNode("http://example.org/o")


@pytest.fixture
def store() -> TripleStore:
    return TripleStore()


@pytest.fixture
def orchestrator(store: TripleStore) -> CognitiveOrchestrator:
    orb = CognitiveOrchestrator(store)
    rule = ProductionRule(
        name="match_test",
        body=[TripleCondition(InferencePattern(S, P, O))],
        head=[LogAction("Matched event")],
    )
    orb.add_rule(rule)
    return orb


def test_orchestrator_fires_rule_on_triple(orchestrator: CognitiveOrchestrator, store: TripleStore):
    """A matching triple triggers the rule and records the execution."""
    orchestrator.start()
    store.add(Triple(S, P, O))

    assert orchestrator.rule_engine._fired_count.get("match_test", 0) == 1
    rec = orchestrator.exec_store.last_execution("match_test")
    assert rec is not None
    assert rec.status == "executed"
    assert rec.action_type == "log"


def test_orchestrator_does_not_fire_when_absent(orchestrator: CognitiveOrchestrator, store: TripleStore):
    orchestrator.start()
    store.add(Triple(S, P, NamedNode("http://example.org/other")))

    assert orchestrator.rule_engine._fired_count.get("match_test", 0) == 0
    assert orchestrator.exec_store.last_execution("match_test") is None


def test_orchestrator_causal_chain(orchestrator: CognitiveOrchestrator, store: TripleStore):
    """get_causal_chain traces an action back to rule and events."""
    orchestrator.start()
    store.add(Triple(S, P, O))

    rec = orchestrator.exec_store.last_execution("match_test")
    assert rec is not None
    chain = orchestrator.get_causal_chain(rec.action_id)
    kinds = {c["type"] for c in chain}
    assert "action" in kinds
    assert "rule" in kinds


def test_orchestrator_rule_status(orchestrator: CognitiveOrchestrator, store: TripleStore):
    orchestrator.start()
    store.add(Triple(S, P, O))

    status = orchestrator.get_rule_status()
    assert len(status) == 1
    assert status[0]["name"] == "match_test"
    assert status[0]["fired"] == 1


def test_orchestrator_stop_disables(orchestrator: CognitiveOrchestrator, store: TripleStore):
    orchestrator.start()
    orchestrator.stop()
    store.add(Triple(S, P, O))

    assert orchestrator.rule_engine._fired_count.get("match_test", 0) == 0


def test_orchestrator_ingest_event_records_transaction(orchestrator: CognitiveOrchestrator):
    orchestrator.start()
    tx = orchestrator.ingest_event("ContainerDelayed", {"container_id": "C-1"})

    assert tx.event_type == "ContainerDelayed"
    assert orchestrator.tx_store.total_count == 1


def test_orchestrator_remove_rule(orchestrator: CognitiveOrchestrator, store: TripleStore):
    orchestrator.start()
    orchestrator.remove_rule("match_test")
    store.add(Triple(S, P, O))

    assert orchestrator.rule_engine._fired_count.get("match_test", 0) == 0
