"""Execution Network — provenance-tracked action records.

Every action fired by the ProductionRuleEngine is recorded as an
ExecutionRecord, enabling full traceability from triggering event
through rule evaluation to action result.

This provides the "Explainability" layer from the architecture:
    Container delayed
      -> ProductionRule "detect-delay" fired
        -> Action: Log ("Container delay detected")
        -> Action: Simulate (result: cash=$45K, backlog=120)
        -> Action: Assert (Decision/evaluating)

Usage::

    from dynafx.knowledge.execution import ExecutionStore, ExecutionRecord

    exec_store = ExecutionStore(triple_store)
    exec_store.record(
        rule_name="detect-delay",
        action_type="simulate",
        bindings={"container": "C-123"},
        output={"end_cash": 45000.0, "steps": 730},
        status="executed",
    )

    # Trace all actions for a rule
    history = exec_store.by_rule("detect-delay")
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from dynafx.core.models import Opinion
from dynafx.knowledge.model import (
    Literal,
    NamedNode,
    Triple,
)
from dynafx.knowledge.store import TripleStore

logger = logging.getLogger(__name__)

NS_EXEC = "http://dynafx.org/execution#"
DEFAULT_GRAPH = "executions"


@dataclass(frozen=True)
class ExecutionRecord:
    """A record of an action that was executed.

    Attributes:
        action_id: Unique identifier for this execution.
        action_type: Type of action ("triple_add", "simulate", "scenario", "log", etc.).
        rule_name: Name of the ProductionRule that fired this action.
        timestamp: When the action was executed.
        bindings: Variable bindings at the time of execution.
        output: Structured result data.
        status: "executed", "failed", or "rolled_back".
        message: Human-readable summary.
    """

    action_id: str
    action_type: str
    rule_name: str
    timestamp: float
    bindings: dict[str, Any]
    output: dict[str, Any]
    status: str = "executed"
    message: str = ""


class ExecutionStore:
    """Stores and queries execution records, with RDF provenance.

    Each execution produces triples in the "executions" named graph:
        <exec/{id}> rdf:type exec:Execution
        <exec/{id}> exec:actionType <string>
        <exec/{id}> exec:ruleName <string>
        <exec/{id}> exec:timestamp <float>
        <exec/{id}> exec:status <string>
        <exec/{id}> exec:hasOutput ...
    """

    def __init__(self, triple_store: TripleStore):
        self._store = triple_store
        self._records: dict[str, ExecutionRecord] = {}
        self._rule_index: dict[str, list[str]] = defaultdict(list)
        self._type_index: dict[str, list[str]] = defaultdict(list)

    def record(
        self,
        rule_name: str,
        action_type: str,
        bindings: dict[str, Any],
        output: dict[str, Any],
        status: str = "executed",
        message: str = "",
        graph: str = DEFAULT_GRAPH,
    ) -> ExecutionRecord:
        """Record an execution with RDF provenance.

        Args:
            rule_name: The ProductionRule that fired.
            action_type: Kind of action.
            bindings: Variable bindings at execution time.
            output: Result data from the action.
            status: "executed", "failed", or "rolled_back".
            message: Optional summary.
            graph: Named graph to store triples in.

        Returns:
            The created ExecutionRecord.
        """
        action_id = str(uuid.uuid4())
        ts = time.time()

        rec = ExecutionRecord(
            action_id=action_id,
            action_type=action_type,
            rule_name=rule_name,
            timestamp=ts,
            bindings=dict(bindings),
            output=dict(output),
            status=status,
            message=message,
        )
        self._records[action_id] = rec
        self._rule_index[rule_name].append(action_id)
        self._type_index[action_type].append(action_id)

        # Store as RDF
        sx = NamedNode(f"{NS_EXEC}{action_id}")
        opin = Opinion(1.0, 0.0, 0.0)

        triples = [
            Triple(sx, NamedNode(f"{NS_EXEC}type"), NamedNode(f"{NS_EXEC}Execution"), opinion=opin),
            Triple(sx, NamedNode(f"{NS_EXEC}actionType"), Literal(action_type), opinion=opin),
            Triple(sx, NamedNode(f"{NS_EXEC}ruleName"), Literal(rule_name), opinion=opin),
            Triple(sx, NamedNode(f"{NS_EXEC}timestamp"), Literal(ts), opinion=opin),
            Triple(sx, NamedNode(f"{NS_EXEC}status"), Literal(status), opinion=opin),
        ]
        if message:
            triples.append(Triple(sx, NamedNode(f"{NS_EXEC}message"), Literal(message), opinion=opin))

        for k, v in output.items():
            val = self._to_literal(v)
            triples.append(Triple(sx, NamedNode(f"{NS_EXEC}output/{k}"), val, opinion=opin))

        with self._store.suppress_callbacks():
            for t in triples:
                self._store.add(t, graph=graph)

        return rec

    def record_action_result(
        self,
        rule_name: str,
        action_result: Any,
        bindings: dict[str, Any],
    ) -> ExecutionRecord:
        """Convenience: record an ActionResult from production.py."""
        return self.record(
            rule_name=rule_name,
            action_type=getattr(action_result, "action_type", type(action_result).__name__),
            bindings=bindings,
            output=getattr(action_result, "output", {}),
            status="executed" if getattr(action_result, "success", True) else "failed",
            message=getattr(action_result, "message", ""),
            action_id=getattr(action_result, "action_id", ""),
        )

    def get(self, action_id: str) -> Optional[ExecutionRecord]:
        return self._records.get(action_id)

    def by_rule(self, rule_name: str) -> list[ExecutionRecord]:
        """Return all execution records for a given rule, newest first."""
        ids = self._rule_index.get(rule_name, [])
        records = [self._records[i] for i in ids if i in self._records]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    def by_type(self, action_type: str) -> list[ExecutionRecord]:
        """Return all execution records of a given action type."""
        ids = self._type_index.get(action_type, [])
        records = [self._records[i] for i in ids if i in self._records]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    def recent(self, n: int = 20) -> list[ExecutionRecord]:
        """Return the N most recent execution records."""
        sorted_records = sorted(self._records.values(), key=lambda r: r.timestamp, reverse=True)
        return sorted_records[:n]

    def last_execution(self, rule_name: str) -> Optional[ExecutionRecord]:
        """Return the most recent execution for a rule."""
        records = self.by_rule(rule_name)
        return records[0] if records else None

    def record_and_return(self, *args, **kwargs) -> ExecutionRecord:
        """Record and return the record (fluent API)."""
        return self.record(*args, **kwargs)

    @property
    def total_count(self) -> int:
        return len(self._records)

    @staticmethod
    def _to_literal(v: Any) -> Literal:
        if isinstance(v, bool):
            return Literal(v, datatype="http://www.w3.org/2001/XMLSchema#boolean")
        if isinstance(v, (int, float)):
            return Literal(float(v), datatype="http://www.w3.org/2001/XMLSchema#double")
        return Literal(str(v))
