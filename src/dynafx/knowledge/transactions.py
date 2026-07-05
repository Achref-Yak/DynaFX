"""Transaction Layer — append-only temporal event log over a TripleStore.

Every external event (sensor reading, ERP update, manual entry) becomes
a Transaction that is:
    1. Recorded as an append-only entry in the transaction log
    2. Stored as RDF triples in the KB's "transactions" named graph
    3. Automatically triggers ProductionRuleEngine evaluation via
       TripleStore.on_add() callbacks

Usage::

    from dynafx.knowledge.transactions import TransactionStore

    tx_store = TransactionStore(triple_store)

    # Record an event
    tx = tx_store.record(
        event_type="ContainerDelayed",
        payload={"container_id": "C-123", "delay_days": 6, "port": "Rotterdam"},
        source="ERP",
    )

    # Query recent events
    recent_delays = tx_store.query(event_type="ContainerDelayed", n=10)

    # Count events in a time window
    count = tx_store.count_by_type("ContainerDelayed", since=time.time() - 86400)
"""

from __future__ import annotations

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

NS_TX = "http://dynafx.org/transaction#"
NS_EVENT = "http://dynafx.org/event#"
NS_PAYLOAD = "http://dynafx.org/payload#"

DEFAULT_GRAPH = "transactions"


@dataclass(frozen=True)
class Transaction:
    """An immutable record of an event or observation.

    Attributes:
        id: Unique transaction ID (UUID string).
        timestamp: Unix timestamp of when the event occurred.
        event_type: Categorizes the event (e.g. "ContainerDelayed").
        payload: Key-value data attached to the event.
        source: Origin of the event ("ERP", "IoT", "manual", "simulation", etc.).
        confidence: Confidence in the event's accuracy [0, 1].
    """

    id: str
    timestamp: float
    event_type: str
    payload: dict[str, Any]
    source: str
    confidence: float


@dataclass(frozen=True)
class TransactionQuery:
    """Filter parameters for querying transactions."""
    event_type: Optional[str] = None
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    source: Optional[str] = None
    n: int = 0  # 0 = unlimited


class TransactionStore:
    """Append-only log of events, backed by RDF triples in the KB.

    Each transaction produces triples in the "transactions" named graph:
        <tx/{id}> rdf:type <event/{type}>
        <tx/{id}> tx:timestamp <float>
        <tx/{id}> tx:source <string>
        <tx/{id}> tx:confidence <float>
        <tx/{id}> payload/{key} <value>
    """

    def __init__(self, triple_store: TripleStore):
        self._store = triple_store
        self._transactions: list[Transaction] = []
        self._event_index: dict[str, list[int]] = defaultdict(list)
        self._source_index: dict[str, list[int]] = defaultdict(list)

    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str = "external",
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
        graph: str = DEFAULT_GRAPH,
    ) -> Transaction:
        """Record a transaction and store it as RDF triples.

        The triples are added to the KB's "transactions" graph, which
        automatically triggers any ProductionRuleEngine listeners.

        Args:
            event_type: Event category (e.g. "ContainerDelayed").
            payload: Key-value data.
            source: Origin identifier.
            confidence: Accuracy confidence [0, 1].
            timestamp: Unix timestamp (default: time.time()).
            graph: Named graph to store triples in.

        Returns:
            The created Transaction.
        """
        tx_id = str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()

        sx = NamedNode(f"{NS_TX}{tx_id}")
        event_node = NamedNode(f"{NS_EVENT}{event_type}")

        opin = Opinion(confidence, 1.0 - confidence, 0.0)

        triples: list[Triple] = [
            Triple(sx, NamedNode(f"{NS_TX}type"), event_node, opinion=opin),
            Triple(sx, NamedNode(f"{NS_TX}timestamp"), Literal(ts), opinion=opin),
            Triple(sx, NamedNode(f"{NS_TX}source"), Literal(source), opinion=opin),
            Triple(sx, NamedNode(f"{NS_TX}confidence"), Literal(confidence), opinion=opin),
        ]

        for k, v in payload.items():
            val = self._to_literal(v)
            triples.append(Triple(
                sx, NamedNode(f"{NS_PAYLOAD}{k}"), val, opinion=opin,
            ))

        with self._store.suppress_callbacks():
            for t in triples:
                self._store.add(t, graph=graph)

        tx = Transaction(
            id=tx_id,
            timestamp=ts,
            event_type=event_type,
            payload=dict(payload),
            source=source,
            confidence=confidence,
        )
        self._transactions.append(tx)
        self._event_index[event_type].append(len(self._transactions) - 1)
        self._source_index[source].append(len(self._transactions) - 1)

        return tx

    def query(
        self,
        event_type: Optional[str] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        source: Optional[str] = None,
        n: int = 0,
    ) -> list[Transaction]:
        """Query transactions by filters, newest first.

        Args:
            event_type: Filter by event type.
            t_start: Include transactions at or after this timestamp.
            t_end: Include transactions at or before this timestamp.
            source: Filter by source.
            n: Max results (0 = unlimited).

        Returns:
            List of matching transactions, newest first.
        """
        candidates = list(range(len(self._transactions)))

        if event_type is not None:
            candidates = [i for i in candidates
                          if i in self._event_index.get(event_type, [])]
        if source is not None:
            candidates = [i for i in candidates
                          if i in self._source_index.get(source, [])]
        if t_start is not None:
            candidates = [i for i in candidates
                          if self._transactions[i].timestamp >= t_start]
        if t_end is not None:
            candidates = [i for i in candidates
                          if self._transactions[i].timestamp <= t_end]

        results = [self._transactions[i] for i in candidates]
        results.sort(key=lambda t: t.timestamp, reverse=True)

        if n > 0:
            results = results[:n]
        return results

    def recent(self, n: int = 10) -> list[Transaction]:
        """Return the N most recent transactions."""
        return self.query(n=n)

    def count_by_type(
        self,
        event_type: str,
        since: Optional[float] = None,
    ) -> int:
        """Count events of a given type since a timestamp."""
        return len(self.query(event_type=event_type, t_start=since))

    def count_by_source(
        self,
        source: str,
        since: Optional[float] = None,
    ) -> int:
        """Count events from a given source since a timestamp."""
        return len(self.query(source=source, t_start=since))

    @property
    def total_count(self) -> int:
        return len(self._transactions)

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _to_literal(v: Any) -> Literal:
        if isinstance(v, bool):
            return Literal(v, datatype="http://www.w3.org/2001/XMLSchema#boolean")
        if isinstance(v, int):
            return Literal(float(v), datatype="http://www.w3.org/2001/XMLSchema#double")
        if isinstance(v, float):
            return Literal(v, datatype="http://www.w3.org/2001/XMLSchema#double")
        return Literal(str(v))
