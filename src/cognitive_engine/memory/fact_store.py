"""FactStore — structured fact persistence with SCD Type 2 versioning.

Replaces the dead node_embeddings table. Stores facts in SQLite with
temporal validity (valid_from/valid_to) for cross-session persistence.

The article's principle: "Structured retrieval wins because it is
interpretable, auditable, and composable."
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from cognitive_engine.core.concept import (
    Appraisal,
    ConceptRegistry,
    Provenance,
    default_registry,
)
from cognitive_engine.memory.models import Fact, FactArchive

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    concept TEXT NOT NULL,
    value TEXT NOT NULL,
    original_text TEXT,
    provenance TEXT NOT NULL,
    appraisal_json TEXT,
    confidence REAL DEFAULT 0.5,
    valid_from REAL NOT NULL,
    valid_to REAL,
    session_id TEXT,
    node_id TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS facts_archive (
    id TEXT PRIMARY KEY,
    concept TEXT NOT NULL,
    value TEXT NOT NULL,
    original_text TEXT,
    provenance TEXT NOT NULL,
    appraisal_json TEXT,
    confidence REAL DEFAULT 0.5,
    valid_from REAL NOT NULL,
    valid_to REAL,
    session_id TEXT,
    node_id TEXT,
    archived_at REAL,
    archive_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_concept ON facts(concept);
CREATE INDEX IF NOT EXISTS idx_facts_provenance ON facts(provenance);
CREATE INDEX IF NOT EXISTS idx_facts_valid ON facts(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);
CREATE INDEX IF NOT EXISTS idx_facts_node ON facts(node_id);
CREATE INDEX IF NOT EXISTS idx_archive_concept ON facts_archive(concept);
CREATE INDEX IF NOT EXISTS idx_archive_reason ON facts_archive(archive_reason);
"""


class FactStore:
    """SQLite-backed persistent fact store with SCD Type 2 versioning.

    Each fact belongs to a concept (BUDGET, NAME, etc.) and has a provenance
    (USER_STATED, TOOL_RETURNED, etc.). When a fact is invalidated, valid_to
    is set and the fact is moved to facts_archive.

    Usage:
        store = FactStore("facts.db", session_id="abc-123")
        store.store(fact)                           # insert active fact
        store.invalidate(fact_id, reason="misleading")  # archive
        facts = store.query(concept="BUDGET")       # structured retrieval
        discarded = store.weeding()                  # MUSTIE criteria
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        session_id: str = "",
        concepts: Optional[ConceptRegistry] = None,
    ):
        self.db_path = Path(db_path) if db_path != ":memory:" else None
        self.session_id = session_id or uuid4().hex
        self.concepts = concepts or default_registry()
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> FactStore:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def store(self, fact: Fact) -> Fact:
        """Insert a new active fact (valid_to = NULL)."""
        if not fact.valid_from:
            fact.valid_from = time.time()
        if not fact.session_id:
            fact.session_id = self.session_id

        self._conn.execute(
            "INSERT OR REPLACE INTO facts "
            "(id, concept, value, original_text, provenance, appraisal_json, "
            "confidence, valid_from, valid_to, session_id, node_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact.id.hex,
                fact.concept,
                fact.value,
                fact.original_text,
                fact.provenance.name,
                json.dumps({
                    "uniqueness": fact.appraisal.uniqueness,
                    "replaceability": fact.appraisal.replaceability,
                    "actionability": fact.appraisal.actionability,
                    "stability": fact.appraisal.stability,
                    "sensitivity": fact.appraisal.sensitivity,
                }),
                fact.confidence,
                fact.valid_from,
                fact.valid_to,
                fact.session_id,
                fact.node_id.hex if fact.node_id else None,
                time.time(),
            ),
        )
        self._conn.commit()
        return fact

    def invalidate(self, fact_id: UUID, reason: str = "conflict") -> bool:
        """SCD Type 2 close: set valid_to, archive the fact.

        Returns True if the fact was found and invalidated.
        """
        now = time.time()
        row = self._conn.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id.hex,)
        ).fetchone()
        if row is None:
            return False

        # Close the fact (SCD Type 2)
        self._conn.execute(
            "UPDATE facts SET valid_to = ? WHERE id = ?",
            (now, fact_id.hex),
        )

        # Archive it
        self._conn.execute(
            "INSERT INTO facts_archive "
            "(id, concept, value, original_text, provenance, appraisal_json, "
            "confidence, valid_from, valid_to, session_id, node_id, "
            "archived_at, archive_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"],
                row["concept"],
                row["value"],
                row["original_text"],
                row["provenance"],
                row["appraisal_json"],
                row["confidence"],
                row["valid_from"],
                row["valid_to"],
                row["session_id"],
                row["node_id"],
                now,
                reason,
            ),
        )
        self._conn.commit()
        return True

    def query(
        self,
        concept: Optional[str] = None,
        provenance_min: Optional[Provenance] = None,
        active_only: bool = True,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[Fact]:
        """Structured retrieval by concept + provenance + temporal validity.

        This is the article's primary retrieval path: SQL against the fact
        table, not vector similarity.
        """
        clauses = []
        params: list = []

        if concept:
            clauses.append("concept = ?")
            params.append(concept)

        if provenance_min is not None:
            # Provenance weights: USER_STATED=1.0, TOOL_RETURNED=0.85, etc.
            # We want facts with provenance >= the minimum
            min_weight = provenance_min.value
            prov_names = [
                p.name for p in Provenance if p.value >= min_weight
            ]
            placeholders = ",".join("?" * len(prov_names))
            clauses.append(f"provenance IN ({placeholders})")
            params.extend(prov_names)

        if active_only:
            clauses.append("valid_to IS NULL")

        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM facts WHERE {where} ORDER BY valid_from DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def list_active(self, concept: Optional[str] = None) -> list[Fact]:
        """List all active facts, optionally filtered by concept."""
        return self.query(concept=concept, active_only=True)

    def get_by_id(self, fact_id: UUID) -> Optional[Fact]:
        """Get a single fact by ID."""
        row = self._conn.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id.hex,)
        ).fetchone()
        return self._row_to_fact(row) if row else None

    def count_active(self, concept: Optional[str] = None) -> int:
        """Count active facts, optionally filtered by concept."""
        if concept:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE concept = ? AND valid_to IS NULL",
                (concept,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE valid_to IS NULL"
            ).fetchone()
        return row[0] if row else 0

    def count_archived(self, concept: Optional[str] = None) -> int:
        """Count archived facts."""
        if concept:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts_archive WHERE concept = ?",
                (concept,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts_archive"
            ).fetchone()
        return row[0] if row else 0

    def stats(self) -> dict:
        """Return statistics about the fact store."""
        return {
            "active_facts": self.count_active(),
            "archived_facts": self.count_archived(),
            "total_facts": self.count_active() + self.count_archived(),
            "concepts": list(
                r[0] for r in self._conn.execute(
                    "SELECT DISTINCT concept FROM facts WHERE valid_to IS NULL"
                ).fetchall()
            ),
        }

    def weeding(self) -> list[FactArchive]:
        """Apply CREW/MUSTIE criteria to discard low-value facts.

        Criteria:
            Misleading — contradicts a newer, higher-provenance fact
            Ugly — empty or malformed value
            Superseded — already invalidated (valid_to IS NOT NULL)
            Trivial — low confidence (< 0.2) and never accessed
            Irrelevant — concept not in current ConceptRegistry
            Elsewhere — duplicate with higher provenance exists

        Returns list of archived facts with reasons.
        """
        archived: list[FactArchive] = []
        now = time.time()

        # 1. UGLY — empty or whitespace-only values
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL AND (value IS NULL OR TRIM(value) = '')"
        ).fetchall()
        for row in rows:
            fact = self._row_to_fact(row)
            archive = self._archive_fact(fact, "ugly", now)
            archived.append(archive)

        # 2. SUPERSEDED — already has valid_to set but not yet archived
        rows = self._conn.execute(
            "SELECT f.* FROM facts f "
            "LEFT JOIN facts_archive fa ON f.id = fa.id "
            "WHERE f.valid_to IS NOT NULL AND fa.id IS NULL"
        ).fetchall()
        for row in rows:
            fact = self._row_to_fact(row)
            archive = self._archive_fact(fact, "superseded", now)
            archived.append(archive)

        # 3. TRIVIAL — low confidence and never accessed (confidence < 0.2)
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL AND confidence < 0.2"
        ).fetchall()
        for row in rows:
            fact = self._row_to_fact(row)
            archive = self._archive_fact(fact, "trivial", now)
            archived.append(archive)

        # 4. IRRELEVANT — concept not in ConceptRegistry
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL"
        ).fetchall()
        for row in rows:
            fact = self._row_to_fact(row)
            if not self.concepts.has(fact.concept):
                archive = self._archive_fact(fact, "irrelevant", now)
                archived.append(archive)

        # 5. ELSEWHERE — duplicate with higher provenance exists
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE valid_to IS NULL"
        ).fetchall()
        seen: dict[str, Fact] = {}
        for row in rows:
            fact = self._row_to_fact(row)
            key = f"{fact.concept}:{fact.value}"
            if key in seen:
                existing = seen[key]
                # Archive the one with lower provenance
                if fact.provenance.value < existing.provenance.value:
                    archive = self._archive_fact(fact, "elsewhere", now)
                    archived.append(archive)
                else:
                    archive = self._archive_fact(existing, "elsewhere", now)
                    archived.append(archive)
                    seen[key] = fact
            else:
                seen[key] = fact

        # 6. MISLEADING — contradicts a newer, higher-provenance fact
        # For each concept, find the highest-provenance active fact.
        # Archive any lower-provenance fact that contradicts it.
        concepts_with_facts = self._conn.execute(
            "SELECT DISTINCT concept FROM facts WHERE valid_to IS NULL"
        ).fetchall()
        for (concept_name,) in concepts_with_facts:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE concept = ? AND valid_to IS NULL "
                "ORDER BY provenance DESC, valid_from DESC",
                (concept_name,),
            ).fetchall()
            if len(rows) < 2:
                continue
            best = self._row_to_fact(rows[0])
            for row in rows[1:]:
                fact = self._row_to_fact(row)
                # Different values + lower provenance = misleading
                if fact.value != best.value and fact.provenance.value < best.provenance.value:
                    archive = self._archive_fact(fact, "misleading", now)
                    archived.append(archive)

        return archived

    def _archive_fact(self, fact: Fact, reason: str, now: float) -> FactArchive:
        """Move a fact to the archive table and invalidate it."""
        archive = FactArchive.from_fact(fact, reason=reason, archived_at=now)

        # Close the fact
        self._conn.execute(
            "UPDATE facts SET valid_to = ? WHERE id = ?",
            (now, fact.id.hex),
        )

        # Insert into archive
        self._conn.execute(
            "INSERT INTO facts_archive "
            "(id, concept, value, original_text, provenance, appraisal_json, "
            "confidence, valid_from, valid_to, session_id, node_id, "
            "archived_at, archive_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                archive.id.hex,
                archive.concept,
                archive.value,
                archive.original_text,
                archive.provenance.name,
                json.dumps({
                    "uniqueness": archive.appraisal.uniqueness,
                    "replaceability": archive.appraisal.replaceability,
                    "actionability": archive.appraisal.actionability,
                    "stability": archive.appraisal.stability,
                    "sensitivity": archive.appraisal.sensitivity,
                }),
                archive.confidence,
                archive.valid_from,
                archive.valid_to,
                archive.session_id,
                archive.node_id.hex if archive.node_id else None,
                archive.archived_at,
                archive.archive_reason,
            ),
        )
        self._conn.commit()
        return archive

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        """Convert a database row to a Fact object."""
        appraisal_data = json.loads(row["appraisal_json"]) if row["appraisal_json"] else {}
        return Fact(
            id=UUID(row["id"]),
            concept=row["concept"],
            value=row["value"],
            original_text=row["original_text"] or "",
            provenance=Provenance[row["provenance"]],
            appraisal=Appraisal(
                uniqueness=appraisal_data.get("uniqueness", 0.5),
                replaceability=appraisal_data.get("replaceability", 0.5),
                actionability=appraisal_data.get("actionability", 0.5),
                stability=appraisal_data.get("stability", 0.5),
                sensitivity=appraisal_data.get("sensitivity", 0.0),
            ),
            confidence=row["confidence"],
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
            session_id=row["session_id"] or "",
            node_id=UUID(row["node_id"]) if row["node_id"] else None,
        )


def _row_to_fact(row: sqlite3.Row) -> Fact:
    """Convert a database row to a Fact object."""
    appraisal_data = json.loads(row["appraisal_json"]) if row["appraisal_json"] else {}
    return Fact(
        id=UUID(row["id"]),
        concept=row["concept"],
        value=row["value"],
        original_text=row["original_text"] or "",
        provenance=Provenance[row["provenance"]],
        appraisal=Appraisal(
            uniqueness=appraisal_data.get("uniqueness", 0.5),
            replaceability=appraisal_data.get("replaceability", 0.5),
            actionability=appraisal_data.get("actionability", 0.5),
            stability=appraisal_data.get("stability", 0.5),
            sensitivity=appraisal_data.get("sensitivity", 0.0),
        ),
        confidence=row["confidence"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        session_id=row["session_id"] or "",
        node_id=UUID(row["node_id"]) if row["node_id"] else None,
    )
