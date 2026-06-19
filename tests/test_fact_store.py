"""Tests for FactStore — structured fact persistence with SCD Type 2."""

import time
import pytest
from uuid import uuid4

from cognitive_engine.core.concept import Appraisal, Provenance
from cognitive_engine.memory.models import Fact, FactArchive
from cognitive_engine.memory.fact_store import FactStore


class TestFact:
    def test_create_fact(self):
        fact = Fact(
            concept="BUDGET",
            value="$500",
            original_text="My budget is $500",
            provenance=Provenance.USER_STATED,
        )
        assert fact.concept == "BUDGET"
        assert fact.value == "$500"
        assert fact.is_active()

    def test_fact_is_active(self):
        fact = Fact(concept="BUDGET", value="$500")
        assert fact.is_active()
        fact.valid_to = time.time()
        assert not fact.is_active()

    def test_fact_to_dict(self):
        fact = Fact(
            id=uuid4(),
            concept="BUDGET",
            value="$500",
            provenance=Provenance.USER_STATED,
        )
        d = fact.to_dict()
        assert d["concept"] == "BUDGET"
        assert d["value"] == "$500"
        assert d["provenance"] == "USER_STATED"


class TestFactArchive:
    def test_from_fact(self):
        fact = Fact(
            concept="BUDGET",
            value="$500",
            provenance=Provenance.USER_STATED,
        )
        archive = FactArchive.from_fact(fact, reason="misleading", archived_at=time.time())
        assert archive.concept == "BUDGET"
        assert archive.archive_reason == "misleading"
        assert archive.archived_at > 0


class TestFactStore:
    def test_create_store(self):
        store = FactStore(":memory:", session_id="test-session")
        assert store is not None
        store.close()

    def test_store_and_retrieve(self):
        store = FactStore(":memory:", session_id="test-session")
        fact = Fact(
            concept="BUDGET",
            value="$500",
            original_text="My budget is $500",
            provenance=Provenance.USER_STATED,
        )
        store.store(fact)

        # Retrieve active facts
        facts = store.query(concept="BUDGET")
        assert len(facts) == 1
        assert facts[0].value == "$500"
        assert facts[0].is_active()
        store.close()

    def test_scd_type_2_invalidation(self):
        store = FactStore(":memory:", session_id="test-session")
        fact = Fact(
            concept="BUDGET",
            value="$500",
            provenance=Provenance.USER_STATED,
        )
        store.store(fact)

        # Invalidate the fact
        success = store.invalidate(fact.id, reason="conflict")
        assert success

        # Should no longer be active
        facts = store.query(concept="BUDGET", active_only=True)
        assert len(facts) == 0

        # Should be in archive
        archived = store.count_archived(concept="BUDGET")
        assert archived == 1
        store.close()

    def test_query_by_provenance(self):
        store = FactStore(":memory:", session_id="test-session")

        # Store facts with different provenance
        fact1 = Fact(
            concept="BUDGET",
            value="$500",
            provenance=Provenance.AGENT_INFERRED,
        )
        fact2 = Fact(
            concept="BUDGET",
            value="$600",
            provenance=Provenance.USER_STATED,
        )
        store.store(fact1)
        store.store(fact2)

        # Query with minimum provenance
        facts = store.query(concept="BUDGET", provenance_min=Provenance.TOOL_RETURNED)
        assert len(facts) == 1
        assert facts[0].value == "$600"  # Only USER_STATED (1.0) >= TOOL_RETURNED (0.85)
        store.close()

    def test_count_active(self):
        store = FactStore(":memory:", session_id="test-session")
        assert store.count_active() == 0

        fact = Fact(concept="BUDGET", value="$500")
        store.store(fact)
        assert store.count_active() == 1
        assert store.count_active(concept="BUDGET") == 1
        assert store.count_active(concept="NAME") == 0
        store.close()

    def test_stats(self):
        store = FactStore(":memory:", session_id="test-session")
        fact = Fact(concept="BUDGET", value="$500")
        store.store(fact)

        stats = store.stats()
        assert stats["active_facts"] == 1
        assert stats["archived_facts"] == 0
        assert "BUDGET" in stats["concepts"]
        store.close()

    def test_get_by_id(self):
        store = FactStore(":memory:", session_id="test-session")
        fact = Fact(concept="BUDGET", value="$500")
        store.store(fact)

        retrieved = store.get_by_id(fact.id)
        assert retrieved is not None
        assert retrieved.value == "$500"
        store.close()

    def test_list_active(self):
        store = FactStore(":memory:", session_id="test-session")
        store.store(Fact(concept="BUDGET", value="$500"))
        store.store(Fact(concept="NAME", value="Alice"))

        budget_facts = store.list_active(concept="BUDGET")
        assert len(budget_facts) == 1

        all_facts = store.list_active()
        assert len(all_facts) == 2
        store.close()


class TestMustieWeeding:
    def test_weeding_ugly(self):
        store = FactStore(":memory:", session_id="test-session")
        fact = Fact(concept="BUDGET", value="")  # Empty value = ugly
        store.store(fact)

        archived = store.weeding()
        assert len(archived) >= 1
        assert any(a.archive_reason == "ugly" for a in archived)
        store.close()

    def test_weeding_superseded(self):
        store = FactStore(":memory:", session_id="test-session")
        fact = Fact(concept="BUDGET", value="$500")
        store.store(fact)
        store.invalidate(fact.id, reason="conflict")

        # Fact is already archived by invalidate(), so weeding should find 0 superseded
        archived = store.weeding()
        # The fact was already archived by invalidate(), so no new superseded archives
        assert len(archived) == 0
        # But the fact should be in the archive table
        assert store.count_archived(concept="BUDGET") == 1
        store.close()

    def test_weeding_trivial(self):
        store = FactStore(":memory:", session_id="test-session")
        fact = Fact(concept="BUDGET", value="$500", confidence=0.1)  # Low confidence
        store.store(fact)

        archived = store.weeding()
        assert len(archived) >= 1
        assert any(a.archive_reason == "trivial" for a in archived)
        store.close()

    def test_weeding_elsewhere(self):
        store = FactStore(":memory:", session_id="test-session")
        # Same concept and value, different provenance
        fact1 = Fact(
            concept="BUDGET",
            value="$500",
            provenance=Provenance.AGENT_INFERRED,
        )
        fact2 = Fact(
            concept="BUDGET",
            value="$500",
            provenance=Provenance.USER_STATED,
        )
        store.store(fact1)
        store.store(fact2)

        archived = store.weeding()
        # The lower-provenance fact should be archived
        assert any(a.archive_reason == "elsewhere" for a in archived)
        store.close()
