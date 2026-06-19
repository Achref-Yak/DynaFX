"""Tests for graph analysis modules: traversal, classification, evidence chains,
labels, and verifiable summary generation."""

from __future__ import annotations

from uuid import uuid4

import pytest

from cognitive_engine.core.models import (
    Edge,
    EdgeType,
    Graph,
    Node,
    NodeType,
    Opinion,
)
from cognitive_engine.analysis import (
    EvidenceChain,
    EvidenceClassification,
    EvidenceItem,
    Fact,
    EdgeRecord,
    GraphAggregates,
    PathEntry,
    TraversalResult,
    VerifiableSummary,
    build_verifiable_summary,
    classify_evidence,
    find_evidence_chains,
    generate_label,
    traverse,
    _compute_belief_tier,
    _compute_aggregates,
    _ATTACKING,
    _CONTEXTUAL,
    _EVIDENCE_TYPES,
    _SUPPORTING,
)


# ── Helpers ─────────────────────────────────────────────────────


def _opinion(b: float = 0.0, d: float = 0.0, u: float = 0.0) -> Opinion:
    return Opinion(belief=b, disbelief=d, uncertainty=u)


def _make_graph() -> tuple[Graph, UUID, UUID, UUID, UUID]:
    """Create a simple graph with 4 nodes and supporting/attacking edges."""
    root_id = uuid4()
    ev1_id = uuid4()
    ev2_id = uuid4()
    weak_id = uuid4()

    g = Graph()
    g.nodes = {
        root_id: Node(id=root_id, type=NodeType.CLAIM, text="root claim", opinion=_opinion(0.8)),
        ev1_id: Node(id=ev1_id, type=NodeType.EVIDENCE, text="supporting evidence A", opinion=_opinion(0.9)),
        ev2_id: Node(id=ev2_id, type=NodeType.AXIOM, text="supporting evidence B", opinion=_opinion(0.7)),
        weak_id: Node(id=weak_id, type=NodeType.COUNTERCLAIM, text="weakening counter", opinion=_opinion(0.2)),
    }
    g.edges = {
        uuid4(): Edge(source_id=root_id, target_id=ev1_id, type=EdgeType.SUPPORTS),
        uuid4(): Edge(source_id=root_id, target_id=ev2_id, type=EdgeType.INFERS),
        uuid4(): Edge(source_id=root_id, target_id=weak_id, type=EdgeType.ATTACKS),
    }
    return g, root_id, ev1_id, ev2_id, weak_id


def _make_empty_graph() -> tuple[Graph, UUID]:
    root_id = uuid4()
    g = Graph()
    g.nodes = {root_id: Node(id=root_id, type=NodeType.CLAIM, text="lonely claim")}
    return g, root_id


def _make_entity_graph() -> tuple[Graph, UUID, UUID]:
    """Graph with an ENTITY node (should not appear in supports/contradictions)."""
    root_id = uuid4()
    entity_id = uuid4()
    g = Graph()
    g.nodes = {
        root_id: Node(id=root_id, type=NodeType.CLAIM, text="claim about entity", opinion=_opinion(0.5)),
        entity_id: Node(id=entity_id, type=NodeType.ENTITY, text="the entity", opinion=_opinion(0.0)),
    }
    g.edges = {
        uuid4(): Edge(source_id=root_id, target_id=entity_id, type=EdgeType.SUPPORTS),
    }
    return g, root_id, entity_id


def _make_contextual_graph() -> tuple[Graph, UUID, UUID]:
    """Graph with a contextual edge (ASSOCIATED_WITH)."""
    root_id = uuid4()
    ctx_id = uuid4()
    g = Graph()
    g.nodes = {
        root_id: Node(id=root_id, type=NodeType.CLAIM, text="main claim", opinion=_opinion(0.6)),
        ctx_id: Node(id=ctx_id, type=NodeType.EVIDENCE, text="contextual note", opinion=_opinion(0.4)),
    }
    g.edges = {
        uuid4(): Edge(source_id=root_id, target_id=ctx_id, type=EdgeType.ASSOCIATED_WITH),
    }
    return g, root_id, ctx_id


def _make_contradiction_graph() -> tuple[Graph, UUID, UUID]:
    """Graph with an incoming ATTACKS edge (contradicting direction)."""
    root_id = uuid4()
    attacker_id = uuid4()
    g = Graph()
    g.nodes = {
        root_id: Node(id=root_id, type=NodeType.CLAIM, text="attacked claim", opinion=_opinion(0.3)),
        attacker_id: Node(id=attacker_id, type=NodeType.COUNTERCLAIM, text="attacker", opinion=_opinion(0.7)),
    }
    g.edges = {
        uuid4(): Edge(source_id=attacker_id, target_id=root_id, type=EdgeType.ATTACKS),
    }
    return g, root_id, attacker_id


# ── Traverse ────────────────────────────────────────────────────


class TestTraverse:
    def test_returns_root_info(self):
        g, root_id, *_ = _make_graph()
        result = traverse(g, root_id, max_depth=2)
        assert isinstance(result, TraversalResult)
        assert result.root_id == root_id
        assert result.root_text == "root claim"
        assert result.root_belief == pytest.approx(0.8)

    def test_follows_outgoing_supports(self):
        g, root_id, ev1_id, ev2_id, _ = _make_graph()
        result = traverse(g, root_id, max_depth=2)
        target_ids = {p[-1].node_id for p in result.paths}
        assert ev1_id in target_ids
        assert ev2_id in target_ids

    def test_follows_outgoing_attacks(self):
        g, root_id, _, _, weak_id = _make_graph()
        result = traverse(g, root_id, max_depth=2)
        target_ids = {p[-1].node_id for p in result.paths}
        assert weak_id in target_ids

    def test_max_depth_limits(self):
        g, root_id, *_ = _make_graph()
        result = traverse(g, root_id, max_depth=0)
        assert len(result.paths) == 0

    def test_returns_path_entries(self):
        g, root_id, ev1_id, _, _ = _make_graph()
        result = traverse(g, root_id, max_depth=2)
        ev_paths = [p for p in result.paths if p[-1].node_id == ev1_id]
        assert len(ev_paths) >= 1
        entry = ev_paths[0][-1]
        assert entry.text == "supporting evidence A"
        assert entry.edge_type == "SUPPORTS"
        assert entry.direction == "outgoing"
        assert entry.belief == pytest.approx(0.9)

    def test_empty_graph_single_node(self):
        g, root_id = _make_empty_graph()
        result = traverse(g, root_id, max_depth=2)
        assert result.paths == []

    def test_include_contextual(self):
        g, root_id, ctx_id = _make_contextual_graph()
        result = traverse(g, root_id, max_depth=2, include_contextual=True)
        target_ids = {p[-1].node_id for p in result.paths}
        assert ctx_id in target_ids

    def test_exclude_contextual(self):
        g, root_id, ctx_id = _make_contextual_graph()
        result = traverse(g, root_id, max_depth=2, include_contextual=False)
        target_ids = {p[-1].node_id for p in result.paths}
        assert ctx_id not in target_ids

    def test_incoming_edge_traversal(self):
        g, root_id, attacker_id = _make_contradiction_graph()
        result = traverse(g, root_id, max_depth=2)
        source_ids = {p[-1].node_id for p in result.paths}
        assert attacker_id in source_ids

    def test_incoming_entry_direction(self):
        g, root_id, attacker_id = _make_contradiction_graph()
        result = traverse(g, root_id, max_depth=2)
        incoming = [p for p in result.paths if p[-1].node_id == attacker_id]
        assert len(incoming) >= 1
        assert incoming[0][-1].direction == "incoming"


# ── Classify evidence ───────────────────────────────────────────


class TestClassifyEvidence:
    def test_supporting_from_outgoing(self):
        g, root_id, ev1_id, _, _ = _make_graph()
        result = classify_evidence(g, root_id, max_depth=2)
        assert isinstance(result, EvidenceClassification)
        support_ids = {i.path[-1].node_id for i in result.supporting}
        assert ev1_id in support_ids

    def test_weakening_from_outgoing_attack(self):
        g, root_id, _, _, weak_id = _make_graph()
        result = classify_evidence(g, root_id, max_depth=2)
        weaken_ids = {i.path[-1].node_id for i in result.weakening}
        assert weak_id in weaken_ids

    def test_contradicting_from_incoming_attack(self):
        g, root_id, attacker_id = _make_contradiction_graph()
        result = classify_evidence(g, root_id, max_depth=2)
        contra_ids = {i.path[-1].node_id for i in result.contradicting}
        assert attacker_id in contra_ids

    def test_entity_excluded(self):
        g, root_id, entity_id = _make_entity_graph()
        result = classify_evidence(g, root_id, max_depth=2)
        all_ids = set()
        for lst in (result.supporting, result.weakening, result.contradicting, result.contextual):
            all_ids.update(i.path[-1].node_id for i in lst)
        assert entity_id not in all_ids

    def test_deduplication(self):
        g, root_id, ev1_id, _, _ = _make_graph()
        result = classify_evidence(g, root_id, max_depth=3)
        seen = set()
        for lst in (result.supporting, result.weakening, result.contradicting, result.contextual):
            for item in lst:
                tid = item.path[-1].node_id
                assert tid not in seen
                seen.add(tid)

    def test_root_belief_captured(self):
        g, root_id, *_ = _make_graph()
        result = classify_evidence(g, root_id)
        assert result.root_belief == pytest.approx(0.8)
        assert result.root_text == "root claim"

    def test_item_classification_field(self):
        g, root_id, ev1_id, _, _ = _make_graph()
        result = classify_evidence(g, root_id, max_depth=2)
        for item in result.supporting:
            assert item.classification == "supporting"
        for item in result.weakening:
            assert item.classification == "weakening"
        for item in result.contradicting:
            assert item.classification == "contradicting"

    def test_contextual_classification(self):
        g, root_id, ctx_id = _make_contextual_graph()
        result = classify_evidence(g, root_id, max_depth=2, include_contextual=True)
        ctx_ids = {i.path[-1].node_id for i in result.contextual}
        assert ctx_id in ctx_ids


# ── Generate label ──────────────────────────────────────────────


class TestGenerateLabel:
    def test_concept_metadata(self):
        node = Node(id=uuid4(), type=NodeType.CLAIM, text="some text",
                     metadata={"concept": "DATA_PRIVACY"})
        assert generate_label(node) == "Data Privacy"

    def test_entity_kind_metadata(self):
        node = Node(id=uuid4(), type=NodeType.ENTITY, text="some text",
                     metadata={"entity_kind": "Person"})
        assert generate_label(node) == "Person"

    def test_fallback_to_text(self):
        node = Node(id=uuid4(), type=NodeType.CLAIM, text="the quick brown fox jumps")
        label = generate_label(node, max_words=3)
        assert label == "the quick brown..."

    def test_short_text_no_ellipsis(self):
        node = Node(id=uuid4(), type=NodeType.CLAIM, text="short")
        label = generate_label(node, max_words=8)
        assert label == "short"
        assert "..." not in label

    def test_none_node(self):
        assert generate_label(None) == "evidence"

    def test_default_max_words(self):
        words = ["word"] * 20
        node = Node(id=uuid4(), type=NodeType.CLAIM, text=" ".join(words))
        label = generate_label(node)
        assert label.endswith("...")
        # "..." is appended after the words, so split gives 8 words + "..."
        # but "..." is a separate token only if preceded by a space
        assert len(label) > 0


# ── Find evidence chains ────────────────────────────────────────


class TestFindEvidenceChains:
    def test_returns_chains(self):
        g, root_id, *_ = _make_graph()
        chains = find_evidence_chains(g, root_id, max_depth=2)
        assert len(chains) >= 2
        assert all(isinstance(c, EvidenceChain) for c in chains)

    def test_classifications_present(self):
        g, root_id, *_ = _make_graph()
        chains = find_evidence_chains(g, root_id, max_depth=2)
        classes = {c.classification for c in chains}
        assert "supporting" in classes

    def test_root_text_propagated(self):
        g, root_id, *_ = _make_graph()
        chains = find_evidence_chains(g, root_id, max_depth=2)
        for c in chains:
            assert c.root_text == "root claim"
            assert c.root_belief == pytest.approx(0.8)

    def test_chain_length(self):
        g, root_id, *_ = _make_graph()
        chains = find_evidence_chains(g, root_id, max_depth=2)
        for c in chains:
            assert c.chain_length >= 2

    def test_full_text_preserved(self):
        g, root_id, *_ = _make_graph()
        chains = find_evidence_chains(g, root_id, max_depth=2)
        for c in chains:
            assert len(c.evidence_text) > 0

    def test_all_root_ids_if_none(self):
        g, root_id, *_ = _make_graph()
        chains = find_evidence_chains(g, root_id=None, max_depth=2)
        assert len(chains) >= 2

    def test_label_populated(self):
        g, root_id, *_ = _make_graph()
        chains = find_evidence_chains(g, root_id, max_depth=2)
        for c in chains:
            assert len(c.label) > 0


# ── Belief tier ─────────────────────────────────────────────────


class TestBeliefTier:
    def test_high_tier(self):
        assert _compute_belief_tier(0.7) == "high"
        assert _compute_belief_tier(1.0) == "high"

    def test_medium_tier(self):
        assert _compute_belief_tier(0.3) == "medium"
        assert _compute_belief_tier(0.69) == "medium"

    def test_low_tier(self):
        assert _compute_belief_tier(0.01) == "low"
        assert _compute_belief_tier(0.29) == "low"

    def test_uninitialized(self):
        assert _compute_belief_tier(0.0) == "uninitialized"


# ── Compute aggregates ──────────────────────────────────────────


class TestComputeAggregates:
    def test_returns_aggregates(self):
        g, *_ = _make_graph()
        agg = _compute_aggregates(g)
        assert isinstance(agg, GraphAggregates)

    def test_node_count(self):
        g, *_ = _make_graph()
        agg = _compute_aggregates(g)
        assert agg.node_count_by_type.get("CLAIM") == 1
        assert agg.node_count_by_type.get("EVIDENCE") == 1

    def test_edge_count(self):
        g, *_ = _make_graph()
        agg = _compute_aggregates(g)
        assert agg.edge_count_by_type.get("SUPPORTS") == 1
        assert agg.edge_count_by_type.get("INFERS") == 1
        assert agg.edge_count_by_type.get("ATTACKS") == 1

    def test_max_belief_node(self):
        g, *_ = _make_graph()
        agg = _compute_aggregates(g)
        assert agg.max_belief_node is not None
        assert agg.max_belief_node.belief == pytest.approx(0.9)

    def test_min_belief_node(self):
        g, *_ = _make_graph()
        agg = _compute_aggregates(g)
        assert agg.min_belief_node is not None
        assert agg.min_belief_node.belief == pytest.approx(0.2)

    def test_max_depth(self):
        g, *_ = _make_graph()
        agg = _compute_aggregates(g)
        assert agg.max_depth_path >= 1

    def test_edge_polarity_balance(self):
        g, *_ = _make_graph()
        agg = _compute_aggregates(g)
        # 2 supporting + 1 attacking out of 3 polar edges = 2/3
        assert agg.edge_polarity_balance == pytest.approx(2 / 3, abs=0.01)

    def test_no_polar_edges(self):
        g, root_id = _make_empty_graph()
        agg = _compute_aggregates(g)
        assert agg.edge_polarity_balance == 0.0

    def test_entity_excluded_from_supports(self):
        g, root_id, entity_id = _make_entity_graph()
        s = build_verifiable_summary(g, root_id)
        all_edge_ids = set()
        for e in s.supports:
            all_edge_ids.add(e.source_node_id)
            all_edge_ids.add(e.target_node_id)
        assert entity_id.hex not in all_edge_ids


# ── Build verifiable summary ────────────────────────────────────


class TestBuildVerifiableSummary:
    def test_returns_summary(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        assert isinstance(s, VerifiableSummary)

    def test_root_id_is_hex(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        assert s.root_id == root_id.hex

    def test_facts_populated(self):
        g, root_id, ev1_id, ev2_id, weak_id = _make_graph()
        s = build_verifiable_summary(g, root_id)
        fact_ids = {f.node_id for f in s.facts}
        assert root_id.hex in fact_ids
        assert ev1_id.hex in fact_ids

    def test_entity_in_facts(self):
        g, root_id, entity_id = _make_entity_graph()
        s = build_verifiable_summary(g, root_id)
        fact_ids = {f.node_id for f in s.facts}
        assert entity_id.hex in fact_ids

    def test_entity_not_in_supports(self):
        g, root_id, entity_id = _make_entity_graph()
        s = build_verifiable_summary(g, root_id)
        all_edge_ids = set()
        for e in s.supports:
            all_edge_ids.add(e.source_node_id)
            all_edge_ids.add(e.target_node_id)
        assert entity_id.hex not in all_edge_ids

    def test_supports_populated(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        assert len(s.supports) >= 2

    def test_contradictions_populated(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        assert len(s.contradictions) >= 1

    def test_aggregates_populated(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        assert isinstance(s.aggregates, GraphAggregates)
        assert s.aggregates.node_count_by_type.get("CLAIM") == 1

    def test_facts_sorted_by_belief(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        beliefs = [f.belief for f in s.facts]
        assert beliefs == sorted(beliefs, reverse=True)

    def test_to_dict_roundtrip(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        d = s.to_dict()
        assert d["root_id"] == root_id.hex
        assert isinstance(d["facts"], list)
        assert isinstance(d["supports"], list)
        assert isinstance(d["contradictions"], list)
        assert isinstance(d["aggregates"], dict)

    def test_to_dict_belief_tiers(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        d = s.to_dict()
        for fact in d["facts"]:
            assert fact["belief_tier"] in ("high", "medium", "low", "uninitialized")

    def test_to_dict_polarity_balance(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        d = s.to_dict()
        assert 0.0 <= d["aggregates"]["edge_polarity_balance"] <= 1.0

    def test_edge_record_fields(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        for e in s.supports:
            assert isinstance(e, EdgeRecord)
            assert len(e.source_node_id) == 32  # UUID hex without dashes
            assert len(e.target_node_id) == 32
            assert e.edge_type in ("SUPPORTS", "INFERS", "JUSTIFIES", "ENABLES", "DIRECT")

    def test_contradiction_edge_types(self):
        g, root_id, *_ = _make_graph()
        s = build_verifiable_summary(g, root_id)
        for e in s.contradictions:
            assert e.edge_type in ("ATTACKS", "REBUTS", "CONTRADICTS")

    def test_empty_graph(self):
        g, root_id = _make_empty_graph()
        s = build_verifiable_summary(g, root_id)
        assert len(s.facts) == 1
        assert len(s.supports) == 0
        assert len(s.contradictions) == 0
