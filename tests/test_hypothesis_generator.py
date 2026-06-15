"""Tests for perception/hypothesis_generator.py."""

from uuid import uuid4

import pytest

from cognitive_engine.perception.hypothesis_generator import (
    CandidateHypothesis, HypothesisGenerator,
)
from cognitive_engine.kernel.assertion_gate import Assertion


class TestCandidateHypothesis:
    def test_defaults(self):
        ch = CandidateHypothesis()
        assert ch.relation_type == "SUPPORTS"
        assert ch.score == 0.5
        assert ch.premise == ""
        assert ch.hypothesis == ""

    def test_fields(self):
        src, tgt = uuid4(), uuid4()
        ch = CandidateHypothesis(
            source_id=src,
            target_id=tgt,
            relation_type="ATTACKS",
            score=0.8,
            premise="source text",
            hypothesis="target text",
        )
        assert ch.source_id == src
        assert ch.target_id == tgt
        assert ch.relation_type == "ATTACKS"
        assert ch.score == 0.8

    def test_to_assertion(self):
        ch = CandidateHypothesis(
            premise="This is evidence",
            hypothesis="This is a claim",
            score=0.75,
            relation_type="SUPPORTS",
        )
        a = ch.to_assertion()
        assert isinstance(a, Assertion)
        assert a.source == "hypothesis_generator"
        assert a.opinion == (0.5, 0.2, 0.3, 0.5)
        assert a.metadata["hypothesis"] is True
        assert a.metadata["score"] == 0.75
        assert a.metadata["relation_type"] == "SUPPORTS"

    def test_to_assertion_has_ids(self):
        ch = CandidateHypothesis()
        a = ch.to_assertion()
        assert a.id == ch.id


class TestHypothesisGenerator:
    def test_default_threshold(self):
        hg = HypothesisGenerator()
        assert hg.threshold == 0.3

    def test_custom_threshold(self):
        hg = HypothesisGenerator(threshold=0.7)
        assert hg.threshold == 0.7

    def test_generate_empty(self):
        hg = HypothesisGenerator()
        candidates = hg.generate({}, {}, [])
        assert candidates == []

    def test_generate_single_pair(self):
        hg = HypothesisGenerator(threshold=0.0)
        src, tgt = uuid4(), uuid4()
        candidates = hg.generate(
            {src: "evidence supports claim", tgt: "claim is true"},
            {src: "EVIDENCE", tgt: "CLAIM"},
            [],
        )
        assert len(candidates) >= 1
        assert candidates[0].relation_type == "SUPPORTS"

    def test_generate_respects_max(self):
        hg = HypothesisGenerator(threshold=0.0)
        ids = [uuid4() for _ in range(5)]
        texts = {nid: f"text_{i}" for i, nid in enumerate(ids)}
        types = {nid: "CLAIM" for nid in ids}
        candidates = hg.generate(texts, types, [], max_candidates=3)
        assert len(candidates) <= 3

    def test_generate_skips_existing_edges(self):
        hg = HypothesisGenerator(threshold=0.0)
        src, tgt = uuid4(), uuid4()
        from cognitive_engine.core.models import Edge, EdgeType
        existing = [Edge(source_id=src, target_id=tgt, type=EdgeType.SUPPORTS)]
        candidates = hg.generate(
            {src: "text a", tgt: "text b"},
            {src: "CLAIM", tgt: "EVIDENCE"},
            existing,
        )
        pairs = {(c.source_id, c.target_id) for c in candidates}
        assert (src, tgt) not in pairs

    def test_generate_skips_self_loops(self):
        hg = HypothesisGenerator(threshold=0.0)
        nid = uuid4()
        candidates = hg.generate(
            {nid: "same text", nid: "same text"},
            {nid: "CLAIM"},
            [],
        )
        pairs = {(c.source_id, c.target_id) for c in candidates}
        assert (nid, nid) not in pairs

    def test_score_pair_word_overlap(self):
        hg = HypothesisGenerator()
        score = hg._score_pair("apple banana cherry", "apple banana date", "CLAIM", "EVIDENCE")
        assert 0 < score <= 1.0

    def test_score_pair_type_compatible(self):
        hg = HypothesisGenerator()
        score = hg._score_pair("hello world", "goodbye world", "EVIDENCE", "CLAIM")
        assert score > 0

    def test_score_pair_empty_texts(self):
        hg = HypothesisGenerator()
        score = hg._score_pair("", "", "CLAIM", "CLAIM")
        assert score >= 0

    def test_infer_relation_evidence_to_claim(self):
        hg = HypothesisGenerator()
        assert hg._infer_relation("", "", "EVIDENCE", "CLAIM") == "SUPPORTS"
        assert hg._infer_relation("", "", "EVIDENCE", "HYPOTHESIS") == "SUPPORTS"

    def test_infer_relation_counterclaim(self):
        hg = HypothesisGenerator()
        assert hg._infer_relation("", "", "COUNTERCLAIM", "CLAIM") == "ATTACKS"

    def test_infer_relation_claim_to_evidence(self):
        hg = HypothesisGenerator()
        assert hg._infer_relation("", "", "CLAIM", "EVIDENCE") == "JUSTIFIES"
        assert hg._infer_relation("", "", "HYPOTHESIS", "EVIDENCE") == "JUSTIFIES"

    def test_infer_relation_default(self):
        hg = HypothesisGenerator()
        assert hg._infer_relation("", "", "UNKNOWN", "OTHER") == "SUPPORTS"

    def test_type_compatible(self):
        hg = HypothesisGenerator()
        assert hg._type_compatible("AXIOM", "") is True
        assert hg._type_compatible("FALLACY", "CLAIM") is True
        assert hg._type_compatible("FALLACY", "EVIDENCE") is False
        assert hg._type_compatible("", "") is True

    def test_generate_sorts_by_score(self):
        hg = HypothesisGenerator(threshold=0.0)
        ids = [uuid4(), uuid4(), uuid4()]
        texts = {ids[0]: "major crime evidence", ids[1]: "legal claim against defendant",
                 ids[2]: "unrelated minor detail"}
        types = {nid: "CLAIM" for nid in ids}
        candidates = hg.generate(texts, types, [])
        for i in range(len(candidates) - 1):
            assert candidates[i].score >= candidates[i + 1].score
