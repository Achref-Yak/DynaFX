"""Tests for EvidenceMatrix — transparent reasoning across sources."""

import pytest
from cognitive_engine.core.models import Opinion
from cognitive_engine.reason.evidence import (
    ClaimAssessment,
    ConsensusLevel,
    EvidenceMatrix,
    EvidenceMatrixResult,
    PairwiseAgreement,
)
from cognitive_engine.reason.fusion import consensus_to_fusion_situation
from cognitive_engine.core.models import FusionSituation


# ── Helpers ──────────────────────────────────────────────────────

def op(b: float, d: float, u: float = 0.0, a: float = 0.5) -> Opinion:
    return Opinion(belief=b, disbelief=d, uncertainty=u, prior=a)


# ── EvidenceMatrix construction ──────────────────────────────────

class TestEvidenceMatrixConstruction:
    def test_empty_matrix(self):
        m = EvidenceMatrix()
        assert m.source_names == []
        assert m.claim_names == []

    def test_add_source(self):
        m = EvidenceMatrix()
        m.add_source("sensor_A", {"temp": op(0.8, 0.1)})
        assert m.source_names == ["sensor_A"]
        assert m.claim_names == ["temp"]

    def test_multiple_sources(self):
        m = EvidenceMatrix()
        m.add_source("A", {"x": op(0.8, 0.1), "y": op(0.3, 0.5)})
        m.add_source("B", {"x": op(0.7, 0.2), "z": op(0.9, 0.0)})
        assert m.source_names == ["A", "B"]
        assert set(m.claim_names) == {"x", "y", "z"}

    def test_remove_source(self):
        m = EvidenceMatrix()
        m.add_source("A", {"x": op(0.8, 0.1)})
        m.add_source("B", {"x": op(0.7, 0.2)})
        m.remove_source("A")
        assert m.source_names == ["B"]

    def test_remove_nonexistent_source(self):
        m = EvidenceMatrix()
        m.remove_source("ghost")  # should not raise


# ── Pairwise agreement ───────────────────────────────────────────

class TestPairwiseAgreement:
    def test_identical_opinions(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.8, 0.1)})
        m.add_source("B", {"claim": op(0.8, 0.1)})
        result = m.compute()
        pw = result.claims["claim"].pairwise
        assert len(pw) == 1
        assert pw[0].agreement > 0.9
        assert not pw[0].conflicts

    def test_conflicting_opinions(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.9, 0.05)})
        m.add_source("B", {"claim": op(0.05, 0.9)})
        result = m.compute()
        pw = result.claims["claim"].pairwise
        assert pw[0].conflicts
        assert pw[0].agreement < 0.3

    def test_uncertain_opinions(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.3, 0.3, 0.4)})
        m.add_source("B", {"claim": op(0.2, 0.2, 0.6)})
        result = m.compute()
        pw = result.claims["claim"].pairwise
        # Similar balanced opinions → high agreement (L1 distance is small)
        assert pw[0].agreement > 0.8
        assert not pw[0].conflicts


# ── Consensus classification ─────────────────────────────────────

class TestConsensusClassification:
    def test_strong_agreement(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.9, 0.05)})
        m.add_source("B", {"claim": op(0.85, 0.1)})
        m.add_source("C", {"claim": op(0.88, 0.07)})
        result = m.compute()
        assert result.claims["claim"].consensus == ConsensusLevel.STRONG_AGREEMENT

    def test_strong_disagreement(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.9, 0.05)})
        m.add_source("B", {"claim": op(0.05, 0.9)})
        m.add_source("C", {"claim": op(0.9, 0.05)})
        result = m.compute()
        assert result.claims["claim"].consensus == ConsensusLevel.STRONG_DISAGREEMENT

    def test_contested(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.7, 0.2)})
        m.add_source("B", {"claim": op(0.2, 0.7)})
        m.add_source("C", {"claim": op(0.5, 0.4)})
        result = m.compute()
        assert result.claims["claim"].consensus == ConsensusLevel.CONTESTED

    def test_mild_agreement(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.5, 0.3)})
        m.add_source("B", {"claim": op(0.4, 0.3)})
        result = m.compute()
        assert result.claims["claim"].consensus == ConsensusLevel.MILD_AGREEMENT

    def test_single_source(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.8, 0.1)})
        result = m.compute()
        assert result.claims["claim"].consensus == ConsensusLevel.STRONG_AGREEMENT


# ── Opinion fusion ───────────────────────────────────────────────

class TestFusion:
    def test_fused_opinion_between_sources(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.8, 0.1)})
        m.add_source("B", {"claim": op(0.6, 0.2)})
        result = m.compute()
        fused = result.claims["claim"].fused
        # Fused belief should be between the two source beliefs
        assert 0.5 < fused.belief < 0.9
        assert fused.disbelief < 0.3

    def test_fused_with_conflict(self):
        m = EvidenceMatrix()
        m.add_source("A", {"claim": op(0.9, 0.05)})
        m.add_source("B", {"claim": op(0.05, 0.9)})
        result = m.compute()
        fused = result.claims["claim"].fused
        # Conflicting opinions fuse to balanced (belief ≈ disbelief)
        assert abs(fused.belief - fused.disbelief) < 0.1


# ── EvidenceMatrixResult ─────────────────────────────────────────

class TestResult:
    def test_contested_claims(self):
        m = EvidenceMatrix()
        m.add_source("A", {"agree": op(0.9, 0.05), "fight": op(0.9, 0.05)})
        m.add_source("B", {"agree": op(0.85, 0.1), "fight": op(0.05, 0.9)})
        result = m.compute()
        assert "fight" in result.contested_claims()
        assert "agree" not in result.contested_claims()

    def test_agreed_claims(self):
        m = EvidenceMatrix()
        m.add_source("A", {"agree": op(0.9, 0.05), "fight": op(0.9, 0.05)})
        m.add_source("B", {"agree": op(0.85, 0.1), "fight": op(0.05, 0.9)})
        result = m.compute()
        assert "agree" in result.agreed_claims()

    def test_summary(self):
        m = EvidenceMatrix()
        m.add_source("A", {"x": op(0.9, 0.05)})
        m.add_source("B", {"x": op(0.85, 0.1)})
        result = m.compute()
        s = result.summary()
        assert "2 sources" in s
        assert "1 claims" in s
        assert "strong_agreement" in s

    def test_to_dict_roundtrip(self):
        m = EvidenceMatrix()
        m.add_source("A", {"x": op(0.8, 0.1), "y": op(0.3, 0.5)})
        m.add_source("B", {"x": op(0.7, 0.2)})
        result = m.compute()
        d = result.to_dict()
        assert d["source_count"] == 2
        assert d["claim_count"] == 2
        assert "x" in d["claims"]
        assert "y" in d["claims"]


# ── Serialization ────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_from_dict(self):
        m = EvidenceMatrix()
        m.add_source("sensor_A", {"temp": op(0.8, 0.1), "pressure": op(0.6, 0.3)})
        m.add_source("sensor_B", {"temp": op(0.7, 0.2)})
        d = m.to_dict()
        m2 = EvidenceMatrix.from_dict(d)
        assert set(m2.source_names) == {"sensor_A", "sensor_B"}
        assert set(m2.claim_names) == {"pressure", "temp"}
        # Verify opinions round-tripped
        r1 = m.compute()
        r2 = m2.compute()
        assert abs(r1.claims["temp"].fused.belief - r2.claims["temp"].fused.belief) < 1e-6


# ── Integration: real-world scenario ─────────────────────────────

class TestRealWorldScenario:
    def test_multi_sensor_fusion(self):
        """Simulate: 3 sensors reporting on 5 claims, 2 conflict."""
        m = EvidenceMatrix()
        m.add_source("sensor_1", {
            "fire": op(0.9, 0.05),
            "smoke": op(0.85, 0.1),
            "heat": op(0.7, 0.2),
            "co2": op(0.3, 0.1),
            "glitch": op(0.8, 0.1),
        })
        m.add_source("sensor_2", {
            "fire": op(0.8, 0.1),
            "smoke": op(0.9, 0.05),
            "heat": op(0.75, 0.15),
            "co2": op(0.35, 0.15),
            "glitch": op(0.1, 0.8),  # disagrees with sensor_1
        })
        m.add_source("sensor_3", {
            "fire": op(0.85, 0.08),
            "smoke": op(0.88, 0.07),
            "heat": op(0.65, 0.25),
            "co2": op(0.4, 0.1),
            "glitch": op(0.15, 0.75),  # agrees with sensor_2
        })

        result = m.compute()

        # fire, smoke, heat should agree
        assert result.claims["fire"].consensus == ConsensusLevel.STRONG_AGREEMENT
        assert result.claims["smoke"].consensus == ConsensusLevel.STRONG_AGREEMENT

        # glitch should be contested (sensor_1 says yes, others say no)
        assert result.claims["glitch"].consensus in (
            ConsensusLevel.CONTESTED, ConsensusLevel.STRONG_DISAGREEMENT
        )

        # co2 should be mild agreement (all have low belief/disbelief)
        assert result.claims["co2"].consensus in (
            ConsensusLevel.MILD_AGREEMENT, ConsensusLevel.STRONG_AGREEMENT
        )

    def test_source_weight_affects_fusion(self):
        """Verify that the fused opinion reflects source agreement."""
        m = EvidenceMatrix()
        # Three sources agree on high belief
        m.add_source("A", {"claim": op(0.8, 0.1)})
        m.add_source("B", {"claim": op(0.75, 0.15)})
        m.add_source("C", {"claim": op(0.85, 0.05)})
        result = m.compute()
        fused = result.claims["claim"].fused
        assert fused.belief > 0.7

    def test_full_pipeline(self):
        """End-to-end: build, compute, serialize, interpret."""
        m = EvidenceMatrix()
        m.add_source("expert_A", {"policy_x": op(0.7, 0.2), "policy_y": op(0.3, 0.5)})
        m.add_source("expert_B", {"policy_x": op(0.65, 0.25), "policy_y": op(0.6, 0.3)})
        m.add_source("data_C", {"policy_x": op(0.8, 0.1), "policy_y": op(0.4, 0.4)})

        result = m.compute()
        d = result.to_dict()

        assert d["source_count"] == 3
        # policy_x: all agree (strong belief), policy_y: mild (no active conflicts)
        assert "policy_x" in d["agreed"]
        # policy_y opinions differ but don't actively conflict (no b>0.5 AND d>0.5 pairs)


# ── Integration: fusion situation classification ─────────────────

class TestFusionSituationIntegration:
    def test_consensus_to_fusion_situation(self):
        assert consensus_to_fusion_situation(ConsensusLevel.STRONG_AGREEMENT, 3) == FusionSituation.INDEPENDENT_SOURCES
        assert consensus_to_fusion_situation(ConsensusLevel.MILD_AGREEMENT, 3) == FusionSituation.INDEPENDENT_SOURCES
        assert consensus_to_fusion_situation(ConsensusLevel.CONTESTED, 3) == FusionSituation.CONFLICTING_VIEWS
        assert consensus_to_fusion_situation(ConsensusLevel.STRONG_DISAGREEMENT, 3) == FusionSituation.CONFLICTING_VIEWS

    def test_single_source_returns_independent(self):
        assert consensus_to_fusion_situation(ConsensusLevel.STRONG_AGREEMENT, 1) == FusionSituation.INDEPENDENT_SOURCES

    def test_classify_fusion_situations(self):
        m = EvidenceMatrix()
        m.add_source("A", {"agree": op(0.9, 0.05), "fight": op(0.9, 0.05)})
        m.add_source("B", {"agree": op(0.85, 0.1), "fight": op(0.05, 0.9)})
        result = m.compute()
        situations = result.classify_fusion_situations()
        assert situations["agree"] == FusionSituation.INDEPENDENT_SOURCES
        assert situations["fight"] == FusionSituation.CONFLICTING_VIEWS

    def test_evidence_matrix_importable(self):
        from cognitive_engine.reason import EvidenceMatrix, EvidenceMatrixResult, ConsensusLevel
        m = EvidenceMatrix()
        m.add_source("A", {"x": op(0.8, 0.1)})
        result = m.compute()
        assert isinstance(result, EvidenceMatrixResult)
