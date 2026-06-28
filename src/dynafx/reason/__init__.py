"""Reasoning package — evidence matrix, fusion, argumentation, KBT, and analysis."""

from dynafx.reason.evidence import (
    ClaimAssessment,
    ConsensusLevel,
    EvidenceMatrix,
    EvidenceMatrixResult,
    PairwiseAgreement,
)
from dynafx.reason.fusion import (
    consensus_compromise,
    consensus_to_fusion_situation,
    cumulative_fusion,
)
from dynafx.reason.argumentation import (
    Argument,
    ArgumentationFramework,
    Attack,
    AttackType,
    SupportType,
    build_framework,
)
from dynafx.reason.kbt import KBTResult, compute_kbt

__all__ = [
    "Argument",
    "ArgumentationFramework",
    "Attack",
    "AttackType",
    "ClaimAssessment",
    "ConsensusLevel",
    "EvidenceMatrix",
    "EvidenceMatrixResult",
    "KBTResult",
    "PairwiseAgreement",
    "SupportType",
    "build_framework",
    "compute_kbt",
    "consensus_compromise",
    "consensus_to_fusion_situation",
    "cumulative_fusion",
]
