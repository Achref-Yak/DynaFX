"""Epistemics — Subjective Logic algebra, evidence matrix, argumentation, KBT.

Core SL algebra::

    from dynafx.epistemics import Opinion, cumulative_fusion, consensus_compromise

Evidence and argumentation::

    from dynafx.epistemics import EvidenceMatrix, build_framework
"""

from dynafx.epistemics.evidence import (
    ClaimAssessment,
    ConsensusLevel,
    EvidenceMatrix,
    EvidenceMatrixResult,
    PairwiseAgreement,
)
from dynafx.epistemics.fusion import (
    consensus_compromise,
    consensus_to_fusion_situation,
    cumulative_fusion,
)
from dynafx.epistemics.argumentation import (
    Argument,
    ArgumentationFramework,
    Attack,
    AttackType,
    SupportType,
    build_framework,
)
from dynafx.epistemics.kbt import KBTResult, compute_kbt
from dynafx.epistemics.graph_ops import (
    list_nodes,
    get_node,
    get_edge,
    query_contested,
    query_by_role,
    get_trace_history,
    create_node,
    create_edge,
    set_role,
    set_parameter,
    merge_nodes,
    retract,
)
from dynafx.epistemics.sl_validation import (
    ArgumentType,
    ValidationResult,
    ValidationArgument,
    ValidationAttack,
    ValidationResultDetail,
    validate_system_internal,
    get_validation_summary,
)
from dynafx.epistemics.sl_params import (
    bind_parameters,
    get_parameter_summary,
)

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
    # SL graph utilities
    "list_nodes",
    "get_node",
    "get_edge",
    "query_contested",
    "query_by_role",
    "get_trace_history",
    "create_node",
    "create_edge",
    "set_role",
    "set_parameter",
    "merge_nodes",
    "retract",
    # SL validation
    "ArgumentType",
    "ValidationResult",
    "ValidationArgument",
    "ValidationAttack",
    "ValidationResultDetail",
    "validate_system_internal",
    "get_validation_summary",
    # SL params
    "bind_parameters",
    "get_parameter_summary",
]
