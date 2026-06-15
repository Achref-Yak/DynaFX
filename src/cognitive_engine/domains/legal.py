"""Legal domain configuration for the cognitive engine.

Overrides DomainConfig defaults with legal-domain-specific thresholds,
warrants, categories, type mappings, lens defaults, and Level 0 rules.

Usage:
    from cognitive_engine.domain import Domain, domain
    from cognitive_engine.domains.legal import LegalConfig, LegalCoefficients, legal_logic_rules

    legal = Domain("legal", LegalConfig)
    with legal:
        cfg = domain.active()
        assert cfg.conflict_threshold == 0.25
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cognitive_engine.domain import DomainConfig


@dataclass
class LegalCoefficients:
    """Legal-domain specific coefficients for cognitive reasoning.

    Replaces the old Coefficients dataclass from unified.coefficients.
    """
    alpha: float = 0.25
    beta: float = 0.35
    gamma: float = 0.25
    delta: float = 0.15
    level0_rule_strength_threshold: float = 0.6
    level4_max_iterations: int = 30
    level4_convergence_threshold: float = 1e-4
    level5_discount_factor: float = 0.85
    level5_acceptance_threshold: float = 0.1
    level6_lambda_neural: float = 0.4
    level6_logic_penalty_weight: float = 0.15
    level7_lambda_violations: float = 0.15
    level7_max_iterations: int = 50

# ── Legal Logic Rules (Level 0) ──────────────────────────────────
# These rules are loaded into the SymbolicLevel for constraint checking.

legal_logic_rules = [
    # Evidence admissibility rules
    {"antecedents": ["hearsay"], "consequent": "inadmissible", "strength": 0.9},
    {"antecedents": ["no_exception"], "consequent": "inadmissible", "strength": 0.85},
    {"antecedents": ["lacks_foundation"], "consequent": "inadmissible", "strength": 0.85},
    {"antecedents": ["relevance", "probative_value"], "consequent": "admissible", "strength": 0.8},

    # Burden of proof rules
    {"antecedents": ["preponderance", "evidence_strong"], "consequent": "claim_proven", "strength": 0.9},
    {"antecedents": ["clear_and_convincing", "evidence_strong"], "consequent": "claim_proven", "strength": 0.95},
    {"antecedents": ["beyond_reasonable_doubt", "evidence_strong"], "consequent": "claim_proven", "strength": 0.99},

    # Witness credibility rules
    {"antecedents": ["hostile_witness", "prior_inconsistency"], "consequent": "credibility_low", "strength": 0.7},
    {"antecedents": ["expert_witness", "field_matches"], "consequent": "credibility_high", "strength": 0.8},
    {"antecedents": ["eyewitness", "good_visibility"], "consequent": "credibility_moderate", "strength": 0.6},

    # Legal inference rules
    {"antecedents": ["direct_evidence", "admissible"], "consequent": "supports_claim", "strength": 0.85},
    {"antecedents": ["circumstantial_evidence", "admissible"], "consequent": "supports_claim", "strength": 0.6},
    {"antecedents": ["hearsay_evidence", "no_exception"], "consequent": "weakens_claim", "strength": 0.7},

    # Cross-examination rules
    {"antecedents": ["cross_examination", "inconsistent_statement"], "consequent": "credibility_low", "strength": 0.75},
    {"antecedents": ["cross_examination", "evasive_answer"], "consequent": "credibility_low", "strength": 0.65},

    # Expert testimony rules
    {"antecedents": ["expert_qualification", "methodology_valid"], "consequent": "expert_reliable", "strength": 0.85},
    {"antecedents": ["expert_qualification", "methodology_invalid"], "consequent": "expert_unreliable", "strength": 0.8},
]

# LegalCoefficients is defined as the dataclass above.
# Instantiate with defaults: LegalCoefficients()
# Override per-field: LegalCoefficients(alpha=0.3, ...)

# ── Legal Domain Config ───────────────────────────────────────────
LegalConfig = DomainConfig(
    # ── Legal thresholds ──────────────────────────────────────────
    opinion_positive_threshold=-0.1,
    conflict_threshold=0.25,
    analogy_uncertainty_delta=0.3,
    uncertainty_pseudocount=3.0,

    # ── Legal category hierarchy (5 levels) ──────────────────────
    category_levels={
        1: "Necessity",
        2: "Fact",
        3: "Inference",
        4: "Opinion",
        5: "Speculation",
    },

    # ── Legal node type mappings ──────────────────────────────────
    source_type_map={
        "EVIDENCE": "empirical_pattern",
        "CLAIM": "consensus_principle",
        "COUNTERCLAIM": "observational_claim",
        "CONDITION": "cognitive_hypothesis",
        "AXIOM": "consensus_principle",
        "FALLACY": "observational_claim",
        "JUSTIFICATION": "empirical_pattern",
        "TESTIMONY": "observational_claim",
    },

    # ── Legal edge warrants ───────────────────────────────────────
    edge_warrants={
        "SUPPORTS": ((0.85, 0.1, 0.05, 0.5), (0.15, 0.8, 0.05, 0.5)),
        "CONTRADICTS": ((0.1, 0.85, 0.05, 0.5), (0.8, 0.15, 0.05, 0.5)),
        "QUALIFIES": ((0.6, 0.2, 0.2, 0.5), (0.4, 0.4, 0.2, 0.5)),
        "INFERS": ((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5)),
        "JUSTIFIES": ((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5)),
        "ATTACKS": ((0.05, 0.85, 0.1, 0.5), (0.85, 0.1, 0.05, 0.5)),
        "REBUTS": ((0.6, 0.3, 0.1, 0.5), (0.3, 0.6, 0.1, 0.5)),
        # Legal-specific edge types
        "DIRECT": ((0.95, 0.03, 0.02, 0.5), (0.05, 0.93, 0.02, 0.5)),
        "CIRCUMSTANTIAL": ((0.60, 0.15, 0.25, 0.5), (0.30, 0.55, 0.15, 0.5)),
        "HEARSAY": ((0.40, 0.30, 0.30, 0.5), (0.50, 0.30, 0.20, 0.5)),
    },

    # ── Legal mode→edge mappings ──────────────────────────────────
    mode_active_edges={
        "CAUSAL": {"INFERS", "SUPPORTS", "SUPPORT", "ENABLES", "TEMPORAL", "PART_OF", "FLOWS_TO"},
        "CONDITIONAL": {"QUALIFIES", "INFERS", "DEPENDS", "ENABLES"},
        "ARGUMENT": {
            "SUPPORTS", "SUPPORT", "CONTRADICTS", "ATTACKS", "REBUTS",
            "DIRECT", "CIRCUMSTANTIAL", "HEARSAY", "EVIDENCE", "CITES",
        },
        "ANALOGY": {"JUSTIFIES", "SUPPORTS", "SIMILAR"},
    },

    # ── Legal edge roles ──────────────────────────────────────────
    positive_edge_types={"INFERS", "SUPPORTS", "JUSTIFIES", "DIRECT"},
    parent_edge_types={"INFERS", "SUPPORTS", "JUSTIFIES", "DIRECT"},

    # ── Legal lens→mode defaults ──────────────────────────────────
    lens_default_mode={
        "classification": "ARGUMENT",
        "funnel": "CAUSAL",
        "decision-tree": "CONDITIONAL",
        "outlier": "ARGUMENT",
        "aggregation": "ARGUMENT",
        "admissibility": "ARGUMENT",
        "burden_of_proof": "CONDITIONAL",
        "cross_examination": "ARGUMENT",
        "standards_of_proof": "ARGUMENT",
    },
)
