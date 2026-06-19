"""Domain configuration framework.

Provides:
    - DomainConfig: all tunable knobs with current-environment defaults.
    - Domain: context-scoped activation via contextvars (thread-safe).
    - domain.active(): read the current domain config from anywhere.

Usage:
    from cognitive_engine.domain import Domain, DomainConfig, domain

    cfg = domain.active()                     # default config
    threshold = cfg.conflict_threshold         # 0.3

    with Domain("legal", DomainConfig(...)):   # activate legal domain
        cfg = domain.active()                  # legal config
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DomainConfig:
    """All tunable configuration knobs for the cognitive engine.

    Every field has a default matching the current hard-coded behavior.
    Domain packs subclass or instantiate this with domain-specific overrides.
    """

    # ── Thresholds ────────────────────────────────────────────────
    opinion_positive_threshold: float = 0.0
    """Offset for 'b > d' classification (evidence.py, store.py)."""

    conflict_threshold: float = 0.3
    """Minimum belief+disbelief for opinions to be conflicting (fusion.py)."""

    analogy_uncertainty_delta: float = 0.2
    """Fraction of belief shifted to uncertainty in ANALOGY mode (mode_operators.py)."""

    uncertainty_pseudocount: float = 2.0
    """Dirichlet pseudocount W (models.py, evidence.py, store.py)."""

    trust_weight_alpha: float = 0.5
    """Coefficient on uncertainty in trust weight: b + alpha * u (sl_operators.py)."""

    # ── Numeric tolerances ────────────────────────────────────────
    clamp_epsilon: float = 1e-9
    """Epsilon for b+d+u == 1 invariant check in _clamp (fusion.py, sl_operators.py)."""

    validation_opinion_tolerance: float = 0.01
    """Tolerance for b+d+u == 1 in validators (validators.py)."""

    validation_u_tolerance: float = -0.001
    """Lower bound for uncertainty in validators (validators.py)."""

    # ── Category hierarchy ────────────────────────────────────────
    category_levels: dict[int, str] = field(default_factory=lambda: {
        1: "Necessity",
        2: "Fact",
        3: "Belief",
        4: "Concept",
    })

    # ── Entity Linking (Soft Wordlists) ───────────────────────────
    entity_linking_threshold: float = 0.85
    """Cosine similarity threshold for soft wordlist entity linking."""

    canonical_concepts: dict[str, str] = field(default_factory=lambda: {
        "Database": "Database",
        "Algorithm": "Algorithm",
        "Analysis": "Analysis",
        "Application": "Application",
        "Architecture": "Architecture",
        "Attack": "Attack",
        "Authentication": "Authentication",
        "Compliance": "Compliance",
        "Environment": "Environment",
        "Infrastructure": "Infrastructure",
        "Network": "Network",
        "Security": "Security",
        "System": "System",
        "User": "User",
        "Access": "Access",
        "Approach": "Approach",
        "Argument": "Argument",
        "Audit": "Audit",
        "Client": "Client",
        "Compatibility": "Compatibility",
        "Complexity": "Complexity",
        "Component": "Component",
        "Configuration": "Configuration",
        "Consistency": "Consistency",
        "Data": "Data",
        "Deadline": "Deadline",
        "Endpoint": "Endpoint",
        "Failure": "Failure",
        "Feature": "Feature",
        "Function": "Function",
        "Gateway": "Gateway",
        "Instance": "Instance",
        "Interface": "Interface",
        "Interval": "Interval",
        "Layer": "Layer",
        "Library": "Library",
        "Limit": "Limit",
        "Middleware": "Middleware",
        "Module": "Module",
        "Node": "Node",
        "Optimization": "Optimization",
        "Pipeline": "Pipeline",
        "Platform": "Platform",
        "Policy": "Policy",
        "Process": "Process",
        "Protocol": "Protocol",
        "Request": "Request",
        "Resource": "Resource",
        "Role": "Role",
        "Route": "Route",
        "Schema": "Schema",
        "Server": "Server",
        "Service": "Service",
        "Session": "Session",
        "Strategy": "Strategy",
        "Task": "Task",
        "Team": "Team",
        "Template": "Template",
        "Throttling": "Throttling",
        "Token": "Token",
        "Validator": "Validator",
        "Version": "Version",
        "Worker": "Worker",
    })

    # ── NodeType → opinion template mapping ──────────────────────
    source_type_map: dict[str, str] = field(default_factory=lambda: {
        "EVIDENCE": "empirical_pattern",
        "CLAIM": "consensus_principle",
        "COUNTERCLAIM": "observational_claim",
        "CONDITION": "cognitive_hypothesis",
        "AXIOM": "consensus_principle",
        "FALLACY": "observational_claim",
        "JUSTIFICATION": "empirical_pattern",
    })

    # ── Default opinion templates ─────────────────────────────────
    default_opinions: dict[str, tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "empirical_pattern": (0.8, 0.1, 0.1, 0.5),
            "cognitive_hypothesis": (0.5, 0.2, 0.3, 0.5),
            "consensus_principle": (0.7, 0.1, 0.2, 0.5),
            "observational_claim": (0.4, 0.3, 0.3, 0.5),
            "total_ignorance": (0.0, 0.0, 1.0, 0.5),
        }
    )

    default_warrant: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = ((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5))

    default_template: str = "observational_claim"
    """Fallback template name when source_type_map lookup fails."""

    total_ignorance: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.5)

    # ── Edge warrants ─────────────────────────────────────────────
    edge_warrants: dict[
        str,
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ],
    ] = field(default_factory=lambda: {
        "SUPPORTS": ((0.85, 0.1, 0.05, 0.5), (0.15, 0.8, 0.05, 0.5)),
        "CONTRADICTS": ((0.1, 0.85, 0.05, 0.5), (0.8, 0.15, 0.05, 0.5)),
        "QUALIFIES": ((0.6, 0.2, 0.2, 0.5), (0.4, 0.4, 0.2, 0.5)),
        "INFERS": ((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5)),
        "JUSTIFIES": ((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5)),
        "ATTACKS": ((0.05, 0.85, 0.1, 0.5), (0.85, 0.1, 0.05, 0.5)),
        "REBUTS": ((0.6, 0.3, 0.1, 0.5), (0.3, 0.6, 0.1, 0.5)),
        "DIRECT": ((0.90, 0.05, 0.05, 0.5), (0.10, 0.85, 0.05, 0.5)),
        "CIRCUMSTANTIAL": ((0.60, 0.20, 0.20, 0.5), (0.30, 0.55, 0.15, 0.5)),
        "HEARSAY": ((0.40, 0.30, 0.30, 0.5), (0.50, 0.30, 0.20, 0.5)),
    })

    # ── Mode → active edge string names ───────────────────────────
    mode_active_edges: dict[str, set[str]] = field(default_factory=lambda: {
        "CAUSAL": {"INFERS", "SUPPORTS", "SUPPORT", "CAUSES",
                    "ENABLES", "TEMPORAL", "PART_OF", "FLOWS_TO",
                    "LOCATED_AT", "TRANSFORMS", "PRODUCES", "CONSUMES",
                    "USES", "HAS_GOAL", "INTENDS", "KNOWS",
                    "COMMUNICATED", "PREFERS"},
        "CONDITIONAL": {"QUALIFIES", "INFERS", "DEPENDS", "ENABLES",
                        "HAS_ATTRIBUTE", "EMPLOYED_BY", "CONTACT_OF"},
        "ARGUMENT": {"SUPPORTS", "SUPPORT", "CONTRADICTS", "ATTACKS", "REBUTS",
                     "DIRECT", "CIRCUMSTANTIAL", "HEARSAY", "EVIDENCE", "CITES",
                     "HAS_ATTRIBUTE", "EMPLOYED_BY", "CONTACT_OF", "LOCATED_AT"},
        "ANALOGY": {"JUSTIFIES", "SUPPORTS", "SIMILAR", "ASSOCIATED_WITH"},
    })

    # ── Edge role definitions ─────────────────────────────────────
    positive_edge_types: set[str] = field(
        default_factory=lambda: {"INFERS", "SUPPORTS", "JUSTIFIES", "DIRECT"}
    )
    """Edges used for cycle-breaking and parent-child tree construction."""

    parent_edge_types: set[str] = field(
        default_factory=lambda: {"INFERS", "SUPPORTS", "JUSTIFIES", "DIRECT"}
    )
    """Edges that establish parent-child relationships in ConversationTree."""

    # ── Lens → mode defaults ──────────────────────────────────────
    lens_default_mode: dict[str, str] = field(default_factory=dict)
    """Maps lens name to default reasoning mode name.

    Populated by domain packs (e.g. a domain maps classification→ARGUMENT).
    """

    # ── CLI defaults ──────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 128
    merge_margin: int = 20
    glob_pattern: str = "*.txt"
    db_default: str = "evidence.db"

    # ── Sweep / sensitivity analysis ──────────────────────────────
    sweep_delta: float = 0.1
    """Step size for prior sweeping in sweep_priors (config.py)."""


_default_config: DomainConfig = DomainConfig()
"""Cached default config so domain.active() is cheap."""


_current_domain: contextvars.ContextVar[Optional["Domain"]] = (
    contextvars.ContextVar("_current_domain", default=None)
)


class Domain:
    """A named domain configuration with context-scoped activation.

    Thread-safe via contextvars — each thread/context can activate
    a different domain without affecting others.

    Usage:
        custom = Domain("custom", DomainConfig(conflict_threshold=0.25))
        with custom:
            cfg = domain.active()   # custom config
        cfg = domain.active()       # back to default
    """

    def __init__(self, name: str, config: DomainConfig) -> None:
        self._name = name
        self._config = config
        Domain._instances[name] = self

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> DomainConfig:
        return self._config

    def __enter__(self) -> Domain:
        self._token = _current_domain.set(self)
        return self

    def __exit__(self, *args) -> None:
        _current_domain.reset(self._token)

    @staticmethod
    def active() -> DomainConfig:
        """Return the current domain config, or default if none is active."""
        d = _current_domain.get()
        return d._config if d is not None else _default_config

    @staticmethod
    def get(name: str) -> Optional[Domain]:
        """Retrieve a registered domain by name, or None."""
        return Domain._instances.get(name)

    _instances: dict[str, Domain] = {}


# Module-level convenience: domain.active(), with domain: ...
domain = Domain
