"""Builtin operator policies for common domains.

Each policy is a set of rules that map state conditions
to operator selections. Replaces hardcoded heuristic rules
with declarative YAML policies.
"""

from __future__ import annotations

from cognitive_engine.policy.schema import (
    OperatorPolicy, PolicyRule, WhenCondition, ThenAction,
)

# ── Default policy: general-purpose reasoning ──────────────────────

DEFAULT_POLICY = OperatorPolicy(
    name="default",
    description="General-purpose reasoning policy",
    rules=[
        PolicyRule(
            when=WhenCondition(cycle="==1"),
            then=ThenAction(operators=["extract", "schema", "graph"], order="sequential"),
        ),
        PolicyRule(
            when=WhenCondition(graph_has_contradictions=True),
            then=ThenAction(operators=["constraint", "debate", "verify"], order="sequential"),
        ),
        PolicyRule(
            when=WhenCondition(graph_mean_uncertainty=">0.4"),
            then=ThenAction(operators=["abduce", "induce"], order="parallel"),
        ),
        PolicyRule(
            when=WhenCondition(convergence_stalled=True),
            then=ThenAction(operators=["compress", "merge", "reason"], order="sequential"),
        ),
    ],
    fallback=ThenAction(operators=["propagate", "verify"], order="sequential"),
)

# ── Scientific policy: hypothesis evaluation ──────────────────────

SCIENTIFIC_POLICY = OperatorPolicy(
    name="scientific",
    description="Hypothesis evaluation and evidence fusion",
    rules=[
        PolicyRule(
            when=WhenCondition(cycle="==1"),
            then=ThenAction(operators=["extract", "schema", "graph"], order="sequential"),
        ),
        PolicyRule(
            when=WhenCondition(domain="scientific", cycle=">1"),
            then=ThenAction(operators=["propagate", "induce", "abduce"], order="sequential"),
        ),
        PolicyRule(
            when=WhenCondition(graph_mean_uncertainty=">0.5"),
            then=ThenAction(operators=["reason", "compare", "verify"], order="sequential"),
        ),
    ],
    fallback=ThenAction(operators=["propagate", "verify"], order="sequential"),
)

# ── Policy registry ────────────────────────────────────────────────

BUILTIN_POLICIES: dict[str, OperatorPolicy] = {
    "default": DEFAULT_POLICY,
    "scientific": SCIENTIFIC_POLICY,
}
