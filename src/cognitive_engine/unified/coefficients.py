"""Learnable coefficients for the Master Equation and all levels.

Coefficients are fixed initially (heuristic defaults). Learning via
gradient descent on labeled (graph, correct_beliefs) pairs is planned
for a later phase.

Usage:
    from cognitive_engine.unified.coefficients import Coefficients
    coeffs = Coefficients()  # defaults
    coeffs.alpha = 0.4       # tune
    coeffs.save("coeffs.json")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Coefficients:
    """All learnable coefficients for the unified reasoning framework.

    Master Equation:
        R(x) = αP(x) + βG(x) + γL(x) - δA(x)

    Where:
        P(x) = probability evidence  (Level 2)
        G(x) = graph propagation     (Level 4)
        L(x) = logic consistency     (Level 0)
        A(x) = adversarial contradiction (Level 5)
    """

    # ── Master equation weights ───────────────────────────────────
    alpha: float = 0.3
    """Weight on probability evidence P(x)."""

    beta: float = 0.3
    """Weight on graph propagation G(x)."""

    gamma: float = 0.2
    """Weight on logic consistency L(x)."""

    delta: float = 0.2
    """Weight on adversarial contradiction A(x)."""

    # ── Level 0: Symbolic Logic ───────────────────────────────────
    level0_rule_strength_threshold: float = 0.5
    """Minimum rule strength to fire Modus Ponens."""

    # ── Level 1: Cognitive Architecture ───────────────────────────
    level1_temperature: float = 1.0
    """Softmax temperature for retrieval probability."""

    level1_decay_rate: float = 0.05
    """Base-level activation decay per time step."""

    level1_firing_threshold: float = 0.0
    """Minimum utility for production rule to fire."""

    # ── Level 2: Probabilistic Reasoning ──────────────────────────
    level2_prior_weight: float = 0.5
    """Weight on prior vs likelihood in Bayesian update."""

    # ── Level 3: Neural Reasoning ─────────────────────────────────
    level3_embedding_dim: int = 64
    """Dimension of node/edge embeddings."""

    level3_attention_heads: int = 4
    """Number of multi-head attention layers."""

    level3_hidden_dim: int = 128
    """Hidden dimension in GNN MLP."""

    level3_num_layers: int = 2
    """Number of GNN propagation layers."""

    level3_dropout: float = 0.1
    """Dropout rate in GNN layers."""

    level3_lambda_neural: float = 0.5
    """Neural vs symbolic weight in Level 6 fusion."""

    # ── Level 4: Graph Propagation ────────────────────────────────
    level4_learning_rate: float = 0.01
    """Learning rate for graph propagation updates."""

    level4_max_iterations: int = 50
    """Maximum propagation iterations before forced stop."""

    level4_convergence_threshold: float = 1e-4
    """L2 norm threshold for convergence detection."""

    # ── Level 5: Argumentation ────────────────────────────────────
    level5_discount_factor: float = 0.9
    """Discount factor for argument strength propagation."""

    level5_acceptance_threshold: float = 0.0
    """Net acceptability threshold for Dung acceptance."""

    # ── Level 6: Neuro-Symbolic ───────────────────────────────────
    level6_lambda_neural: float = 0.5
    """Fusion weight: neural contribution vs symbolic."""

    level6_logic_penalty_weight: float = 0.1
    """Weight on logic penalty in total loss."""

    # ── Level 7: Unified Graph Truth ──────────────────────────────
    level7_lambda_violations: float = 0.1
    """Penalty weight for constraint violations in global objective."""

    level7_max_iterations: int = 100
    """Maximum fixed-point iterations."""

    # ── Global convergence ────────────────────────────────────────
    convergence_threshold: float = 1e-6
    """Global convergence threshold across all levels."""

    # ── Persistence ───────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Save coefficients to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Coefficients:
        """Load coefficients from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Coefficients:
        """Deserialize from dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
