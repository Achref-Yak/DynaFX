"""Operators — reusable cognitive transformations.

Every operator satisfies:
    O: (State, **kwargs) -> State

This is the industry standard pattern:
    - External: uniform interface (composable)
    - Internal: typed parameters (safe)

Operators:
    Ξ (Extract)      — Text -> Graph
    Σ (Schema)       — Apply domain schema
    Ĝ (Graph)        — Build/assign edges
    ⊗ (Propagate)    — Run 8-level reasoning
    ΠΩ (Constraint)  — Filter inconsistencies
    π (Attention)    — Select relevant subgraph
    κ (Compress)     — Summarize graph
    Δ (Update)       — Track state changes
    M (Merge)        — Combine multiple graphs
    T (Temporal)     — Time-series alignment
    Sim (Simulate)   — What-if analysis
    D (Compare)      — Structured diff between graphs
    Align            — Semantic concept alignment
    𝒜 (Abduce)      — Infer best explanation from causes
    Ι (Induce)       — Generalize from observations
    Analogy          — Transfer structure between domains
    FeedbackLoops    — Detect reinforcing/balancing loops
    LeveragePoints   — Identify high-leverage interventions
    SystemArchetypes — Classify system archetypes
    CausalSCM        — Structural causal model operations
"""

from cognitive_engine.core.operator import Operator
from cognitive_engine.core.pipeline import Pipeline
from cognitive_engine.core.state import State, StateDelta
from cognitive_engine.core.schema import Schema

__all__ = [
    "Operator",
    "Pipeline",
    "State",
    "StateDelta",
    "Schema",
]
