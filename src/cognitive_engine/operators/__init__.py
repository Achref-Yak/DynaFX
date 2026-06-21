"""Operators — reusable cognitive transformations.

Every operator satisfies:
    O: (State, **kwargs) -> State

Operator implementations live in sub-modules:
    extract.py        — Text -> Graph
    systems.py        — Feedback loops, leverage points, archetypes
    propagate.py      — 8-level belief propagation
    schema.py (core)  — Schema application (Σ)
    graph.py          — Graph building/edge assignment
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
