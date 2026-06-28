"""Operators — reusable cognitive transformations.

Every operator satisfies:
    O: (State, **kwargs) -> State

Operator implementations live in sub-modules:
    systems.py        — Feedback loops, leverage points, archetypes
    propagate.py      — 8-level belief propagation
    schema.py (core)  — Schema application (Σ)
    graph.py          — Graph building/edge assignment
"""

from dynafx.core.operator import Operator
from dynafx.core.pipeline import Pipeline
from dynafx.core.state import State, StateDelta
from dynafx.core.schema import Schema

__all__ = [
    "Operator",
    "Pipeline",
    "State",
    "StateDelta",
    "Schema",
]
