"""Monterey Phoenix (MP) — simplified scenario generator.

Generates all possible event orderings permitted by precedence
and inclusion constraints, enabling behavioral stress-testing.

Based on:
    - Monterey Phoenix (MP) event grammar
    - Precedence relations (A must happen before B)
    - Inclusion relations (A is part of B)
    - Assertion checking (formal safety rules)
"""

from cognitive_engine.mp.scenario import Event, ScenarioGenerator, Assertion

__all__ = ["Event", "ScenarioGenerator", "Assertion"]
