"""Monterey Phoenix Scenario Generator — simplified Python implementation.

Generates all valid event orderings permitted by an event grammar.
The grammar defines:
    - Events: activities governed by precedence and inclusion
    - Precedence: A must happen before B (A → B)
    - Inclusion: A is part of B (A ⊂ B)

Usage:
    from dynafx.mp.scenario import Event, ScenarioGenerator

    gen = ScenarioGenerator()
    arrival = gen.add_event("Arrival")
    security = gen.add_event("Security Check")
    loading = gen.add_event("Loading")
    departure = gen.add_event("Departure")

    gen.set_precedence(arrival, security)
    gen.set_precedence(security, loading)
    gen.set_precedence(loading, departure)

    scenarios = gen.generate_scenarios(max_scenarios=100)
    # Returns all valid orderings like:
    # [Arrival, Security, Loading, Departure]

Reference: Statecharts and Monterey Phoenix event grammar.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """An activity governed by precedence and inclusion.

    Attributes:
        id: Unique event identifier.
        name: Human-readable event name.
        metadata: Additional event metadata.
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id.hex,
            "name": self.name,
        }


@dataclass
class Assertion:
    """A formal safety rule to check against generated scenarios.

    Attributes:
        name: Human-readable assertion name.
        description: What the assertion checks.
        event_a: First event name.
        event_b: Second event name.
        assertion_type: Type of assertion (must_precede, must_not_cause, etc.).
    """
    name: str = ""
    description: str = ""
    event_a: str = ""
    event_b: str = ""
    assertion_type: str = "must_precede"  # must_precede, must_not_cause, must_include

    def check(self, scenario: list[str]) -> bool:
        """Check if a scenario satisfies this assertion.

        Returns:
            True if assertion is satisfied, False otherwise.
        """
        if self.assertion_type == "must_precede":
            return self._check_must_precede(scenario)
        elif self.assertion_type == "must_not_cause":
            return self._check_must_not_cause(scenario)
        elif self.assertion_type == "must_include":
            return self._check_must_include(scenario)
        return True

    def _check_must_precede(self, scenario: list[str]) -> bool:
        """A must happen before B."""
        try:
            idx_a = scenario.index(self.event_a)
            idx_b = scenario.index(self.event_b)
            return idx_a < idx_b
        except ValueError:
            return True  # If either event not in scenario, assertion holds

    def _check_must_not_cause(self, scenario: list[str]) -> bool:
        """A must not cause B (B must not follow A directly)."""
        try:
            idx_a = scenario.index(self.event_a)
            idx_b = scenario.index(self.event_b)
            return abs(idx_a - idx_b) > 1
        except ValueError:
            return True

    def _check_must_include(self, scenario: list[str]) -> bool:
        """Scenario must include both events."""
        return self.event_a in scenario and self.event_b in scenario


class ScenarioGenerator:
    """Simplified Monterey Phoenix scenario generator.

    Generates all valid event orderings permitted by an event grammar.
    The grammar defines events with precedence and inclusion constraints.

    Usage:
        gen = ScenarioGenerator()
        arr = gen.add_event("Arrival")
        sec = gen.add_event("Security")
        load = gen.add_event("Loading")
        dep = gen.add_event("Departure")

        gen.set_precedence(arr, sec)
        gen.set_precedence(sec, load)
        gen.set_precedence(load, dep)

        scenarios = gen.generate_scenarios()
    """

    def __init__(self) -> None:
        self.events: dict[UUID, Event] = {}
        self.event_names: dict[str, UUID] = {}
        self._precedes: dict[UUID, set[UUID]] = defaultdict(set)
        self._included_in: dict[UUID, set[UUID]] = defaultdict(set)
        self._includes: dict[UUID, set[UUID]] = defaultdict(set)

    def add_event(self, name: str, **metadata) -> UUID:
        """Add an event to the grammar.

        Args:
            name: Event name (must be unique).
            **metadata: Additional metadata.

        Returns:
            Event ID.
        """
        if name in self.event_names:
            return self.event_names[name]

        event = Event(name=name, metadata=metadata)
        self.events[event.id] = event
        self.event_names[name] = event.id
        return event.id

    def set_precedence(self, before: UUID, after: UUID) -> None:
        """A must happen before B.

        Args:
            before: Event that must happen first.
            after: Event that must happen after.
        """
        self._precedes[before].add(after)

    def set_inclusion(self, container: UUID, contained: UUID) -> None:
        """A is part of B (A must happen within B's execution).

        Args:
            container: The containing event.
            contained: The contained event.
        """
        self._included_in[contained].add(container)
        self._includes[container].add(contained)

    def generate_scenarios(
        self,
        max_scenarios: int = 1000,
        include_names: bool = True,
    ) -> list[list[str]]:
        """Generate all valid event orderings.

        Uses topological sort with backtracking to generate all
        orderings that satisfy precedence constraints.

        Args:
            max_scenarios: Maximum number of scenarios to generate.
            include_names: If True, return event names; else return IDs.

        Returns:
            List of scenarios, where each scenario is a list of event
            names (or IDs) in execution order.
        """
        n = len(self.events)
        if n == 0:
            return []

        # Build adjacency list for precedence
        adj: dict[UUID, list[UUID]] = defaultdict(list)
        in_degree: dict[UUID, int] = {eid: 0 for eid in self.events}

        for before, afters in self._precedes.items():
            for after in afters:
                adj[before].append(after)
                in_degree[after] = in_degree.get(after, 0) + 1

        # Topological sort with backtracking
        scenarios: list[list[UUID]] = []
        self._backtrack(
            adj, in_degree, [], set(), scenarios, max_scenarios
        )

        # Convert to names if requested
        if include_names:
            return [
                [self.events[eid].name for eid in scenario]
                for scenario in scenarios
            ]
        return [
            [eid.hex for eid in scenario]
            for scenario in scenarios
        ]

    def _backtrack(
        self,
        adj: dict[UUID, list[UUID]],
        in_degree: dict[UUID, int],
        current: list[UUID],
        used: set[UUID],
        scenarios: list[list[UUID]],
        max_scenarios: int,
    ) -> None:
        """Recursive backtracking to generate scenarios."""
        if len(scenarios) >= max_scenarios:
            return

        if len(current) == len(self.events):
            scenarios.append(current[:])
            return

        # Find all events with in_degree 0 (ready to execute)
        ready = [
            eid for eid, deg in in_degree.items()
            if deg == 0 and eid not in used
        ]

        for event_id in ready:
            # Execute this event
            current.append(event_id)
            used.add(event_id)

            # Reduce in_degree for successors
            for successor in adj[event_id]:
                in_degree[successor] -= 1

            # Recurse
            self._backtrack(adj, in_degree, current, used, scenarios, max_scenarios)

            # Undo
            current.pop()
            used.remove(event_id)
            for successor in adj[event_id]:
                in_degree[successor] += 1

    def check_assertions(
        self,
        scenarios: list[list[str]],
        assertions: list[Assertion],
    ) -> dict[str, list[str]]:
        """Check assertions against generated scenarios.

        Args:
            scenarios: List of scenarios to check.
            assertions: List of assertions to verify.

        Returns:
            Dict mapping assertion name → list of violating scenarios.
        """
        violations: dict[str, list[str]] = defaultdict(list)

        for scenario in scenarios:
            for assertion in assertions:
                if not assertion.check(scenario):
                    violations[assertion.name].append(
                        " → ".join(scenario)
                    )

        return dict(violations)

    def to_dict(self) -> dict:
        """Serialize grammar to dictionary."""
        return {
            "events": {
                eid.hex: event.to_dict()
                for eid, event in self.events.items()
            },
            "precedes": {
                before.hex: [after.hex for after in afters]
                for before, afters in self._precedes.items()
            },
            "includes": {
                container.hex: [contained.hex for contained in contained_set]
                for container, contained_set in self._includes.items()
            },
        }
