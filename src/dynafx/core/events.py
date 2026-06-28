"""Event system — Pub/Sub event bus for semantic interfaces.

Provides a lightweight event-driven architecture where components
can subscribe to and publish events. Enables real-time reactions
to model changes (Jena-style event-based listeners).

Usage:
    from dynafx.core.events import EventBus, Event

    bus = EventBus()

    # Subscribe to events
    bus.subscribe("port_closed", lambda e: reroute_ships(e))
    bus.subscribe("port_closed", lambda e: estimate_costs(e))

    # Publish events
    bus.publish(Event("port_closed", data={"port": "Port A"}))
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A model change event.

    Attributes:
        id: Unique event identifier.
        event_type: Type of event (e.g., "blob_added", "port_closed").
        source: Source component that emitted the event.
        data: Event payload data.
    """
    id: UUID = field(default_factory=uuid4)
    event_type: str = ""
    source: Any = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id.hex,
            "event_type": self.event_type,
            "source": str(self.source) if self.source else None,
            "data": self.data,
        }


class EventBus:
    """Pub/Sub event bus for semantic interfaces.

    Implements the Observer pattern:
        - Publishers emit events without knowing who receives them
        - Subscribers register interest in specific event types
        - All matching subscribers are notified automatically

    Supports:
        - Wildcard subscriptions ("*" receives all events)
        - Multiple subscribers per event type
        - Subscriber priority (lower number = higher priority)
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[tuple[int, Callable]]] = defaultdict(list)
        self._history: list[Event] = []
        self._max_history: int = 1000

    def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], None],
        priority: int = 0,
    ) -> None:
        """Register interest in an event type.

        Args:
            event_type: Event type to subscribe to (or "*" for all).
            callback: Function to call when event is published.
            priority: Lower number = higher priority (default 0).
        """
        self._subscribers[event_type].append((priority, callback))
        # Sort by priority
        self._subscribers[event_type].sort(key=lambda x: x[0])

    def unsubscribe(
        self,
        event_type: str,
        callback: Callable[[Event], None],
    ) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                (p, cb) for p, cb in self._subscribers[event_type]
                if cb != callback
            ]

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Notifies:
            1. Specific event type subscribers
            2. Wildcard ("*") subscribers
        """
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Notify specific subscribers
        for priority, callback in self._subscribers.get(event.event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(
                    "Subscriber error for event %s: %s",
                    event.event_type,
                    e,
                    exc_info=True,
                )

        # Notify wildcard subscribers (skip if already notified via specific)
        if event.event_type != "*":
            for priority, callback in self._subscribers.get("*", []):
                try:
                    callback(event)
                except Exception as e:
                    logger.error(
                        "Wildcard subscriber error: %s",
                        e,
                        exc_info=True,
                    )

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[Event]:
        """Get event history.

        Args:
            event_type: Filter by event type (None for all).
            limit: Maximum number of events to return.

        Returns:
            List of events, most recent first.
        """
        events = self._history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return list(reversed(events[-limit:]))

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()

    def subscriber_count(self, event_type: Optional[str] = None) -> int:
        """Count subscribers.

        Args:
            event_type: Count for specific type (None for all).

        Returns:
            Number of subscribers.
        """
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(subs) for subs in self._subscribers.values())


# Singleton instance for global event bus
_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
