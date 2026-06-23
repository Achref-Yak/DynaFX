"""Discrete Event Simulation (DES) engine.

Provides event scheduling, FIFO queues with capacity, resource pools
with utilization tracking, and statistics collection.

Core pattern: priority-queue event scheduler processes events in
chronological order. Each event can enqueue/dequeue entities from
queues, request/release resources, and trigger side effects.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class QueueStats:
    """Statistics collected for a queue during simulation."""
    name: str = ""
    total_arrivals: int = 0
    total_departures: int = 0
    total_dropped: int = 0
    max_length: int = 0
    total_wait_time: float = 0.0
    total_served: int = 0
    length_history: list[tuple[float, int]] = field(default_factory=list)

    @property
    def avg_wait(self) -> float:
        return self.total_wait_time / self.total_served if self.total_served > 0 else 0.0

    @property
    def avg_length(self) -> float:
        if not self.length_history:
            return 0.0
        total = sum(length for _, length in self.length_history)
        return total / len(self.length_history)

    @property
    def utilization(self) -> float:
        """Fraction of time the queue had entities waiting."""
        if not self.length_history:
            return 0.0
        busy = sum(1 for _, length in self.length_history if length > 0)
        return busy / len(self.length_history)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_arrivals": self.total_arrivals,
            "total_departures": self.total_departures,
            "total_dropped": self.total_dropped,
            "max_length": self.max_length,
            "avg_wait": self.avg_wait,
            "avg_length": self.avg_length,
            "utilization": self.utilization,
        }


@dataclass
class ResourceStats:
    """Statistics collected for a resource during simulation."""
    name: str = ""
    capacity: int = 1
    total_requests: int = 0
    total_granted: int = 0
    total_denied: int = 0
    total_released: int = 0
    busy_time: float = 0.0
    utilization_history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        if not self.utilization_history:
            return 0.0
        total = sum(u for _, u in self.utilization_history)
        return total / len(self.utilization_history)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capacity": self.capacity,
            "total_requests": self.total_requests,
            "total_granted": self.total_granted,
            "total_denied": self.total_denied,
            "utilization": self.utilization,
        }


@dataclass
class DESClock:
    """Simulation clock."""
    time: float = 0.0

    def advance(self, dt: float) -> float:
        self.time += dt
        return self.time


@dataclass
class Event:
    """A scheduled event."""
    time: float
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # lower = higher priority

    def __lt__(self, other: Event) -> bool:
        if self.time != other.time:
            return self.time < other.time
        return self.priority < other.priority


class EventQueue:
    """Priority-queue based event scheduler."""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = 0

    def schedule(self, event: Event) -> None:
        heapq.heappush(self._heap, event)

    def schedule_at(self, time: float, name: str, payload: dict[str, Any] | None = None) -> None:
        self.schedule(Event(time=time, name=name, payload=payload or {}))

    def next(self) -> Optional[Event]:
        if self._heap:
            return heapq.heappop(self._heap)
        return None

    def peek(self) -> Optional[Event]:
        return self._heap[0] if self._heap else None

    @property
    def empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)


class Queue:
    """FIFO queue with optional capacity limit."""

    def __init__(self, name: str, capacity: int = -1, service_time: str = ""):
        self.name = name
        self.capacity = capacity  # -1 = unlimited
        self.service_time = service_time
        self._compiled_service_time: Optional[Callable[[], float]] = None
        self._entities: list[dict[str, Any]] = []
        self._enter_times: list[float] = []
        self.stats = QueueStats(name=name)

    def length(self) -> int:
        return len(self._entities)

    def is_full(self) -> bool:
        return self.capacity > 0 and len(self._entities) >= self.capacity

    def _get_service_time(self) -> float:
        """Evaluate compiled service time expression."""
        if self._compiled_service_time is not None:
            try:
                return max(0.0, self._compiled_service_time())
            except Exception:
                return 1.0
        return 1.0  # default service time

    def enqueue(self, entity: dict[str, Any], t: float, event_queue: Optional[EventQueue] = None) -> bool:
        """Add entity to queue. Returns False if queue is full.

        If event_queue is provided, schedules a departure event at t + service_time.
        """
        if self.is_full():
            self.stats.total_dropped += 1
            return False
        self._entities.append(entity)
        self._enter_times.append(t)
        self.stats.total_arrivals += 1
        self.stats.max_length = max(self.stats.max_length, len(self._entities))
        # Schedule departure event if service_time is configured and event_queue available
        if event_queue is not None and self._compiled_service_time is not None:
            svc_time = self._get_service_time()
            event_queue.schedule_at(
                t + svc_time,
                f"_queue_depart_{self.name}",
                {"queue": self.name},
            )
        return True

    def dequeue(self, t: float) -> Optional[dict[str, Any]]:
        """Remove and return front entity. Returns None if empty."""
        if not self._entities:
            return None
        entity = self._entities.pop(0)
        enter_t = self._enter_times.pop(0)
        wait = t - enter_t
        self.stats.total_wait_time += wait
        self.stats.total_served += 1
        self.stats.total_departures += 1
        return entity

    def peek(self) -> Optional[dict[str, Any]]:
        return self._entities[0] if self._entities else None

    def record_length(self, t: float) -> None:
        """Record current length at time t for statistics."""
        self.stats.length_history.append((t, len(self._entities)))


class Resource:
    """Resource pool with capacity constraint (e.g., servers)."""

    def __init__(self, name: str, capacity: int = 1, cost_per_unit: float = 0.0):
        self.name = name
        self.capacity = capacity
        self.cost_per_unit = cost_per_unit
        self._busy: int = 0
        self.stats = ResourceStats(name=name, capacity=capacity)

    @property
    def available(self) -> int:
        return self.capacity - self._busy

    @property
    def busy(self) -> int:
        return self._busy

    @property
    def utilization(self) -> float:
        return self._busy / self.capacity if self.capacity > 0 else 0.0

    def request(self, t: float) -> bool:
        """Try to acquire one unit. Returns True if granted."""
        self.stats.total_requests += 1
        if self._busy < self.capacity:
            self._busy += 1
            self.stats.total_granted += 1
            return True
        self.stats.total_denied += 1
        return False

    def release(self, t: float) -> None:
        """Release one unit."""
        if self._busy > 0:
            self._busy -= 1
            self.stats.total_released += 1

    def record_utilization(self, t: float) -> None:
        """Record current utilization at time t."""
        self.stats.utilization_history.append((t, self.utilization))


class DESEngine:
    """Discrete Event Simulation engine.

    Orchestrates clock, event queue, queues, and resources.
    Processes events in chronological order.
    """

    def __init__(self) -> None:
        self.clock = DESClock()
        self.event_queue = EventQueue()
        self.queues: dict[str, Queue] = {}
        self.resources: dict[str, Resource] = {}
        self._handlers: dict[str, Callable] = {}
        self._event_log: list[tuple[float, str, dict[str, Any]]] = []

    def add_queue(self, queue: Queue) -> None:
        self.queues[queue.name] = queue

    def add_resource(self, resource: Resource) -> None:
        self.resources[resource.name] = resource

    def register_handler(self, event_name: str, handler: Callable) -> None:
        """Register a handler function for an event name."""
        self._handlers[event_name] = handler

    def schedule_event(self, time: float, name: str, payload: dict[str, Any] | None = None) -> None:
        self.event_queue.schedule_at(time, name, payload)

    def process_event(self, event: Event) -> dict[str, float]:
        """Process a single event. Returns metrics dict."""
        self._event_log.append((event.time, event.name, event.payload))
        handler = self._handlers.get(event.name)
        if handler:
            return handler(event, self)
        return {}

    def _process_queue_departures(self, t: float) -> dict[str, float]:
        """Process any queue departures that should have happened by time t."""
        metrics: dict[str, float] = {}
        for q in self.queues.values():
            if q._compiled_service_time is None:
                continue
            # Process departures for entities whose service time has elapsed
            while q.length() > 0:
                # Check if front entity's service time has elapsed
                enter_t = q._enter_times[0] if q._enter_times else t
                svc_time = q._get_service_time()
                if t >= enter_t + svc_time:
                    entity = q.dequeue(t)
                    if entity is not None:
                        metrics[f"{q.name}_departed"] = metrics.get(f"{q.name}_departed", 0) + 1
                else:
                    break
        return metrics

    def step(self, t: float, dt: float) -> dict[str, float]:
        """Process all events in [t, t+dt). Returns aggregated metrics."""
        end_time = t + dt
        metrics: dict[str, float] = {}

        while not self.event_queue.empty:
            event = self.event_queue.peek()
            if event is None or event.time >= end_time:
                break
            event = self.event_queue.next()
            self.clock.time = event.time
            event_metrics = self.process_event(event)
            metrics.update(event_metrics)

        # Process queue departures based on service_time
        departure_metrics = self._process_queue_departures(end_time)
        metrics.update(departure_metrics)

        self.clock.time = end_time

        # Record queue lengths and resource utilization
        for q in self.queues.values():
            q.record_length(end_time)
            metrics[f"{q.name}_length"] = float(q.length())
            metrics[f"{q.name}_arrivals"] = float(q.stats.total_arrivals)
            metrics[f"{q.name}_dropped"] = float(q.stats.total_dropped)

        for r in self.resources.values():
            r.record_utilization(end_time)
            metrics[f"{r.name}_utilization"] = r.utilization
            metrics[f"{r.name}_available"] = float(r.available)

        return metrics

    def get_all_stats(self) -> dict[str, Any]:
        """Return all queue and resource statistics."""
        stats: dict[str, Any] = {}
        for name, q in self.queues.items():
            stats[name] = q.stats.summary()
        for name, r in self.resources.items():
            stats[name] = r.stats.summary()
        return stats

    @property
    def event_log(self) -> list[tuple[float, str, dict[str, Any]]]:
        return list(self._event_log)
