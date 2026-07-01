"""Discrete Event Simulation (DES) engine.

Provides event scheduling, FIFO queues with capacity, resource pools
with utilization tracking, multi-step service operations, resource
wait queues, recurring events, and statistics collection.

Core pattern: priority-queue event scheduler processes events in
chronological order. Each event can enqueue/dequeue entities from
queues, request/release resources, and trigger side effects.
"""

from __future__ import annotations

import heapq
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
    cost_per_unit: float = 0.0
    total_requests: int = 0
    total_granted: int = 0
    total_denied: int = 0
    total_released: int = 0
    busy_time: float = 0.0
    total_cost: float = 0.0
    utilization_history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        if not self.utilization_history:
            return 0.0
        total = sum(u for _, u in self.utilization_history)
        return total / len(self.utilization_history)

    @property
    def avg_cost_per_unit_time(self) -> float:
        if self.busy_time > 0:
            return self.total_cost / self.busy_time
        return 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capacity": self.capacity,
            "cost_per_unit": self.cost_per_unit,
            "total_requests": self.total_requests,
            "total_granted": self.total_granted,
            "total_denied": self.total_denied,
            "utilization": self.utilization,
            "total_cost": self.total_cost,
            "busy_time": self.busy_time,
            "avg_cost_per_unit_time": self.avg_cost_per_unit_time,
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


# ── Entity types ──────────────────────────────────────────────────

EntityData = dict[str, Any]


@dataclass
class Entity:
    """Generic entity base class for DES simulation.

    Can be used directly or subclassed for typed entities.
    Provides basic identity, timing, and metadata fields.
    """
    id: int
    type: str = "generic"
    created_at: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access for backward compatibility."""
        if key == "id":
            return self.id
        if key == "type":
            return self.type
        if key == "created_at":
            return self.created_at
        if key == "properties":
            return self.properties
        if key == "attributes":
            return self.attributes
        if key in self.properties:
            return self.properties[key]
        if key in self.attributes:
            return self.attributes[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@dataclass
class Order(Entity):
    """An order entity with quantity, priority, and SKU."""
    quantity: float = 1.0
    priority: int = 0
    sku: str = ""
    due_date: float = 0.0
    type: str = "order"

    def __getitem__(self, key: str) -> Any:
        if key == "quantity":
            return self.quantity
        if key == "priority":
            return self.priority
        if key == "sku":
            return self.sku
        if key == "due_date":
            return self.due_date
        return super().__getitem__(key)


@dataclass
class Shipment(Entity):
    """A shipment entity with source, destination, and items."""
    source: str = ""
    destination: str = ""
    items: list[Any] = field(default_factory=list)
    status: str = "pending"
    type: str = "shipment"

    def __getitem__(self, key: str) -> Any:
        if key == "source":
            return self.source
        if key == "destination":
            return self.destination
        if key == "items":
            return self.items
        if key == "status":
            return self.status
        return super().__getitem__(key)


@dataclass
class ServiceRecord:
    """Tracks a multi-step service operation in a queue.

    Tracks the front entity and its remaining service duration,
    decremented each time step.
    """
    entity: EntityData | Entity
    remaining: float
    total_duration: float
    start_time: float


class EventQueue:
    """Priority-queue based event scheduler."""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter = 0

    def schedule(self, event: Event) -> None:
        heapq.heappush(self._heap, event)

    def schedule_at(self, time: float, name: str, payload: dict[str, Any] | None = None) -> None:
        self.schedule(Event(time=time, name=name, payload=payload or {}))

    def schedule_recurring(self, name: str, interval: float, start_time: float = 0.0,
                           payload: dict[str, Any] | None = None) -> None:
        """Schedule a recurring event at fixed intervals.

        The event auto-reschedules itself each time it fires.
        """
        self.schedule(Event(
            time=start_time, name=name,
            payload={"__recurring__": True, "_interval": interval, **(payload or {})},
        ))

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
    """Queue with optional capacity, multi-server multi-step service, and ordering disciplines.

    Disciplines:
        FIFO (default): First-In, First-Out.
        SPT: Shortest Processing Time first. Evaluates service time at enqueue.
        EDD: Earliest Due Date first. Uses entity.due_date or entity["due_date"].

    Supports multiple servers that process entities in parallel.
    With ``servers=1`` (default), behavior is identical to single-server.
    With ``servers=N``, up to N entities can be in service simultaneously.

    Supports state-based routing: when dequeueing, the entity can be
    routed to another queue based on compiled condition expressions.
    """

    def __init__(self, name: str, capacity: int = -1, service_time: str = "",
                 discipline: str = "FIFO", servers: int = 1, event_driven: bool = False):
        self.name = name
        self.capacity = capacity  # -1 = unlimited
        self.service_time = service_time
        self.discipline = discipline.upper()
        self.servers = max(1, servers)
        self.event_driven = event_driven
        self._compiled_service_time: Optional[Callable[[], float]] = None
        self._entities: list[EntityData | Entity] = []
        self._enter_times: list[float] = []
        self._service_records: list[Optional[ServiceRecord]] = [None] * self.servers
        self._in_service: set[int] = set()  # id() of entities currently being served
        self.stats = QueueStats(name=name)
        # Routing rules: list of (condition_expression, target_queue_name)
        self._routing_rules: list[tuple[str, str]] = []
        self._compiled_routes: list[tuple[Any, str]] | None = None

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

    def _get_sort_key(self, entity: EntityData | Entity) -> float:
        """Get sort key for the entity based on current discipline."""
        if self.discipline == "SPT":
            return self._get_entity_service_time(entity)
        if self.discipline == "EDD":
            if isinstance(entity, dict):
                return entity.get("due_date", float("inf"))
            return getattr(entity, "due_date", float("inf"))
        return 0.0  # FIFO: all equal

    def _make_eval_ns(self, entity: EntityData | Entity,
                      state: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "entity": entity,
            "e": entity,
            "state": state or {},
            "len": len,
            "abs": abs,
            "min": min,
            "max": max,
        }

    def _get_entity_service_time(self, entity: EntityData | Entity) -> float:
        """Evaluate service time expression for a specific entity."""
        if self._compiled_service_time is not None:
            try:
                st = self._compiled_service_time()
                if st > 0:
                    return st
            except Exception:
                pass
        if self.service_time:
            try:
                ns = self._make_eval_ns(entity)
                result = eval(self.service_time, {"__builtins__": {}}, ns)
                return max(0.0, float(result))
            except Exception:
                pass
        return 1.0

    def _insert_sorted(self, entity: EntityData | Entity, t: float) -> None:
        """Insert entity in sorted order based on discipline key."""
        key = self._get_sort_key(entity)
        for i in range(len(self._entities)):
            existing = self._entities[i]
            existing_key = self._get_sort_key(existing)
            if key < existing_key:
                self._entities.insert(i, entity)
                self._enter_times.insert(i, t)
                return
        self._entities.append(entity)
        self._enter_times.append(t)

    def enqueue(self, entity: EntityData | Entity, t: float,
                event_queue: Optional[EventQueue] = None) -> bool:
        """Add entity to queue. Returns False if queue is full.

        Respects ordering discipline (FIFO/SPT/EDD).
        Multi-server: starts service immediately if a server is free.
        """
        success = True
        if self.is_full():
            self.stats.total_dropped += 1
            success = False
        else:
            if self.discipline == "FIFO":
                self._entities.append(entity)
                self._enter_times.append(t)
            else:
                self._insert_sorted(entity, t)
            self.stats.total_arrivals += 1
            self.stats.max_length = max(self.stats.max_length, len(self._entities))
            # Start service if a server is free
            if self._compiled_service_time is not None:
                self.fill_servers(t, event_queue)

        from dynafx.registry import get_queue_hooks
        for hook in get_queue_hooks("enqueue"):
            hook(self, entity=entity, t=t, event_queue=event_queue,
                 success=success)
        return success

    def dequeue(self, t: float) -> Optional[EntityData | Entity]:
        """Remove and return front entity. Returns None if empty.
        Clears the service record for this entity so the server is freed.
        Does NOT start new service — call fill_servers() separately."""
        if not self._entities:
            return None
        entity = self._entities.pop(0)
        enter_t = self._enter_times.pop(0)
        wait = t - enter_t
        self.stats.total_wait_time += wait
        self.stats.total_served += 1
        self.stats.total_departures += 1
        self._in_service.discard(id(entity))
        # Clear the service record for this entity
        for i in range(self.servers):
            sr = self._service_records[i]
            if sr is not None and sr.entity is entity:
                self._service_records[i] = None
                break

        from dynafx.registry import get_queue_hooks
        for hook in get_queue_hooks("dequeue"):
            hook(self, entity=entity, t=t)
        return entity

    def peek(self) -> Optional[EntityData | Entity]:
        return self._entities[0] if self._entities else None

    def advance_service(self, dt: float) -> int:
        """Advance all active services by dt. Returns count of completions.
        For event-driven queues, returns 0 (completions handled by events)."""
        if self.event_driven:
            return 0
        completed = 0
        for i in range(self.servers):
            sr = self._service_records[i]
            if sr is None:
                continue
            sr.remaining -= dt
            if sr.remaining <= 0.0:
                completed += 1
        return completed

    def fill_servers(self, t: float, event_queue: Optional[EventQueue] = None) -> None:
        """Fill any free servers from the front of the queue.
        
        For event-driven queues, schedules a departure event for each
        newly started service so completion happens at the exact time
        rather than via time-sliced dt decrement.
        """
        for i in range(self.servers):
            if self._service_records[i] is not None:
                continue
            if not self._entities:
                break
            # Find first entity not already in service (multi-server fix)
            entity = None
            for e in self._entities:
                if id(e) not in self._in_service:
                    entity = e
                    break
            if entity is None:
                break
            st = self._get_service_time()
            if st <= 0:
                continue
            self._service_records[i] = ServiceRecord(
                entity=entity, remaining=st,
                total_duration=st, start_time=t,
            )
            self._in_service.add(id(entity))
            if self.event_driven and event_queue is not None:
                event_queue.schedule_at(t + st, f"_dep_{self.name}", {"server": i})

    def is_service_active(self) -> bool:
        """True if any server is currently busy."""
        return any(sr is not None for sr in self._service_records)

    def active_servers(self) -> int:
        """Count of currently busy servers."""
        return sum(1 for sr in self._service_records if sr is not None)

    def record_length(self, t: float) -> None:
        """Record current length at time t for statistics."""
        self.stats.length_history.append((t, len(self._entities)))

    # ── Routing ────────────────────────────────────────────────────

    def add_route(self, condition: str, target_queue: str) -> None:
        """Register a routing rule.

        When an entity is dequeued, rules are evaluated in registration
        order. The first matching rule determines the target queue.

        Args:
            condition: Expression string evaluated against the entity.
                Available in eval scope: entity (dict|Entity), e (alias),
                t (time), state (shared dict), len, abs, min, max.
            target_queue: Name of the target queue to route to.
        """
        self._routing_rules.append((condition, target_queue))
        self._compiled_routes = None  # invalidate cache

    def _ensure_routes_compiled(self) -> None:
        if self._compiled_routes is not None:
            return
        self._compiled_routes = []
        for cond, target in self._routing_rules:
            try:
                code = compile(cond, f"<route:{self.name}>", "eval")
                self._compiled_routes.append((code, target))
            except SyntaxError:
                self._compiled_routes.append((None, target))

    def route(self, entity: EntityData | Entity, t: float, state: dict[str, Any] | None = None) -> Optional[str]:
        """Evaluate routing rules and return target queue name.

        Returns None if no rule matches.
        """
        self._ensure_routes_compiled()
        ns: dict[str, Any] = {
            "entity": entity,
            "e": entity,
            "t": t,
            "state": state or {},
            "len": len,
            "abs": abs,
            "min": min,
            "max": max,
        }
        from dynafx.registry import get_registered_builtins
        ns.update(get_registered_builtins())
        target = None
        for code, target_name in self._compiled_routes:
            if code is None:
                continue
            try:
                if eval(code, {"__builtins__": {}}, ns):
                    target = target_name
                    break
            except Exception:
                continue

        from dynafx.registry import get_queue_hooks
        for hook in get_queue_hooks("route"):
            hook(self, entity=entity, t=t, target_queue=target, state=state)
        return target

    def dequeue_routed(self, t: float, state: dict[str, Any] | None = None) -> tuple[EntityData | Entity | None, Optional[str]]:
        """Dequeue and route entity. Returns (entity, target_queue).

        target_queue is None if no routing rule matches.
        """
        entity = self.dequeue(t)
        if entity is None:
            return None, None
        target = self.route(entity, t, state)
        return entity, target


class Resource:
    """Resource pool with capacity constraint and optional wait queue."""

    def __init__(self, name: str, capacity: int = 1, cost_per_unit: float = 0.0):
        self.name = name
        self.capacity = capacity
        self.cost_per_unit = cost_per_unit
        self._busy: int = 0
        self._waiting: list[dict[str, Any]] = []
        self._last_cost_time: Optional[float] = None
        self.stats = ResourceStats(name=name, capacity=capacity, cost_per_unit=cost_per_unit)

    @property
    def available(self) -> int:
        return self.capacity - self._busy

    @property
    def busy(self) -> int:
        return self._busy

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    @property
    def utilization(self) -> float:
        return self._busy / self.capacity if self.capacity > 0 else 0.0

    def request(self, t: float, wait: bool = False, priority: int = 0,
                quantity: int = 1) -> bool:
        """Try to acquire units.

        Args:
            t: Current time.
            wait: If True and resource is busy, add to wait queue.
            priority: Lower = higher priority for wait queue ordering.
            quantity: Number of units to acquire (default 1).

        Returns:
            True if granted immediately, False if denied (and optionally queued).
        """
        from dynafx.registry import get_resource_hooks
        denied_by_hook = False
        for hook in get_resource_hooks("pre_request"):
            try:
                hook(self, t=t, wait=wait, priority=priority, quantity=quantity)
            except Exception:
                denied_by_hook = True

        if denied_by_hook:
            self.stats.total_requests += 1
            self.stats.total_denied += 1
            return False

        quantity = max(1, quantity)
        self.stats.total_requests += 1
        granted = False
        if self._busy + quantity <= self.capacity:
            self._busy += quantity
            self.stats.total_granted += 1
            granted = True
        else:
            self.stats.total_denied += 1
            if wait:
                self._waiting.append({"time": t, "priority": priority,
                                      "quantity": quantity})
                self._waiting.sort(key=lambda w: (w["priority"], w["time"]))

        for hook in get_resource_hooks("post_request"):
            hook(self, t=t, wait=wait, priority=priority, quantity=quantity,
                 granted=granted)
        return granted

    def release(self, t: float) -> bool:
        """Release one unit. Auto-grants the next waiting requestor.

        Returns:
            True if a unit was released, False if none were busy.
        """
        from dynafx.registry import get_resource_hooks
        for hook in get_resource_hooks("pre_release"):
            hook(self, t=t)

        released = False
        if self._busy > 0:
            self._busy -= 1
            self.stats.total_released += 1
            # Auto-grant next waiter (respects quantity)
            while (self._waiting and self._busy < self.capacity):
                waiter = self._waiting[0]
                w_qty = waiter.get("quantity", 1)
                if self._busy + w_qty <= self.capacity:
                    self._waiting.pop(0)
                    self._busy += w_qty
                    self.stats.total_granted += 1
                else:
                    break
            released = True

        for hook in get_resource_hooks("post_release"):
            hook(self, t=t, released=released)
        return released

    def record_utilization(self, t: float) -> None:
        """Record current utilization at time t and accrue cost for elapsed time."""
        if self._last_cost_time is not None:
            dt = t - self._last_cost_time
            if dt > 0:
                cost_inc = self.utilization * self.capacity * self.cost_per_unit * dt
                self.stats.total_cost += cost_inc
                self.stats.busy_time += self.utilization * self.capacity * dt
        self._last_cost_time = t
        self.stats.utilization_history.append((t, self.utilization))


class DESEngine:
    """Discrete Event Simulation engine.

    Orchestrates clock, event queue, queues, and resources.
    Processes events in chronological order with multi-step service support.
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
        if queue.event_driven:
            self.register_handler(f"_dep_{queue.name}", self._make_departure_handler(queue.name))

    @staticmethod
    def _make_departure_handler(queue_name: str) -> Callable:
        """Create a handler for departure events for the given queue.

        The handler dequeues the completed entity and starts service
        on the next entity in line (which schedules another departure event).
        This chain reaction enables high throughput within a single step.
        Departure counts are aggregated by step() from queue stats.
        """
        def handler(event: Event, engine: DESEngine) -> dict[str, float]:
            queue = engine.queues[queue_name]
            entity = queue.dequeue(event.time)
            if entity is not None:
                queue.fill_servers(event.time, engine.event_queue)
            return {}
        return handler

    def add_resource(self, resource: Resource) -> None:
        self.resources[resource.name] = resource

    def register_handler(self, event_name: str, handler: Callable) -> None:
        """Register a handler function for an event name."""
        self._handlers[event_name] = handler

    def schedule_event(self, time: float, name: str, payload: dict[str, Any] | None = None) -> None:
        self.event_queue.schedule_at(time, name, payload)

    def schedule_recurring(self, name: str, interval: float, start_time: float = 0.0,
                           payload: dict[str, Any] | None = None) -> None:
        self.event_queue.schedule_recurring(name, interval, start_time, payload)

    def process_event(self, event: Event) -> dict[str, float]:
        """Process a single event. Returns metrics dict."""
        self._event_log.append((event.time, event.name, event.payload))
        # Handle recurring events: reschedule
        if event.payload.get("__recurring__"):
            interval = event.payload["_interval"]
            self.event_queue.schedule_at(
                event.time + interval,
                event.name,
                dict(event.payload),  # keep __recurring__ and _interval for continued rescheduling
            )
        handler = self._handlers.get(event.name)
        if handler:
            return handler(event, self)
        return {}

    def _process_queue_departures(self, t: float, dt: float) -> dict[str, float]:
        """Process queue departures using multi-step service tracking.

        Advances each queue's services by dt. When a service completes,
        the entity is dequeued. Free servers immediately start serving
        the next entity in line.

        With multi-server queues, up to N completions per step are
        possible (one per server).

        Args:
            t: Current end-of-step time.
            dt: Time step size.

        Returns:
            Metrics dict with departure counts.
        """
        metrics: dict[str, float] = {}
        for q in self.queues.values():
            if q._compiled_service_time is None:
                continue
            if q.event_driven:
                # Departures handled by events — skip time-sliced processing
                continue
            if q.length() == 0 and not q.is_service_active():
                continue
            completed = q.advance_service(dt)
            for _ in range(completed):
                entity = q.dequeue(t)
                if entity is not None:
                    metrics[f"{q.name}_departed"] = metrics.get(f"{q.name}_departed", 0) + 1
            q.fill_servers(t)
        return metrics

    def step(self, t: float, dt: float) -> dict[str, float]:
        """Process all events in [t, t+dt). Returns aggregated metrics."""
        end_time = t + dt
        metrics: dict[str, float] = {}

        # Initialize cost tracking on first call
        if all(r._last_cost_time is None for r in self.resources.values()):
            for r in self.resources.values():
                r.record_utilization(t)

        # Record baseline departure counts before processing
        dep_baseline: dict[str, int] = {}
        for name, q in self.queues.items():
            if q._compiled_service_time is not None:
                dep_baseline[name] = q.stats.total_departures

        while not self.event_queue.empty:
            event = self.event_queue.peek()
            if event is None or event.time >= end_time:
                break
            event = self.event_queue.next()
            self.clock.time = event.time
            self.process_event(event)

        # Process time-sliced queue departures
        self._process_queue_departures(end_time, dt)

        self.clock.time = end_time

        # Aggregate departure counts from stats (handles both time-sliced
        # and event-driven queues without double-counting)
        for name, q in self.queues.items():
            if name in dep_baseline:
                deps = q.stats.total_departures - dep_baseline[name]
                if deps > 0:
                    metrics[f"{name}_departed"] = deps

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
            metrics[f"{r.name}_waiting"] = float(r.waiting)

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
