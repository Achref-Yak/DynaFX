"""Tests for Discrete Event Simulation (DES) engine."""

import pytest
from cognitive_engine.system.des import (
    Event, EventQueue, Queue, Resource, DESEngine,
    QueueStats, ResourceStats, DESClock,
    Entity, Order, Shipment,
)


# ── DESClock ─────────────────────────────────────────────────────

class TestDESClock:
    def test_initial_time(self):
        clock = DESClock()
        assert clock.time == 0.0

    def test_advance(self):
        clock = DESClock()
        clock.advance(1.0)
        assert clock.time == 1.0
        clock.advance(2.5)
        assert clock.time == 3.5


# ── EventQueue ───────────────────────────────────────────────────

class TestEventQueue:
    def test_empty_queue(self):
        eq = EventQueue()
        assert eq.empty
        assert len(eq) == 0
        assert eq.next() is None

    def test_schedule_and_pop(self):
        eq = EventQueue()
        eq.schedule_at(1.0, "arrive")
        eq.schedule_at(0.5, "depart")
        assert len(eq) == 2
        first = eq.next()
        assert first.name == "depart"
        assert first.time == 0.5

    def test_fifo_order(self):
        eq = EventQueue()
        eq.schedule_at(1.0, "a")
        eq.schedule_at(2.0, "b")
        eq.schedule_at(3.0, "c")
        assert eq.next().name == "a"
        assert eq.next().name == "b"
        assert eq.next().name == "c"

    def test_priority_breaks_ties(self):
        eq = EventQueue()
        eq.schedule(Event(time=1.0, name="low", priority=10))
        eq.schedule(Event(time=1.0, name="high", priority=0))
        assert eq.next().name == "high"

    def test_payload(self):
        eq = EventQueue()
        eq.schedule_at(1.0, "test", {"key": "value"})
        event = eq.next()
        assert event.payload == {"key": "value"}


# ── Queue ────────────────────────────────────────────────────────

class TestQueue:
    def test_enqueue_dequeue(self):
        q = Queue("test", capacity=-1)
        q.enqueue({"id": 1}, t=0.0)
        q.enqueue({"id": 2}, t=1.0)
        assert q.length() == 2
        entity = q.dequeue(t=3.0)
        assert entity["id"] == 1
        assert q.length() == 1

    def test_capacity_limit(self):
        q = Queue("test", capacity=2)
        assert q.enqueue({"id": 1}, t=0.0)
        assert q.enqueue({"id": 2}, t=0.0)
        assert not q.enqueue({"id": 3}, t=0.0)  # dropped
        assert q.stats.total_dropped == 1

    def test_unlimited_capacity(self):
        q = Queue("test", capacity=-1)
        for i in range(100):
            q.enqueue({"id": i}, t=0.0)
        assert q.length() == 100

    def test_stats_tracking(self):
        q = Queue("test", capacity=-1)
        q.enqueue({"id": 1}, t=0.0)
        q.enqueue({"id": 2}, t=1.0)
        q.dequeue(t=3.0)
        assert q.stats.total_arrivals == 2
        assert q.stats.total_departures == 1
        assert q.stats.total_wait_time == 3.0
        assert q.stats.total_served == 1
        assert q.stats.avg_wait == 3.0

    def test_is_full(self):
        q = Queue("test", capacity=2)
        assert not q.is_full()
        q.enqueue({}, t=0.0)
        assert not q.is_full()
        q.enqueue({}, t=0.0)
        assert q.is_full()

    def test_length_history(self):
        q = Queue("test", capacity=-1)
        q.record_length(0.0)
        q.enqueue({}, t=0.0)
        q.record_length(1.0)
        assert q.stats.length_history == [(0.0, 0), (1.0, 1)]

    def test_peek(self):
        q = Queue("test")
        assert q.peek() is None
        q.enqueue({"id": 1}, t=0.0)
        assert q.peek()["id"] == 1
        assert q.length() == 1

    def test_advance_service_completes(self):
        q = Queue("svc", capacity=-1, service_time="4.0")
        q._compiled_service_time = lambda: 4.0
        q.enqueue({"id": 1}, t=0.0)
        assert q.is_service_active()
        assert q.advance_service(2.0) is False
        assert q.is_service_active()
        assert q.advance_service(2.0) is True
        # Service record stays until dequeue clears it
        assert q.is_service_active()
        entity = q.dequeue(t=4.0)
        assert entity["id"] == 1
        assert not q.is_service_active()

    def test_advance_service_no_compiled_time(self):
        q = Queue("svc", capacity=-1)
        q.enqueue({"id": 1}, t=0.0)
        assert not q.is_service_active()
        assert q.advance_service(2.0) is False

    def test_service_starts_on_first_enqueue(self):
        q = Queue("svc", capacity=-1, service_time="3.0")
        q._compiled_service_time = lambda: 3.0
        q.enqueue({"id": 1}, t=0.0)
        assert q._service_record is not None
        assert q._service_record.remaining == 3.0

    def test_service_starts_on_next_entity_after_dequeue(self):
        q = Queue("svc", capacity=-1, service_time="2.0")
        q._compiled_service_time = lambda: 2.0
        q.enqueue({"id": 1}, t=0.0)
        q.enqueue({"id": 2}, t=0.0)
        # complete service on entity 1
        while q.advance_service(5.0):
            break
        q.dequeue(t=5.0)
        # service should start on entity 2
        assert q.is_service_active()
        assert q._service_record is not None
        assert q._service_record.entity["id"] == 2


# ── Resource Cost Tracking ────────────────────────────────────────

class TestResourceCost:
    def test_cost_per_unit_default(self):
        r = Resource("r", capacity=2)
        assert r.cost_per_unit == 0.0

    def test_cost_per_unit_custom(self):
        r = Resource("r", capacity=2, cost_per_unit=10.0)
        assert r.cost_per_unit == 10.0

    def test_cost_accrues_over_steps(self):
        engine = DESEngine()
        r = Resource("cpu", capacity=2, cost_per_unit=5.0)
        engine.add_resource(r)
        r.request(t=0.0)  # 1 of 2 units busy → utilization = 0.5
        engine.step(0.0, 1.0)
        engine.step(1.0, 1.0)
        # Cost = utilization * capacity * cost_per_unit * total_time
        # 0.5 * 2 * 5.0 * 2.0 = 10.0
        assert r.stats.total_cost == pytest.approx(10.0)

    def test_cost_zero_when_idle(self):
        engine = DESEngine()
        r = Resource("cpu", capacity=2, cost_per_unit=10.0)
        engine.add_resource(r)
        engine.step(0.0, 5.0)  # idle
        assert r.stats.total_cost == 0.0

    def test_cost_increases_with_utilization(self):
        engine = DESEngine()
        r = Resource("cpu", capacity=1, cost_per_unit=10.0)
        engine.add_resource(r)
        r.request(t=0.0)  # 100% busy
        engine.step(0.0, 3.0)
        # 1.0 * 1 * 10.0 * 3.0 = 30.0
        assert r.stats.total_cost == pytest.approx(30.0)

    def test_cost_in_summary(self):
        r = Resource("r", capacity=2, cost_per_unit=5.0)
        s = r.stats.summary()
        assert "total_cost" in s
        assert "busy_time" in s
        assert "avg_cost_per_unit_time" in s

    def test_cost_per_unit_time(self):
        engine = DESEngine()
        r = Resource("r", capacity=1, cost_per_unit=8.0)
        engine.add_resource(r)
        r.request(t=0.0)
        engine.step(0.0, 4.0)
        # 4 units of busy time, total cost = 1.0 * 1 * 8.0 * 4.0 = 32.0
        # avg_cost_per_unit_time = 32.0 / 4.0 = 8.0
        assert r.stats.avg_cost_per_unit_time == pytest.approx(8.0)

    def test_multiple_resources_cost(self):
        engine = DESEngine()
        r1 = Resource("r1", capacity=2, cost_per_unit=10.0)
        r2 = Resource("r2", capacity=1, cost_per_unit=20.0)
        engine.add_resource(r1)
        engine.add_resource(r2)
        r1.request(t=0.0)  # 1/2 busy
        r2.request(t=0.0)  # 1/1 busy
        engine.step(0.0, 2.0)
        #         r1: 0.5 * 2 * 10.0 * 2.0 = 20.0
        # r2: 1.0 * 1 * 20.0 * 2.0 = 40.0
        assert r1.stats.total_cost == pytest.approx(20.0)
        assert r2.stats.total_cost == pytest.approx(40.0)


# ── Entity Types ──────────────────────────────────────────────────

class TestEntity:
    def test_entity_creation(self):
        e = Entity(id=1)
        assert e.id == 1
        assert e.type == "generic"
        assert e.properties == {}

    def test_entity_dict_access(self):
        e = Entity(id=1, properties={"color": "red"})
        assert e["id"] == 1
        assert e["type"] == "generic"
        assert e["properties"]["color"] == "red"

    def test_entity_get(self):
        e = Entity(id=1, properties={"x": 10})
        assert e.get("x") == 10
        assert e.get("missing", "fallback") == "fallback"

    def test_entity_created_at(self):
        e = Entity(id=1, created_at=100.0)
        assert e["created_at"] == 100.0
        assert e.get("created_at") == 100.0


class TestOrder:
    def test_order_creation(self):
        o = Order(id=1, quantity=5.0, sku="ABC-123", priority=2)
        assert o.id == 1
        assert o.quantity == 5.0
        assert o.sku == "ABC-123"
        assert o.priority == 2
        assert o.type == "order"

    def test_order_dict_access(self):
        o = Order(id=1, quantity=10.0)
        assert o["quantity"] == 10.0
        assert o["priority"] == 0
        assert o["type"] == "order"
        assert o["id"] == 1

    def test_order_get(self):
        o = Order(id=1, quantity=3.0)
        assert o.get("quantity") == 3.0
        assert o.get("due_date") == 0.0

    def test_order_with_properties(self):
        o = Order(id=1, properties={"notes": "urgent"}, priority=5)
        assert o["priority"] == 5
        assert o.get("notes") == "urgent"


class TestShipment:
    def test_shipment_creation(self):
        s = Shipment(id=1, source="Factory", destination="Warehouse")
        assert s.id == 1
        assert s.source == "Factory"
        assert s.destination == "Warehouse"
        assert s.status == "pending"
        assert s.type == "shipment"

    def test_shipment_dict_access(self):
        s = Shipment(id=1, source="A", destination="B")
        assert s["source"] == "A"
        assert s["destination"] == "B"
        assert s["status"] == "pending"

    def test_shipment_items(self):
        items = [Order(id=101, sku="X"), Order(id=102, sku="Y")]
        s = Shipment(id=1, items=items)
        assert len(s["items"]) == 2
        assert s.items[0].sku == "X"

    def test_shipment_entity_in_queue(self):
        from cognitive_engine.system.des import Queue
        q = Queue("dock", capacity=10)
        s = Shipment(id=1, source="A", destination="B")
        assert q.enqueue(s, t=0.0)
        dequeued = q.dequeue(t=5.0)
        assert isinstance(dequeued, Shipment)
        if isinstance(dequeued, Shipment):
            assert dequeued.source == "A"
            assert dequeued.destination == "B"
        assert dequeued.get("source") == "A"


# ── Queue Routing ─────────────────────────────────────────────────

class TestQueueRouting:
    def test_add_route(self):
        q = Queue("q", capacity=10)
        q.add_route("e.id > 5", "large")
        assert len(q._routing_rules) == 1
        assert q._routing_rules[0][1] == "large"

    def test_route_by_entity_id(self):
        q = Queue("q")
        q.add_route("entity.id > 5", "large")
        e1 = Entity(id=10)
        assert q.route(e1, t=0.0) == "large"
        e2 = Entity(id=1)
        assert q.route(e2, t=0.0) is None

    def test_route_by_entity_type(self):
        q = Queue("q")
        q.add_route("entity.type == 'order'", "order_queue")
        o = Order(id=1)
        assert q.route(o, t=0.0) == "order_queue"
        s = Shipment(id=2)
        assert q.route(s, t=0.0) is None

    def test_route_order_by_quantity(self):
        q = Queue("q")
        q.add_route("entity.quantity > 100", "bulk")
        q.add_route("entity.quantity > 10", "medium")
        o1 = Order(id=1, quantity=200)
        assert q.route(o1, t=0.0) == "bulk"
        o2 = Order(id=2, quantity=50)
        assert q.route(o2, t=0.0) == "medium"
        o3 = Order(id=3, quantity=5)
        assert q.route(o3, t=0.0) is None

    def test_route_with_state(self):
        q = Queue("q")
        q.add_route("state.get('mode') == 'express'", "express")
        state = {"mode": "express"}
        e = Entity(id=1)
        assert q.route(e, t=0.0, state=state) == "express"
        assert q.route(e, t=0.0, state={}) is None

    def test_dequeue_routed(self):
        q = Queue("q")
        q.add_route("e.type == 'order'", "orders")
        o = Order(id=1)
        assert q.enqueue(o, t=0.0)
        entity, target = q.dequeue_routed(t=5.0)
        assert entity is not None
        assert target == "orders"
        if isinstance(entity, Order):
            assert entity.id == 1

    def test_dequeue_routed_empty(self):
        q = Queue("q")
        entity, target = q.dequeue_routed(t=0.0)
        assert entity is None
        assert target is None

    def test_route_by_sku(self):
        q = Queue("q")
        q.add_route("e.sku == 'ABC'", "abc_pick")
        o1 = Order(id=1, sku="ABC")
        o2 = Order(id=2, sku="XYZ")
        assert q.route(o1, t=0.0) == "abc_pick"
        assert q.route(o2, t=0.0) is None

    def test_route_dict_entity(self):
        q = Queue("q")
        q.add_route("e.get('priority', 0) < 3", "high_priority")
        d = {"id": 1, "priority": 1}
        assert q.route(d, t=0.0) == "high_priority"
        d2 = {"id": 2, "priority": 5}
        assert q.route(d2, t=0.0) is None

    def test_route_malformed_condition_does_not_crash(self):
        q = Queue("q")
        q.add_route("syntax error {{{", "nowhere")
        e = Entity(id=1)
        assert q.route(e, t=0.0) is None



class TestQueueDiscipline:
    def test_default_is_fifo(self):
        q = Queue("q")
        assert q.discipline == "FIFO"

    def test_spt_orders_by_service_time(self):
        q = Queue("q", discipline="SPT")
        # service_time is eval'd per entity; shorter expected time = higher priority
        q.service_time = "e.id * 2.0"  # entity 1 → 2, entity 2 → 4
        assert q.enqueue(Entity(id=2), t=0.0)
        assert q.enqueue(Entity(id=1), t=0.0)
        e1 = q.dequeue(t=0.0)
        e2 = q.dequeue(t=0.0)
        assert e1 is not None and e1.get("id") == 1  # shorter key first
        assert e2 is not None and e2.get("id") == 2

    def test_edd_orders_by_due_date(self):
        q = Queue("q", discipline="EDD")
        assert q.enqueue(Order(id=1, due_date=10.0), t=0.0)
        assert q.enqueue(Order(id=2, due_date=5.0), t=0.0)
        e1 = q.dequeue(t=0.0)
        e2 = q.dequeue(t=0.0)
        assert e1 is not None and e1.get("id") == 2  # earlier due date first
        assert e2 is not None and e2.get("id") == 1

    def test_edd_with_dict_entity(self):
        q = Queue("q", discipline="EDD")
        assert q.enqueue({"id": 1, "due_date": 10.0}, t=0.0)
        assert q.enqueue({"id": 2, "due_date": 3.0}, t=0.0)
        e1 = q.dequeue(t=0.0)
        assert e1 is not None and e1["id"] == 2

    def test_edd_default_inf_for_missing_due_date(self):
        q = Queue("q", discipline="EDD")
        assert q.enqueue(Entity(id=1), t=0.0)  # no due_date → inf
        assert q.enqueue(Order(id=2, due_date=5.0), t=0.0)
        e1 = q.dequeue(t=0.0)
        assert e1 is not None and e1.get("id") == 2  # entity 2 has finite due date

    def test_spt_with_mixed_service_times(self):
        q = Queue("q", discipline="SPT")
        q.service_time = "e.get('size', 1)"
        assert q.enqueue({"id": 1, "size": 10}, t=0.0)
        assert q.enqueue({"id": 2, "size": 3}, t=0.0)
        assert q.enqueue({"id": 3, "size": 7}, t=0.0)
        ids = []
        while q.length() > 0:
            e = q.dequeue(t=0.0)
            ids.append(e["id"])
        assert ids == [2, 3, 1]  # sorted by size ascending

    def test_spt_with_order_quantity(self):
        q = Queue("q", discipline="SPT")
        q.service_time = "e.quantity * 0.5"
        assert q.enqueue(Order(id=1, quantity=100.0), t=0.0)
        assert q.enqueue(Order(id=2, quantity=10.0), t=0.0)
        e1 = q.dequeue(t=0.0)
        assert e1 is not None and e1.get("id") == 2  # smaller qty first
        q = Queue("q")
        q.add_route("e.id > 0", "somewhere")
        # Trigger compilation
        assert q.route(Entity(id=1), t=0.0) == "somewhere"
        q.add_route("e.id > 5", "big")
        # Compilation restarted, old cache invalidated
        assert q.route(Entity(id=10), t=0.0) == "somewhere"  # first rule matches first


# ── Resource with Wait Queue ──────────────────────────────────────

class TestResourceWaitQueue:
    def test_wait_queue_adds_when_busy(self):
        r = Resource("server", capacity=1)
        assert r.request(t=0.0)
        assert r.request(t=0.0, wait=True) is False  # denied, but queued
        assert r.waiting == 1

    def test_no_wait_queue_when_not_waiting(self):
        r = Resource("server", capacity=1)
        assert r.request(t=0.0)
        assert r.request(t=0.0) is False  # denied, not queued
        assert r.waiting == 0

    def test_release_auto_grants_waiter(self):
        r = Resource("server", capacity=1)
        r.request(t=0.0)  # busy
        r.request(t=0.0, wait=True)  # queued
        r.release(t=2.0)  # releases busy, grants waiter
        assert r.busy == 1  # waiter auto-granted
        assert r.waiting == 0
        assert r.stats.total_granted == 2

    def test_priority_ordering(self):
        r = Resource("server", capacity=1)
        r.request(t=0.0)  # busy
        r.request(t=1.0, wait=True, priority=10)  # low priority
        r.request(t=2.0, wait=True, priority=0)   # high priority
        r.release(t=3.0)  # grants high priority first
        assert r.busy == 1
        # Next release grants low priority
        r.release(t=4.0)
        assert r.busy == 1
        r.release(t=5.0)
        assert r.busy == 0
        assert r.waiting == 0


# ── Recurring Events ──────────────────────────────────────────────

class TestRecurringEvents:
    def test_recurring_auto_reschedules(self):
        eq = EventQueue()
        eq.schedule_recurring("tick", interval=1.0, start_time=0.0)
        e1 = eq.next()
        assert e1.name == "tick"
        assert e1.time == 0.0
        assert e1.payload.get("__recurring__") is True
        # Reschedule (same pattern as DESEngine.process_event)
        interval = e1.payload["_interval"]
        eq.schedule_at(e1.time + interval, e1.name, dict(e1.payload))
        e2 = eq.next()
        assert e2.time == 1.0
        assert e2.name == "tick"
        # Next event also carries recurring metadata
        assert e2.payload.get("__recurring__") is True

    def test_recurring_engine_integration(self):
        engine = DESEngine()
        count = [0]
        def handler(event, eng):
            count[0] += 1
            return {"tick_count": float(count[0])}
        engine.register_handler("tick", handler)
        engine.schedule_recurring("tick", interval=0.5, start_time=0.5)
        engine.step(0.0, 2.0)
        # Events at 0.5, 1.0, 1.5 (2.0 excluded by half-open window [t, t+dt))
        assert count[0] == 3

    def test_recurring_with_payload(self):
        eq = EventQueue()
        eq.schedule_recurring("check", interval=2.0, start_time=1.0,
                               payload={"key": "val"})
        e = eq.next()
        assert e.payload["key"] == "val"
        assert e.payload["__recurring__"] is True


# ── Multi-Step Service ────────────────────────────────────────────

class TestMultiStepService:
    def test_queue_processes_entities_over_multiple_steps(self):
        engine = DESEngine()
        q = Queue("process", capacity=-1, service_time="3.0")
        q._compiled_service_time = lambda: 3.0
        engine.add_queue(q)
        q.enqueue({"id": 1}, t=0.0)
        q.enqueue({"id": 2}, t=0.0)
        # Step 1: dt=1, service advances by 1
        metrics = engine.step(0.0, 1.0)
        assert q.length() == 2
        assert q._service_record is not None
        assert q._service_record.remaining == pytest.approx(2.0)
        # Step 2: dt=1, remaining=1
        metrics = engine.step(1.0, 1.0)
        assert q.length() == 2
        assert q._service_record.remaining == pytest.approx(1.0)
        # Step 3: dt=1, remaining=0 → entity 1 departs, service starts on entity 2
        metrics = engine.step(2.0, 1.0)
        assert q.length() == 1
        assert q.stats.total_departures == 1
        assert q.stats.total_served == 1
        assert q._service_record is not None
        assert q._service_record.entity["id"] == 2

    def test_multi_step_departure_metrics(self):
        engine = DESEngine()
        q = Queue("svc", capacity=-1, service_time="2.0")
        q._compiled_service_time = lambda: 2.0
        engine.add_queue(q)
        q.enqueue({"id": 1}, t=0.0)
        # dt=1.5: not enough to complete
        metrics = engine.step(0.0, 1.5)
        assert "svc_departed" not in metrics
        # dt=0.5: completes
        metrics = engine.step(1.5, 0.5)
        assert metrics.get("svc_departed", 0) == 1.0
        assert q.length() == 0

    def test_no_service_time_no_departure(self):
        q = Queue("inbox", capacity=-1)
        q.enqueue({"id": 1}, t=0.0)
        # No compiled service time → entity stays forever
        assert q.advance_service(100.0) is False
        assert q.length() == 1


# ── DES Engine Integration (Phase 3) ──────────────────────────────

class TestEngineStepMultiStep:
    def test_step_writes_departure_metrics(self):
        engine = DESEngine()
        q = Queue("svc", capacity=-1, service_time="2.0")
        q._compiled_service_time = lambda: 2.0
        engine.add_queue(q)
        q.enqueue({"id": 1}, t=0.0)
        metrics1 = engine.step(0.0, 2.0)
        assert metrics1.get("svc_departed", 0) == 1.0
        assert metrics1["svc_length"] == 0.0

    def test_step_with_multiple_entities(self):
        engine = DESEngine()
        q = Queue("svc", capacity=-1, service_time="1.0")
        q._compiled_service_time = lambda: 1.0
        engine.add_queue(q)
        q.enqueue({"id": 1}, t=0.0)
        q.enqueue({"id": 2}, t=0.0)
        q.enqueue({"id": 3}, t=0.0)
        # dt=1: first completes
        engine.step(0.0, 1.0)
        assert q.length() == 2
        assert q.stats.total_departures == 1
        # dt=1: second completes
        engine.step(1.0, 1.0)
        assert q.length() == 1
        assert q.stats.total_departures == 2
        # dt=1: third completes
        engine.step(2.0, 1.0)
        assert q.length() == 0
        assert q.stats.total_departures == 3


# ── Resource ─────────────────────────────────────────────────────

class TestResource:
    def test_request_grant(self):
        r = Resource("server", capacity=2)
        assert r.request(t=0.0)
        assert r.request(t=0.0)
        assert r.busy == 2
        assert r.available == 0

    def test_request_denied(self):
        r = Resource("server", capacity=1)
        assert r.request(t=0.0)
        assert not r.request(t=0.0)
        assert r.stats.total_denied == 1

    def test_release(self):
        r = Resource("server", capacity=2)
        r.request(t=0.0)
        r.request(t=0.0)
        r.release(t=1.0)
        assert r.busy == 1
        assert r.available == 1

    def test_utilization(self):
        r = Resource("server", capacity=4)
        r.request(t=0.0)
        r.request(t=0.0)
        assert r.utilization == 0.5

    def test_stats(self):
        r = Resource("server", capacity=2)
        r.request(t=0.0)
        r.request(t=0.0)
        assert r.stats.total_requests == 2
        assert r.stats.total_granted == 2


# ── QueueStats / ResourceStats ───────────────────────────────────

class TestStats:
    def test_queue_stats_summary(self):
        s = QueueStats(name="q1")
        s.total_arrivals = 10
        s.total_departures = 8
        s.total_dropped = 2
        s.max_length = 5
        s.total_wait_time = 40.0
        s.total_served = 8
        summary = s.summary()
        assert summary["name"] == "q1"
        assert summary["avg_wait"] == 5.0

    def test_resource_stats_summary(self):
        s = ResourceStats(name="r1", capacity=3)
        summary = s.summary()
        assert summary["capacity"] == 3
        assert summary["utilization"] == 0.0

    def test_queue_avg_length(self):
        s = QueueStats()
        s.length_history = [(0.0, 0), (1.0, 3), (2.0, 1)]
        assert s.avg_length == pytest.approx(4.0 / 3.0)

    def test_queue_utilization(self):
        s = QueueStats()
        s.length_history = [(0.0, 0), (1.0, 1), (2.0, 0), (3.0, 1)]
        assert s.utilization == pytest.approx(0.5)


# ── DESEngine ────────────────────────────────────────────────────

class TestDESEngine:
    def test_step_processes_events(self):
        engine = DESEngine()
        q = Queue("inbox", capacity=-1)
        engine.add_queue(q)
        processed = []

        def handler(event, e):
            processed.append(event.name)
            return {}

        engine.register_handler("arrive", handler)
        engine.schedule_event(0.5, "arrive")
        engine.schedule_event(1.5, "arrive")
        engine.step(0.0, 2.0)
        assert len(processed) == 2

    def test_step_writes_metrics(self):
        engine = DESEngine()
        q = Queue("inbox", capacity=-1)
        engine.add_queue(q)
        q.enqueue({}, t=0.0)
        metrics = engine.step(0.0, 1.0)
        assert "inbox_length" in metrics
        assert metrics["inbox_length"] == 1.0

    def test_resource_metrics(self):
        engine = DESEngine()
        r = Resource("cpu", capacity=4)
        engine.add_resource(r)
        r.request(t=0.0)
        r.request(t=0.0)
        metrics = engine.step(0.0, 1.0)
        assert metrics["cpu_utilization"] == pytest.approx(0.5)

    def test_get_all_stats(self):
        engine = DESEngine()
        q = Queue("q1")
        r = Resource("r1")
        engine.add_queue(q)
        engine.add_resource(r)
        stats = engine.get_all_stats()
        assert "q1" in stats
        assert "r1" in stats

    def test_events_outside_window_not_processed(self):
        engine = DESEngine()
        processed = []
        engine.register_handler("arrive", lambda e, eng: processed.append(e.time))
        engine.schedule_event(5.0, "arrive")
        engine.step(0.0, 2.0)
        assert len(processed) == 0

    def test_event_log(self):
        engine = DESEngine()
        engine.register_handler("arrive", lambda e, eng: {})
        engine.schedule_event(1.0, "arrive", {"x": 1})
        engine.step(0.0, 2.0)
        assert len(engine.event_log) == 1
        assert engine.event_log[0] == (1.0, "arrive", {"x": 1})

    def test_clock_advances(self):
        engine = DESEngine()
        engine.step(0.0, 3.0)
        assert engine.clock.time == 3.0


# ── DSL Integration ──────────────────────────────────────────────

class TestDesDslIntegration:
    def test_des_engine_created_when_queues_defined(self):
        from cognitive_engine.system.dsl import parse_sysd
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'queue \"Q\": capacity 5\n'
        )
        r = m.simulate()
        assert r.des_engine is not None
        assert "Q" in r.des_engine.queues

    def test_des_engine_created_when_resources_defined(self):
        from cognitive_engine.system.dsl import parse_sysd
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'resource \"Server\": capacity 2\n'
        )
        r = m.simulate()
        assert r.des_engine is not None
        assert "Server" in r.des_engine.resources

    def test_no_des_when_no_queues_resources(self):
        from cognitive_engine.system.dsl import parse_sysd
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'stock "S" = 10\n'
            'flow in = 1\n'
        )
        r = m.simulate()
        assert r.des_engine is None

    def test_des_coexists_with_sd(self):
        from cognitive_engine.system.dsl import parse_sysd
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 5\n'
            'stock "population" = 1000\n'
            'flow growth = population * 0.1\n'
            'queue \"Line\": capacity 10\n'
            'resource \"Server\": capacity 3\n'
        )
        r = m.simulate()
        assert "population" in r.values
        assert r.des_engine is not None
        assert "Line" in r.des_engine.queues
        assert "Server" in r.des_engine.resources

    def test_queue_capacity_respected(self):
        from cognitive_engine.system.dsl import parse_sysd
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'queue \"Q\": capacity 2\n'
        )
        r = m.simulate()
        q = r.des_engine.queues["Q"]
        assert q.capacity == 2

    def test_resource_capacity_respected(self):
        from cognitive_engine.system.dsl import parse_sysd
        m = parse_sysd(
            'Test\ndt 1\nfrom 0 to 3\n'
            'resource \"R\": capacity 5\n'
        )
        r = m.simulate()
        stats = r.des_engine.get_all_stats()
        assert stats["R"]["capacity"] == 5
