"""Tests for Discrete Event Simulation (DES) engine."""

import pytest
from cognitive_engine.system.des import (
    Event, EventQueue, Queue, Resource, DESEngine,
    QueueStats, ResourceStats, DESClock,
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
