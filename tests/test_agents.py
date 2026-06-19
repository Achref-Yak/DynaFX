"""Tests for the multi-agent system (Manager, Blackboard, Specialist)."""

import pytest
from cognitive_engine.core.models import Graph, Node, NodeType, EdgeType, Edge
from cognitive_engine.agents.manager import ManagerAgent, HealthStatus
from cognitive_engine.agents.blackboard import BlackboardAgent
from cognitive_engine.agents.specialist import SpecialistAgent, HyperHeuristic


def _make_graph(n_nodes: int = 5) -> Graph:
    """Create a simple test graph."""
    import uuid
    g = Graph(source_text="test")
    for i in range(n_nodes):
        nid = uuid.uuid4()
        g.nodes[nid] = Node(
            id=nid,
            text=f"node_{i}",
            type=NodeType.CLAIM,
            opinion=(0.5 + i * 0.1, 0.1, 0.3, 0.5),
        )
    return g


class TestManagerAgent:
    def test_initialize(self):
        mgr = ManagerAgent()
        mgr.initialize({"domain": "test"})
        assert mgr._start_time > 0

    def test_terminate(self):
        mgr = ManagerAgent()
        mgr.initialize()
        mgr.terminate()  # should not raise

    def test_health_check_returns_status(self):
        mgr = ManagerAgent()
        graph = _make_graph()
        health = mgr.health_check(graph)
        assert isinstance(health, HealthStatus)
        assert isinstance(health.healthy, bool)

    def test_health_check_tracks_history(self):
        mgr = ManagerAgent()
        graph = _make_graph()
        mgr.health_check(graph)
        mgr.health_check(graph)
        assert len(mgr.health_history) == 2

    def test_health_history_capped(self):
        mgr = ManagerAgent()
        graph = _make_graph()
        for _ in range(25):
            mgr.health_check(graph)
        assert len(mgr.health_history) <= 20

    def test_health_check_bottlenecks(self):
        mgr = ManagerAgent()
        graph = _make_graph(1)  # very sparse graph
        health = mgr.health_check(graph)
        assert "sparse_graph" in health.bottlenecks

    def test_reconfigure(self):
        from cognitive_engine.kernel.inference_cycle import InferenceCycle, InferenceCycleConfig
        config = InferenceCycleConfig(max_cycles=10)
        cycle = InferenceCycle(operators={}, config=config)
        mgr = ManagerAgent(inference_cycle=cycle)
        mgr.reconfigure({"max_cycles": 5})
        assert cycle.config.max_cycles == 5


class TestBlackboardAgent:
    def test_publish_and_get(self):
        bb = BlackboardAgent()
        bb.publish("key1", "value1", publisher="test")
        assert bb.get("key1") == "value1"

    def test_publish_latest(self):
        bb = BlackboardAgent()
        bb.publish("key1", "first")
        bb.publish("key1", "second")
        assert bb.get("key1", latest=True) == "second"

    def test_publish_all_versions(self):
        bb = BlackboardAgent()
        bb.publish("key1", "first")
        bb.publish("key1", "second")
        result = bb.get("key1", latest=False)
        assert result == ["first", "second"]

    def test_get_all(self):
        bb = BlackboardAgent()
        bb.publish("a", 1)
        bb.publish("b", 2)
        result = bb.get_all()
        assert result == {"a": 1, "b": 2}

    def test_subscribe(self):
        bb = BlackboardAgent()
        received = []
        bb.subscribe("key1", lambda v: received.append(v))
        bb.publish("key1", "hello")
        assert received == ["hello"]

    def test_query(self):
        bb = BlackboardAgent()
        bb.publish("k1", "v1", publisher="agent1")
        results = bb.query(key="k1")
        assert len(results) == 1
        assert results[0]["value"] == "v1"

    def test_close(self):
        bb = BlackboardAgent()
        bb.close()  # should not raise


class TestSpecialistAgent:
    def test_can_handle_default(self):
        spec = SpecialistAgent(name="test", operator_names=["propagate"])
        assert spec.can_handle({}) == 0.5

    def test_execute(self):
        from cognitive_engine.core.state import State
        spec = SpecialistAgent(name="noop", operator_names=[])
        state = State(graph=Graph(source_text="test"))
        result = spec.execute(state, {})
        assert result is state  # no operators, state unchanged

    def test_record_performance(self):
        spec = SpecialistAgent(name="test", operator_names=[])
        spec.record_performance(0.8)
        spec.record_performance(0.9)
        assert spec.avg_performance == pytest.approx(0.85)

    def test_avg_performance_empty(self):
        spec = SpecialistAgent(name="test", operator_names=[])
        assert spec.avg_performance == 0.0


class TestHyperHeuristic:
    def test_select_best_agent(self):
        hh = HyperHeuristic()
        spec1 = SpecialistAgent(name="agent1", operator_names=[])
        spec2 = SpecialistAgent(name="agent2", operator_names=[])
        hh.register(spec1)
        hh.register(spec2)
        result = hh.select({})
        assert result is not None
        assert result.name in ("agent1", "agent2")

    def test_select_empty(self):
        hh = HyperHeuristic()
        assert hh.select({}) is None

    def test_update_performance(self):
        hh = HyperHeuristic()
        spec = SpecialistAgent(name="agent1", operator_names=[])
        hh.register(spec)
        hh.update_performance("agent1", {}, 0.9)
        # Should influence future selections
        result = hh.select({})
        assert result is not None
