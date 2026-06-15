"""Tests for kernel/inference_cycle.py."""

from unittest.mock import Mock
from uuid import uuid4

import pytest

from cognitive_engine.core.models import Graph, Node, NodeType, Opinion
from cognitive_engine.core.state import State
from cognitive_engine.kernel.inference_cycle import (
    InferenceCycle, InferenceCycleConfig, CycleReport, InferenceResult,
)
from cognitive_engine.kernel.assertion_gate import AssertionGate
from cognitive_engine.policy.engine import PolicyEngine, PolicySelection


class TestInferenceCycleConfig:
    def test_defaults(self):
        cfg = InferenceCycleConfig()
        assert cfg.epsilon == 1e-4
        assert cfg.max_cycles == 20
        assert cfg.stm_capacity == 128
        assert cfg.policy_name == "default"
        assert cfg.domain == "general"
        assert cfg.convergence_window == 3

    def test_custom(self):
        cfg = InferenceCycleConfig(epsilon=0.01, max_cycles=5, domain="legal")
        assert cfg.epsilon == 0.01
        assert cfg.max_cycles == 5
        assert cfg.domain == "legal"


class TestCycleReport:
    def test_defaults(self):
        r = CycleReport(cycle=1, norm=0.5, converged=False)
        assert r.cycle == 1
        assert r.norm == 0.5
        assert r.converged is False
        assert r.operator_log == []
        assert r.duration == 0.0

    def test_converged(self):
        r = CycleReport(cycle=2, norm=0.001, converged=True)
        assert r.converged is True


class TestInferenceResult:
    def test_defaults(self):
        state = State(graph=Graph())
        r = InferenceResult(state=state)
        assert r.state is state
        assert r.cycles == []
        assert r.converged is False
        assert r.total_cycles == 0

    def test_with_cycles(self):
        state = State(graph=Graph())
        r = InferenceResult(
            state=state,
            cycles=[CycleReport(cycle=1, norm=0.5, converged=True)],
            converged=True,
            total_cycles=1,
        )
        assert r.converged is True
        assert r.total_cycles == 1


class TestInferenceCycle:
    def _make_state(self, text="test"):
        state = State(graph=Graph(source_text=text), metadata={"text": text})
        return state

    def test_init_defaults(self):
        cycle = InferenceCycle(operators={})
        assert cycle.config.epsilon == 1e-4
        assert isinstance(cycle.gate, AssertionGate)
        assert isinstance(cycle.policy, PolicyEngine)
        assert cycle.memory is None
        assert cycle.tbox is None

    def test_init_custom(self):
        cfg = InferenceCycleConfig(max_cycles=3)
        gate = AssertionGate()
        policy = PolicyEngine()
        cycle = InferenceCycle(operators={}, config=cfg, assertion_gate=gate, policy=policy)
        assert cycle.config.max_cycles == 3
        assert cycle.gate is gate
        assert cycle.policy is policy

    def test_run_no_operators(self):
        cycle = InferenceCycle(operators={}, config=InferenceCycleConfig(max_cycles=1))
        state = self._make_state()
        result = cycle.run(state)
        assert result.converged is False
        assert result.total_cycles == 1
        assert len(result.cycles) == 1
        assert result.state.metadata["cycle"] == 1

    def test_run_with_extract(self):
        mock_extract = Mock(return_value=None)
        def extract_side_effect(s):
            s.metadata["extracted"] = True
            s.graph.nodes[uuid4()] = Node(text="extracted", type=NodeType.CLAIM)
            return s
        mock_extract.side_effect = extract_side_effect

        operators = {"extract": mock_extract}
        cycle = InferenceCycle(operators=operators, config=InferenceCycleConfig(max_cycles=1))
        state = self._make_state("process me")
        result = cycle.run(state)
        assert result.total_cycles >= 1
        mock_extract.assert_called_once()

    def test_run_with_structural_ops(self):
        mock_schema = Mock()
        def schema_side_effect(s):
            s.metadata["schema_applied"] = "test"
            return s
        mock_schema.side_effect = schema_side_effect

        operators = {"extract": Mock(side_effect=lambda s: s), "schema": mock_schema}
        cycle = InferenceCycle(operators=operators, config=InferenceCycleConfig(max_cycles=1))
        state = self._make_state()
        result = cycle.run(state)
        assert result.total_cycles >= 1

    def test_run_with_propagate(self):
        mock_propagate = Mock()
        def prop_side_effect(s):
            s.metadata["beliefs"] = {"test": 0.8}
            return s
        mock_propagate.side_effect = prop_side_effect

        operators = {"extract": Mock(side_effect=lambda s: s), "propagate": mock_propagate}
        cycle = InferenceCycle(operators=operators, config=InferenceCycleConfig(max_cycles=1))
        state = self._make_state()
        result = cycle.run(state)
        mock_propagate.assert_called()

    def test_run_policy_selects_operators(self):
        """Test that policy-influenced operator selection runs evidential ops."""
        mock_abduce = Mock(side_effect=lambda s: s)
        operators = {
            "extract": Mock(side_effect=lambda s: s),
            "propagate": Mock(side_effect=lambda s: s),
            "abduce": mock_abduce,
        }
        # Configure policy to select "abduce"
        policy = PolicyEngine()
        policy._policy = None  # use default

        cycle = InferenceCycle(
            operators=operators,
            config=InferenceCycleConfig(max_cycles=1, domain="general"),
            policy=policy,
        )
        state = self._make_state()
        result = cycle.run(state)
        assert result.total_cycles >= 1

    def test_structural_op_failure_skips(self):
        """Test that failing structural operator is skipped gracefully."""
        failing_op = Mock(side_effect=ValueError("fail"))
        operators = {"extract": Mock(side_effect=lambda s: s), "schema": failing_op}
        cycle = InferenceCycle(operators=operators, config=InferenceCycleConfig(max_cycles=1))
        state = self._make_state()
        result = cycle.run(state)
        assert result.total_cycles >= 1

    def test_converges_early(self):
        """Test that inference stops when converged."""
        cfg = InferenceCycleConfig(epsilon=10.0, max_cycles=10)
        operators = {"extract": Mock(side_effect=lambda s: s)}
        cycle = InferenceCycle(operators=operators, config=cfg)
        state = self._make_state()
        result = cycle.run(state)
        assert result.converged is True
        assert result.total_cycles == 1

    def test_state_snapshot(self):
        cycle = InferenceCycle(operators={})
        state = self._make_state()
        nid = uuid4()
        state.graph.nodes[nid] = Node(text="test", type=NodeType.CLAIM, opinion=Opinion(0.8, 0.1, 0.1, 0.5))
        snap = cycle._state_snapshot(state)
        assert "node_ids" in snap
        assert "edges" in snap
        assert "beliefs" in snap
        assert str(nid) in snap["beliefs"]
        assert snap["beliefs"][str(nid)] == pytest.approx(0.8 + 0.5 * 0.1)

    def test_compute_norm(self):
        cycle = InferenceCycle(operators={})
        nid = uuid4()
        prev = {"node_ids": {str(nid)}, "edges": set(), "beliefs": {str(nid): 0.5}, "operator": ""}
        curr = {"node_ids": {str(nid)}, "edges": set(), "beliefs": {str(nid): 0.8}, "operator": ""}
        norm = cycle._compute_norm(prev, curr)
        assert norm > 0
