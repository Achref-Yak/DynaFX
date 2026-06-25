"""Tests for SimulationController and BatchRunner."""
import pytest
import os
from cognitive_engine.system.dsl import parse_sysd
from cognitive_engine.system.controller import (
    SimulationController,
    batch_run,
)


SIMPLE_MODEL = """
model 'Simple'
  dt 1
  from 0 to 10
  stock 'X': 100
    - 'out': X * 0.1
"""


class TestSimulationController:
    def test_initialize(self):
        """Controller initializes without error."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        assert ctrl.t == 0.0
        assert ctrl.step_count == 0
        assert ctrl.is_running

    def test_step_advances_time(self):
        """Single step advances time by dt."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        ctrl.step()
        assert ctrl.t == 1.0
        assert ctrl.step_count == 1

    def test_step_decreases_stock(self):
        """Stock decreases due to outflow."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        state0 = ctrl.get_stock_state()["X"]
        ctrl.step()
        state1 = ctrl.get_stock_state()["X"]
        assert state1 < state0

    def test_multiple_steps(self):
        """Multiple steps accumulate."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        for _ in range(5):
            ctrl.step()
        assert ctrl.step_count == 5
        assert ctrl.t == 5.0

    def test_run_until(self):
        """run_until advances to specified time."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize(dt=0.5)
        ctrl.run_until(3.0)
        assert ctrl.t == 3.0
        assert ctrl.step_count == 6

    def test_get_state(self):
        """get_state returns all names and values."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        state = ctrl.get_state()
        assert "X" in state
        assert state["X"] == 100.0

    def test_set_param_hot_swap(self):
        """set_param changes parameter mid-simulation."""
        m = parse_sysd("""
model 'ParamSwap'
  dt 1
  from 0 to 10
  aux rate: my_rate
  stock 'X': 100
    - 'out': X * rate
""")
        ctrl = SimulationController(m)
        ctrl.initialize(params={"my_rate": 0.1})
        ctrl.step()
        x_before = ctrl.get_stock_state()["X"]
        ctrl.set_param("my_rate", 0.5)
        ctrl.step()
        x_after = ctrl.get_stock_state()["X"]
        # With rate=0.5, more should be drained than with rate=0.1
        assert x_after < x_before - 10

    def test_inject_state(self):
        """inject_state directly sets a stock value."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        ctrl.step()
        ctrl.inject_state("X", 999.0)
        assert ctrl.get_stock_state()["X"] == 999.0

    def test_snapshot_restore(self):
        """snapshot and restore allows checkpoint/rollback."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        for _ in range(3):
            ctrl.step()
        snap = ctrl.snapshot()
        for _ in range(2):
            ctrl.step()
        assert ctrl.t == 5.0
        ctrl.restore(snap)
        assert ctrl.t == 3.0
        assert ctrl.step_count == 3

    def test_get_result(self):
        """get_result builds correct SysdModelResult."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        ctrl.run_until(10.0)
        result = ctrl.get_result()
        assert len(result.times) == 11  # t=0..10
        assert "X" in result.values
        assert result.steps == 10

    def test_on_step_callback(self):
        """on_step callback fires each step."""
        called = []
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model, on_step=lambda c: called.append(c.t))
        ctrl.initialize()
        for _ in range(3):
            ctrl.step()
        assert len(called) == 3
        assert called == [1.0, 2.0, 3.0]

    def test_is_done(self):
        """Controller reports done at end of time span."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        while ctrl.step():
            pass
        assert ctrl.is_running is False

    def test_step_returns_false_when_done(self):
        """step() returns False when simulation is finished."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        ctrl.initialize()
        while ctrl.step():
            pass
        assert ctrl.step() is False

    def test_not_initialized_raises(self):
        """step() before initialize() raises RuntimeError."""
        model = parse_sysd(SIMPLE_MODEL)
        ctrl = SimulationController(model)
        with pytest.raises(RuntimeError):
            ctrl.step()


class TestBatchRunner:
    def test_batch_single_scenario(self):
        """batch_run with single scenario returns one result."""
        model = parse_sysd(SIMPLE_MODEL)
        results = batch_run(model, [{}])
        assert len(results) == 1
        assert "X" in results[0].values

    def test_batch_multiple_scenarios(self):
        """batch_run with multiple scenarios returns all results."""
        model = parse_sysd(SIMPLE_MODEL)
        results = batch_run(model, [{}, {}])
        assert len(results) == 2

    def test_batch_different_params(self):
        """Different params produce different results."""
        m = parse_sysd("""
model 'BatchDiff'
  dt 1
  from 0 to 5
  aux rate: my_rate
  stock 'X': 100
    - 'out': X * rate
""")
        results = batch_run(m, [{"my_rate": 0.1}, {"my_rate": 0.5}])
        final_0 = results[0].values["X"][-1]
        final_1 = results[1].values["X"][-1]
        assert final_0 > final_1  # slower decay with lower rate

    def test_batch_t_span_override(self):
        """batch_run respects t_span override."""
        model = parse_sysd(SIMPLE_MODEL)
        results = batch_run(model, [{}], t_span=(0, 3))
        assert len(results[0].times) == 4  # t=0,1,2,3
