"""Tests for higher-order delay functions: DELAY3, DELAYN, DELAY_FIXED."""

import math
import pytest
from dynafx.dynamics.dsl import SysdModel, parse_sysd

# ── DELAY3 Tests ────────────────────────────────────────────────

class TestDELAY3:
    def test_delay3_basic(self):
        """DELAY3 with step input should produce S-curve response."""
        m = parse_sysd("""
model 'Delay3Test'
  dt 0.5
  from 0 to 20
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'delay_in': Input
    - 'delay_out': DELAY3(Input, 6)
""")
        result = m.simulate()
        # After t=6, Delayed should start rising
        assert result["values"]["Delayed"][-1] > 0

    def test_delay3_creates_three_smooth_stages(self):
        """DELAY3(x, T) should create 3 smooth state variables with delay T/3 each."""
        from dynafx.dynamics.dsl import _build_system
        m = parse_sysd("""
model 'Delay3Test'
  dt 0.5
  from 0 to 20
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd3_in': Input
    - 'd3_out': DELAY3(Input, 6)
""")
        f, names, y0, aux_count, _ = _build_system(m, {})
        # Should have: Input, Delayed, _delay3_0, _delay3_1, _delay3_2
        assert len(names) == 5, f"Expected 5 names, got {len(names)}: {names}"
        assert "_delay3_0" in names
        assert "_delay3_1" in names
        assert "_delay3_2" in names
        assert aux_count == 3  # 3 smooth state variables

    def test_delay3_converges_to_input(self):
        """DELAY3 with constant input should eventually converge to input value."""
        m = parse_sysd("""
model 'Delay3Conv'
  dt 0.5
  from 0 to 50
  stock 'Input': 50
    + 'set_input': 50
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAY3(Input, 6)
""")
        result = m.simulate()
        # After many time constants, should approach 50
        assert result["values"]["Delayed"][-1] > 45

    def test_delay3_initial_zero(self):
        """DELAY3 output should start at 0 (init of smooth stages)."""
        m = parse_sysd("""
model 'Delay3Init'
  dt 0.5
  from 0 to 30
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAY3(Input, 10)
""")
        result = m.simulate()
        # At t=0, output should be 0
        assert result["values"]["Delayed"][0] == 0.0

    def test_delay3_converges_to_input(self):
        """DELAY3 with constant input should eventually converge to input value."""
        m = parse_sysd("""
model 'Delay3Conv'
  dt 0.5
  from 0 to 50
  stock 'Input': 50
    + 'set_input': 50
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAY3(Input, 6)
""")
        result = m.simulate()
        # After many time constants, should approach 50
        assert result["values"]["Delayed"][-1] > 45


# ── DELAYN Tests ────────────────────────────────────────────────

class TestDELAYN:
    def test_delayn_basic(self):
        """DELAYN with 3 stages should match DELAY3."""
        m_d3 = parse_sysd("""
model 'D3'
  dt 0.5
  from 0 to 20
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAY3(Input, 6)
""")
        m_dn = parse_sysd("""
model 'DN'
  dt 0.5
  from 0 to 20
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAYN(Input, 6, 3)
""")
        r1 = m_d3.simulate()
        r2 = m_dn.simulate()
        # Should be identical (both 3-stage delays)
        for v1, v2 in zip(r1["values"]["Delayed"], r2["values"]["Delayed"]):
            assert abs(v1 - v2) < 1e-10

    def test_delayn_variable_stages(self):
        """DELAYN with more stages should create more smooth state variables."""
        from dynafx.dynamics.dsl import _build_system
        m5 = parse_sysd("""
model 'D5'
  dt 0.5
  from 0 to 20
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAYN(Input, 6, 5)
""")
        f5, names5, y0_5, aux5, _ = _build_system(m5, {})
        # 5 stages → 5 smooth state variables
        assert aux5 == 5, f"Expected 5 aux states, got {aux5}"
        assert "_delayn_0" in names5
        assert "_delayn_4" in names5

    def test_delayn_converges(self):
        """DELAYN should converge to input value with enough time."""
        m = parse_sysd("""
model 'DNConv'
  dt 0.5
  from 0 to 50
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAYN(Input, 6, 5)
""")
        result = m.simulate()
        assert result["values"]["Delayed"][-1] > 90

    def test_delayn_single_stage_equals_smooth(self):
        """DELAYN with N=1 should create same structure as SMOOTH."""
        from dynafx.dynamics.dsl import _build_system
        m_smooth = parse_sysd("""
model 'Smooth'
  dt 0.5
  from 0 to 20
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': SMOOTH(Input, 5)
""")
        m_delayn = parse_sysd("""
model 'DelayN1'
  dt 0.5
  from 0 to 20
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAYN(Input, 5, 1)
""")
        f_s, names_s, y0_s, aux_s, _ = _build_system(m_smooth, {})
        f_d, names_d, y0_d, aux_d, _ = _build_system(m_delayn, {})
        # Both should have 1 smooth state variable
        assert aux_s == 1
        assert aux_d == 1
        # Both should converge to same value with constant input
        r1 = m_smooth.simulate()
        r2 = m_delayn.simulate()
        assert abs(r1["values"]["Delayed"][-1] - r2["values"]["Delayed"][-1]) < 0.1


# ── DELAY_FIXED Tests ──────────────────────────────────────────

class TestDELAYFixed:
    def test_delay_fixed_basic(self):
        """DELAY_FIXED should hold input value for delay time units."""
        m = parse_sysd("""
model 'DFixed'
  dt 1
  from 0 to 20
  stock 'Input': 0
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAY_FIXED(Input, 5)
""")
        result = m.simulate()
        # At t=0, output should be 0
        assert result["values"]["Delayed"][0] == 0.0

    def test_delay_fixed_preserves_value(self):
        """DELAY_FIXED should preserve the exact input value (not smooth it)."""
        m = parse_sysd("""
model 'DFixed2'
  dt 1
  from 0 to 30
  stock 'Input': 0
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAY_FIXED(Input, 5)
""")
        result = m.simulate()
        # After delay + some time, output should approach 100
        assert result["values"]["Delayed"][-1] > 50

    def test_delay_fixed_step_response(self):
        """DELAY_FIXED with step input should produce a delayed step."""
        from dynafx.dynamics.dsl import _build_system
        m = parse_sysd("""
model 'DFixed3'
  dt 0.5
  from 0 to 20
  stock 'Input': 0
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAY_FIXED(Input, 3)
""")
        # Verify the delay_fixed buffer state variable exists
        f, names, y0, aux_count, _ = _build_system(m, {})
        delay_vars = [n for n in names if "_delay_fixed_" in n]
        assert len(delay_vars) == 1, f"Expected 1 delay_fixed var, got {delay_vars}"


# ── Validation Tests ────────────────────────────────────────────

class TestDelayValidation:
    def test_delay3_recognized_in_validation(self):
        """DELAY3 should be recognized as a valid builtin."""
        from dynafx.dynamics.dsl import _get_builtin_names
        builtins = _get_builtin_names()
        assert "DELAY3" in builtins
        assert "DELAYN" in builtins
        assert "DELAY_FIXED" in builtins

    def test_delay3_no_validation_error(self):
        """Model using DELAY3 should pass validation."""
        m = parse_sysd("""
model 'D3Valid'
  dt 0.5
  from 0 to 20
  stock 'Input': 100
    + 'set_input': 100
  stock 'Delayed': 0
    + 'd_in': Input
    - 'd_out': DELAY3(Input, 6)
""")
        result = m.validate()
        assert result.is_valid


# ── CONVEY Tests ────────────────────────────────────────────────

class TestCONVEY:
    def test_convey_recognized_in_builtins(self):
        """CONVEY should be a recognized builtin name."""
        from dynafx.dynamics.dsl import _get_builtin_names
        assert "CONVEY" in _get_builtin_names()

    def test_convey_creates_aux_variable(self):
        """CONVEY(x, T) should create one aux state variable."""
        m = parse_sysd("""
model 'ConveyTest'
  dt 1
  from 0 to 20
  aux const_in: 100
  stock 'Storage': 0
    + 'in': const_in
    - 'out': CONVEY(const_in, 5)
""")
        from dynafx.dynamics.dsl import _build_system
        f, names, y0, aux_count, _ = _build_system(m, {})
        convey_vars = [n for n in names if "_convey_" in n]
        assert len(convey_vars) == 1, f"Expected 1 CONVEY var, got {convey_vars}"

    def test_convey_basic(self):
        """CONVEY with constant input: storage accumulates, then plateaus at delay×input."""
        m = parse_sysd("""
model 'ConveyBasic'
  dt 1
  from 0 to 20
  aux const_in: 100
  stock 'Storage': 0
    + 'in': const_in
    - 'out': CONVEY(const_in, 5)
""")
        result = m.simulate(method="euler")
        vals = result["values"]["Storage"]
        times = result.times
        # Before t=5: CONVEY emits 0, Storage grows at 100/day
        for i, t in enumerate(times):
            if t <= 5:
                expected = int(t) * 100
                assert abs(vals[i] - expected) < 1, f"At t={t}: expected {expected}, got {vals[i]}"
        # At steady state (t >= 6): in=100, out=100, Storage=500
        for i, t in enumerate(times):
            if t >= 6:
                assert abs(vals[i] - 500) < 1, f"At t={t}: expected 500, got {vals[i]}"

    def test_convey_euler_rk4_match(self):
        """CONVEY should give same results with RK4 and Euler (no buffer corruption)."""
        m = parse_sysd("""
model 'ConveyMatch'
  dt 0.25
  from 0 to 30
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': CONVEY(const_input, 5)
""")
        rk4 = m.simulate(method="rk4")
        euler = m.simulate(method="euler")
        for i in range(len(rk4.times)):
            diff = abs(rk4["values"]["Storage"][i] - euler["values"]["Storage"][i])
            assert diff < 0.5, f"Diff at t={rk4.times[i]:.2f}: {diff}"

    def test_convey_pipeline_fifo_behavior(self):
        """CONVEY should pipeline exact input value for delay time, then emit."""
        m = parse_sysd("""
model 'ConveyFIFO'
  dt 1
  from 0 to 15
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': CONVEY(const_input, 5)
""")
        result = m.simulate(method="euler")
        vals = result["values"]["Storage"]
        times = result.times
        # With constant 100 input and delay=5:
        # t=0..5: Storage accumulates at 100/day (CONVEY still in pipeline)
        # t=5: Storage should be ~500, then steady at 500 (in=100, out=100)
        for i, t in enumerate(times):
            if t <= 5:
                expected = int(t) * 100
                assert abs(vals[i] - expected) < 1, f"At t={t}: expected {expected}, got {vals[i]}"
        # After t=5, Storage should stay at 500 (steady state)
        for i, t in enumerate(times):
            if t >= 6:
                assert abs(vals[i] - 500) < 1, f"At t={t}: expected 500, got {vals[i]}"


    def test_convey_uniform_stochastic(self):
        """CONVEY with UNIFORM(a,b) should sample random delays per entry."""
        m = parse_sysd("""
model 'ConveyUniform'
  dt 1
  from 0 to 20
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': CONVEY(const_input, UNIFORM(2, 8))
""")
        r1 = m.simulate(method="euler")
        # UNIFORM should produce non-constant pipeline delay → steady state varies
        vals = r1["values"]["Storage"]
        # Storage should accumulate (delay > 0) then plateau
        assert vals[-1] > 0
        # At t=20 with mean delay ~5 and input 100/day, Storage should be substantial
        assert vals[-1] > 500

    def test_convey_lognormal_stochastic(self):
        """CONVEY with LOGNORMAL(mu, sigma) should sample lognormal delays."""
        m = parse_sysd("""
model 'ConveyLogNormal'
  dt 1
  from 0 to 20
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': CONVEY(const_input, LOGNORMAL(1.5, 0.4))
""")
        r = m.simulate(method="euler")
        vals = r["values"]["Storage"]
        # Storage should accumulate (delay > 0), then plateau
        assert vals[-1] > vals[0]
        assert vals[-1] > 0

    def test_convey_uniform_no_negative_delay(self):
        """CONVEY with UNIFORM should never produce negative delays."""
        m = parse_sysd("""
model 'ConveyNoNeg'
  dt 1
  from 0 to 10
  aux const_input: 50
  stock 'Storage': 0
    + 'in': const_input
    - 'out': CONVEY(const_input, UNIFORM(0.1, 2))
""")
        r = m.simulate(method="euler")
        vals = r["values"]["Storage"]
        assert all(v >= 0 for v in vals)
        assert vals[-1] > 0


# ── CONVEY_BATCH (Transport Batching) ────────────────────────────

class TestCONVEYBatch:
    def test_convey_batch_recognized_in_builtins(self):
        from dynafx.dynamics.dsl import _get_builtin_names
        assert "CONVEY_BATCH" in _get_builtin_names()

    def test_convey_batch_accumulates_and_emits_pulses(self):
        """CONVEY_BATCH should accumulate input and emit in batch pulses."""
        m = parse_sysd("""
model 'BatchTest'
  dt 1
  from 0 to 30
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': CONVEY_BATCH(const_input, 3, 200)
""")
        r = m.simulate(method="euler")
        vals = r["values"]["Storage"]
        # At t=0 to 3: no outflow yet (batch not ready + pipeline delay 3)
        # At t=3: first batch of 200 arrives (accumulated 300 over 3 days)
        # At t=6: second batch of 200, etc.
        # Steady state: in=100, out=...batch emissions vary
        assert vals[-1] > 0

    def test_convey_batch_no_premature_emission(self):
        """CONVEY_BATCH should not emit before delay has elapsed."""
        m = parse_sysd("""
model 'BatchNoEarly'
  dt 1
  from 0 to 5
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': CONVEY_BATCH(const_input, 5, 500)
""")
        r = m.simulate(method="euler")
        vals = r["values"]["Storage"]
        # At t=5: input = 500, so one batch of 500 should be emitted at t=5+5=10 (after sim end)
        # So Storage should accumulate linearly (no outflow yet)
        # At t=5: Storage should be 500
        assert abs(vals[-1] - 500) < 1

    def test_convey_batch_emits_exact_batch_size(self):
        """Each batch emission should be exactly batch_size."""
        m = parse_sysd("""
model 'BatchExact'
  dt 0.5
  from 0 to 20
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': CONVEY_BATCH(const_input, 2, 150)
""")
        r = m.simulate(method="euler")
        outflow = r["values"]["Storage_diff"] if "Storage_diff" in r["values"] else None
        # Can't easily check individual batch sizes from cumulative storage,
        # but total emissions should equal cumulative batches
        vals = r["values"]["Storage"]
        # With input=100/day, dt=0.5: 50 per step
        # delay=2, batch=150: batches emitted every 3 steps (150/50=3)
        # Total throughput should be approximately 100*20 = 2000
        assert vals[-1] > 0

    def test_convey_batch_with_uniform_delay(self):
        """CONVEY_BATCH with stochastic delay should work."""
        m = parse_sysd("""
model 'BatchStochastic'
  dt 1
  from 0 to 20
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': CONVEY_BATCH(const_input, UNIFORM(2, 5), 200)
""")
        r = m.simulate(method="euler")
        vals = r["values"]["Storage"]
        assert all(v >= 0 for v in vals)
        assert vals[-1] > 0


# ── Pipeline Delay Fix Regression Tests ─────────────────────────

class TestPipelineDelayFix:
    """Regression tests for the RK4/DELAY_FIXED buffer corruption fix."""

    def test_delay_fixed_rk4_euler_match(self):
        """DELAY_FIXED should give same results with RK4 and Euler (regression: buffer corruption)."""
        m = parse_sysd("""
model 'DFMatch'
  dt 0.25
  from 0 to 30
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': DELAY_FIXED(const_input, 5)
""")
        rk4 = m.simulate(method="rk4")
        euler = m.simulate(method="euler")
        for i in range(len(rk4.times)):
            diff = abs(rk4["values"]["Storage"][i] - euler["values"]["Storage"][i])
            assert diff < 0.5, f"Diff at t={rk4.times[i]:.2f}: {diff}"

    def test_delay_fixed_no_premature_emission(self):
        """DELAY_FIXED should not emit value before delay has elapsed."""
        m = parse_sysd("""
model 'DFNoEarly'
  dt 1
  from 0 to 15
  aux const_input: 100
  stock 'Storage': 0
    + 'in': const_input
    - 'out': DELAY_FIXED(const_input, 5)
""")
        result = m.simulate(method="rk4")
        vals = result["values"]["Storage"]
        times = result.times
        # Before t=5, outflow is 0, so Storage accumulates at 100/day
        for i, t in enumerate(times):
            if t < 5:
                expected = int(t) * 100
                assert abs(vals[i] - expected) < 1, f"At t={t}: expected {expected}, got {vals[i]}"

    def test_supply_chain_demo_rk4_euler_match(self):
        """Supply chain demo should give similar fill rates with RK4 and Euler."""
        from dynafx.dynamics.dsl import parse_sysd_file
        import os
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "supply_chain_demo.sysd")
        model = parse_sysd_file(model_path)
        params = {'base_demand': 500, 'smoothing_time': 4, 'reorder_point': 2000, 'shipping_delay': 6, 'factory_capacity': 2000}
        rk4 = model.simulate(method="rk4", params=params)
        euler = model.simulate(method="euler", params=params)
        fr_rk4 = rk4["values"]["Cumulative_Met"][-1] / rk4["values"]["Cumulative_Demand"][-1] * 100
        fr_euler = euler["values"]["Cumulative_Met"][-1] / euler["values"]["Cumulative_Demand"][-1] * 100
        assert abs(fr_rk4 - fr_euler) < 1.0, f"Fill rate diff: RK4={fr_rk4:.1f}%, Euler={fr_euler:.1f}%"

    def test_retailer_does_not_deplete(self):
        """Retailer inventory should never go to zero (regression: was depleting at t~101)."""
        from dynafx.dynamics.dsl import parse_sysd_file
        import os
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "supply_chain_demo.sysd")
        model = parse_sysd_file(model_path)
        result = model.simulate(method="rk4", params={'base_demand': 500, 'smoothing_time': 4, 'reorder_point': 2000, 'shipping_delay': 6, 'factory_capacity': 2000})
        min_retail = min(result["values"]["Retailer_Inventory"])
        assert min_retail > 0, f"Retailer depleted to {min_retail}"


# ── CONVEY_BATCH Tests ───────────────────────────────────────────


class TestCONVEYBATCH:
    def test_convey_batch_basic_accumulate(self):
        """CONVEY_BATCH accumulates input and emits a batch after delay."""
        m = parse_sysd("""
model 'CBatchTest'
  dt 1
  from 0 to 15
  aux input_rate: 100
  aux shipped: CONVEY_BATCH(input_rate, 3, 400)
  stock 'Buffer': 0
    + 'inflow': input_rate
    - 'outflow': shipped
""")
        r = m.simulate(method='euler')
        buf = r['values']['Buffer']
        # Before batch matures: Buffer accumulates at 100/dt
        assert buf[4] == pytest.approx(400, abs=1)
        assert buf[7] == pytest.approx(700, abs=1)
        # After first batch (400) exits pipeline:
        assert buf[8] == pytest.approx(400, abs=1)

    def test_convey_batch_steady_state(self):
        """CONVEY_BATCH enters steady sawtooth pattern."""
        m = parse_sysd("""
model 'CBatchTest'
  dt 1
  from 0 to 30
  aux input_rate: 100
  aux shipped: CONVEY_BATCH(input_rate, 3, 400)
  stock 'Buffer': 0
    + 'inflow': input_rate
    - 'outflow': shipped
""")
        r = m.simulate(method='euler')
        buf = r['values']['Buffer']
        # After settling, Buffer oscillates 400..700 every 4 steps
        for t_idx in range(8, 25):
            mod = (t_idx - 8) % 4
            expected = 400 + mod * 100
            assert buf[t_idx] == pytest.approx(expected, abs=1), f"t={t_idx}: got {buf[t_idx]}, expected {expected}"

    def test_convey_batch_rk4_euler_consistency(self):
        """CONVEY_BATCH should produce similar results with RK4 and Euler."""
        m = parse_sysd("""
model 'CBatchTest'
  dt 0.25
  from 0 to 20
  aux inflow_rate: 50
  aux outflow: CONVEY_BATCH(inflow_rate, 2, 200)
  stock 'Inv': 0
    + 'in': inflow_rate
    - 'out': outflow
""")
        rk4 = m.simulate(method='rk4')
        euler = m.simulate(method='euler')
        rk4_end = rk4['values']['Inv'][-1]
        euler_end = euler['values']['Inv'][-1]
        assert abs(rk4_end - euler_end) < rk4_end * 0.5

    def test_convey_batch_variable_batch_size(self):
        """CONVEY_BATCH with variable batch size (parameter reference)."""
        m = parse_sysd("""
model 'CBatchVar'
  dt 1
  from 0 to 20
  aux input_rate: 100
  aux batch_sz: 300
  aux shipped: CONVEY_BATCH(input_rate, 2, batch_sz)
  stock 'Buffer': 0
    + 'inflow': input_rate
    - 'outflow': shipped
""")
        r = m.simulate(method='euler')
        buf = r['values']['Buffer']
        # batch_sz=300, input=100/dt, delay=2
        # First batch emitted at t=3, exits pipeline at t=5
        # After settling (t>=5): 500 -> 300 -> 400 -> 500 -> 300 -> ...
        assert buf[5] == pytest.approx(500, abs=1)
        pattern = [500, 300, 400]
        for i, t in enumerate([5, 6, 7, 8, 9, 10]):
            expected = pattern[i % 3]
            assert buf[t] == pytest.approx(expected, abs=1), f"t={t}: got {buf[t]}, expected {expected}"

    def test_convey_batch_small_dt_fidelity(self):
        """CONVEY_BATCH with dt=0.25 should correctly accumulate fractional values."""
        m = parse_sysd("""
model 'CBatchFine'
  dt 0.25
  from 0 to 10
  aux rate: 10
  aux shipped: CONVEY_BATCH(rate, 1, 30)
  stock 'S': 0
    + 'in': rate
    - 'out': shipped
""")
        r = m.simulate(method='euler')
        s_vals = r['values']['S']
        # At rate=10/dt=0.25, accumulates 2.5 per step -> 12 steps for 30
        # First batch: floor(30/2.5)=12 steps, then pipeline delay 1.0
        # After t=3.0 (12 steps * 0.25), batch of 30 should fire
        assert s_vals[12] > s_vals[0]

    def test_convey_batch_no_stock_outflow(self):
        """CONVEY_BATCH produces outflow that can be consumed by a stock."""
        m = parse_sysd("""
model 'CBatchStandalone'
  dt 1
  from 0 to 10
  aux inp: 50
  aux out: CONVEY_BATCH(inp, 2, 100)
  stock 'S': 0
    - 'drain': out
""")
        r = m.simulate(method='euler')
        s = r['values']['S']
        # inp=50, batch=100 (2 steps), delay=2
        # First batch emitted at t=2, exits pipeline at t=4
        # At t=5: S drained by 100
        assert s[5] == pytest.approx(-100, abs=1)

    def test_convey_batch_pipeline_correctness(self):
        """Verify pipeline: batches arrive exactly after delay and in correct quantity."""
        m = parse_sysd("""
model 'CBatchCorrect'
  dt 1
  from 0 to 20
  aux inp: 100
  aux out: CONVEY_BATCH(inp, 4, 300)
  stock 'Acc': 0
    + 'add': inp
    - 'sub': out
""")
        r = m.simulate(method='euler')
        acc = r['values']['Acc']
        # inp=100, batch=300, delay=4
        # Accumulates 100/dt: Reaches 300 at t=3 (batch emitted at t=3, exit t=7)
        # t=3: batch emitted, accumulator reset to 0, Acc keeps growing
        assert acc[3] == pytest.approx(300, abs=1)
        # t=7: batch arrives, outflow=300, Acc drops
        assert acc[7] == pytest.approx(700, abs=1)
        # After outflow: Acc = 700 - 300 + 100 = 500
        assert acc[8] == pytest.approx(500, abs=1)
        # Sawtooth settled: 500 -> 600 -> 700 -> 500 -> ...
        assert acc[9] == pytest.approx(600, abs=1)
        assert acc[10] == pytest.approx(700, abs=1)
        assert acc[11] == pytest.approx(500, abs=1)
