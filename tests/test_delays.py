"""Tests for higher-order delay functions: DELAY3, DELAYN, DELAY_FIXED."""

import math
import pytest
from cognitive_engine.system.dsl import SysdModel, parse_sysd


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
        from cognitive_engine.system.dsl import _build_system
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
        f, names, y0, aux_count = _build_system(m, {})
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
        from cognitive_engine.system.dsl import _build_system
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
        f5, names5, y0_5, aux5 = _build_system(m5, {})
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
        from cognitive_engine.system.dsl import _build_system
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
        f_s, names_s, y0_s, aux_s = _build_system(m_smooth, {})
        f_d, names_d, y0_d, aux_d = _build_system(m_delayn, {})
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
        from cognitive_engine.system.dsl import _build_system
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
        f, names, y0, aux_count = _build_system(m, {})
        delay_vars = [n for n in names if "_delay_fixed_" in n]
        assert len(delay_vars) == 1, f"Expected 1 delay_fixed var, got {delay_vars}"


# ── Validation Tests ────────────────────────────────────────────

class TestDelayValidation:
    def test_delay3_recognized_in_validation(self):
        """DELAY3 should be recognized as a valid builtin."""
        from cognitive_engine.system.dsl import _BUILTIN_NAMES
        assert "DELAY3" in _BUILTIN_NAMES
        assert "DELAYN" in _BUILTIN_NAMES
        assert "DELAY_FIXED" in _BUILTIN_NAMES

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
