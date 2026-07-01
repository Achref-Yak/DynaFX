"""Tests for Phase 1: math functions, time functions, SMOOTHI."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TMPDIR", "/tmp")

from dynafx.dynamics.dsl import parse_sysd, SysdModelResult


# ── Helpers ──────────────────────────────────────────────────────

def _sim(code: str, dt: float = 1.0, t_span: tuple = (0, 100), **params) -> SysdModelResult:
    m = parse_sysd(code)
    return m.simulate(dt=dt, t_span=t_span, params=params)


def _val(r: SysdModelResult, name: str) -> float:
    return r.values[name][-1]


# ── ABS ──────────────────────────────────────────────────────────

def test_abs_positive():
    r = _sim("""
stock X: 10
  - drain: ABS(-5)
""", dt=0.1, t_span=(0, 0.1))
    # drain = 5, one step of 0.1 => X = 10 - 5*0.1 = 9.5
    assert abs(_val(r, "X") - 9.5) < 0.01


def test_abs_negative_input():
    r = _sim("""
stock X: 10
  - drain: ABS(X - 20)
""", dt=0.1, t_span=(0, 0.1))
    # ABS(X-20) changes as X changes; verify it's between 8 and 10
    assert 8.0 < _val(r, "X") < 10.0


# ── EXP / LN ────────────────────────────────────────────────────

def test_exp():
    r = _sim("""
stock X: 0
  - drain: 0
  + growth: EXP(1) * 0.1
""", dt=0.1, t_span=(0, 0.1))
    # growth = e * 0.1 ≈ 0.2718, one step => X = 0.2718*0.1 ≈ 0.02718
    assert abs(_val(r, "X") - math.e * 0.1 * 0.1) < 0.001


def test_ln():
    r = _sim("""
aux val: 10
stock X: 0
  + growth: LN(val) * 0.1
""", dt=0.1, t_span=(0, 0.1))
    assert abs(_val(r, "X") - math.log(10) * 0.1 * 0.1) < 0.001


# ── SQRT ────────────────────────────────────────────────────────

def test_sqrt():
    r = _sim("""
stock X: 0
  + growth: SQRT(16) * 0.5
""", dt=0.1, t_span=(0, 0.1))
    # sqrt(16)=4, growth=2, one step => X = 2*0.1 = 0.2
    assert abs(_val(r, "X") - 0.2) < 0.01


# ── SIN / COS / PI ──────────────────────────────────────────────

def test_sin():
    r = _sim("""
stock X: 0
  + growth: SIN(PI / 2) * 10
""", dt=0.1, t_span=(0, 0.1))
    # sin(PI/2) = 1.0, growth=10, one step => X = 10*0.1 = 1.0
    assert abs(_val(r, "X") - 1.0) < 0.01


def test_cos():
    r = _sim("""
stock X: 0
  + growth: COS(0) * 10
""", dt=0.1, t_span=(0, 0.1))
    # cos(0) = 1.0, growth=10, one step => X = 10*0.1 = 1.0
    assert abs(_val(r, "X") - 1.0) < 0.01


def test_pi():
    r = _sim("""
stock X: 0
  + growth: PI * 10
""", dt=0.1, t_span=(0, 0.1))
    assert abs(_val(r, "X") - math.pi * 10 * 0.1) < 0.01


# ── Combined math ───────────────────────────────────────────────

def test_combined_math():
    r = _sim("""
stock X: 0
  + growth: SQRT(ABS(-9)) + SIN(PI) + COS(0)
""", dt=0.1, t_span=(0, 0.1))
    # sqrt(9) + sin(pi) + cos(0) = 3 + 0 + 1 = 4
    assert abs(_val(r, "X") - 4.0 * 0.1) < 0.01


# ── PULSE ───────────────────────────────────────────────────────

def test_pulse_inside():
    r = _sim("""
stock X: 0
  + inflow: PULSE(100, 5, 2)
""", dt=1.0, t_span=(0, 10))
    # t=5..7: pulse=100, dt=1 => ~200 added in 2 steps
    assert _val(r, "X") >= 190.0


def test_pulse_outside():
    r = _sim("""
stock X: 0
  + inflow: PULSE(100, 5, 2)
""", dt=1.0, t_span=(0, 10))
    # After t=7, pulse stops => stock stays at ~200
    assert _val(r, "X") < 300.0


def test_pulse_width_one():
    r = _sim("""
stock X: 0
  + inflow: PULSE(50, 3, 1)
""", dt=1.0, t_span=(0, 10))
    # t=3: pulse fires once, adds ~50
    assert abs(_val(r, "X") - 50.0) < 1.0


# ── STEP ────────────────────────────────────────────────────────

def test_step_before():
    r = _sim("""
stock X: 0
  + inflow: STEP(10, 5)
""", dt=1.0, t_span=(0, 4))
    # t < 5 always, step = 0
    assert _val(r, "X") == 0.0


def test_step_after():
    r = _sim("""
stock X: 0
  + inflow: STEP(10, 5)
""", dt=1.0, t_span=(0, 10))
    # t >= 5 for t=5,6,7,8,9 => 5 steps of 10 = 50
    assert _val(r, "X") >= 49.0


# ── RAMP ────────────────────────────────────────────────────────

def test_ramp_before():
    r = _sim("""
stock X: 0
  + inflow: RAMP(2, 3, 8)
""", dt=1.0, t_span=(0, 2))
    # t < 3 always, ramp = 0
    assert _val(r, "X") == 0.0


def test_ramp_during():
    r = _sim("""
stock X: 0
  + inflow: RAMP(2, 0, 5)
""", dt=1.0, t_span=(0, 3))
    # t=0: ramp=0, t=1: ramp=2, t=2: ramp=4 => sum = 0+2+4=6
    assert _val(r, "X") >= 5.0


def test_ramp_after():
    r = _sim("""
stock X: 0
  + inflow: RAMP(2, 0, 3)
""", dt=1.0, t_span=(0, 10))
    # After t=3, ramp = 2*3 = 6 constant, stock grows
    assert _val(r, "X") > 10.0


# ── NOISE ───────────────────────────────────────────────────────

def test_noise_bounded():
    r = _sim("""
stock X: 0
  + inflow: NOISE(1)
""", dt=0.1, t_span=(0, 10))
    # NOISE in [-1,1], dt=0.1, 100 steps => X in [-10, 10]
    assert -10.1 < _val(r, "X") < 10.1


# ── SMOOTHI ─────────────────────────────────────────────────────
# SMOOTHI creates internal state variables stripped from output,
# so test through its effect on stock values.

def test_smoothi_feeds_stock():
    """SMOOTHI(X, 10, 10) starts at 10, converges toward X=100.
    The smoothed value drives an outflow, so stock X decreases."""
    r = _sim("""
stock X: 100
  - outflow: SMOOTHI(X, 10, 10) * 0.01
""", dt=0.5, t_span=(0, 10))
    assert _val(r, "X") < 100.0
    assert _val(r, "X") > 30.0  # not all drained


def test_smoothi_converges():
    """SMOOTH and SMOOTHI with init=0 should produce same stock trajectory."""
    r1 = _sim("""
stock X: 100
  - outflow: SMOOTH(X, 10) * 0.01
""")
    r2 = _sim("""
stock X: 100
  - outflow: SMOOTHI(X, 10, 0) * 0.01
""")
    for v1, v2 in zip(r1.values["X"], r2.values["X"]):
        assert abs(v1 - v2) < 0.01


def test_smoothi_different_init_diverge():
    """Different initial values produce different trajectories."""
    r1 = _sim("""
stock X: 100
  - outflow: SMOOTHI(X, 10, 10) * 0.01
""")
    r2 = _sim("""
stock X: 100
  - outflow: SMOOTHI(X, 10, 80) * 0.01
""")
    assert abs(_val(r1, "X") - _val(r2, "X")) > 0.1


# ── Validation recognizes builtins ──────────────────────────────

def test_validation_recognizes_all_builtins():
    m = parse_sysd("""
stock X: 10
  - drain: ABS(X) + EXP(1) + LN(2) + SQRT(4) + SIN(0) + COS(0) + PI
""")
    result = m.validate()
    # No "unknown identifier" errors for builtins
    assert result.is_valid or not any("unknown" in e.lower() for e in result.errors)


# ── Stochastic functions ─────────────────────────────────────────

def test_uniform_in_expression():
    """UNIFORM(a,b) produces values in [a, b) range when sampled during sim."""
    vals = []
    for _ in range(50):
        r = _sim("""
stock X: 0
  + inflow: UNIFORM(5, 10)
""")
        vals.append(r["values"]["X"][-1])
    # All values should be between 5*100 and 10*100 (if uniform produced ~avg 7.5)
    for v in vals:
        assert 400 < v < 1000, f"UNIFORM({v}) out of expected range"


def test_lognormal_in_expression():
    """LOGNORMAL(mu, sigma) produces positive values."""
    for _ in range(20):
        r = _sim("""
stock X: 0
  + inflow: LOGNORMAL(2, 0.5)
""")
        assert r["values"]["X"][-1] > 0
