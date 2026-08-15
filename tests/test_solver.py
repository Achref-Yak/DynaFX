"""Tests for numerical integration primitives (RK4, Euler)."""

from dynafx.dynamics.equations import (
    _parse_expression,
    euler_step,
    rk4_step,
)

# ── Expression parser ────────────────────────────────────────


def test_parse_constant():
    fn = _parse_expression("demand(200)")
    assert fn(0.0, {}) == 200.0


def test_parse_raw_number():
    fn = _parse_expression("150")
    assert fn(0.0, {}) == 150.0


def test_parse_unknown_fallback():
    fn = _parse_expression("?")
    assert fn(0.0, {}) == 0.0


def test_parse_empty_fallback():
    fn = _parse_expression("")
    assert fn(0.0, {}) == 0.0


def test_parse_negative():
    fn = _parse_expression("leak(-50)")
    assert fn(0.0, {}) == -50.0


def test_parse_float():
    fn = _parse_expression("rate(3.14)")
    assert abs(fn(0.0, {}) - 3.14) < 1e-9


# ── Solver steps ─────────────────────────────────────────────


def _const_f(_t, y, _p):
    """dy/dt = 2"""
    return [2.0]


def _linear_f(_t, y, _p):
    """dy/dt = y"""
    return [y[0]]


def test_rk4_zero_derivative():
    y = [10.0]
    y1 = rk4_step(_const_f, 0.0, y, 1.0, {})
    assert abs(y1[0] - 12.0) < 1e-9


def test_rk4_linear_growth():
    """dy/dt = y → y(t) = y0 * exp(t)."""
    y = [1.0]
    y1 = rk4_step(_linear_f, 0.0, y, 0.01, {})
    expected = 1.0 * 1.010050167  # exp(0.01)
    assert abs(y1[0] - expected) < 1e-6


def test_euler_linear_growth():
    """Euler is less accurate but should be close."""
    y = [1.0]
    y1 = euler_step(_linear_f, 0.0, y, 0.01, {})
    assert abs(y1[0] - 1.01) < 1e-9


def test_rk4_and_euler_agree_linear():
    """Both methods should agree on a pure linear ODE (dy/dt = 2)."""
    y = [0.0]
    rk4 = rk4_step(_const_f, 0.0, list(y), 0.5, {})
    eu = euler_step(_const_f, 0.0, list(y), 0.5, {})
    assert abs(rk4[0] - eu[0]) < 1e-12
