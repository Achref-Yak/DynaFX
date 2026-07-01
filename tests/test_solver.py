"""Tests for numerical integration solvers (RK4, Euler)."""

from uuid import uuid4

from dynafx.dynamics.equations import (
    Equation,
    _parse_expression,
    euler_step,
    rk4_step,
    simulate_equations,
)


def _make_eq(
    name: str,
    inflow: str = "0",
    outflow: str = "0",
    stock_value: float = 0.0,
) -> Equation:
    return Equation(
        stock_id=uuid4(),
        stock_name=name,
        equation_type="stock_flow",
        inflow_ids=[],
        outflow_ids=[],
        inflow_expression=inflow,
        outflow_expression=outflow,
        full_expression=f"d({name})/dt = {inflow} - {outflow}",
        metadata={"stock_value": stock_value},
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


# ── Full simulate_equations ──────────────────────────────────


def test_simulate_single_stock():
    """Single stock with inflow 10, no outflow → linear growth."""
    eq = _make_eq("inventory", inflow="inflow(10)", stock_value=0.0)
    result = simulate_equations([eq], t_span=(0.0, 5.0), dt=1.0, method="rk4")

    assert result["stocks"] == ["inventory"]
    assert result["method"] == "rk4"
    assert result["steps"] == 5
    assert len(result["times"]) == 6
    assert result["times"][0] == 0.0
    assert result["times"][-1] == 5.0
    assert abs(result["final_state"][0] - 50.0) < 1e-9


def test_simulate_two_stocks():
    """Two independent stocks."""
    eqs = [
        _make_eq("stock_a", inflow="a(10)", stock_value=0.0),
        _make_eq("stock_b", inflow="b(5)", outflow="b_out(1)", stock_value=10.0),
    ]
    result = simulate_equations(eqs, t_span=(0.0, 10.0), dt=1.0, method="rk4")

    assert len(result["stocks"]) == 2
    assert len(result["values"]) == 2
    # stock_a: starts 0, inflow 10 → ends at 100
    assert abs(result["values"]["stock_a"][-1] - 100.0) < 1e-9
    # stock_b: starts 10, net 5-1=4 → ends at 10 + 40 = 50
    assert abs(result["values"]["stock_b"][-1] - 50.0) < 1e-9


def test_simulate_euler_method():
    """Euler should produce same result as RK4 for linear ODE."""
    eq = _make_eq("x", inflow="const(3)", stock_value=0.0)
    rk4_result = simulate_equations([eq], t_span=(0.0, 10.0), dt=1.0, method="rk4")
    euler_result = simulate_equations([eq], t_span=(0.0, 10.0), dt=1.0, method="euler")

    assert abs(rk4_result["final_state"][0] - euler_result["final_state"][0]) < 1e-9


def test_simulate_negative_dt():
    """Backward integration should work."""
    eq = _make_eq("x", inflow="const(5)", stock_value=100.0)
    result = simulate_equations([eq], t_span=(0.0, -10.0), dt=1.0, method="rk4")

    # Starting at 100, net rate -5 (since dt is negative)
    # After going back 10 steps: 100 + 5*(-10) = 50
    assert abs(result["final_state"][0] - 50.0) < 1e-9


def test_simulate_no_stock_value():
    """Missing stock_value defaults to 0.0."""
    eq = Equation(
        stock_id=uuid4(),
        stock_name="empty",
        equation_type="stock_flow",
        inflow_ids=[],
        outflow_ids=[],
        inflow_expression="ten(10)",
        outflow_expression="0",
        full_expression="d(empty)/dt = tenor(10) - 0",
        metadata={},
    )
    result = simulate_equations([eq], t_span=(0.0, 3.0), dt=1.0, method="rk4")
    assert abs(result["final_state"][0] - 30.0) < 1e-9
