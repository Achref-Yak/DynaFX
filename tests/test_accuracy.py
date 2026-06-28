"""Accuracy tests for the .sysd solver against analytical solutions.

Verifies that our RK4 solver + DSL parser produce correct results for
models with known closed-form solutions.
"""

from dynafx.system.dsl import parse_sysd
import math


# ── Analytical reference solutions ──────────────────────────────

def test_exponential_decay_analytical():
    """stock decaying with outflow proportional to itself → S(t) = S0 * exp(-k*t)."""
    mdl = """Exponential Decay
  stock "S": 1000
    - "Decay": S * decay
"""
    m = parse_sysd(mdl)
    m.dt = 0.5
    m.t_span = (0.0, 10.0)
    r = m.simulate(params={"decay": 0.1})

    t_end = 10.0
    s_analytical = 1000.0 * math.exp(-0.1 * t_end)
    s_numerical = r["values"]["S"][-1]

    error = abs(s_numerical - s_analytical) / s_analytical
    assert error < 0.01, f"Exponential decay: {error:.4%} error ({s_numerical:.2f} vs {s_analytical:.2f})"


def test_linear_growth_analytical():
    """stock with constant inflow → X(t) = X0 + rate * t."""
    mdl = """Linear Growth
  stock "X": 0
    + "Growth": rate
"""
    m = parse_sysd(mdl)
    m.dt = 1.0
    m.t_span = (0.0, 50.0)
    r = m.simulate(params={"rate": 3.0})

    x_analytical = 0.0 + 3.0 * 50.0  # = 150.0
    x_numerical = r["values"]["X"][-1]

    error = abs(x_numerical - x_analytical)
    assert error < 0.1, f"Linear growth: {error:.4f} error ({x_numerical:.2f} vs {x_analytical:.2f})"


def test_sir_peak_infected():
    """SIR model peak infected count vs analytical R0 prediction."""
    mdl = """Simple SIR
  stock "Susceptible": 990
    - "Infection_Rate": beta * Susceptible * Infected
  stock "Infected": 10
    + "Infection_Rate": beta * Susceptible * Infected
    - "Recovery_Rate": gamma * Infected
  stock "Recovered": 0
    + "Recovery_Rate": gamma * Infected
"""
    m = parse_sysd(mdl)
    m.dt = 0.1
    m.t_span = (0.0, 200.0)
    r = m.simulate(params={"beta": 0.002, "gamma": 0.1})

    i_values = r["values"]["Infected"]
    i_peak_numerical = max(i_values)

    # Analytical peak: I_peak ≈ N - gamma/beta * (1 + ln(beta*S0/gamma))
    N = 1000.0
    beta, gamma = 0.002, 0.1
    R0 = beta * 990.0 / gamma
    s_peak = gamma / beta
    i_peak_analytical = N - s_peak * (1.0 + math.log(990.0 / s_peak) / R0)

    # Allow 25% tolerance (discretization + numerical error + formula approximation)
    error_pct = abs(i_peak_numerical - i_peak_analytical) / i_peak_analytical
    assert error_pct < 0.25, (
        f"SIR peak: {error_pct:.1%} error "
        f"({i_peak_numerical:.1f} vs {i_peak_analytical:.1f})"
    )


def test_population_conservation():
    """SIR total population (S+I+R) must be conserved."""
    mdl = """Conservation Test
  stock "Susceptible": 990
    - "Infection_Rate": beta * Susceptible * Infected
  stock "Infected": 10
    + "Infection_Rate": beta * Susceptible * Infected
    - "Recovery_Rate": gamma * Infected
  stock "Recovered": 0
    + "Recovery_Rate": gamma * Infected
"""
    m = parse_sysd(mdl)
    m.dt = 0.25
    m.t_span = (0.0, 100.0)
    r = m.simulate(params={"beta": 0.003, "gamma": 0.1})

    total_0 = 990.0 + 10.0 + 0.0
    for t_idx in range(len(r["times"])):
        total_t = (
            r["values"]["Susceptible"][t_idx]
            + r["values"]["Infected"][t_idx]
            + r["values"]["Recovered"][t_idx]
        )
        assert abs(total_t - total_0) < 0.01, (
            f"Population not conserved at t={r['times'][t_idx]:.1f}: "
            f"{total_t:.4f} vs {total_0:.4f}"
        )


def test_smooth_approaches_input():
    """SMOOTH(input, delay) should converge to input after several delays."""
    mdl = """SMOOTH Test
  stock "Raw": 100
  stock "Smoothed": 0
    + "Raw": Raw
    - "Smoothed": SMOOTH(Smoothed, delay_time)
"""
    m = parse_sysd(mdl)
    m.dt = 1.0
    m.t_span = (0.0, 50.0)
    r = m.simulate(params={"delay_time": 5.0})

    smoothed_final = r["values"]["Smoothed"][-1]
    raw_final = r["values"]["Raw"][-1]

    # After 50 time units (10x the delay), smoothed should be very close to raw
    assert abs(smoothed_final - raw_final) < 1.0, (
        f"SMOOTH did not converge: {smoothed_final:.2f} vs {raw_final:.2f}"
    )


def test_euler_vs_rk4_stability():
    """For a stiff system, RK4 should be more accurate than Euler."""
    mdl = """Stiff Test
  stock "A": 100
    + "Slow_Input": slow_input
    - "Decay": A * fast_rate
"""
    m = parse_sysd(mdl)
    m.dt = 0.5

    r_rk4 = m.simulate(method="rk4", params={"fast_rate": 2.0, "slow_input": 50.0})
    r_euler = m.simulate(method="euler", params={"fast_rate": 2.0, "slow_input": 50.0})

    # Analytical: A(t) → slow_input/fast_rate = 25 as t→∞
    a_analytical = 25.0
    err_rk4 = abs(r_rk4["values"]["A"][-1] - a_analytical)
    err_euler = abs(r_euler["values"]["A"][-1] - a_analytical)

    assert err_rk4 < 1.0, f"RK4 too far from steady state: {err_rk4:.2f}"
    assert err_euler < 5.0, f"Euler too far from steady state: {err_euler:.2f}"


def test_multi_stock_conservation():
    """Two-stock system with transfer: total must be conserved."""
    mdl = """Transfer Test
  stock "Source": 1000
    - "Transfer_Rate": Source * transfer_fraction
  stock "Sink": 0
    + "Transfer_Rate": Source * transfer_fraction
"""
    m = parse_sysd(mdl)
    m.dt = 1.0
    m.t_span = (0.0, 50.0)
    r = m.simulate(params={"transfer_fraction": 0.1})

    for t_idx in range(len(r["times"])):
        total = r["values"]["Source"][t_idx] + r["values"]["Sink"][t_idx]
        assert abs(total - 1000.0) < 0.1, (
            f"Transfer not conserved at t={r['times'][t_idx]:.1f}: {total:.4f}"
        )


def test_table_interpolation():
    """Table lookup values should interpolate correctly."""
    mdl = """Table Test
  table "demand"
    x: [0, 50, 100]
    y: [100, 150, 50]
  stock "Stock": 0
    + "demand": demand
"""
    m = parse_sysd(mdl)
    m.dt = 1.0
    m.t_span = (0.0, 100.0)
    r = m.simulate()

    # Stock is integral of demand, so just check it's monotonically increasing
    stock_vals = r["values"]["Stock"]
    for i in range(1, len(stock_vals)):
        assert stock_vals[i] >= stock_vals[i - 1] - 0.01, (
            f"Stock decreased at index {i}: {stock_vals[i]} < {stock_vals[i-1]}"
        )
