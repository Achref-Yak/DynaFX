"""Tests for sensitivity analysis — SensitivityResult, SensitivityAnalyzer, and ensemble simulation."""

import math
import numpy as np

from dynafx.dynamics import (
    SensitivityAnalyzer,
    SensitivityResult,
    parse_sysd,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Existing ensemble tests (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def test_simulate_ensemble_basic():
    m = parse_sysd("""
model 'SM'
  dt 1
  from 0 to 10
  stock 'S': 100
    - 'Out': 0.1 * S
""")
    ens = m.simulate_ensemble(params={"dt": (0.5, 1.5)}, n=5)
    assert "mean" in ens
    assert "std" in ens
    assert "p5" in ens
    assert "p95" in ens
    assert "trajectories" in ens
    assert len(ens["trajectories"]) == 5
    assert ens["stocks"] == ["S"]


def test_ensemble_mean_reasonable():
    m = parse_sysd("""
model 'EM'
  dt 1
  from 0 to 5
  stock 'X': 0
    + 'In': 10
""")
    ens = m.simulate_ensemble(params={}, n=3)
    assert abs(ens["mean"]["X"][-1] - 50.0) < 1e-9


def test_ensemble_with_uncertain_param():
    m = parse_sysd("""
model 'UP'
  dt 1
  from 0 to 10
  stock 'X': 0
    + 'In': rate
  aux 'rate': growth * dt
""")
    ens = m.simulate_ensemble(params={"growth": (0.5, 1.5)}, n=10)
    final_mean = ens["mean"]["X"][-1]
    assert 0 < final_mean < 150


def test_ensemble_with_normal_dist():
    m = parse_sysd("""
model 'ND'
  dt 1
  from 0 to 5
  stock 'S': 100
    - 'Out': S * decay
  aux 'decay': 0.1
""")
    param_spec: tuple[float, float, str] = (0.05, 0.15, "normal")
    ens = m.simulate_ensemble(params={"decay": param_spec}, n=10, seed=42)
    assert len(ens["trajectories"]) == 10


def test_ensemble_with_lognormal_dist():
    m = parse_sysd("""
model 'LN'
  dt 1
  from 0 to 5
  stock 'S': 100
    - 'Out': S * rate
  aux 'rate': 0.1
""")
    param_spec: tuple[float, float, str] = (0.05, 0.2, "lognormal")
    ens = m.simulate_ensemble(params={"rate": param_spec}, n=10, seed=42)
    assert len(ens["trajectories"]) == 10


def test_ensemble_seed_reproducibility():
    m = parse_sysd("""
model 'RP'
  dt 1
  from 0 to 5
  stock 'X': 0
    + 'In': rate
  aux 'rate': growth
""")
    ens1 = m.simulate_ensemble(params={"growth": (0.5, 1.5)}, n=5, seed=42)
    ens2 = m.simulate_ensemble(params={"growth": (0.5, 1.5)}, n=5, seed=42)
    assert ens1["mean"]["X"][-1] == ens2["mean"]["X"][-1]


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityResult
# ═══════════════════════════════════════════════════════════════════════════════

def _model_linear() -> str:
    """Deterministic linear model: y = 2*A + 3*B."""
    return """
model 'linear'
  dt 1
  from 0 to 1
  stock 'x': 0
    + 'dx': 2 * A + 3 * B
  aux 'y': 2 * A + 3 * B
"""


def _model_interaction() -> str:
    """y = A * B  (pure interaction, no main effects at zero-mean)."""
    return """
model 'interact'
  dt 1
  from 0 to 1
  stock 'x': 0
    + 'dx': A * B
  aux 'y': A * B
"""


class TestSensitivityResult:
    def test_to_dict_sobol(self):
        r = SensitivityResult(
            method="sobol",
            param_names=["A", "B"],
            output="y",
            n_samples=100,
            first_order={"A": 0.3, "B": 0.6},
            total_order={"A": 0.35, "B": 0.65},
        )
        d = r.to_dict()
        assert d["method"] == "sobol"
        assert d["first_order"]["A"] == 0.3
        assert d["total_order"]["B"] == 0.65

    def test_to_dict_morris(self):
        r = SensitivityResult(
            method="morris",
            param_names=["A", "B"],
            output="y",
            n_samples=30,
            mu_star={"A": 2.0, "B": 5.0},
            sigma={"A": 0.1, "B": 0.3},
        )
        d = r.to_dict()
        assert d["mu_star"]["B"] == 5.0

    def test_to_dict_ci(self):
        r = SensitivityResult(
            method="sobol",
            param_names=["A"],
            output="y",
            n_samples=100,
            first_order={"A": 0.3},
            first_order_ci={"A": (0.1, 0.5)},
        )
        d = r.to_dict()
        assert d["first_order_ci"]["A"] == [0.1, 0.5]

    def test_ranking_orders_by_value(self):
        r = SensitivityResult(
            method="sobol",
            param_names=["C", "A", "B"],
            output="y",
            n_samples=100,
            first_order={"A": 0.4, "B": 0.8, "C": 0.1},
        )
        ranked = r.ranking("first_order")
        assert ranked[0] == ("B", 0.8)
        assert ranked[1] == ("A", 0.4)
        assert ranked[2] == ("C", 0.1)

    def test_ranking_uses_absolute_value(self):
        r = SensitivityResult(
            method="prcc",
            param_names=["A", "B"],
            output="y",
            n_samples=100,
            prcc={"A": -0.7, "B": 0.3},
        )
        ranked = r.ranking("prcc")
        assert ranked[0] == ("A", -0.7)  # | -0.7 | > | 0.3 |

    def test_ranking_raises_for_missing_method(self):
        r = SensitivityResult(method="sobol", param_names=[], output="y", n_samples=0)
        import re
        import pytest
        with pytest.raises(ValueError, match="No index 'prcc'"):
            r.ranking("prcc")


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — Sobol
# ═══════════════════════════════════════════════════════════════════════════════

class TestSobol:
    def test_sobol_linear_recovers_first_order(self):
        """For y = 2*A + 3*B, first-order should be proportional to variance contribution.

        Var(2A) = 4 * Var(A) = 4 * (1/12) = 1/3  (uniform on [0,1])
        Var(3B) = 9 * Var(B) = 9 * (1/12) = 3/4
        Total = 1/3 + 3/4 = 13/12
        S_A = (1/3) / (13/12) = 4/13 = 0.3077
        S_B = (3/4) / (13/12) = 9/13 = 0.6923
        """
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.sobol({"A": (0, 1), "B": (0, 1)}, output="y", n_base=256, seed=42)

        assert result.method == "sobol"
        assert abs(result.first_order["A"] - 0.3077) < 0.1
        assert abs(result.first_order["B"] - 0.6923) < 0.1
        assert abs(result.total_order["A"] - 0.3077) < 0.15
        assert abs(result.total_order["B"] - 0.6923) < 0.15

    def test_sobol_sums_to_approx_one(self):
        """First-order indices should sum to ~1 for additive model."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.sobol({"A": (0, 1), "B": (0, 1)}, output="y", n_base=256, seed=42)
        total = result.first_order["A"] + result.first_order["B"]
        assert abs(total - 1.0) < 0.15

    def test_sobol_total_order_ge_first(self):
        """Total-order >= first-order for each parameter."""
        m = parse_sysd(_model_interaction())
        sa = SensitivityAnalyzer(m)
        result = sa.sobol({"A": (0, 1), "B": (0, 1)}, output="y", n_base=256, seed=42)
        for p in ("A", "B"):
            assert result.total_order[p] >= result.first_order[p] - 0.05

    def test_sobol_with_aux_output(self):
        """Sobol on an aux variable (not a stock)."""
        m = parse_sysd("""
model 'aux_test'
  dt 1
  from 0 to 1
  stock 'x': 0
    + 'dx': foo
  aux 'foo': A + B
""")
        sa = SensitivityAnalyzer(m)
        result = sa.sobol({"A": (0, 1), "B": (0, 1)}, output="foo", n_base=128, seed=42)
        assert abs(result.first_order["A"] + result.first_order["B"] - 1.0) < 0.2

    def test_sobol_confidence_intervals(self):
        """Bootstrap CIs should contain the true value."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.sobol({"A": (0, 1), "B": (0, 1)}, output="y",
                          n_base=256, seed=42, n_bootstrap=50)
        assert result.first_order_ci is not None
        assert "A" in result.first_order_ci
        lo, hi = result.first_order_ci["A"]
        assert lo <= result.first_order["A"] <= hi

    def test_sobol_deterministic(self):
        """Same seed produces identical results."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        r1 = sa.sobol({"A": (0, 1), "B": (0, 1)}, output="y", n_base=128, seed=99)
        r2 = sa.sobol({"A": (0, 1), "B": (0, 1)}, output="y", n_base=128, seed=99)
        for p in ("A", "B"):
            assert abs(r1.first_order[p] - r2.first_order[p]) < 1e-10

    def test_sobol_raises_on_missing_output(self):
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        import pytest
        with pytest.raises(ValueError, match="not found"):
            sa.sobol({"A": (0, 1)}, output="nonexistent", n_base=32)


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — Morris
# ═══════════════════════════════════════════════════════════════════════════════

class TestMorris:
    def test_morris_linear_model(self):
        """Morris on y = 2A + 3B. B should have larger mu_star."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.morris({"A": (0, 1), "B": (0, 1)}, output="y",
                           n_trajectories=15, seed=42)
        assert result.mu_star["B"] > result.mu_star["A"]
        assert result.sigma is not None

    def test_morris_nonzero_effects(self):
        """All params should have nonzero mu_star."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.morris({"A": (0, 1), "B": (0, 1)}, output="y",
                           n_trajectories=10, seed=42)
        for p in ("A", "B"):
            assert result.mu_star[p] > 0

    def test_morris_returns_mu_as_well(self):
        """mu (signed) is available alongside mu_star (absolute)."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.morris({"A": (0, 1), "B": (0, 1)}, output="y",
                           n_trajectories=10, seed=42)
        assert result.mu is not None
        assert result.mu["A"] != 0


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — PRCC
# ═══════════════════════════════════════════════════════════════════════════════

class TestPRCC:
    def test_prcc_positive_correlation(self):
        """y = 2*A + 3*B, both positive. PRCC should be > 0."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.prcc({"A": (0, 1), "B": (0, 1)}, output="y",
                         n_samples=200, seed=42)
        assert result.prcc["A"] > 0.3
        assert result.prcc["B"] > 0.3

    def test_prcc_pvalue_available(self):
        """P-values should be returned."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.prcc({"A": (0, 1), "B": (0, 1)}, output="y",
                         n_samples=200, seed=42)
        assert result.prcc_pvalue is not None
        for p in ("A", "B"):
            assert 0 <= result.prcc_pvalue[p] <= 1

    def test_prcc_rank_invariant(self):
        """PRCC is rank-based, so monotonic transform doesn't change it."""
        m = parse_sysd("""
model 'mono'
  dt 1
  from 0 to 1
  stock 'x': 0
    + 'dx': A * A + B
  aux 'y': A * A + B
""")
        sa = SensitivityAnalyzer(m)
        result = sa.prcc({"A": (0, 1), "B": (0, 1)}, output="y",
                         n_samples=200, seed=42)
        # Both should be positive (monotonic)
        assert result.prcc["A"] > 0
        assert result.prcc["B"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — OAT
# ═══════════════════════════════════════════════════════════════════════════════

class TestOAT:
    def test_oat_returns_low_high(self):
        """OAT should return low and high values per param."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.oat({"A": (0, 1), "B": (0, 1)}, output="y")
        assert result.oat_low is not None
        assert result.oat_high is not None
        assert "A" in result.oat_low
        assert "B" in result.oat_high

    def test_oat_increasing_param_increases_output(self):
        """For y = 2*A + 3*B, raising A should increase output."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.oat({"A": (0, 2), "B": (0, 2)}, output="y")
        assert result.oat_high["A"] > result.oat_low["A"]

    def test_oat_spread_ranked(self):
        """Parameter with larger coefficient has larger spread."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.oat({"A": (0, 2), "B": (0, 2)}, output="y")
        spread_a = result.oat_high["A"] - result.oat_low["A"]
        spread_b = result.oat_high["B"] - result.oat_low["B"]
        # B should have larger spread since coefficient 3 > 2
        assert spread_b > spread_a


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — SRC
# ═══════════════════════════════════════════════════════════════════════════════

class TestSRC:
    def test_src_linear_recovers_coefficients(self):
        """For y = 2*A + 3*B (standardized), SRC = standardized coefficients.

        When A and B have same range, SRC should be proportional to
        their coefficients (2 and 3).
        """
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.src({"A": (0, 1), "B": (0, 1)}, output="y",
                        n_samples=200, seed=42)
        assert result.src is not None
        # B should have larger |SRC|
        assert abs(result.src["B"]) > abs(result.src["A"])

    def test_src_sign_preserved(self):
        """Positive coefficient -> positive SRC."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.src({"A": (0, 1), "B": (0, 1)}, output="y",
                        n_samples=200, seed=42)
        assert result.src["A"] > 0
        assert result.src["B"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — Plotting
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlots:
    def test_plot_sobol_returns_figure(self):
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.sobol({"A": (0, 1), "B": (0, 1)}, output="y", n_base=128, seed=42)
        fig = sa.plot_sobol(result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_sobol_saves_file(self, tmp_path):
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.sobol({"A": (0, 1), "B": (0, 1)}, output="y", n_base=128, seed=42)
        p = tmp_path / "sobol.png"
        out = sa.plot_sobol(result, path=str(p))
        assert out is None
        assert p.exists()

    def test_plot_sobol_raises_on_wrong_method(self):
        import pytest
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        r = SensitivityResult(method="morris", param_names=[], output="y", n_samples=0)
        with pytest.raises(ValueError, match="Expected sobol"):
            sa.plot_sobol(r)

    def test_plot_morris_returns_figure(self):
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.morris({"A": (0, 1), "B": (0, 1)}, output="y",
                           n_trajectories=10, seed=42)
        fig = sa.plot_morris(result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_tornado_returns_figure(self):
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.oat({"A": (0, 1), "B": (0, 1)}, output="y")
        fig = sa.plot_tornado(result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_prcc_heatmap_returns_figure(self):
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        r = sa.prcc({"A": (0, 1), "B": (0, 1)}, output="y", n_samples=100, seed=42)
        fig = sa.plot_prcc_heatmap(r)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_tornado_saves_file(self, tmp_path):
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        result = sa.oat({"A": (0, 1), "B": (0, 1)}, output="y")
        p = tmp_path / "tornado.png"
        out = sa.plot_tornado(result, path=str(p))
        assert out is None
        assert p.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# SensitivityAnalyzer — Edge cases & integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_parameter(self):
        """All methods should work with a single parameter."""
        m = parse_sysd(_model_linear())
        sa = SensitivityAnalyzer(m)
        sobol_r = sa.sobol({"A": (0, 1)}, output="y", n_base=128, seed=42)
        assert abs(sobol_r.first_order["A"] - 1.0) < 0.1
        morris_r = sa.morris({"A": (0, 1)}, output="y", n_trajectories=10, seed=42)
        assert morris_r.mu_star["A"] > 0
        oat_r = sa.oat({"A": (0, 1)}, output="y")
        assert oat_r.oat_low is not None

    def test_time_point_interpolation(self):
        """Sensitivity at t=0.5 of a linear ramp should match analytic."""
        m = parse_sysd("""
model 'ramp'
  dt 0.1
  from 0 to 1
  stock 'x': 0
    + 'dx': rate
""")
        sa = SensitivityAnalyzer(m)
        # x(t) = rate * t. At t=0.5, x = rate * 0.5
        result = sa.oat({"rate": (0, 2)}, output="x", t=0.5)
        assert result.oat_low is not None
        # rate=0: x(0.5)=0; rate=2: x(0.5)=1
        assert abs(result.oat_low["rate"] - 0.0) < 0.05
        assert abs(result.oat_high["rate"] - 1.0) < 0.05

    def test_many_parameters(self):
        """Sobol with 6 parameters should complete quickly."""
        model_str = """
model 'big'
  dt 1
  from 0 to 1
  stock 'x': 0
    + 'dx': """ + "+".join(f"{chr(65+i)}" for i in range(6))
        m = parse_sysd(model_str)
        sa = SensitivityAnalyzer(m)
        spec = {chr(65+i): (0, 1) for i in range(6)}
        result = sa.sobol(spec, output="x", n_base=128, seed=42)
        total = sum(result.first_order.values())
        assert abs(total - 1.0) < 0.2
