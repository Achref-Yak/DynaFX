"""Tests for linear programming and optimization."""

import pytest
import numpy as np
from dynafx.dynamics.dsl import parse_sysd
from dynafx.dynamics.optimization import (
    lp_minimize,
    lp_maximize,
    calibrate,
    optimize,
    pareto_optimize,
    LPResult,
    CalibrationResult,
    OptimizationResult,
    ParetoResult,
)


# ── Linear Programming Tests ──────────────────────────────────

class TestLP:
    def test_lp_minimize_simple(self):
        """Minimize x + y subject to x >= 1, y >= 2."""
        result = lp_minimize(
            c=[1, 1],
            A_ub=[[-1, 0], [0, -1]],
            b_ub=[-1, -2],
        )
        assert result.success
        assert abs(result.x[0] - 1.0) < 1e-6
        assert abs(result.x[1] - 2.0) < 1e-6
        assert abs(result.objective_value - 3.0) < 1e-6

    def test_lp_maximize_simple(self):
        """Maximize 2x + 3y subject to x + y <= 10, x >= 0, y >= 0."""
        result = lp_maximize(
            c=[2, 3],
            A_ub=[[1, 1]],
            b_ub=[10],
            bounds=[(0, None), (0, None)],
        )
        assert result.success
        # Optimal: x=0, y=10, objective=30
        assert abs(result.objective_value - 30.0) < 1e-6

    def test_lp_equality(self):
        """Minimize x + y subject to x + y = 5, x >= 0, y >= 0."""
        result = lp_minimize(
            c=[1, 1],
            A_eq=[[1, 1]],
            b_eq=[5],
            bounds=[(0, None), (0, None)],
        )
        assert result.success
        assert abs(result.objective_value - 5.0) < 1e-6

    def test_lp_bounds(self):
        """Minimize x subject to 2 <= x <= 5."""
        result = lp_minimize(
            c=[1],
            bounds=[(2, 5)],
        )
        assert result.success
        assert abs(result.x[0] - 2.0) < 1e-6

    def test_lp_infeasible(self):
        """Infeasible problem should return failure."""
        result = lp_minimize(
            c=[1],
            A_ub=[[-1]],
            b_ub=[-10],  # x >= 10
            A_eq=[[1]],
            b_eq=[5],    # x = 5
        )
        assert not result.success

    def test_lp_to_dict(self):
        """LPResult should serialize to dict."""
        result = lp_minimize(c=[1], bounds=[(0, 10)])
        d = result.to_dict()
        assert "x" in d
        assert "objective_value" in d
        assert "success" in d


# ── Calibration Tests ──────────────────────────────────────────

class TestCalibration:
    def test_calibrate_simple(self):
        """Calibrate a simple model to match data."""
        m = parse_sysd("""
model 'Calibrate'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
    - 'Out': Rate * S
""")
        # Generate "observed" data from known parameters
        ref_result = m.simulate(params={"Rate": 0.05})
        data = {
            "S": [(t, v) for t, v in zip(ref_result["times"], ref_result["values"]["S"])],
        }

        cal_result = calibrate(
            m,
            data=data,
            param_bounds={"Rate": (0.01, 0.2)},
            method="nelder-mead",
            max_iterations=1000,
        )
        # Error should be very small (near-perfect fit)
        assert cal_result.best_error < 0.1
        # Rate should be in valid range
        rate = cal_result.best_params.get("Rate", 0)
        assert 0.01 <= rate <= 0.2

    def test_calibrate_sse(self):
        """Calibration with SSE objective should minimize squared errors."""
        m = parse_sysd("""
model 'SSE'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
    - 'Out': K * S
""")
        data = {"S": [(0, 100), (5, 80), (10, 60)]}
        result = calibrate(
            m, data=data, param_bounds={"K": (0.01, 0.5)},
            objective="sse", max_iterations=500,
        )
        assert result.method == "nelder-mead"
        assert result.best_error >= 0

    def test_calibrate_to_dict(self):
        """CalibrationResult should serialize to dict."""
        m = parse_sysd("""
model 'Dict'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': Rate
""")
        data = {"S": [(0, 100), (10, 200)]}
        result = calibrate(m, data=data, param_bounds={"Rate": (1, 20)}, max_iterations=50)
        d = result.to_dict()
        assert "best_params" in d
        assert "best_error" in d


# ── Pareto Multi-Objective Tests ──────────────────────────────────

class TestPareto:
    def test_pareto_simple(self):
        """Pareto optimization returns non-dominated solutions."""
        def obj1(p):
            return (p["x"] - 2) ** 2
        def obj2(p):
            return (p["x"] + 1) ** 2

        result = pareto_optimize(
            objective_fns=[obj1, obj2],
            objective_names=["Near 2", "Near -1"],
            param_bounds={"x": (-5, 5)},
            population_size=20,
            generations=10,
            seed=42,
        )
        assert len(result.solutions) == 20
        assert result.objective_names == ["Near 2", "Near -1"]
        assert result.generations == 10
        # Should have non-dominated solutions
        nondom = [s for s in result.solutions if s["rank"] == 0]
        assert len(nondom) > 0
        # Objectives should cover the range between the two targets
        objs = [s["objectives"][0] for s in nondom]
        assert min(objs) < 1  # Some near x=2
        objs2 = [s["objectives"][1] for s in nondom]
        assert min(objs2) < 1  # Some near x=-1

    def test_pareto_with_model(self):
        """Pareto optimization works with SysdModel simulation."""
        m = parse_sysd("""
model 'ParetoModel'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': Rate
""")
        def obj_neg_final(p):
            r = m.simulate(params=p)
            return -r["values"]["S"][-1]  # Want negative final (maximize)

        def obj_cost(p):
            return p.get("Rate", 0)  # Minimize rate

        result = pareto_optimize(
            objective_fns=[obj_neg_final, obj_cost],
            objective_names=["-Final Stock", "Rate Cost"],
            param_bounds={"Rate": (1, 20)},
            population_size=20,
            generations=10,
            seed=42,
        )
        assert len(result.solutions) == 20
        nondom = [s for s in result.solutions if s["rank"] == 0]
        assert len(nondom) > 0
        # All solutions should have valid rates
        for s in result.solutions:
            assert 1 <= s["params"]["Rate"] <= 20

    def test_pareto_to_dict(self):
        """ParetoResult serializes to dict."""
        def obj1(p):
            return p["x"]
        def obj2(p):
            return -p["x"]
        result = pareto_optimize(
            objective_fns=[obj1, obj2],
            objective_names=["X", "Neg X"],
            param_bounds={"x": (-10, 10)},
            population_size=10,
            generations=5,
            seed=42,
        )
        d = result.to_dict()
        assert "objective_names" in d
        assert "solutions" in d
        assert len(d["solutions"]) == 10
        s = d["solutions"][0]
        assert "params" in s
        assert "objectives" in s
        assert "rank" in s
        assert "crowding" in s

    def test_pareto_requires_two_objectives(self):
        """Must have at least 2 objectives."""
        with pytest.raises(ValueError, match="at least 2 objectives"):
            pareto_optimize(
                objective_fns=[lambda p: 0],
                objective_names=["Single"],
                param_bounds={"x": (0, 1)},
                generations=1,
            )

    def test_pareto_mismatched_names(self):
        """objective_fns and objective_names must match length."""
        with pytest.raises(ValueError):
            pareto_optimize(
                objective_fns=[lambda p: 0, lambda p: 1],
                objective_names=["Only one"],
                param_bounds={"x": (0, 1)},
                generations=1,
            )


# ── Policy Optimization Tests ──────────────────────────────────

class TestOptimization:
    def test_optimize_simple(self):
        """Optimize a parameter to minimize model output."""
        m = parse_sysd("""
model 'Optimize'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
    - 'Out': Rate * S
""")
        def objective(params):
            result = m.simulate(params=params)
            return result["values"]["S"][-1]  # Minimize final stock value

        opt_result = optimize(
            m,
            objective_fn=objective,
            param_bounds={"Rate": (0.01, 1.0)},
            max_iterations=500,
        )
        # With high rate, stock should be lower than initial 100
        assert opt_result.best_objective < 100
        assert opt_result.best_params.get("Rate", 0) > 0.01

    def test_optimize_with_constraints(self):
        """Optimize with constraint checking."""
        m = parse_sysd("""
model 'Constrained'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
    - 'Out': Rate * S
""")
        def objective(params):
            r = max(0.01, params.get("Rate", 0.1))
            result = m.simulate(params={"Rate": r})
            return -result["values"]["S"][-1]

        constraints = [{"type": "ineq", "fun": lambda p: p.get("Rate", 0) - 0.05}]

        opt_result = optimize(
            m,
            objective_fn=objective,
            param_bounds={"Rate": (0.01, 0.5)},
            constraints=constraints,
            method="differential-evolution",
            max_iterations=100,
        )
        # Should produce valid result
        assert "Rate" in opt_result.best_params
        assert 0.01 <= opt_result.best_params["Rate"] <= 0.5

    def test_optimize_to_dict(self):
        """OptimizationResult should serialize to dict."""
        m = parse_sysd("""
model 'OptDict'
  dt 1
  from 0 to 10
  stock 'S': 100
    + 'In': 10
    - 'Out': Rate * S
""")
        result = optimize(
            m,
            objective_fn=lambda p: p.get("Rate", 0.1),
            param_bounds={"Rate": (0.01, 1.0)},
            max_iterations=50,
        )
        d = result.to_dict()
        assert "best_params" in d
        assert "best_objective" in d


