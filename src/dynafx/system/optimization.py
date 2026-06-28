"""Optimization module for system dynamics models.

Provides linear programming, parameter calibration, and policy optimization.

Linear Programming:
- lp_minimize / lp_maximize: wrapper around scipy.optimize.linprog

Model Calibration:
- calibrate: fit model parameters to observed data

Policy Optimization:
- optimize: find optimal parameter values subject to constraints

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class LPResult:
    """Result of a linear programming solve."""
    x: list[float]
    objective_value: float
    success: bool
    message: str = ""
    shadow_prices: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "objective_value": self.objective_value,
            "success": self.success,
            "message": self.message,
            "shadow_prices": self.shadow_prices,
        }


@dataclass
class CalibrationResult:
    """Result of model calibration."""
    best_params: dict[str, float]
    best_error: float
    iterations: int
    method: str
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_error": self.best_error,
            "iterations": self.iterations,
            "method": self.method,
        }


@dataclass
class OptimizationResult:
    """Result of policy optimization."""
    best_params: dict[str, float]
    best_objective: float
    constraints_satisfied: bool
    iterations: int
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_objective": self.best_objective,
            "constraints_satisfied": self.constraints_satisfied,
            "iterations": self.iterations,
            "method": self.method,
        }


def lp_minimize(
    c: list[float],
    A_ub: Optional[list[list[float]]] = None,
    b_ub: Optional[list[float]] = None,
    A_eq: Optional[list[list[float]]] = None,
    b_eq: Optional[list[float]] = None,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
) -> LPResult:
    """Minimize c^T x subject to constraints.

    Wrapper around scipy.optimize.linprog.

    Args:
        c: Coefficients of objective function (minimize c^T x)
        A_ub: Inequality constraint matrix (A_ub @ x <= b_ub)
        b_ub: Inequality constraint vector
        A_eq: Equality constraint matrix (A_eq @ x == b_eq)
        b_eq: Equality constraint vector
        bounds: Variable bounds [(low, high), ...] for each variable

    Returns:
        LPResult with optimal x, objective value, and status
    """
    from scipy.optimize import linprog

    c_arr = np.array(c, dtype=float)
    result = linprog(
        c_arr,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    return LPResult(
        x=result.x.tolist() if result.x is not None else [],
        objective_value=result.fun if result.fun is not None else float("inf"),
        success=result.success,
        message=result.message if hasattr(result, "message") else "",
    )


def lp_maximize(
    c: list[float],
    A_ub: Optional[list[list[float]]] = None,
    b_ub: Optional[list[float]] = None,
    A_eq: Optional[list[list[float]]] = None,
    b_eq: Optional[list[float]] = None,
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
) -> LPResult:
    """Maximize c^T x subject to constraints.

    Equivalent to minimizing -c^T x.
    """
    neg_c = [-ci for ci in c]
    result = lp_minimize(neg_c, A_ub, b_ub, A_eq, b_eq, bounds)
    if result.success:
        result.objective_value = -result.objective_value
    return result


def _simulate_model(model, params: dict[str, float], variable: str) -> list[float]:
    """Run model simulation and return time series for a variable."""
    result = model.simulate(params=params)
    if variable in result["values"]:
        return result["values"][variable]
    # Try stocks list
    if variable in result["stocks"]:
        idx = result["stocks"].index(variable)
        return [row[idx] for row in result["values"].values()]
    return []


def calibrate(
    model,
    data: dict[str, list[tuple[float, float]]],
    param_bounds: dict[str, tuple[float, float]],
    objective: str = "sse",
    method: str = "nelder-mead",
    max_iterations: int = 1000,
    seed: int = 42,
) -> CalibrationResult:
    """Fit model parameters to observed data.

    Args:
        model: SysdModel to calibrate
        data: {variable_name: [(time, value), ...]} observed data points
        param_bounds: {param_name: (min, max)} bounds for each parameter
        objective: "sse" (sum squared error), "mae" (mean absolute error), or "max_error"
        method: "nelder-mead", "least-squares", or "differential-evolution"
        max_iterations: Maximum number of iterations
        seed: Random seed for reproducibility

    Returns:
        CalibrationResult with best parameters and error
    """
    from scipy.optimize import minimize, differential_evolution

    param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in param_names]

    def objective_fn(x: np.ndarray) -> float:
        params = {name: float(xi) for name, xi in zip(param_names, x)}
        total_error = 0.0

        for variable, observations in data.items():
            times = [t for t, _ in observations]
            values = [v for _, v in observations]

            # Run simulation
            result = model.simulate(params=params)
            sim_times = result["times"]
            sim_values = result["values"].get(variable, [])

            if not sim_values:
                total_error += 1e10
                continue

            # Interpolate simulation to observation times
            for obs_t, obs_v in observations:
                # Find closest simulation time
                sim_v = 0.0
                for i, st in enumerate(sim_times):
                    if abs(st - obs_t) < 1e-10:
                        sim_v = sim_values[i]
                        break
                    if i > 0 and sim_times[i - 1] <= obs_t <= st:
                        # Linear interpolation
                        alpha = (obs_t - sim_times[i - 1]) / (st - sim_times[i - 1])
                        sim_v = sim_values[i - 1] + alpha * (sim_values[i] - sim_values[i - 1])
                        break

                if objective == "sse":
                    total_error += (sim_v - obs_v) ** 2
                elif objective == "mae":
                    total_error += abs(sim_v - obs_v)
                elif objective == "max_error":
                    total_error = max(total_error, abs(sim_v - obs_v))

        return total_error

    best_params = {}
    best_error = float("inf")
    iterations = 0
    history = []

    if method == "differential-evolution":
        result = differential_evolution(
            objective_fn,
            bounds,
            maxiter=max_iterations,
            seed=seed,
            tol=1e-8,
        )
        best_params = {name: float(x) for name, x in zip(param_names, result.x)}
        best_error = float(result.fun)
        iterations = result.nit
    else:
        # Use scipy.optimize.minimize with Nelder-Mead or L-BFGS-B
        x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])

        if method == "least-squares":
            # Use least_squares for better convergence
            from scipy.optimize import least_squares

            def residual_fn(x: np.ndarray) -> np.ndarray:
                params = {name: float(xi) for name, xi in zip(param_names, x)}
                residuals = []
                for variable, observations in data.items():
                    result = model.simulate(params=params)
                    sim_times = result["times"]
                    sim_values = result["values"].get(variable, [])
                    for obs_t, obs_v in observations:
                        sim_v = 0.0
                        for i, st in enumerate(sim_times):
                            if abs(st - obs_t) < 1e-10:
                                sim_v = sim_values[i]
                                break
                            if i > 0 and sim_times[i - 1] <= obs_t <= st:
                                alpha = (obs_t - sim_times[i - 1]) / (st - sim_times[i - 1])
                                sim_v = sim_values[i - 1] + alpha * (sim_values[i] - sim_values[i - 1])
                                break
                        residuals.append(sim_v - obs_v)
                return np.array(residuals)

            ls_bounds = ([lo for lo, _ in bounds], [hi for _, hi in bounds])
            result = least_squares(residual_fn, x0, bounds=ls_bounds, max_nfev=max_iterations)
            best_params = {name: float(x) for name, x in zip(param_names, result.x)}
            best_error = float(np.sum(result.fun ** 2))
            iterations = result.nfev
        else:
            # Nelder-Mead
            scipy_bounds = bounds if method != "nelder-mead" else None
            result = minimize(
                objective_fn,
                x0,
                method="Nelder-Mead",
                bounds=scipy_bounds if method != "nelder-mead" else None,
                options={"maxiter": max_iterations, "xatol": 1e-8, "fatol": 1e-8},
            )
            best_params = {name: float(x) for name, x in zip(param_names, result.x)}
            best_error = float(result.fun)
            iterations = result.nit

    return CalibrationResult(
        best_params=best_params,
        best_error=best_error,
        iterations=iterations,
        method=method,
        history=history,
    )


def optimize(
    model,
    objective_fn: Callable[[dict[str, float]], float],
    param_bounds: dict[str, tuple[float, float]],
    constraints: Optional[list[dict[str, Any]]] = None,
    method: str = "nelder-mead",
    max_iterations: int = 1000,
    seed: int = 42,
) -> OptimizationResult:
    """Find optimal parameter values subject to constraints.

    Args:
        model: SysdModel to optimize
        objective_fn: Function mapping params dict to scalar objective (minimize)
        param_bounds: {param_name: (min, max)} bounds for each parameter
        constraints: List of constraint dicts:
            {"type": "ineq", "fun": callable} or
            {"type": "eq", "fun": callable}
        method: Optimization method
        max_iterations: Maximum iterations
        seed: Random seed

    Returns:
        OptimizationResult with best parameters and objective value
    """
    from scipy.optimize import minimize, differential_evolution

    param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in param_names]

    def full_objective(x: np.ndarray, apply_penalties: bool = True) -> float:
        params = {name: float(xi) for name, xi in zip(param_names, x)}
        obj = objective_fn(params)
        # Add penalty for constraint violations
        if apply_penalties and constraints:
            penalty = 0.0
            for c in constraints:
                if "fun" in c:
                    val = c["fun"](params)
                    if c.get("type") == "ineq" and val < 0:
                        penalty += 10000.0 * abs(val)
                    elif c.get("type") == "eq" and abs(val) > 1e-6:
                        penalty += 10000.0 * val ** 2
            obj += penalty
        return obj

    best_params = {}
    best_objective = float("inf")
    iterations = 0

    if method == "differential-evolution":
        result = differential_evolution(
            full_objective,
            bounds,
            maxiter=max_iterations,
            seed=seed,
            tol=1e-8,
        )
        best_params = {name: float(x) for name, x in zip(param_names, result.x)}
        best_objective = float(result.fun)
        iterations = result.nit
    else:
        # Nelder-Mead doesn't respect bounds, so we clamp
        x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])

        def clamped_objective(x):
            x_clamped = np.array([
                max(lo, min(hi, xi))
                for xi, (lo, hi) in zip(x, bounds)
            ])
            return full_objective(x_clamped)

        result = minimize(
            clamped_objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": max_iterations, "xatol": 1e-8, "fatol": 1e-8},
        )
        # Clamp final result to bounds
        x_final = np.array([
            max(lo, min(hi, xi))
            for xi, (lo, hi) in zip(result.x, bounds)
        ])
        best_params = {name: float(x) for name, x in zip(param_names, x_final)}
        best_objective = float(result.fun)
        iterations = result.nit

    # Check constraints
    constraints_satisfied = True
    if constraints:
        test_params = best_params.copy()
        for c in constraints:
            if "fun" in c:
                val = c["fun"](test_params)
                if c.get("type") == "ineq" and val < 0:
                    constraints_satisfied = False
                elif c.get("type") == "eq" and abs(val) > 1e-6:
                    constraints_satisfied = False

    return OptimizationResult(
        best_params=best_params,
        best_objective=best_objective,
        constraints_satisfied=constraints_satisfied,
        iterations=iterations,
        method=method,
    )
