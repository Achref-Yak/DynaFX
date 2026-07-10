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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional


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
    from scipy.optimize import differential_evolution, minimize

    param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in param_names]

    def objective_fn(x: np.ndarray) -> float:
        params = {name: float(xi) for name, xi in zip(param_names, x, strict=False)}
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
        best_params = {name: float(x) for name, x in zip(param_names, result.x, strict=False)}
        best_error = float(result.fun)
        iterations = result.nit
    else:
        # Use scipy.optimize.minimize with Nelder-Mead or L-BFGS-B
        x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])

        if method == "least-squares":
            # Use least_squares for better convergence
            from scipy.optimize import least_squares

            def residual_fn(x: np.ndarray) -> np.ndarray:
                params = {name: float(xi) for name, xi in zip(param_names, x, strict=False)}
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
            best_params = {name: float(x) for name, x in zip(param_names, result.x, strict=False)}
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
            best_params = {name: float(x) for name, x in zip(param_names, result.x, strict=False)}
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
    from scipy.optimize import differential_evolution, minimize

    param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in param_names]

    def full_objective(x: np.ndarray, apply_penalties: bool = True) -> float:
        params = {name: float(xi) for name, xi in zip(param_names, x, strict=False)}
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
        best_params = {name: float(x) for name, x in zip(param_names, result.x, strict=False)}
        best_objective = float(result.fun)
        iterations = result.nit
    else:
        # Nelder-Mead doesn't respect bounds, so we clamp
        x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])

        def clamped_objective(x):
            x_clamped = np.array([
                max(lo, min(hi, xi))
                for xi, (lo, hi) in zip(x, bounds, strict=False)
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
            for xi, (lo, hi) in zip(result.x, bounds, strict=False)
        ])
        best_params = {name: float(x) for name, x in zip(param_names, x_final, strict=False)}
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


# ── Multi-Objective Pareto Optimization ────────────────────────────


@dataclass
class ParetoResult:
    """Result of multi-objective Pareto optimization.

    Contains a set of non-dominated solutions (Pareto frontier) plus
    the rest of the final population with rank/crowding information.
    """
    solutions: list[dict[str, Any]]
    objective_names: list[str]
    generations: int
    population_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_names": self.objective_names,
            "generations": self.generations,
            "population_size": self.population_size,
            "solutions": [
                {
                    "params": s["params"],
                    "objectives": [round(o, 4) for o in s["objectives"]],
                    "rank": s["rank"],
                    "crowding": round(s["crowding"], 6),
                }
                for s in self.solutions
            ],
        }


def _random_population(
    param_bounds: dict[str, tuple[float, float]],
    size: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Generate a random population of parameter sets within bounds."""
    pop: list[dict[str, float]] = []
    for _ in range(size):
        ind: dict[str, float] = {}
        for name, (lo, hi) in param_bounds.items():
            ind[name] = rng.uniform(lo, hi)
        pop.append(ind)
    return pop


def _non_dominated_sort(
    values: list[list[float]],
) -> list[list[int]]:
    """Fast non-dominated sort (NSGA-II).

    Returns fronts: list of lists of indices, front[0] = Pareto front.
    All objectives are MINIMIZED.
    """
    n = len(values)
    dominates = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vi, vj = values[i], values[j]
            better_any = False
            worse_any = False
            for a, b in zip(vi, vj, strict=False):
                if a < b:
                    better_any = True
                elif a > b:
                    worse_any = True
            if not worse_any and better_any:
                dominates[i][j] = True

    fronts: list[list[int]] = []
    remaining = set(range(n))
    while remaining:
        front: set[int] = set()
        for i in remaining:
            dominated = False
            for j in remaining:
                if i != j and dominates[j][i]:
                    dominated = True
                    break
            if not dominated:
                front.add(i)
        if not front:
            break
        fronts.append(sorted(front))
        remaining -= front
    return fronts


def _crowding_distance(
    values: list[list[float]],
    front_indices: list[int],
) -> dict[int, float]:
    """Compute crowding distance for individuals in a front.

    Returns: {original_index: crowding_distance}
    """
    n = len(front_indices)
    dists: dict[int, float] = {idx: 0.0 for idx in front_indices}
    if n <= 2:
        for idx in front_indices:
            dists[idx] = float("inf")
        return dists

    n_obj = len(values[0])
    for obj in range(n_obj):
        sorted_idx = sorted(front_indices, key=lambda i: values[i][obj])
        obj_min = values[sorted_idx[0]][obj]
        obj_max = values[sorted_idx[-1]][obj]
        span = obj_max - obj_min
        if span < 1e-10:
            continue
        dists[sorted_idx[0]] = float("inf")
        dists[sorted_idx[-1]] = float("inf")
        for k in range(1, n - 1):
            idx = sorted_idx[k]
            prev_val = values[sorted_idx[k - 1]][obj]
            next_val = values[sorted_idx[k + 1]][obj]
            dists[idx] += (next_val - prev_val) / span
    return dists


def _tournament_selection(
    rank_of: dict[int, int],
    crowding_of: dict[int, float],
    k: int,
    rng: np.random.Generator,
) -> list[int]:
    """Binary tournament selection. Prefer lower rank, then higher crowding."""
    indices = list(rank_of.keys())
    selected: list[int] = []
    for _ in range(k):
        i, j = rng.choice(indices, 2, replace=False)
        if rank_of[i] < rank_of[j] or (rank_of[i] == rank_of[j] and crowding_of.get(i, 0) > crowding_of.get(j, 0)):
            selected.append(i)
        else:
            selected.append(j)
    return selected


def _blx_crossover(
    p1: dict[str, float],
    p2: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> dict[str, float]:
    """Blend crossover (BLX-alpha)."""
    child: dict[str, float] = {}
    for name, (lo, hi) in bounds.items():
        y1, y2 = p1[name], p2[name]
        if y1 > y2:
            y1, y2 = y2, y1
        spread = max(y2 - y1, 0.001) * alpha
        c = rng.uniform(y1 - spread, y2 + spread)
        child[name] = max(lo, min(hi, c))
    return child


def _gaussian_mutation(
    ind: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    rng: np.random.Generator,
    prob: float = 0.2,
    sigma_scale: float = 0.1,
) -> dict[str, float]:
    """Gaussian mutation with per-param probability."""
    child = dict(ind)
    for name, (lo, hi) in bounds.items():
        if rng.random() < prob:
            delta = rng.normal(0, sigma_scale * (hi - lo))
            child[name] = max(lo, min(hi, child[name] + delta))
    return child


def pareto_optimize(
    objective_fns: list[Callable[[dict[str, float]], float]],
    objective_names: list[str],
    param_bounds: dict[str, tuple[float, float]],
    population_size: int = 30,
    generations: int = 15,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.2,
    seed: int = 42,
) -> ParetoResult:
    """Multi-objective Pareto optimization using NSGA-II-inspired algorithm.

    Finds the set of non-dominated (Pareto-optimal) parameter sets that
    represent tradeoffs across multiple objectives. All objectives are
    MINIMIZED.

    Args:
        objective_fns: List of functions mapping params -> scalar objective (minimized)
        objective_names: Labels for each objective (for display)
        param_bounds: {param_name: (min, max)} for each parameter
        population_size: Number of individuals per generation
        generations: Number of generations to evolve
        crossover_prob: Probability of crossover between selected parents
        mutation_prob: Per-parameter mutation probability
        seed: Random seed

    Returns:
        ParetoResult with ranked solutions and objective values
    """
    if len(objective_fns) != len(objective_names):
        raise ValueError("objective_fns and objective_names must have same length")
    if not objective_fns:
        raise ValueError("At least one objective function required")
    if len(objective_fns) < 2:
        raise ValueError("Pareto optimization requires at least 2 objectives (use optimize() for single-objective)")

    rng = np.random.default_rng(seed)

    # Initialize population
    population = _random_population(param_bounds, population_size, rng)

    n_objectives = len(objective_fns)

    for gen in range(generations):
        # Evaluate all objectives
        values: list[list[float]] = []
        for ind in population:
            obj_vals = [fn(ind) for fn in objective_fns]
            values.append([max(float("-inf"), min(float("inf"), v)) for v in obj_vals])

        # Non-dominated sort
        fronts = _non_dominated_sort(values)

        # Rank assignment and crowding distance
        rank_of: dict[int, int] = {}
        crowding_of: dict[int, float] = {}
        for r, front in enumerate(fronts):
            cd = _crowding_distance(values, front)
            for idx in front:
                rank_of[idx] = r
                crowding_of[idx] = cd.get(idx, 0.0)

        # Selection (tournament)
        selected_indices = _tournament_selection(rank_of, crowding_of, population_size, rng)

        # Generate offspring
        offspring: list[dict[str, float]] = []
        for i in range(0, len(selected_indices) - 1, 2):
            p1_idx = selected_indices[i]
            p2_idx = selected_indices[i + 1]
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            if rng.random() < crossover_prob:
                c1 = _blx_crossover(p1, p2, param_bounds, rng)
                c2 = _blx_crossover(p2, p1, param_bounds, rng)
            else:
                c1 = dict(p1)
                c2 = dict(p2)
            c1 = _gaussian_mutation(c1, param_bounds, rng, prob=mutation_prob)
            c2 = _gaussian_mutation(c2, param_bounds, rng, prob=mutation_prob)
            offspring.append(c1)
            offspring.append(c2)

        # Ensure we have exactly population_size offspring
        while len(offspring) < population_size:
            idx = rng.integers(0, population_size)
            c = _gaussian_mutation(
                dict(population[idx]), param_bounds, rng, prob=mutation_prob
            )
            offspring.append(c)
        offspring = offspring[:population_size]

        # Elitism: combine parent + offspring, sort by rank, keep top population_size
        combined = population + offspring
        combined_values = values + [
            [fn(ind) for fn in objective_fns] for ind in offspring
        ]
        combined_values = [
            [max(float("-inf"), min(float("inf"), v)) for v in vals]
            for vals in combined_values
        ]

        comb_fronts = _non_dominated_sort(combined_values)
        new_population: list[dict[str, float]] = []
        for front in comb_fronts:
            cd = _crowding_distance(combined_values, front)
            # Sort front by crowding descending
            front_sorted = sorted(front, key=lambda i: cd.get(i, 0), reverse=True)
            for idx in front_sorted:
                if len(new_population) >= population_size:
                    break
                new_population.append(combined[idx])
            if len(new_population) >= population_size:
                break
        population = new_population

    # Final evaluation
    final_values: list[list[float]] = []
    for ind in population:
        obj_vals = [fn(ind) for fn in objective_fns]
        final_values.append([max(float("-inf"), min(float("inf"), v)) for v in obj_vals])

    final_fronts = _non_dominated_sort(final_values)
    final_ranks: dict[int, int] = {}
    final_crowding: dict[int, float] = {}
    for r, front in enumerate(final_fronts):
        cd = _crowding_distance(final_values, front)
        for idx in front:
            final_ranks[idx] = r
            final_crowding[idx] = cd.get(idx, 0.0)

    solutions: list[dict[str, Any]] = []
    for i in range(population_size):
        solutions.append({
            "params": dict(population[i]),
            "objectives": final_values[i],
            "rank": final_ranks.get(i, 999),
            "crowding": final_crowding.get(i, 0.0),
        })

    return ParetoResult(
        solutions=solutions,
        objective_names=list(objective_names),
        generations=generations,
        population_size=population_size,
    )


# ── KB-Constrained Optimization ──────────────────────────────────


def _float_or_inf(val: Any) -> float:
    try:
        f = float(val)
        if f is None or (isinstance(val, str) and val.strip() == ""):
            return float("inf")
        return f
    except (ValueError, TypeError):
        return float("inf")


def kb_lp_minimize(
    store: Any,
    c_query: str,
    bounds_query: str,
    A_ub_query: Optional[str] = None,
    b_ub_query: Optional[str] = None,
    A_eq_query: Optional[str] = None,
    b_eq_query: Optional[str] = None,
        var_name: str = "v",
        var_count: Optional[int] = None,
) -> LPResult:
    """Solve LP with objective and constraints read from SPARQL queries.

    Args:
        store: TripleStore to query.
        c_query: SPARQL SELECT returning ?v values for objective coeffs.
        bounds_query: SPARQL SELECT returning ?v (lower), ?v2 (upper),
            ?name for each variable's bounds. If ?name is absent, uses
            positional ordering. Use -inf/+inf for unbounded.
        A_ub_query: Optional SPARQL for inequality constraint matrix rows.
            Returns ?v, ?v2, ?v3,... for each row.
        b_ub_query: Optional SPARQL for inequality RHS values.
        A_eq_query: Optional SPARQL for equality constraint matrix rows.
        b_eq_query: Optional SPARQL for equality RHS values.
        var_name: Variable name in SPARQL results for coefficients.
        var_count: Number of decision variables. If None, inferred from
            c_query result count.

    Returns:
        LPResult.
    """
    from dynafx.knowledge import parse_sparql, sparql_evaluate

    def _eval_q(q: str) -> list[list[float]]:
        ast = parse_sparql(q)
        qr = sparql_evaluate(ast, store)
        rows: list[list[float]] = []
        for binding in qr.bindings:
            vals = []
            for k, v in binding.items():
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(0.0)
            if vals:
                rows.append(vals)
        return rows

    # Objective coefficients
    c_rows = _eval_q(c_query)
    if not c_rows:
        return LPResult(x=[], objective_value=float("inf"), success=False, message="Empty c_query results")
    n_vars = var_count or len(c_rows)
    c = [c_rows[i][0] if i < len(c_rows) else 0.0 for i in range(n_vars)]

    # Bounds
    bounds_rows = _eval_q(bounds_query)
    bounds: list[tuple[Optional[float], Optional[float]]] = [(None, None)] * n_vars
    for i, row in enumerate(bounds_rows):
        if i < n_vars:
            lo = _float_or_inf(row[0]) if len(row) > 0 else None
            hi = _float_or_inf(row[1]) if len(row) > 1 else None
            if lo == float("inf"):
                lo = None
            if hi == float("inf"):
                hi = None
            bounds[i] = (lo, hi)

    # Constraints
    A_ub, b_ub, A_eq, b_eq = None, None, None, None

    if A_ub_query and b_ub_query:
        A_rows = _eval_q(A_ub_query)
        b_rows = _eval_q(b_ub_query)
        if A_rows and b_rows:
            m = min(len(A_rows), len(b_rows))
            A_ub = [A_rows[i][:n_vars] for i in range(m)]
            b_ub = [b_rows[i][0] for i in range(m)]

    if A_eq_query and b_eq_query:
        A_rows = _eval_q(A_eq_query)
        b_rows = _eval_q(b_eq_query)
        if A_rows and b_rows:
            m = min(len(A_rows), len(b_rows))
            A_eq = [A_rows[i][:n_vars] for i in range(m)]
            b_eq = [b_rows[i][0] for i in range(m)]

    return lp_minimize(c, A_ub, b_ub, A_eq, b_eq, bounds)


def kb_lp_maximize(
    store: Any,
    c_query: str,
    bounds_query: str,
    **kwargs: Any,
) -> LPResult:
    """Maximize with objective coeffs from SPARQL.

    Delegates to kb_lp_minimize with negated coefficients.
    """
    result = kb_lp_minimize(store, c_query, bounds_query, **kwargs)
    if result.success:
        result.objective_value = -result.objective_value
    return result


def kb_calibrate(
    model,
    store: Any,
    data_query: str,
    param_bounds_query: str,
    var_name: str = "v",
    **cal_kwargs: Any,
) -> CalibrationResult:
    """Calibrate model with parameter bounds from SPARQL.

    Args:
        model: SysdModel to calibrate.
        store: TripleStore with bounds data.
        data_query: SPARQL SELECT returning ?time, ?value, ?variable
            for observed data points.
        param_bounds_query: SPARQL SELECT returning ?name, ?lo, ?hi
            for parameter bounds.
        var_name: Variable name for data values.
        **cal_kwargs: Additional kwargs passed to calibrate().

    Returns:
        CalibrationResult.
    """
    from dynafx.knowledge import parse_sparql, sparql_evaluate

    def _extract_str(b: Any, key: str, default: str = "") -> str:
        val = b.get(key, default)
        if hasattr(val, "value"):
            return str(val.value)
        return str(val)

    def _extract_float(b: Any, key: str, default: float = 0.0) -> float:
        val = b.get(key, default)
        if hasattr(val, "value"):
            return float(val.value)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Extract param bounds
    ast = parse_sparql(param_bounds_query)
    qr = sparql_evaluate(ast, store)
    param_bounds: dict[str, tuple[float, float]] = {}
    for binding in qr.bindings:
        name = _extract_str(binding, "name", "")
        lo = _extract_float(binding, "lo", 0)
        hi = _extract_float(binding, "hi", 1)
        if name:
            param_bounds[name] = (lo, hi)

    # Extract data
    ast = parse_sparql(data_query)
    qr = sparql_evaluate(ast, store)
    data: dict[str, list[tuple[float, float]]] = {}
    for binding in qr.bindings:
        variable = _extract_str(binding, "variable", "")
        t = _extract_float(binding, "time", 0)
        v = _extract_float(binding, var_name, 0)
        if variable:
            if variable not in data:
                data[variable] = []
            data[variable].append((t, v))

    return calibrate(model, data, param_bounds, **cal_kwargs)


def kb_optimize(
    model,
    objective_fn: Callable[[dict[str, float]], float],
    store: Any,
    param_bounds_query: str,
    constraints_query: Optional[str] = None,
    var_name: str = "v",
    **opt_kwargs: Any,
) -> OptimizationResult:
    """Optimize with parameter bounds and constraints from SPARQL.

    Args:
        model: SysdModel to optimize.
        objective_fn: Objective function params → scalar.
        store: TripleStore with bounds/constraints.
        param_bounds_query: SPARQL SELECT returning ?name, ?lo, ?hi.
        constraints_query: Optional SPARQL SELECT returning ?type
            ("ineq"/"eq"), ?expr_name, ?expr.
        var_name: Variable name in query results.
        **opt_kwargs: Additional kwargs for optimize().

    Returns:
        OptimizationResult.
    """
    from dynafx.knowledge import parse_sparql, sparql_evaluate

    def _extract_str(b: Any, key: str, default: str = "") -> str:
        val = b.get(key, default)
        if hasattr(val, "value"):
            return str(val.value)
        return str(val)

    def _extract_float(b: Any, key: str, default: float = 0.0) -> float:
        val = b.get(key, default)
        if hasattr(val, "value"):
            return float(val.value)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Param bounds
    ast = parse_sparql(param_bounds_query)
    qr = sparql_evaluate(ast, store)
    param_bounds: dict[str, tuple[float, float]] = {}
    for binding in qr.bindings:
        name = _extract_str(binding, "name", "")
        lo = _extract_float(binding, "lo", 0)
        hi = _extract_float(binding, "hi", 1)
        if name:
            param_bounds[name] = (lo, hi)

    # Constraints
    constraints: list[dict[str, Any]] = []
    if constraints_query:
        ast = parse_sparql(constraints_query)
        qr = sparql_evaluate(ast, store)
        for binding in qr.bindings:
            ctype = _extract_str(binding, "type", "ineq")
            raw_expr = binding.get("expr", "0")
            expr_str = str(raw_expr.value) if hasattr(raw_expr, "value") else str(raw_expr)
            constraints.append({
                "type": ctype,
                "fun": lambda params, _es=expr_str: _safe_eval_constraint(_es, params),
            })

    return optimize(model, objective_fn, param_bounds, constraints=constraints or None, **opt_kwargs)


def _safe_eval_constraint(expr: str, params: dict[str, float]) -> float:
    """Safely evaluate a numeric expression with param substitution."""
    import math
    _safe_ns = {"__builtins__": {}, "abs": abs, "min": min, "max": max,
                "math": math}
    _safe_ns.update(params)
    try:
        return float(eval(expr, _safe_ns))
    except Exception:
        return 0.0
