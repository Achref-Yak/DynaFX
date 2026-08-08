# Optimization

**What it is:** Linear programming, parameter calibration, and policy optimization.

**When to use it:** When you want to find optimal parameter values or fit model to data.

**Quick start:**
```python
from dynafx.dynamics import lp_minimize, calibrate, optimize

# Linear programming
result = lp_minimize(c=[1, 2], A_ub=[[1, 1]], b_ub=[10])
print(result.x, result.objective_value)

# Calibration
result = calibrate(model, observed_data, params=["growth_rate"])
print(result.best_params)
```

---

## Import

```python
from dynafx.dynamics import (
    lp_minimize, lp_maximize, calibrate, optimize, pareto_optimize,
    kb_lp_minimize, kb_lp_maximize, kb_calibrate, kb_optimize,
    LPResult, CalibrationResult, OptimizationResult, ParetoResult,
)
```

## Linear Programming

### `lp_minimize(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None)`

Minimize `c^T x` subject to constraints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c` | `list[float]` | — | Objective coefficients (required) |
| `A_ub` | `list[list[float]] \| None` | `None` | Inequality constraint matrix |
| `b_ub` | `list[float] \| None` | `None` | Inequality constraint vector |
| `A_eq` | `list[list[float]] \| None` | `None` | Equality constraint matrix |
| `b_eq` | `list[float] \| None` | `None` | Equality constraint vector |
| `bounds` | `list[tuple] \| None` | `None` | Variable bounds |

**Returns:** `LPResult`

---

### `lp_maximize(c, ...)`

Maximize `c^T x` subject to constraints. Same signature as `lp_minimize`.

---

## Calibration

### `calibrate(model, observed, params, method="nelder", max_iter=100)`

Fit model parameters to observed data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `SysdModel` | — | The model (required) |
| `observed` | `dict[str, list]` | — | Observed data `{time: [values]}` |
| `params` | `list[str]` | — | Parameters to calibrate |
| `method` | `str` | `"nelder"` | Optimization method |
| `max_iter` | `int` | `100` | Max iterations |

**Returns:** `CalibrationResult`

---

## Policy Optimization

### `optimize(model, objective, params, bounds, constraints=None, method="nelder")`

Find optimal parameter values subject to constraints.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `SysdModel` | — | The model (required) |
| `objective` | `Callable` | — | Objective function |
| `params` | `dict[str, tuple]` | — | Parameter bounds `{name: (low, high)}` |
| `constraints` | `list[dict] \| None` | `None` | Constraints |
| `method` | `str` | `"nelder"` | Optimization method |

**Returns:** `OptimizationResult`

---

## Multi-Objective

### `pareto_optimize(model, objectives, params, bounds)`

Multi-objective Pareto search.

**Returns:** `ParetoResult`

---

## KB-Constrained Versions

All functions have `kb_` prefixed versions that read coefficients from SPARQL:

```python
from dynafx.dynamics import kb_lp_minimize, kb_calibrate, kb_optimize

result = kb_lp_minimize(store, claim_map=[...], ...)
result = kb_calibrate(model, observed, params, store=store)
result = kb_optimize(model, objective, params, store=store)
```

---

## Result Types

### LPResult

```python
LPResult(
    x: list[float],             # Optimal values
    objective_value: float,      # Objective value
    success: bool,               # Whether solved
    message: str = "",           # Status message
    shadow_prices: list[float],  # Shadow prices
)
```

### CalibrationResult

```python
CalibrationResult(
    best_params: dict[str, float],
    best_error: float,
    iterations: int,
    method: str,
)
```

### OptimizationResult

```python
OptimizationResult(
    best_params: dict[str, float],
    best_objective: float,
    constraints_satisfied: bool,
    iterations: int,
    method: str,
)
```

---

## See Also

- [`SysdModel`](SysdModel.md) — The model to optimize
- [Tutorial 10: Publishing Results](../../tutorials/10-publishing-results.md) — Optimization deep dive
