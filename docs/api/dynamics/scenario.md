# Scenario Comparison

**What it is:** Compare multiple simulation scenarios — see how different parameters affect outcomes.

**When to use it:** When testing "what if" questions — e.g., "what if demand increases by 50%?"

**Quick start:**
```python
from dynafx.dynamics import ScenarioComparison, ScenarioDef

comp = ScenarioComparison(model, [
    ScenarioDef("Baseline", {}),
    ScenarioDef("High demand", {"demand": 150}),
    ScenarioDef("Low capacity", {"capacity": 50}),
])
comp.plot_comparison("comparison.png", stocks=["Inventory"])
```

---

## Import

```python
from dynafx.dynamics import ScenarioComparison, ScenarioDef, ScenarioResult
```

## Classes

### ScenarioDef

Definition of a scenario to run.

```python
ScenarioDef(
    name: str,                  # Scenario name
    params: dict[str, Any],     # Parameter overrides
)
```

### ScenarioResult

A single scenario run result.

```python
ScenarioResult(
    name: str,                  # Scenario name
    result: SysdModelResult,    # Simulation result
    params: dict[str, Any],     # Parameters used
)
```

### ScenarioCompare

Compare multiple scenarios.

```python
ScenarioComparison(
    model: SysdModel,
    scenarios: list[ScenarioDef],
    method: str = "rk4",
    **sim_kwargs,
)
```

#### Methods

```python
comp.plot_comparison(path, stocks=None, title=None)
comp.plot_deviation(path, stocks=None, baseline="Baseline")
comp.tornado(path, param_ranges, output_stock)
comp.summary()
comp.get(name)  # Get specific scenario result
comp.names      # List scenario names
comp.times      # Time points
```

---

## Common Patterns

### Pattern 1: Basic Comparison

```python
comp = ScenarioComparison(model, [
    ScenarioDef("Baseline", {}),
    ScenarioDef("Optimistic", {"growth_rate": 0.05}),
    ScenarioDef("Pessimistic", {"growth_rate": 0.01}),
])
comp.plot_comparison("scenarios.png", stocks=["Population"])
```

### Pattern 2: Tornado Diagram

```python
comp.tornado("tornado.png", {
    "growth_rate": (0.01, 0.05),
    "capacity": (500, 2000),
}, output_stock="Population")
```

---

## See Also

- [`SysdModel`](SysdModel.md) — The model to compare
- [`SensitivityAnalyzer`](sensitivity.md) — Variance-based analysis
- [Tutorial 8: Scenarios & Sensitivity](../../tutorials/08-scenarios-and-sensitivity.md) — Full walkthrough
