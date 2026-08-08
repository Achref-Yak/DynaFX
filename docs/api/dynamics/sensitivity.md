# Sensitivity Analysis

**What it is:** Find which parameters matter most — Sobol, Morris, PRCC, OAT, and SRC methods.

**When to use it:** When you have many uncertain parameters and want to know which ones drive the output.

**Quick start:**
```python
from dynafx.dynamics import SensitivityAnalyzer

sa = SensitivityAnalyzer(model)
result = sa.sobol(param_spec={"demand": (50, 200)}, output="Inventory")
print(result.ranking("total_order"))
```

---

## Import

```python
from dynafx.dynamics import SensitivityAnalyzer, SensitivityResult
```

## Classes

### SensitivityAnalyzer

```python
SensitivityAnalyzer(model: SysdModel)
```

#### Methods

```python
sa.sobol(param_spec, output, n_base=512)
sa.morris(param_spec, output, n_trajectories=10)
sa.prcc(param_spec, output, n_samples=256)
sa.oat(param_spec, output, n_steps=10)
sa.src(param_spec, output, n_samples=256)
```

### SensitivityResult

```python
SensitivityResult(
    method: str,
    param_names: list[str],
    output: str,
    n_samples: int,
    first_order: dict[str, float] = None,
    total_order: dict[str, float] = None,
    mu_star: dict[str, float] = None,
    sigma: dict[str, float] = None,
    prcc: dict[str, float] = None,
    src: dict[str, float] = None,
)
```

#### Methods

```python
result.ranking(index_type)  # Rank parameters by sensitivity
result.plot(path)           # Plot sensitivity indices
```

---

## Common Patterns

### Pattern 1: Sobol Analysis

```python
sa = SensitivityAnalyzer(model)
result = sa.sobol(
    param_spec={"demand": (50, 200), "capacity": (100, 500)},
    output="Inventory",
    n_base=512,
)
print(result.ranking("total_order"))
```

### Pattern 2: Morris Screening

```python
result = sa.morris(
    param_spec={"demand": (50, 200)},
    output="Inventory",
)
print(result.mu_star)
```

---

## See Also

- [`SysdModel`](SysdModel.md) — The model to analyze
- [`ScenarioComparison`](scenario.md) — Compare specific scenarios
- [Tutorial 8: Scenarios & Sensitivity](../../tutorials/08-scenarios-and-sensitivity.md) — Full walkthrough
