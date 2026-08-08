# SysdModelResult

**What it is:** The output of a simulation — holds all the trajectory data (how each stock changed over time) and methods to plot and export results.

**When to use it:** After calling `model.simulate()`, you get a `SysdModelResult` back. Use it to access values, create plots, or export to CSV.

**Quick start:**
```python
result = model.simulate()

# Access stock values
print(result.values["Inventory"][-1])  # final value
print(result.times)                    # time points

# Plot
result.plot("output.png", stocks=["Inventory"])
```

---

## Import

```python
from dynafx.dynamics import SysdModelResult
```

You don't typically import this directly — it's returned by `SysdModel.simulate()`.

---

## Attributes

### `times: list[float]`

The time axis — a list of time points from `t_span[0]` to `t_span[1]`.

**Example:**
```python
result = model.simulate()
print(result.times)        # [0.0, 1.0, 2.0, ..., 100.0]
print(len(result.times))   # 101 (for dt=1.0, t_span=(0, 100))
```

---

### `stocks: list[str]`

Names of all stocks in the model.

**Example:**
```python
print(result.stocks)  # ["Inventory", "Cash", "Population"]
```

---

### `values: dict[str, list[float]]`

The main results — a dictionary mapping stock names to their trajectories.

**Example:**
```python
# Get the full trajectory for a stock
inventory_trajectory = result.values["Inventory"]
print(inventory_trajectory)  # [1000, 1050, 1100, ..., 2500]

# Get the final value
final_inventory = result.values["Inventory"][-1]

# Get the initial value
initial_inventory = result.values["Inventory"][0]

# Get the value at a specific time step
value_at_step_10 = result.values["Inventory"][10]
```

---

### `aux_values: dict[str, list[float]]`

Trajectories for auxiliary variables (if computed).

**Example:**
```python
print(result.aux_values["reliability"])  # [0.85, 0.85, ..., 0.85]
```

---

### `method: str`

The integration method used (`"rk4"` or `"euler"`).

---

### `steps: int`

Number of simulation steps taken.

---

### `model_name: str`

Name of the model that produced this result.

---

### `abm_engine: ABMEngine | None`

The ABM engine if agents were defined, otherwise `None`.

---

### `des_engine: DESEngine | None`

The DES engine if queues/resources were defined, otherwise `None`.

---

## Methods

### `plot(path, stocks=None, subplots=False, title=None)`

Save a plot of stock trajectories to a file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Output file path (required) |
| `stocks` | `list[str] \| None` | `None` | Stocks to plot (default: all) |
| `subplots` | `bool` | `False` | If `True`, each stock gets its own subplot |
| `title` | `str \| None` | `None` | Plot title (default: model name) |

**Returns:** `None` (saves file to disk)

**Raises:**
- `ImportError` — If matplotlib is not installed

**Example:**
```python
# Plot all stocks on one chart
result.plot("output.png")

# Plot specific stocks
result.plot("inventory.png", stocks=["Inventory"])

# Plot with subplots (each stock separate)
result.plot("all_stocks.png", subplots=True)

# Custom title
result.plot("results.png", title="Scenario A Results")
```

---

### `plot_with_bands(path, mean, std, p5, p95)`

Plot trajectories with confidence bands (for sensitivity analysis).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Output file path (required) |
| `mean` | `dict[str, list[float]]` | — | Mean trajectories |
| `std` | `dict[str, list[float]]` | — | Standard deviation |
| `p5` | `dict[str, list[float]]` | — | 5th percentile |
| `p95` | `dict[str, list[float]]` | — | 95th percentile |

**Returns:** `None` (saves file to disk)

**Example:**
```python
from dynafx.dynamics import SensitivityAnalyzer

sa = SensitivityAnalyzer(model)
result = sa.sobol(param_spec={"demand": (50, 200)}, output="Inventory")
result.plot_with_bands("sensitivity.png", result.mean, result.std, result.p5, result.p95)
```

---

### `export_results(path)`

Export simulation results to a CSV file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Output file path (required) |

**Returns:** `None` (saves file to disk)

**CSV format:**
```
time,Stock1,Stock2,Stock3
0.0,1000.0,500.0,100.0
1.0,1050.0,520.0,105.0
2.0,1100.0,540.0,110.0
...
```

**Example:**
```python
result.export_results("results.csv")
```

---

## Dict-Style Access

For backward compatibility, `SysdModelResult` supports dictionary-style access:

```python
# These are equivalent:
result.values["Inventory"]
result["values"]["Inventory"]

# Check if an attribute exists
"Inventory" in result  # True if stocks contains "Inventory"
```

---

## Common Patterns

### Pattern 1: Extract Final Values

```python
result = model.simulate()

# Get final value for each stock
for stock in result.stocks:
    final = result.values[stock][-1]
    print(f"{stock}: {final:.2f}")
```

### Pattern 2: Find Max/Min Values

```python
result = model.simulate()

# Find peak inventory
peak_inventory = max(result.values["Inventory"])
peak_time = result.times[result.values["Inventory"].index(peak_inventory)]
print(f"Peak inventory: {peak_inventory:.0f} at t={peak_time:.1f}")
```

### Pattern 3: Compare Two Runs

```python
result1 = model.simulate(params={"growth_rate": 0.02})
result2 = model.simulate(params={"growth_rate": 0.05})

# Compare final values
print(f"Slow growth: {result1.values['Population'][-1]:.0f}")
print(f"Fast growth: {result2.values['Population'][-1]:.0f}")
```

### Pattern 4: Export and Analyze

```python
result = model.simulate()
result.export_results("simulation_output.csv")

# Later, load and analyze with pandas
import pandas as pd
df = pd.read_csv("simulation_output.csv")
print(df.describe())
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ImportError: matplotlib not installed` | matplotlib not available | `pip install matplotlib` |
| Empty plot | No data in result | Check model has stocks and simulation ran |
| `KeyError` in `result.values[name]` | Stock name doesn't exist | Check `result.stocks` for valid names |
| All values are 0 | Model has no dynamics | Add flows to stocks or check expressions |

---

## See Also

- [`SysdModel`](SysdModel.md) — How to create a model and run simulation
- [`plot_with_bands`](SysdModelResult.md#plot_with_bands) — For sensitivity analysis results
- [Tutorial 1: Hello World](../../tutorials/01-hello-world.md) — First simulation
- [Tutorial 10: Publishing Results](../../tutorials/10-publishing-results.md) — Exporting and visualizing
