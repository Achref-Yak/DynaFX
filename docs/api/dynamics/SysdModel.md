# SysdModel

**What it is:** The core simulation model — a container for stocks, flows, agents, queues, and all the pieces of your system.

**When to use it:** Every DynaFX simulation starts here. Create a `SysdModel`, define your variables, and call `.simulate()` to run the simulation.

**Quick start:**
```python
from dynafx.dynamics import SysdModel

model = SysdModel(name="hello", dt=0.1, t_span=(0, 50))
with model.stock("Population", 1000) as s:
    s.inflow("births", "population * 0.02")
    s.outflow("deaths", "population * 0.01")
result = model.simulate()
print(result.values["Population"][-1])
```

---

## Import

```python
from dynafx.dynamics import SysdModel
```

## Constructor

### `SysdModel(name="", dt=1.0, t_span=(0.0, 100.0))`

Creates a new simulation model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `""` | Model name (for display and identification) |
| `dt` | `float` | `1.0` | Time step size for integration |
| `t_span` | `tuple[float, float]` | `(0.0, 100.0)` | `(start, end)` simulation time range |

**Returns:** `SysdModel` instance

**Example:**
```python
# Simple model
model = SysdModel()

# Named model with custom time settings
model = SysdModel(name="inventory", dt=0.5, t_span=(0, 50))
```

---

## Methods

### `simulate(method="rk4", t_span=None, dt=None, params=None, kb=None)`

Run the simulation and return the trajectory results.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"rk4"` | Integration method: `"rk4"` (accurate) or `"euler"` (fast) |
| `t_span` | `tuple[float, float] \| None` | `None` | Override the model's time range |
| `dt` | `float \| None` | `None` | Override the model's time step |
| `params` | `dict[str, Any] \| None` | `None` | Parameter overrides: `{"name": value}` |
| `kb` | `TripleStore \| None` | `None` | Enable `KB_QUERY()`/`KB_ASSERT()` in expressions |

**Returns:** [`SysdModelResult`](SysdModelResult.md) — object with `.values`, `.times`, `.stocks`, `.plot()`

**Raises:**
- `ValueError` — If `method` is not `"rk4"` or `"euler"`
- `RuntimeError` — If model has no stocks defined

**Example:**
```python
# Basic simulation
result = model.simulate()

# With parameter overrides
result = model.simulate(params={"reliability": 0.85, "demand": 200})

# With KB integration
from dynafx.knowledge import TripleStore
store = TripleStore()
result = model.simulate(kb=store)

# Override time settings
result = model.simulate(t_span=(0, 200), dt=0.01)
```

---

### `stock(name, initial=0.0, unit="")`

Define a stock (a variable that accumulates over time). Returns a context manager for adding flows.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Stock name (required, must be unique) |
| `initial` | `float` | `0.0` | Starting value at t=0 |
| `unit` | `str` | `""` | Unit label (for dimensional analysis) |

**Returns:** `_StockCtx` — context manager with `.inflow()` and `.outflow()` methods

**Example:**
```python
# Basic stock with inflow and outflow
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "desired * reliability")
    s.outflow("demand", "sales_rate")

# Stock with unit checking
with model.stock("CO2", 0, unit="tons") as s:
    s.inflow("emissions", "activity * emission_factor")

# Stock with multiple flows
with model.stock("Population", 1000) as s:
    s.inflow("births", "population * birth_rate")
    s.inflow("immigration", "100")
    s.outflow("deaths", "population * death_rate")
    s.outflow("emigration", "20")
```

---

### `aux(name, expr, unit="")`

Define an auxiliary variable — a value computed from an expression each time step.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Variable name (required) |
| `expr` | `str` | — | Expression string (can reference other variables) |
| `unit` | `str` | `""` | Unit label |

**Returns:** `SysdModel` — self (for chaining)

**Example:**
```python
# Simple constants
model.aux("desired", "500")
model.aux("reliability", "0.85")

# Derived values
model.aux("total_cost", "fixed_cost + variable_cost * quantity")
model.aux("utilization", "actual_output / capacity")

# Chained calls
model.aux("demand", "200").aux("supply", "150").aux("shortage", "demand - supply")
```

---

### `agent(name, count=1)`

Define an agent type for agent-based modeling. Returns a context manager for adding properties and rules.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Agent type name (required) |
| `count` | `int` | `1` | Number of agent instances to create |

**Returns:** `_AgentCtx` — context manager with `.prop()`, `.rule()`, `.strategy()`, `.meta_rule()`

**Example:**
```python
# Basic agent with properties and rules
with model.agent("Customer", 100) as a:
    a.prop("satisfaction", 1.0, min_val=0, max_val=1)
    a.prop("budget", 100, min_val=0)
    a.rule("buy", "budget > 10", ["budget -= 10", "satisfaction += 0.1"])
    a.rule("churn", "satisfaction < 0.3", ["churn_risk += 0.1"])

# Agent with strategy switching
with model.agent("Worker", 50) as a:
    a.prop("stress", 0.0, min_val=0, max_val=1)
    a.strategy("normal").rule("work", "stress < 0.7", ["output += 10"])
    a.strategy("crisis").rule("rest", "stress > 0.7", ["stress -= 0.2"])
    a.meta_rule("switch", "KB_QUERY(stress_level) > 0.8", ["SWITCH_STRATEGY('crisis')"])
```

---

### `queue(name, capacity=-1, service_time="", arrival_rate="", servers=1, event_driven=False)`

Define a DES queue for discrete event simulation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Queue name (required) |
| `capacity` | `int` | `-1` | Max queue length (`-1` = unlimited) |
| `service_time` | `str` | `""` | Service time expression or distribution |
| `arrival_rate` | `str` | `""` | Arrival rate expression |
| `servers` | `int` | `1` | Number of parallel service channels |
| `event_driven` | `bool` | `False` | Use event-driven vs time-sliced service |

**Returns:** `SysdModel` — self (for chaining)

**Example:**
```python
# Basic queue
model.queue("checkout", capacity=20, service_time="3 + random()", servers=3)

# Queue with arrival rate
model.queue("assembly", capacity=-1, arrival_rate="10")

# Event-driven queue
model.queue("processing", service_time="5", event_driven=True)
```

---

### `resource(name, capacity=1, cost_per_unit=0.0)`

Define a DES resource (e.g., a machine or worker pool).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Resource name (required) |
| `capacity` | `int` | `1` | Number of units available |
| `cost_per_unit` | `float` | `0.0` | Cost per unit time |

**Returns:** `SysdModel` — self (for chaining)

**Example:**
```python
model.resource("machine", capacity=5, cost_per_unit=100)
model.resource("worker", capacity=20, cost_per_unit=50)
```

---

### `event(name, rate="", target_queue="", effects=None)`

Define a DES event.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Event name (required) |
| `rate` | `str` | `""` | Event rate expression |
| `target_queue` | `str` | `""` | Queue to send entities to |
| `effects` | `list[str] \| None` | `None` | Effects to apply when event fires |

**Returns:** `SysdModel` — self (for chaining)

**Example:**
```python
model.event("arrival", rate="5", target_queue="checkout")
model.event("failure", rate="0.1", effects=["downtime += 1"])
```

---

### `table(name, x, y)`

Define a lookup table for nonlinear relationships.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Table name (required) |
| `x` | `list[float]` | — | Input values (must be sorted ascending) |
| `y` | `list[float]` | — | Output values (same length as x) |

**Returns:** `SysdModel` — self (for chaining)

**Example:**
```python
# Demand decreases as price increases
model.table("demand_curve", [10, 20, 50, 100], [500, 300, 100, 20])

# Use in expressions
with model.stock("Inventory", 100) as s:
    s.outflow("sales", "demand_curve(price)")
```

---

### `param(name, value)`

Set a default parameter value. Can be overridden at simulate time.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Parameter name (required) |
| `value` | `float` | — | Default value |

**Returns:** `SysdModel` — self (for chaining)

**Example:**
```python
model.param("growth_rate", 0.02)
model.param("capacity", 1000)

# Override at simulate time
result = model.simulate(params={"growth_rate": 0.05})
```

---

### `func(name, params, body)`

Define a reusable function for use in expressions (macro, expanded at compile time).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Function name (required) |
| `params` | `list[str]` | — | Parameter names |
| `body` | `str` | — | Expression body |

**Returns:** `SysdModel` — self (for chaining)

**Example:**
```python
model.func("square", ["x"], "x * x")
model.func("clamp", ["val", "lo", "hi"], "MAX(lo, MIN(val, hi))")

model.aux("KE", "0.5 * m * square(v)")
model.aux("clamped", "clamp(value, 0, 100)")
```

---

### `submodel(name)`

Define a submodel template for reusable model components.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Submodel name (required) |

**Returns:** `_SubmodelCtx` — context manager with `.stock()` and `.aux()` methods

**Example:**
```python
with model.submodel("sector") as sm:
    with sm.stock("population", 1000) as s:
        s.inflow("births", "population * birth_rate")
    sm.aux("birth_rate", "0.02")

model.include("sector", alias="north")
model.include("sector", alias="south", params={"birth_rate": 0.03})
```

---

### `include(name, alias="", params=None)`

Instantiate a submodel with optional parameter overrides.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Submodel name (required) |
| `alias` | `str` | `""` | Instance name (default: `"{name}_inst"`) |
| `params` | `dict[str, float] \| None` | `None` | Parameter overrides |

**Returns:** `SysdModel` — self (for chaining)

**Example:**
```python
model.include("sector", alias="north")
model.include("sector", alias="south", params={"birth_rate": 0.03})
```

---

### `import_data(path, fill="forward", time_unit="auto", mode="replace", encoding="utf-8")`

Import time series data from a CSV file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Path to CSV file (required) |
| `fill` | `str` | `"forward"` | Missing data strategy: `"forward"`, `"interpolate"`, `"zero"` |
| `time_unit` | `str` | `"auto"` | Time unit: `"auto"`, `"hours"`, `"days"`, `"seconds"` |
| `mode` | `str` | `"replace"` | `"replace"` overwrites existing imported data; `"merge"` adds new variables and raises `ValueError` on duplicate names |
| `encoding` | `str` | `"utf-8"` | CSV file encoding |

**Returns:** `dict[str, list[tuple[float, float]]]` — imported data

**Example:**
```python
data = model.import_data("data/sales.csv", fill="interpolate")
# data = {"sales": [(0, 100), (1, 120), (2, 115), ...]}

# Merge additional variables without overwriting existing ones
model.import_data("data/prices.csv", mode="merge")

# Non-UTF-8 file
model.import_data("data/legacy.csv", encoding="latin-1")
```

---

### `merge_data(paths, fill="forward", time_unit="auto", encoding="utf-8")`

Import and merge data from multiple CSV files into a single dataset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `paths` | `list[str]` | — | List of CSV file paths (required) |
| `fill` | `str` | `"forward"` | Missing data strategy: `"forward"`, `"interpolate"`, `"zero"` |
| `time_unit` | `str` | `"auto"` | Time unit: `"auto"`, `"hours"`, `"days"`, `"seconds"` |
| `encoding` | `str` | `"utf-8"` | CSV file encoding |

**Returns:** `dict[str, list[tuple[float, float]]]` — merged data

**Raises:** `ValueError` if the same variable name appears in multiple files.

**Example:**
```python
data = model.merge_data([
    "data/demand.csv",
    "data/supply.csv",
    "data/prices.csv",
])
# data = {"demand": [...], "supply": [...], "prices": [...]}
```

---

### `validate()`

Validate the model for structural issues.

**Returns:** `ValidationResult` — list of errors and warnings

**Example:**
```python
result = model.validate()
if result.errors:
    for issue in result.errors:
        print(f"{issue.level}: {issue.message}")
```

---

### `to_dict()`

Export model structure as a plain dict.

**Returns:** `dict` with structure:

```python
{
    "nodes": [
        {"id": "Inventory", "type": "stock", "label": "Inventory", "initial": 1000},
        {"id": "supply",    "type": "flow",  "label": "supply"},
        {"id": "demand",    "type": "flow",  "label": "demand"},
        {"id": "utilization", "type": "auxiliary", "label": "utilization", "expr": "0.85"},
    ],
    "edges": [
        {"source": "supply",    "target": "Inventory", "polarity": "+"},
        {"source": "demand",    "target": "Inventory", "polarity": "-"},
    ],
    "loops": [
        {"name": "L1", "nodes": ["Inventory", "supply"], "polarity": "+"},
    ]
}
```

Internally calls `get_dependencies()` and `detect_feedback_loops()` to build the
edge and loop lists. Auxiliaries are included via dependency analysis.

**Example:**
```python
model = SysdModel(dt=1.0, t_span=(0, 100))
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")
    s.outflow("demand", "150")
model.aux("turnover", "demand / inventory")

d = model.to_dict()
for node in d["nodes"]:
    print(f"{node['type']:10s} {node['id']}")
# stock      Inventory
# flow       supply
# flow       demand
# auxiliary  turnover
```

---

## Common Patterns

### Pattern 1: Minimal Model

```python
from dynafx.dynamics import SysdModel

model = SysdModel(dt=1.0, t_span=(0, 100))
with model.stock("Stock", 100) as s:
    s.inflow("inflow", "10")
    s.outflow("outflow", "5")
result = model.simulate()
```

### Pattern 2: Parameterized Model

```python
model = SysdModel(dt=0.5, t_span=(0, 50))
model.param("growth_rate", 0.02)
model.param("capacity", 1000)

with model.stock("Population", 100) as s:
    s.inflow("births", "population * growth_rate * (1 - population/capacity)")
    s.outflow("deaths", "population * 0.01")

result = model.simulate(params={"growth_rate": 0.05})
```

### Pattern 3: With Lookup Tables

```python
model = SysdModel(dt=0.5, t_span=(0, 50))
model.table("demand_curve", [0, 50, 100, 200], [200, 150, 80, 20])

with model.stock("Inventory", 100) as s:
    s.inflow("supply", "200")
    s.outflow("demand", "demand_curve(inventory)")

result = model.simulate()
```

### Pattern 4: Multi-Paradigm

```python
model = SysdModel(dt=0.5, t_span=(0, 100))

# System Dynamics
with model.stock("Inventory", 500) as s:
    s.inflow("production", "production_rate")
    s.outflow("sales", "demand")

# Agents
with model.agent("Customer", 50) as a:
    a.prop("budget", 100, min_val=0)
    a.rule("buy", "budget > price", ["budget -= price"])

# DES
model.queue("shipping", capacity=100, service_time="2 + random()")

result = model.simulate()
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: method must be "rk4" or "euler"` | Invalid integration method | Use `method="rk4"` or `method="euler"` |
| `RuntimeError: No stocks defined` | Model has no stocks | Add at least one stock with `model.stock()` |
| `KeyError: 'var_name'` | Variable referenced but not defined | Add missing aux/stock/table definition |
| `ZeroDivisionError` in expression | Division by zero in expression | Use `MAX(expr, 0.001)` or check parameters |
| `TypeError: 'NoneType'` | Missing required parameter | Check all required parameters are provided |

---

## See Also

- [`SysdModelResult`](SysdModelResult.md) — What `.simulate()` returns
- [`parse_sysd`](parse_sysd.md) — Load models from `.sysd` files
- [`StockDef`](StockDef.md) — Stock definition details
- [`FlowDef`](FlowDef.md) — Flow definition details
- [`AuxDef`](AuxDef.md) — Auxiliary variable details
- [Tutorial 1: Hello World](../../tutorials/01-hello-world.md) — Build your first model
- [Tutorial 2: System Dynamics](../../tutorials/02-system-dynamics.md) — Stocks, flows, auxiliaries
