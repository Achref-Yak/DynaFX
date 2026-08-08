# dynafx.dynamics

**What it is:** The simulation engine — build models with stocks, flows, agents, and queues, then run them to see how the system evolves over time.

**When to use it:** Every DynaFX simulation starts here. This module gives you the tools to define your system's structure and simulate its behavior.

## Quick Start

```python
from dynafx.dynamics import SysdModel

model = SysdModel(dt=0.1, t_span=(0, 50))
with model.stock("Population", 1000) as s:
    s.inflow("births", "population * 0.02")
    s.outflow("deaths", "population * 0.01")
result = model.simulate()
```

## Import

```python
from dynafx.dynamics import SysdModel, parse_sysd, parse_sysd_file
```

## Core Classes

| Class | What it does | When to use it |
|-------|--------------|----------------|
| [`SysdModel`](SysdModel.md) | The main simulation model | Always — this is your starting point |
| [`SysdModelResult`](SysdModelResult.md) | Simulation output (trajectories, times) | After calling `.simulate()` |
| [`parse_sysd`](parse_sysd.md) | Parse a `.sysd` string into a model | When loading models from files |

## Building Blocks

### Stocks and Flows

| Class | What it does | Example |
|-------|--------------|---------|
| [`StockDef`](StockDef.md) | A variable that accumulates over time | `Inventory`, `Population`, `Cash` |
| [`FlowDef`](FlowDef.md) | A rate that changes a stock | `births`, `sales`, `supply` |
| [`AuxDef`](AuxDef.md) | A derived variable computed each step | `reliability = 0.85` |

```python
# Stocks accumulate, flows change them
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")      # +200 per time step
    s.outflow("demand", "150")     # -150 per time step

# Auxiliaries are computed each step
model.aux("reliability", "0.85")
model.aux("desired", "500")
```

### Agent-Based Modeling

| Class | What it does | Example |
|-------|--------------|---------|
| [`AgentDef`](AgentDef.md) | Agent type with properties and rules | `Customer`, `Worker`, `Vehicle` |
| [`ABMEngine`](ABMEngine.md) | Runtime engine that runs agent steps | Used internally by `SysdModel` |

```python
# Agents have properties and behavioral rules
with model.agent("Customer", 100) as a:
    a.prop("satisfaction", 1.0, min_val=0, max_val=1)
    a.rule("churn", "satisfaction < 0.3", ["churn_risk += 0.1"])
    a.strategy("crisis").rule("ration", "inventory < 10", ["safety_stock += 50"])
```

### Discrete Event Simulation

| Class | What it does | Example |
|-------|--------------|---------|
| [`QueueDef`](QueueDef.md) | A queue with capacity and service time | `checkout`, `assembly_line` |
| [`DESEngine`](DESEngine.md) | Runtime engine for event-driven simulation | Used internally by `SysdModel` |

```python
# Queues process entities with service times
model.queue("checkout", capacity=20, service_time="3 + random()", servers=3)
model.queue("assembly", capacity=-1, arrival_rate="10")
```

## Analysis Tools

| Module | What it does | When to use it |
|--------|--------------|----------------|
| [`causal`](causal.md) | Trace causes and effects through model structure | When you need to understand *why* something happens |
| [`scenario`](scenario.md) | Compare multiple parameter scenarios | When testing "what if" questions |
| [`sensitivity`](sensitivity.md) | Find which parameters matter most | When you have many uncertain parameters |
| [`optimization`](optimization.md) | Find optimal parameter values | When you want the "best" outcome |
| [`units`](units.md) | Check dimensional consistency | When verifying model equations |

```python
# Scenario comparison
from dynafx.dynamics import ScenarioComparison, ScenarioDef

comp = ScenarioComparison(model, [
    ScenarioDef("Baseline", {}),
    ScenarioDef("High demand", {"demand": 150}),
    ScenarioDef("Low capacity", {"capacity": 50}),
])
comp.plot_comparison("comparison.png", stocks=["Inventory"])

# Sensitivity analysis
from dynafx.dynamics import SensitivityAnalyzer

sa = SensitivityAnalyzer(model)
result = sa.sobol(param_spec={"demand": (50, 200)}, output="Inventory")
print(result.ranking("total_order"))
```

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

# Override defaults at simulate time
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

## Available Expressions

DynaFX expressions support these built-in functions:

| Function | Description | Example |
|----------|-------------|---------|
| `MIN(a, b)` | Minimum of two values | `MIN(demand, capacity)` |
| `MAX(a, b)` | Maximum of two values | `MAX(0, inventory)` |
| `IF(cond, a, b)` | Conditional | `IF(inventory > 100, 1, 0)` |
| `ABS(x)` | Absolute value | `ABS(change)` |
| `EXP(x)` | Exponential | `EXP(growth_rate * t)` |
| `LN(x)` | Natural log | `LN(value)` |
| `SQRT(x)` | Square root | `SQRT(variance)` |
| `SIN(x)`, `COS(x)` | Trigonometry | `SIN(2 * PI * t / period)` |
| `PI` | Pi constant | `2 * PI * t` |
| `SMOOTH(x, delay)` | Exponential smoothing | `SMOOTH(demand, 5)` |
| `DELAY3(input, delay)` | 3rd-order delay | `DELAY3 orders, 10` |
| `PULSE(vol, start, width)` | Pulse function | `PULSE(100, 10, 5)` |
| `STEP(height, start)` | Step function | `STEP(200, 20)` |
| `RAMP(slope, start, end)` | Ramp function | `RAMP(10, 0, 50)` |
| `NOISE(amplitude)` | Random noise | `NOISE(5)` |

## See Also

- [`SysdModel`](SysdModel.md) — Full API reference for the core model
- [Tutorial 1: Hello World](../../tutorials/01-hello-world.md) — Your first simulation
- [Tutorial 2: System Dynamics](../../tutorials/02-system-dynamics.md) — Stocks, flows, auxiliaries
- [Tutorial 3: Agent-Based Modeling](../../tutorials/03-agent-based-modeling.md) — Agents and rules
- [Tutorial 4: Discrete Event Simulation](../../tutorials/04-discrete-event-simulation.md) — Queues and resources
