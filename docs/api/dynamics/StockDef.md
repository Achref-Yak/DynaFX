# StockDef

**What it is:** A stock (level variable) — a value that accumulates or depletes over time based on inflows and outflows.

**When to use it:** When you need a variable that represents a quantity that builds up or drains, like inventory, population, cash, or CO2 levels.

**Quick start:**
```python
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")
    s.outflow("demand", "150")
```

---

## Import

```python
from dynafx.dynamics import StockDef
```

You typically don't import this directly — use `model.stock()` context manager instead.

## Constructor

### `StockDef(name, initial=0.0, flows=None, unit="")`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Stock name (required, must be unique) |
| `initial` | `float` | `0.0` | Starting value at t=0 |
| `flows` | `list[FlowDef] \| None` | `None` | List of flow definitions |
| `unit` | `str` | `""` | Unit label for dimensional analysis |

**Returns:** `StockDef` instance

**Example:**
```python
from dynafx.dynamics import StockDef, FlowDef

# Direct construction (rarely needed)
stock = StockDef(
    name="Inventory",
    initial=1000,
    flows=[
        FlowDef(name="supply", direction="+", expr="200"),
        FlowDef(name="demand", direction="-", expr="150"),
    ],
    unit="units"
)
```

---

## Usage via Context Manager

The recommended way to create stocks is through `model.stock()`:

```python
# Basic stock
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")
    s.outflow("demand", "150")

# Stock with unit
with model.stock("CO2", 0, unit="tons") as s:
    s.inflow("emissions", "activity * emission_factor")

# Stock with no flows (state variable)
with model.stock("Temperature", 20.0) as s:
    pass
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Stock name |
| `initial` | `float` | Initial value |
| `flows` | `list[FlowDef]` | List of attached flows |
| `unit` | `str` | Unit label |

---

## How Stocks Work

A stock accumulates based on the net flow:

```
stock(t+dt) = stock(t) + (inflows - outflows) * dt
```

For example, if Inventory starts at 1000, inflow is 200, and outflow is 150, with dt=1:

```
Inventory(1) = 1000 + (200 - 150) * 1 = 1050
Inventory(2) = 1050 + (200 - 150) * 1 = 1100
```

---

## Common Patterns

### Pattern 1: Basic Accumulation

```python
with model.stock("Cash", 10000) as s:
    s.inflow("revenue", "sales * price")
    s.outflow("costs", "units * unit_cost")
```

### Pattern 2: Population Dynamics

```python
with model.stock("Population", 1000) as s:
    s.inflow("births", "population * birth_rate")
    s.outflow("deaths", "population * death_rate")
```

### Pattern 3: With KB Integration

```python
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "KB_QUERY('SELECT ?v WHERE { ... }')")
    s.outflow("demand", "sales_rate")
```

---

## See Also

- [`SysdModel`](SysdModel.md) — How to add stocks to a model
- [`FlowDef`](FlowDef.md) — How flows work
- [`AuxDef`](AuxDef.md) — Auxiliary variables
- [Tutorial 2: System Dynamics](../../tutorials/02-system-dynamics.md) — Stocks and flows deep dive
