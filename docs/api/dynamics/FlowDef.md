# FlowDef

**What it is:** A flow — a rate that changes a stock over time. Flows are the "pipes" that move quantities into or out of stocks.

**When to use it:** When you need to define how a stock changes — what goes in (inflow) and what goes out (outflow).

**Quick start:**
```python
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")      # +200 per time step
    s.outflow("demand", "150")     # -150 per time step
```

---

## Import

```python
from dynafx.dynamics import FlowDef
```

You typically don't import this directly — use `s.inflow()` and `s.outflow()` on a stock context manager.

## Constructor

### `FlowDef(name, direction, expr, unit="")`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Flow name (required) |
| `direction` | `str` | — | `"+"` for inflow, `"-"` for outflow |
| `expr` | `str` | — | Expression for the flow rate |
| `unit` | `str` | `""` | Unit label |

**Returns:** `FlowDef` instance

**Example:**
```python
from dynafx.dynamics import FlowDef

# Direct construction (rarely needed)
inflow = FlowDef(name="supply", direction="+", expr="200")
outflow = FlowDef(name="demand", direction="-", expr="150")
```

---

## Usage via Context Manager

The recommended way to create flows is through a stock's context manager:

```python
# Inflows (add to stock)
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")
    s.inflow("returns", "sold * return_rate")

# Outflows (subtract from stock)
with model.stock("Inventory", 1000) as s:
    s.outflow("sales", "150")
    s.outflow("waste", "inventory * waste_rate")
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Flow name |
| `direction` | `str` | `"+"` (inflow) or `"-"` (outflow) |
| `expr` | `str` | Expression for flow rate |
| `unit` | `str` | Unit label |

---

## How Flows Work

Flows define the rate of change for a stock:

```
stock(t+dt) = stock(t) + sum(inflows) - sum(outflows) * dt
```

For example:

```python
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")      # +200 per time step
    s.outflow("demand", "150")     # -150 per time step
```

With dt=1:
- t=0: Inventory = 1000
- t=1: Inventory = 1000 + (200 - 150) * 1 = 1050
- t=2: Inventory = 1050 + (200 - 150) * 1 = 1100

---

## Expression Examples

```python
# Constant rate
s.inflow("supply", "200")

# Variable rate
s.inflow("supply", "desired * reliability")

# Rate depending on stock itself
s.outflow("depletion", "inventory * depletion_rate")

# Rate with KB query
s.inflow("kb_supply", "KB_QUERY('SELECT ?v WHERE { ... }')")

# Rate with functions
s.outflow("sales", "MAX(0, demand - inventory)")

# Rate with tables
s.outflow("sales", "demand_curve(price)")
```

---

## Common Patterns

### Pattern 1: Constant Flows

```python
with model.stock("Cash", 10000) as s:
    s.inflow("revenue", "5000")
    s.outflow("costs", "3000")
```

### Pattern 2: Variable Flows

```python
with model.stock("Population", 1000) as s:
    s.inflow("births", "population * birth_rate")
    s.outflow("deaths", "population * death_rate")
```

### Pattern 3: Conditional Flows

```python
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "IF(inventory < 500, 300, 100)")
    s.outflow("demand", "MIN(demand, inventory)")
```

### Pattern 4: With KB Integration

```python
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "KB_QUERY('SELECT ?v WHERE { ex:supply_rate ?v }')")
    s.outflow("demand", "KB_QUERY('SELECT ?v WHERE { ex:demand_rate ?v }')")
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ZeroDivisionError` | Flow expression divides by zero | Use `MAX(expr, 0.001)` to avoid division by zero |
| `NameError` | Referenced variable doesn't exist | Add missing aux or stock definition |
| Negative stock value | Outflow exceeds stock | Add constraints: `MIN(outflow, inventory)` |

---

## See Also

- [`SysdModel`](SysdModel.md) — How to add stocks with flows
- [`StockDef`](StockDef.md) — Stocks that flows attach to
- [`AuxDef`](AuxDef.md) — Auxiliary variables
- [Tutorial 2: System Dynamics](../../tutorials/02-system-dynamics.md) — Stocks and flows
