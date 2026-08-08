# AuxDef

**What it is:** An auxiliary variable — a value computed from an expression each time step. Auxiliaries are intermediate calculations that make your model easier to read and maintain.

**When to use it:** When you need to break down complex expressions into named, readable pieces, or when you want to parameterize your model with adjustable constants.

**Quick start:**
```python
model.aux("reliability", "0.85")
model.aux("desired", "500")
model.aux("total_cost", "fixed_cost + variable_cost * quantity")
```

---

## Import

```python
from dynafx.dynamics import AuxDef
```

You typically don't import this directly — use `model.aux()` instead.

## Constructor

### `AuxDef(name, expr, unit="")`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Variable name (required) |
| `expr` | `str` | — | Expression string (required) |
| `unit` | `str` | `""` | Unit label |

**Returns:** `AuxDef` instance

**Example:**
```python
from dynafx.dynamics import AuxDef

# Direct construction (rarely needed)
aux = AuxDef(name="reliability", expr="0.85", unit="ratio")
```

---

## Usage

The recommended way to create auxiliaries is through `model.aux()`:

```python
# Simple constants
model.aux("reliability", "0.85")
model.aux("growth_rate", "0.02")

# Derived values
model.aux("total_cost", "fixed_cost + variable_cost * quantity")
model.aux("utilization", "actual_output / capacity")

# Chained calls
model.aux("demand", "200").aux("supply", "150").aux("shortage", "demand - supply")
```

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Variable name |
| `expr` | `str` | Expression string |
| `unit` | `str` | Unit label |

---

## How Auxiliaries Work

Auxiliaries are recomputed at the start of each time step. They can reference:

- Other auxiliaries
- Stocks (by their current value)
- Parameters
- Tables
- Built-in functions

The order of evaluation is determined automatically based on dependencies.

---

## Expression Examples

```python
# Constants
model.aux("reliability", "0.85")
model.aux("pi_value", "3.14159")

# Derived values
model.aux("total_cost", "fixed_cost + variable_cost * quantity")
model.aux("utilization", "actual_output / capacity")

# Conditional values
model.aux("effective_rate", "IF(inventory > 100, growth_rate, 0)")

# With functions
model.aux("smoothed_demand", "SMOOTH(demand, 5)")
model.aux("delayed_signal", "DELAY3(signal, 10)")

# With tables
model.aux("price", "demand_curve(inventory)")

# With KB queries
model.aux("kb_value", "KB_QUERY('SELECT ?v WHERE { ... }')")
```

---

## Common Patterns

### Pattern 1: Constants

```python
model.aux("birth_rate", "0.02")
model.aux("death_rate", "0.01")
model.aux("capacity", "1000")
```

### Pattern 2: Derived Values

```python
model.aux("total_revenue", "units_sold * price")
model.aux("total_cost", "fixed_cost + variable_cost * units_produced")
model.aux("profit", "total_revenue - total_cost")
```

### Pattern 3: Growth with Limits

```python
model.aux("growth", "population * growth_rate * (1 - population/capacity)")
model.aux("effective_growth", "MAX(0, growth)")
```

### Pattern 4: KB-Connected Auxiliaries

```python
model.aux("reliability", "KB_QUERY('SELECT ?v WHERE { ex:reliability ?v }')")
model.aux("demand", "KB_QUERY('SELECT ?v WHERE { ex:demand ?v }')")
```

---

## Auxiliaries vs Parameters

| Feature | Auxiliaries (`model.aux()`) | Parameters (`model.param()`) |
|---------|----------------------------|------------------------------|
| Value can change | Yes, recomputed each step | No, fixed during simulation |
| Expression support | Yes | No |
| Override at simulate time | No | Yes |
| Use case | Derived calculations | Adjustable constants |

```python
# Aux: computed each step
model.aux("utilization", "actual_output / capacity")

# Parameter: fixed, but overridable
model.param("growth_rate", 0.02)
result = model.simulate(params={"growth_rate": 0.05})  # override
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `NameError` | Referenced variable doesn't exist | Add missing aux/stock definition |
| Circular dependency | Aux references itself | Break the cycle by restructuring |
| `ZeroDivisionError` | Division by zero | Use `MAX(expr, 0.001)` |

---

## See Also

- [`SysdModel`](SysdModel.md) — How to add auxiliaries to a model
- [`StockDef`](StockDef.md) — Stocks that auxiliaries can reference
- [`FlowDef`](FlowDef.md) — Flows that can use auxiliaries
- [Tutorial 2: System Dynamics](../../tutorials/02-system-dynamics.md) — Auxiliaries deep dive
