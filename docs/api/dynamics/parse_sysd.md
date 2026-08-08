# parse_sysd / parse_sysd_file

**What it is:** Functions that load simulation models from `.sysd` text files instead of building them in Python code.

**When to use it:** When you want to define models in a simple text format (easier to share, version control, and iterate on) rather than writing Python.

**Quick start:**
```python
from dynafx.dynamics import parse_sysd

model = parse_sysd("""
    model "Inventory"
      dt 0.5
      from 0 to 100

      stock "Inventory": 1000
        + "supply": desired * reliability
        - "demand": sales_rate

      aux "desired": 500
      aux "reliability": 0.85
      aux "sales_rate": 300
""")
result = model.simulate()
```

---

## Import

```python
from dynafx.dynamics import parse_sysd, parse_sysd_file
```

## Functions

### `parse_sysd(source)`

Parse a `.sysd` string into a `SysdModel`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | — | `.sysd` model definition string (required) |

**Returns:** `SysdModel` — ready to simulate

**Raises:**
- `SyntaxError` — If the `.sysd` syntax is invalid
- `ValueError` — If required fields are missing

**Example:**
```python
from dynafx.dynamics import parse_sysd

model = parse_sysd("""
    model "Population"
      dt 1.0
      from 0 to 100

      stock "Population": 1000
        + "births": population * 0.02
        - "deaths": population * 0.01
""")
result = model.simulate()
```

---

### `parse_sysd_file(path)`

Load a `.sysd` file from disk and parse it into a `SysdModel`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Path to `.sysd` file (required) |

**Returns:** `SysdModel` — ready to simulate

**Raises:**
- `FileNotFoundError` — If the file doesn't exist
- `SyntaxError` — If the `.sysd` syntax is invalid

**Example:**
```python
from dynafx.dynamics import parse_sysd_file

model = parse_sysd_file("models/inventory.sysd")
result = model.simulate()
```

---

## .sysd File Syntax

### Basic Structure

```
model "Model Name"
  dt 0.5
  from 0 to 100

  stock "Stock Name": initial_value
    + "Inflow Name": rate_expression
    - "Outflow Name": rate_expression

  aux "Auxiliary Name": expression

  table "Table Name"
    x: [0, 10, 20, 30]
    y: [5, 15, 5, 15]
```

### Stocks

```
stock "Inventory": 1000
  + "supply": desired * reliability
  - "demand": sales_rate
```

- `stock` keyword
- Name in quotes
- Initial value after colon
- Flows indented below
- `+` for inflows, `-` for outflows

### Auxiliaries

```
aux "desired": 500
aux "reliability": 0.85
aux "total_cost": fixed_cost + variable_cost * quantity
```

### Tables

```
table "demand_curve"
  x: [10, 20, 50, 100]
  y: [500, 300, 100, 20]
```

### Time Settings

```
dt 0.5
from 0 to 100
```

- `dt` — time step
- `from ... to ...` — time range

---

## Common Patterns

### Pattern 1: Simple Model

```python
model = parse_sysd("""
    model "Simple"
      dt 1.0
      from 0 to 50

      stock "Stock": 100
        + "inflow": 10
        - "outflow": 5
""")
```

### Pattern 2: With Parameters

```python
model = parse_sysd("""
    model "Parameterized"
      dt 0.5
      from 0 to 100

      stock "Population": 100
        + "births": population * growth_rate
        - "deaths": population * death_rate

      aux "growth_rate": 0.02
      aux "death_rate": 0.01
""")
result = model.simulate(params={"growth_rate": 0.05})
```

### Pattern 3: With Lookup Tables

```python
model = parse_sysd("""
    model "WithTable"
      dt 0.5
      from 0 to 50

      stock "Inventory": 100
        + "supply": 200
        - "demand": demand_curve(inventory)

      table "demand_curve"
        x: [0, 50, 100, 200]
        y: [200, 150, 80, 20]
""")
```

### Pattern 4: Load from File

```python
# models/inventory.sysd
model = parse_sysd_file("models/inventory.sysd")
result = model.simulate()
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `SyntaxError: Expected 'model'` | File doesn't start with `model` | Add `model "Name"` at the top |
| `SyntaxError: Expected ':'` | Missing colon after stock name | Add `:` after initial value |
| `SyntaxError: Expected '+' or '-'` | Flow missing direction | Add `+` or `-` before flow name |
| `ValueError: No stocks defined` | Model has no stocks | Add at least one `stock` definition |
| `FileNotFoundError` | File path is wrong | Check the file path |

---

## See Also

- [`SysdModel`](SysdModel.md) — Python API for building models programmatically
- [`SysdModelResult`](SysdModelResult.md) — What `.simulate()` returns
- [Tutorial 1: Hello World](../../tutorials/01-hello-world.md) — Your first model
- [Examples](../../examples.md) — Sample `.sysd` files
