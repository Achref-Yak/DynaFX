# Video 2: Building Models with Python (Not Text Files)

**What it does:** Construct models programmatically — stocks, flows, auxiliaries, parameters.

**Duration:** 15-18 minutes

**Hook:** "Text files are great for tutorials. But real models need loops, conditionals, and reuse. Let's build with Python."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "We built a model in 5 lines last time. Today: how to build any model in pure Python."

### 0:30 - Why Python DSL? (2 min)
- Text files are limited — no loops, no conditionals, no reuse
- Python DSL gives you full programming power
- "If your model needs dynamic behavior, build it in Python."

### 2:30 - SysdModel Constructor (2 min)
```python
from dynafx.dynamics import SysdModel

model = SysdModel(name="Inventory", dt=0.5, t_span=(0, 100))
```
- `name` — for display and identification
- `dt` — time step (smaller = more accurate, slower)
- `t_span` — (start, end) simulation time

### 4:30 - Stocks and Flows (4 min)
```python
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")
    s.outflow("demand", "150")
```
- Context manager pattern — clean, readable
- `inflow` adds to stock, `outflow` subtracts
- Expressions can reference other variables

### 8:30 - Auxiliaries (3 min)
```python
model.aux("reliability", "0.85")
model.aux("desired", "500")
model.aux("total_cost", "fixed_cost + variable_cost * quantity")
```
- Derived values recomputed each step
- Can reference stocks, other auxes, parameters
- Order of evaluation is automatic (topological sort)

### 11:30 - Parameters (2 min)
```python
model.param("growth_rate", "0.02")
model.param("capacity", "1000")

# Override at simulate time
result = model.simulate(params={"growth_rate": "0.05"})
```
- Default values set in model
- Overridable at simulate time
- "This is how KB facts become model inputs."

### 13:30 - Lookup Tables (2 min)
```python
model.table("demand_curve", [10, 20, 50, 100], [500, 300, 100, 20])
```
- Nonlinear relationships
- Interpolated automatically
- Use in expressions: `demand_curve(price)`

### 15:30 - Chaining (1 min)
```python
model.aux("demand", "200").aux("supply", "150").aux("shortage", "demand - supply")
```
- Fluent API for clean code

### 16:30 - Challenge (1 min)
- "Build an inventory model: Stock starts at 1000, supply flows in at 200, demand flows out based on a lookup table. Share your code in the comments."

### 17:30 - Outro (30 sec)
- "Next time: what if? Comparing scenarios side by side."

---

## Code Samples

### Basic Inventory Model
```python
from dynafx.dynamics import SysdModel

model = SysdModel(name="Inventory", dt=0.5, t_span=(0, 100))

with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")
    s.outflow("demand", "150")

result = model.simulate()
result.plot("inventory.png")
```

### With Parameters
```python
model = SysdModel(name="Parameterized", dt=0.5, t_span=(0, 100))
model.param("supply_rate", 200)
model.param("demand_rate", 150)

with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "supply_rate")
    s.outflow("demand", "demand_rate")

# Default run
result1 = model.simulate()

# Override parameters
result2 = model.simulate(params={"supply_rate": 300})
```

### With Lookup Table
```python
model = SysdModel(name="WithTable", dt=0.5, t_span=(0, 100))
model.table("demand_curve", [0, 50, 100, 200], [200, 150, 80, 20])

with model.stock("Inventory", 100) as s:
    s.inflow("supply", "200")
    s.outflow("demand", "demand_curve(inventory)")

result = model.simulate()
```

### Chained Auxiliaries
```python
model = SysdModel(dt=1.0, t_span=(0, 50))
model.aux("fixed_cost", "1000")
model.aux("variable_cost", "5")
model.aux("quantity", "100")
model.aux("total_cost", "fixed_cost + variable_cost * quantity")
model.aux("price", "total_cost * 1.2")
model.aux("profit", "(price - total_cost) * quantity")
```

---

## Thumbnail
- Left: Python code editor
- Right: Inventory curve
- Bottom: "Build any model in Python"

---

## Description
```
Build DynaFX models programmatically — stocks, flows, auxiliaries, parameters, and lookup tables.

Docs: https://achref-yak.github.io/DynaFX/api/dynamics/SysdModel/
GitHub: https://github.com/Achref-Yak/DynaFX

Challenge: Build an inventory model with a nonlinear demand curve. Share your code!
```

---

## Pin Comment
```
Challenge #2: Build a population model with:
- Stock: Population (start at 1000)
- Inflow: births (population * birth_rate)
- Outflow: deaths (population * death_rate)
- Aux: birth_rate = 0.03
- Aux: death_rate = 0.01
- Table: carrying_capacity effect (limits growth as population approaches 1500)

What's the final population? Reply with your code!
```
