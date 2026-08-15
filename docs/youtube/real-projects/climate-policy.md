# Real-World Project: Climate Policy Impact

**What it does:** Model how policy changes affect emissions and temperature.

**Duration:** 20-25 minutes

**Hook:** "What if we could see 100 years of climate policy in 20 minutes?"

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Climate models are complex. But the core is simple stocks and flows."

### 0:30 - Problem (2 min)
- CO2 accumulates in atmosphere
- Policy affects emissions
- Temperature responds to CO2
- "What policy reduces temperature most?"

### 2:30 - Step 1: Emissions Model (5 min)
```python
model = SysdModel(dt=1.0, t_span=(0, 100))

model.param("carbon_tax", 0)
model.param("renewable_incentive", 0)

with model.stock("atmospheric_CO2", 400) as s:
    s.inflow("emissions", "50 * (1 - renewable_incentive * 0.5)")
    s.outflow("absorption", "atmospheric_CO2 * 0.02")

with model.stock("ocean_CO2", 1000) as s:
    s.inflow("from_atmosphere", "atmospheric_CO2 * 0.02")
    s.outflow("deep_storage", "ocean_CO2 * 0.001")
```

### 7:30 - Step 2: Temperature Response (3 min)
```python
with model.stock("temperature", 1.0) as s:
    s.inflow("warming", "atmospheric_CO2 * 0.005")
    s.outflow("cooling", "temperature * 0.01")

model.aux("sea_level", "temperature * 0.3")
model.aux("extreme_events", "IF(temperature > 1.5, 2, 1)")
```

### 10:30 - Step 3: Policy Levers (3 min)
```python
model.aux("effective_emissions", "emissions * (1 - carbon_tax * 0.01)")
```

### 13:30 - Step 4: Knowledge Graph (3 min)
```python
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:climate ex:baseline_temp 1.0 .
    ex:climate ex:target_temp 1.5 .
    ex:policy ex:carbon_tax 0 .
""")
```

### 16:30 - Step 5: Scenarios (3 min)
```python
# Baseline
result1 = model.simulate(params={"carbon_tax": 0})

# Moderate policy
result2 = model.simulate(params={"carbon_tax": 50})

# Aggressive policy
result3 = model.simulate(params={"carbon_tax": 100})
```

### 19:30 - Analysis (2 min)
```python
result1.plot_with_bands(result3, "climate_policy.png")
print(f"Baseline temp: {result1.values['temperature'][-1]:.2f}")
print(f"With policy: {result3.values['temperature'][-1]:.2f}")
```

### 21:30 - Outro (1 min)
- "This is how policy makers use simulation."

---

## Full Code

```python
from dynafx.dynamics import SysdModel
from dynafx.knowledge import parse_turtle
from dynafx.bridge import KBSimBridge

# Knowledge graph
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:climate ex:baseline_temp 1.0 .
    ex:climate ex:target_temp 1.5 .
    ex:policy ex:carbon_tax 0 .
""")

# Model
model = SysdModel(dt=1.0, t_span=(0, 100))

model.param("carbon_tax", 0)
model.param("renewable_incentive", 0)

with model.stock("atmospheric_CO2", 400) as s:
    s.inflow("emissions", "50 * (1 - renewable_incentive * 0.5)")
    s.outflow("absorption", "atmospheric_CO2 * 0.02")

with model.stock("ocean_CO2", 1000) as s:
    s.inflow("from_atmosphere", "atmospheric_CO2 * 0.02")
    s.outflow("deep_storage", "ocean_CO2 * 0.001")

with model.stock("temperature", 1.0) as s:
    s.inflow("warming", "atmospheric_CO2 * 0.005")
    s.outflow("cooling", "temperature * 0.01")

model.aux("sea_level", "temperature * 0.3")
model.aux("effective_emissions", "emissions * (1 - carbon_tax * 0.01)")

# Scenarios
result1 = model.simulate(params={"carbon_tax": 0})
result2 = model.simulate(params={"carbon_tax": 50})
result3 = model.simulate(params={"carbon_tax": 100})

# Analyze
result1.plot_with_bands(result3, "climate_policy.png")
print(f"Baseline: {result1.values['temperature'][-1]:.2f}")
print(f"Moderate: {result2.values['temperature'][-1]:.2f}")
print(f"Aggressive: {result3.values['temperature'][-1]:.2f}")
```

---

## Thumbnail
- Left: Earth temperature map
- Right: Temperature curves
- Bottom: "Climate policy simulation"

---

## Description
```
Model climate policy impact in DynaFX.

Docs: https://achref-yak.github.io/DynaFX/
GitHub: https://github.com/Achref-Yak/DynaFX
```
