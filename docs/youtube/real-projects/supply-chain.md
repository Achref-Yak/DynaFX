# Real-World Project: Global Supply Chain Disruption

**What it does:** Model a global supply chain with disruption tracking and recovery.

**Duration:** 20-25 minutes

**Hook:** "A single chip shortage shut down car production worldwide. Let's model why."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Supply chains are fragile. Let's model the fragility."

### 0:30 - Problem (2 min)
- Multiple tiers of suppliers
- Disruptions cascade
- Recovery takes time
- "How resilient is our supply chain?"

### 2:30 - Step 1: Supply Chain Structure (5 min)
```python
model = SysdModel(dt=1.0, t_span=(0, 365))

# Tier 1: Raw materials
with model.stock("raw_materials", 10000) as s:
    s.inflow("mining", "500")
    s.outflow("to_manufacturing", "400")

# Tier 2: Manufacturing
with model.stock("manufacturing", 5000) as s:
    s.inflow("from_raw", "raw_materials * 0.08")
    s.outflow("to_warehouse", "manufacturing * 0.1")

# Tier 3: Warehousing
with model.stock("warehouse", 3000) as s:
    s.inflow("from_manufacturing", "manufacturing * 0.1")
    s.outflow("to_retail", "warehouse * 0.15")

# Tier 4: Retail
with model.stock("retail", 1000) as s:
    s.inflow("from_warehouse", "warehouse * 0.15")
    s.outflow("sales", "retail * 0.2")
```

### 7:30 - Step 2: Disruption Events (4 min)
```python
cascade = DisruptionCascade({
    "raw_materials": {"probability": 0.1, "recovery": 30},
    "manufacturing": {"probability": 0.2, "recovery": 45},
    "logistics": {"probability": 0.3, "recovery": 20},
})
cascade.link("raw_materials", "manufacturing", 0.8)
cascade.link("manufacturing", "logistics", 0.6)
```

### 11:30 - Step 3: Knowledge Graph (3 min)
```python
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:supply_chain ex:tiers 4 .
    ex:supply_chain ex:resilience 0.7 .
    ex:supplier_a ex:reliability 0.95 .
    ex:supplier_b ex:reliability 0.70 .
""")
```

### 14:30 - Step 4: Closed Loop (3 min)
```python
bridge = KBSimBridge(store)
result = bridge.run_with_kb(model)
```

### 17:30 - Step 5: Analysis (3 min)
```python
result.plot("supply_chain.png")
print(f"Final inventory: {result.values['retail'][-1]:.0f}")
```

### 20:30 - Scenario Comparison (2 min)
- Baseline vs disruption
- Recovery time analysis
- "What's the impact?"

### 22:30 - Outro (1 min)
- "This is how companies model resilience."

---

## Full Code

```python
from dynafx.dynamics import SysdModel
from dynafx.knowledge import parse_turtle
from dynafx.bridge import KBSimBridge
from dynafx.patterns import DisruptionCascade

# Knowledge graph
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:supply_chain ex:tiers 4 .
    ex:supply_chain ex:resilience 0.7 .
    ex:supplier_a ex:reliability 0.95 .
    ex:supplier_b ex:reliability 0.70 .
""")

# Model
model = SysdModel(dt=1.0, t_span=(0, 365))

with model.stock("raw_materials", 10000) as s:
    s.inflow("mining", "500")
    s.outflow("to_manufacturing", "400")

with model.stock("manufacturing", 5000) as s:
    s.inflow("from_raw", "raw_materials * 0.08")
    s.outflow("to_warehouse", "manufacturing * 0.1")

with model.stock("warehouse", 3000) as s:
    s.inflow("from_manufacturing", "manufacturing * 0.1")
    s.outflow("to_retail", "warehouse * 0.15")

with model.stock("retail", 1000) as s:
    s.inflow("from_warehouse", "warehouse * 0.15")
    s.outflow("sales", "retail * 0.2")

# Disruption cascade
cascade = DisruptionCascade({
    "raw_materials": {"probability": 0.1, "recovery": 30},
    "manufacturing": {"probability": 0.2, "recovery": 45},
    "logistics": {"probability": 0.3, "recovery": 20},
})
cascade.link("raw_materials", "manufacturing", 0.8)
cascade.link("manufacturing", "logistics", 0.6)

# Run
bridge = KBSimBridge(store)
result = bridge.run_with_kb(model)

# Analyze
result.plot("supply_chain.png")
```

---

## Thumbnail
- Left: Global supply chain map
- Right: Disruption timeline
- Bottom: "Supply chain resilience"

---

## Description
```
Model global supply chain disruptions in DynaFX.

Docs: https://achref-yak.github.io/DynaFX/
GitHub: https://github.com/Achref-Yak/DynaFX
```
