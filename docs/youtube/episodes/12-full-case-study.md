# Video 12: Full Case Study - Global Supply Chain

**What it does:** Build a complete supply chain model with disruption tracking and knowledge graph integration.

**Duration:** 18-22 minutes

**Hook:** "We have covered 11 videos of individual concepts. Now let's put them all together into one complete model."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Everything you have learned comes together here."

### 0:30 - Problem Definition (2 min)
- Model a global supply chain
- Multiple suppliers
- Disruption events
- Track impact through the chain
- Knowledge graph as the source of truth

### 2:30 - Step 1: Knowledge Graph (3 min)
```python
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:supplier_a ex:reliability 0.95 ; ex:capacity 1000 .
    ex:supplier_b ex:reliability 0.70 ; ex:capacity 2000 .
    ex:supplier_c ex:reliability 0.85 ; ex:capacity 800 .
    ex:supply_chain ex:active true .
""")
```

### 5:30 - Step 2: Model Structure (5 min)
```python
model = SysdModel(dt=1.0, t_span=(0, 100))
model.param("reliability", 0.85)
model.param("disruption_active", False)

with model.stock("raw_materials", 5000) as s:
    s.inflow("procurement", "500 * reliability")
    s.outflow("consumption", "300")

with model.stock("production", 2000) as s:
    s.inflow("from_raw", "raw_materials * 0.1")
    s.outflow("to_logistics", "production * 0.05")

with model.stock("logistics", 1000) as s:
    s.inflow("from_production", "production * 0.05")
    s.outflow("delivery", "logistics * 0.1")

with model.stock("retail", 500) as s:
    s.inflow("from_logistics", "logistics * 0.1")
    s.outflow("sales", "retail * 0.2")

model.aux("total_inventory", "raw_materials + production + logistics + retail")
```

### 10:30 - Step 3: Disruption Cascade (3 min)
```python
cascade = DisruptionCascade({
    "supply": {"probability": 0.3, "recovery": 20},
    "logistics": {"probability": 0.5, "recovery": 15},
    "production": {"probability": 0.4, "recovery": 30},
})
cascade.link("supply", "logistics", 0.8)
cascade.link("logistics", "production", 0.6)
```

### 13:30 - Step 4: Closed Loop (3 min)
```python
bridge = KBSimBridge(store)
result = bridge.run_with_kb(model)
```

### 16:30 - Step 5: Analysis (2 min)
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(result.times, result.values['raw_materials'], linewidth=2, color='#2196F3')
axes[0, 0].set_title('Raw Materials')
axes[0, 0].set_ylabel('Units')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(result.times, result.values['production'], linewidth=2, color='#FF9800')
axes[0, 1].set_title('Production')
axes[0, 1].set_ylabel('Units')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(result.times, result.values['logistics'], linewidth=2, color='#4CAF50')
axes[1, 0].set_title('Logistics')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Units')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(result.times, result.values['retail'], linewidth=2, color='#F44336')
axes[1, 1].set_title('Retail')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Units')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Supply Chain Simulation', fontsize=14)
plt.tight_layout()
plt.show()
```

### 18:30 - Full Code Review (2 min)
- Walk through the complete code
- "This is what a production model looks like."

### 20:30 - Challenge (1 min)
- "Build your own supply chain. Add a new supplier with reliability 0.6. How does it change?"

### 21:30 - Outro (30 sec)
- "Congratulations - you can now model anything."

---

## Code Samples

### Complete Supply Chain Model
```python
from dynafx.dynamics import SysdModel
from dynafx.knowledge import TripleStore, parse_turtle
from dynafx.bridge import KBSimBridge
from dynafx.patterns import DisruptionCascade

# 1. Knowledge graph
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:supplier_a ex:reliability 0.95 ; ex:capacity 1000 .
    ex:supplier_b ex:reliability 0.70 ; ex:capacity 2000 .
    ex:supplier_c ex:reliability 0.85 ; ex:capacity 800 .
    ex:supply_chain ex:active true .
""")

# 2. Model
model = SysdModel(dt=1.0, t_span=(0, 100))
model.aux("reliability", "KB_QUERY('PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:supplier_a ex:reliability ?v }')")

with model.stock("raw_materials", 5000) as s:
    s.inflow("procurement", "500 * reliability")
    s.outflow("consumption", "300")

with model.stock("production", 2000) as s:
    s.inflow("from_raw", "raw_materials * 0.1")
    s.outflow("to_logistics", "production * 0.05")

with model.stock("logistics", 1000) as s:
    s.inflow("from_production", "production * 0.05")
    s.outflow("delivery", "logistics * 0.1")

with model.stock("retail", 500) as s:
    s.inflow("from_logistics", "logistics * 0.1")
    s.outflow("sales", "retail * 0.2")

# 3. Disruption cascade
cascade = DisruptionCascade({
    "supply": {"probability": 0.3, "recovery": 20},
    "logistics": {"probability": 0.5, "recovery": 15},
    "production": {"probability": 0.4, "recovery": 30},
})
cascade.link("supply", "logistics", 0.8)
cascade.link("logistics", "production", 0.6)

# 4. Run with knowledge graph
bridge = KBSimBridge(store)
result = bridge.run_with_kb(model)

# 5. Analyze
result.plot("supply_chain.png")
```

---

## Thumbnail
- Left: Complete supply chain diagram
- Right: Final results chart
- Bottom: "The complete model"

---

## Description
```
Build a complete supply chain model in DynaFX - the full case study.

Docs: https://achref-yak.github.io/DynaFX/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Challenge #12: Build your own supply chain with:
- 3 suppliers
- Disruption cascade
- Knowledge graph integration
Share your model in the comments!
```
