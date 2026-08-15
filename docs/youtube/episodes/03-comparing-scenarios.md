# Video 3: What If? Comparing Scenarios Side by Side

**What it does:** Run multiple parameter sets, compare trajectories, find which scenario wins.

**Duration:** 12-15 minutes

**Hook:** "Your model works. Now the boss asks: what if demand doubles? What if capacity drops? Let's find out."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "You've built a model. Now someone asks: what if? That's what scenarios are for."

### 0:30 - ScenarioDef (2 min)
```python
from dynafx.dynamics.scenario import ScenarioDef

baseline = ScenarioDef("Baseline", {})
high_demand = ScenarioDef("High Demand", {"demand_rate": 300})
low_capacity = ScenarioDef("Low Capacity", {"capacity": 500})
```
- Name + parameter overrides
- "A scenario is just a named set of parameters."

### 2:30 - ScenarioComparison (4 min)
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(result1.times, result1.values['Inventory'], linewidth=2, label='Balanced')
ax.plot(result2.times, result2.values['Inventory'], linewidth=2, label='High Production')
ax.plot(result3.times, result3.values['Inventory'], linewidth=2, label='High Demand')
ax.set_xlabel('Time')
ax.set_ylabel('Inventory')
ax.set_title('Scenario Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
- Runs all scenarios automatically
- Stores results for comparison

### 6:30 - Visual Comparison (3 min)
```python
comp.plot_comparison("scenarios.png", stocks=["Inventory"])
comp.plot_deviation("deviation.png", stocks=["Inventory"], baseline="Baseline")
```
- Overlay all trajectories
- Show delta from baseline
- "See the difference at a glance."

### 9:30 - Tornado Diagram (2 min)
```python
comp.tornado("tornado.png", 
    param_ranges={"demand_rate": (100, 400), "capacity": (200, 800)},
    output_stock="Inventory")
```
- Which parameter has the biggest impact?
- "The tornado tells you what to focus on."

### 11:30 - Numerical Summary (1 min)
```python
print(comp.summary())
print(comp.deviation_table(baseline=0, mode="relative"))
```
- Exact numbers for reports

### 12:30 - Challenge (1 min)
- "Build a model with 3 scenarios. Which one gives the highest inventory? Comment below."

### 13:30 - Outro (30 sec)
- "Next time: we go deeper — sensitivity analysis to find which parameters actually matter."

---

## Code Samples

### Basic Scenario Comparison
```python
from dynafx.dynamics import SysdModel, ScenarioComparison, ScenarioDef

model = SysdModel(dt=0.5, t_span=(0, 100))
model.param("supply_rate", 200)
model.param("demand_rate", 150)

with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "supply_rate")
    s.outflow("demand", "demand_rate")

comp = ScenarioComparison(model, [
    ScenarioDef("Baseline", {}),
    ScenarioDef("High Demand", {"demand_rate": 300}),
    ScenarioDef("Low Supply", {"supply_rate": 100}),
    ScenarioDef("Balanced", {"supply_rate": 250, "demand_rate": 200}),
])

for scenario in comp.scenarios:
    final = scenario.result.values["Inventory"][-1]
    print(f"{scenario.name:<18} {final:.0f}")
```

### Visual Comparison
```python
comp.plot_comparison("scenarios.png", stocks=["Inventory"])
comp.plot_deviation("deviation.png", stocks=["Inventory"], baseline="Baseline")
```

### Tornado Diagram
```python
comp.tornado("tornado.png",
    param_ranges={"supply_rate": (100, 400), "demand_rate": (50, 350)},
    output_stock="Inventory")
```

### With KB Grading
```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore

store = TripleStore()
bridge = KBSimBridge(store)

# Grade each scenario
for scenario in comp.scenarios:
    triples = bridge.evidence_from_result(scenario.result, evidence_map)
    for t in triples:
        store.add(t, graph="scenarios")
```

---

## Thumbnail
- Left: 3 scenario curves overlaid
- Right: Tornado diagram
- Bottom: "What if?"

---

## Description
```
Compare multiple scenarios in DynaFX — side-by-side trajectories, deviation charts, and tornado diagrams.

Docs: https://achref-yak.github.io/DynaFX/api/dynamics/scenario/
GitHub: https://github.com/Achref-Yak/DynaFX

Challenge: Build 3 scenarios for your model. Which one wins?
```

---

## Pin Comment
```
Challenge #3: Take the population model from last time. Create 3 scenarios:
1. Baseline (birth_rate=0.03, death_rate=0.01)
2. Baby Boom (birth_rate=0.05)
3. Pandemic (death_rate=0.04)

Which scenario has the highest final population? Comment your answer!
```
