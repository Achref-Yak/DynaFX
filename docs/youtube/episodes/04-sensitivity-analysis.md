# Video 4: Which Parameter Actually Matters?

**What it does:** Find which inputs drive the output using Sobol, Morris, and PRCC methods.

**Duration:** 15-18 minutes

**Hook:** "You have 10 parameters. Which 2 should you focus on? Sensitivity analysis tells you."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Not all parameters are created equal. Some barely matter. Others change everything."

### 0:30 - Why Sensitivity? (2 min)
- You cannot optimize everything - focus on what matters
- Three methods: Sobol (variance-based), Morris (screening), PRCC (correlation)

### 2:30 - SensitivityAnalyzer Setup (2 min)
```python
from dynafx.dynamics import SensitivityAnalyzer
sa = SensitivityAnalyzer(model)
```

### 4:30 - Sobol Analysis (5 min)
```python
result = sa.sobol(
    param_spec={"supply_rate": (100, 400), "demand_rate": (50, 350)},
    output="Inventory",
    n_base=512,
)
print(result.first_order)  # Individual effect
print(result.total_order)  # Including interactions
```

### 9:30 - Morris Screening (3 min)
```python
result = sa.morris(
    param_spec={"supply_rate": (100, 400), "demand_rate": (50, 350)},
    output="Inventory",
)
print(result.mu_star)  # Sensitivity measure
```

### 12:30 - PRCC (2 min)
```python
result = sa.prcc(
    param_spec={"supply_rate": (100, 400), "demand_rate": (50, 350)},
    output="Inventory",
)
print(result.prcc)  # Rank correlation
```

### 14:30 - Ranking (1 min)
```python
print(result.ranking("total_order"))
```

### 15:30 - Challenge (1 min)
- "Run sensitivity on your model. Which parameter matters most?"

### 16:30 - Outro (30 sec)
- "Next time: agents with brains."

---

## Code Samples

### Sobol Analysis
```python
from dynafx.dynamics import SysdModel, SensitivityAnalyzer

model = SysdModel(dt=0.5, t_span=(0, 100))
model.param("supply_rate", 200)
model.param("demand_rate", 150)

with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "supply_rate")
    s.outflow("demand", "demand_rate")

sa = SensitivityAnalyzer(model)
result = sa.sobol(
    param_spec={"supply_rate": (100, 400), "demand_rate": (50, 350)},
    output="Inventory",
    n_base=512,
)

print("First order:", result.first_order)
print("Total order:", result.total_order)
print("Ranking:", result.ranking("total_order"))
```

### Morris Screening
```python
result = sa.morris(
    param_spec={"supply_rate": (100, 400), "demand_rate": (50, 350)},
    output="Inventory",
)
print("Mu star:", result.mu_star)
```

### Visualize
```python
import matplotlib.pyplot as plt

params = list(result.total_order.keys())
values = list(result.total_order.values())

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(params, values, color=['#2196F3', '#FF9800'])
ax.set_xlabel('Sensitivity Index')
ax.set_title('Sensitivity Analysis (Total Order)')
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center')

plt.tight_layout()
plt.show()
```

---

## Thumbnail
- Left: Sensitivity bar chart
- Right: Tornado diagram
- Bottom: "What matters most?"

---

## Description
```
Find which parameters drive your model - Sobol, Morris, and PRCC sensitivity analysis in DynaFX.

Docs: https://achref-yak.github.io/DynaFX/api/dynamics/sensitivity/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Challenge #4: Run sensitivity on your model with 3+ parameters. Which one matters most?
```
