# Video 1: Your First Simulation in 5 Minutes

**What it does:** Parse a model, simulate it, read the result. Zero setup friction.

**Duration:** 8-10 minutes

**Hook:** "Most simulation frameworks need 30 minutes of setup. DynaFX needs 5 lines of Python."

---

## Script Outline

### 0:00 - Hook (30 sec)
- Open with a working model on screen
- "This is DynaFX. In the next 5 minutes, you'll have your first simulation running."

### 0:30 - Install (1 min)
```bash
pip install https://github.com/Achref-Yak/DynaFX/releases/download/v0.3.0/dynafx-0.3.0-py3-none-any.whl
```
- "One command. No databases, no servers, no config files."

### 1:30 - The 5-Line Model (3 min)
```python
from dynafx.dynamics import parse_sysd

model = parse_sysd("""
    Decay
    dt 1
    from 0 to 10
    stock X: 100
      - Out: X * 0.1
""")

result = model.simulate()
print(result.values["X"][-1])  # ~36.8
```

Walk through each line:
1. Import the parser
2. Write a tiny model in `.sysd` format
3. Parse it
4. Simulate it
5. Read the result

### 4:30 - What Just Happened? (2 min)
- Stock `X` starts at 100
- Outflow removes 10% each step
- After 10 steps: `100 * 0.9^10 ≈ 34.87` (discrete) or `100 * e^(-1) ≈ 36.79` (RK4)
- "The default integration method is RK4 — accurate and fast."

### 5:30 - Plot It (1 min)
```python
result.plot("decay.png")
```
- "One line to visualize."

### 6:30 - Challenge (1 min)
- "Your turn: change the decay rate to 0.05. What do you predict the final value will be? Comment below."

### 7:30 - Outro (30 sec)
- "Next time: we'll build models with Python, not text files. Stocks, flows, parameters — the full System Dynamics toolkit."

---

## Code Samples (Copy-Pasteable)

### Minimal Model
```python
from dynafx.dynamics import parse_sysd

model = parse_sysd("""
    Decay
    dt 1
    from 0 to 10
    stock X: 100
      - Out: X * 0.1
""")
result = model.simulate()
print(result.values["X"][-1])
```

### Growth Model
```python
model = parse_sysd("""
    Growth
    dt 0.1
    from 0 to 20
    stock Population: 1000
      + births: Population * 0.02
      - deaths: Population * 0.01
""")
result = model.simulate()
print(f"Start: {result.values['Population'][0]}")
print(f"End: {result.values['Population'][-1]:.0f}")
```

### With Plotting
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(result.times, result.values['Inventory'], linewidth=2, label='Inventory')
ax.set_xlabel('Time')
ax.set_ylabel('Units')
ax.set_title('Inventory Over Time')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Thumbnail
- Left: DynaFX logo
- Right: Decaying exponential curve
- Bottom: "5 lines of Python"

---

## Description
```
Your first DynaFX simulation in 5 minutes. No setup, no config, no servers.

Install: pip install https://github.com/Achref-Yak/DynaFX/releases/download/v0.3.0/dynafx-0.3.0-py3-none-any.whl
Docs: https://achref-yak.github.io/DynaFX/
GitHub: https://github.com/Achref-Yak/DynaFX

Challenge: Change the decay rate to 0.05 and predict the result. Comment below!
```

---

## Pin Comment
```
Challenge #1: Build a model with 2 stocks and 1 flow between them. 
Stock A starts at 100, flows to Stock B at rate 5 per step. 
What are the final values after 20 steps? Reply with your answer!
```
