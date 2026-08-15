# Video 11: Signal Chains and Disruption Cascades

**What it does:** Model how impacts propagate through interconnected systems.

**Duration:** 15-18 minutes

**Hook:** "A hurricane doesn't just damage one building. It disrupts power, which disrupts communications, which disrupts supply chains."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Single-stock models are too simple. Real systems cascade."

### 0:30 - SignalChain (2 min)
```python
from dynafx.patterns import SignalChain
chain = SignalChain(["demand", "price", "profit", "investment"])
```
- Ordered stocks
- Each affects the next
- "Demand affects price affects profit affects investment."

### 2:30 - Build a Chain (4 min)
```python
with model.stock("demand", 1000) as s:
    s.inflow("base_demand", "100")
    s.outflow("decay", "demand * 0.1")

with model.stock("price", 100) as s:
    s.inflow("price_change", "(demand - 1000) * 0.01")

with model.stock("profit", 500) as s:
    s.inflow("revenue", "price * demand * 0.001")
    s.outflow("cost", "200")

with model.stock("investment", 100) as s:
    s.inflow("invest_rate", "profit * 0.1")
```

### 6:30 - Propagation Analysis (3 min)
```python
result = model.simulate()
print(chain.propagation_report(result))
print(chain.max_lag(result))
```
- How far does the shock travel?
- What is the delay between cause and effect?

### 9:30 - DisruptionCascade (4 min)
```python
from dynafx.patterns import DisruptionCascade

cascade = DisruptionCascade({
    "supply": {"probability": 0.3, "recovery": 20},
    "logistics": {"probability": 0.5, "recovery": 15},
    "production": {"probability": 0.4, "recovery": 30},
})
cascade.link("supply", "logistics", 0.8)
cascade.link("logistics", "production", 0.6)
```

- Define disruption types
- Link disruptions
- "Supply failure causes logistics failure."

### 13:30 - Running a Cascade (2 min)
```python
events = cascade.simulate(steps=100)
for event in events:
    print(f"Step {event['step']}: {event['type']} - {event['status']}")
```

### 15:30 - Challenge (1 min)
- "Build a 4-stage cascade: raw materials -> production -> distribution -> retail. What breaks first?"

### 16:30 - Outro (30 sec)
- "Next time: the full case study."

---

## Code Samples

### SignalChain
```python
from dynafx.dynamics import SysdModel
from dynafx.patterns import SignalChain

model = SysdModel(dt=0.5, t_span=(0, 100))

with model.stock("demand", 1000) as s:
    s.inflow("base_demand", "100")
    s.outflow("decay", "demand * 0.1")

with model.stock("price", 100) as s:
    s.inflow("price_change", "(demand - 1000) * 0.01")

with model.stock("profit", 500) as s:
    s.inflow("revenue", "price * demand * 0.001")
    s.outflow("cost", "200")

chain = SignalChain(["demand", "price", "profit"])
result = model.simulate()
print(chain.propagation_report(result))
```

### DisruptionCascade
```python
from dynafx.patterns import DisruptionCascade

cascade = DisruptionCascade({
    "supply": {"probability": 0.3, "recovery": 20},
    "logistics": {"probability": 0.5, "recovery": 15},
    "production": {"probability": 0.4, "recovery": 30},
})
cascade.link("supply", "logistics", 0.8)
cascade.link("logistics", "production", 0.6)

events = cascade.simulate(steps=100)
for event in events:
    print(f"Step {event['step']}: {event['type']} - {event['status']}")
```

---

## Thumbnail
- Left: Domino chain
- Right: Cascade visualization
- Bottom: "When one thing breaks everything"

---

## Description
```
Model cascading disruptions in DynaFX - signal chains and disruption cascades.

Docs: https://achref-yak.github.io/DynaFX/api/patterns/SignalChain/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Challenge #11: Build a 4-stage cascade:
raw materials -> production -> distribution -> retail
Each has probability of failure (0.2-0.5).
Link them with cascading probability (0.6-0.8).
What breaks first?
```
