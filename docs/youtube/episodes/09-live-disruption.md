# Video 9: Live Disruption - Changing the World Mid-Simulation

**What it does:** Inject real-time events into a running simulation via knowledge graph updates.

**Duration:** 15-18 minutes

**Hook:** "What if a typhoon hits your supply chain while the simulation is running? Let's find out."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Real disruptions don't wait for you to restart the model."

### 0:30 - Why Mid-Simulation Changes? (2 min)
- Real systems are dynamic
- Events happen during operation
- "The KB is your source of truth - update it, and the simulation responds."

### 2:30 - KB Flags as Switches (3 min)
```python
store.add(Triple(
    NamedNode("http://ex.org/disruption"),
    NamedNode("http://ex.org/active"),
    Literal(False)
), graph="meta")
```
- Boolean flags in the KB
- "Flip a flag, watch the cascade."

### 5:30 - Live KB_QUERY (4 min)
```python
model.aux("disruption_active",
    "KB_QUERY('PREFIX ex: <http://ex.org/> ASK { ex:disruption ex:active true }')")
model.aux("supply_multiplier", "IF(disruption_active, 0.3, 1.0)")
```
- Re-reads KB each timestep
- "The simulation sees the change immediately."

### 9:30 - Injecting Disruption (3 min)
```python
# Mid-run, flip the flag
store.remove(TriplePattern(
    NamedNode("http://ex.org/disruption"),
    NamedNode("http://ex.org/active"),
    Literal(False)
))
store.add(Triple(
    NamedNode("http://ex.org/disruption"),
    NamedNode("http://ex.org/active"),
    Literal(True)
), graph="meta")
```

### 12:30 - Comparing Before/After (2 min)
- Run baseline first
- Inject disruption
- Compare trajectories
- "See the impact in real time."

### 14:30 - Challenge (1 min)
- "Add a disruption flag to your model. What happens when you flip it?"

### 15:30 - Outro (30 sec)
- "Next time: automated decisions - rules that fire themselves."

---

## Code Samples

### Live Disruption Model
```python
from dynafx.dynamics import SysdModel
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, Triple, TriplePattern
from dynafx.bridge import KBSimBridge

store = TripleStore()
store.add(Triple(
    NamedNode("http://ex.org/disruption"),
    NamedNode("http://ex.org/active"),
    Literal(False)
), graph="meta")
store.add(Triple(
    NamedNode("http://ex.org/portfolio"),
    NamedNode("http://ex.org/reliability"),
    Literal(0.85)
), graph="enterprise")

model = SysdModel(dt=1.0, t_span=(0, 100))
model.aux("disruption_active",
    "KB_QUERY('PREFIX ex: <http://ex.org/> ASK { ex:disruption ex:active true }')")
model.aux("supply_multiplier", "IF(disruption_active, 0.3, 1.0)")

with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "500 * supply_multiplier")
    s.outflow("demand", "300")

bridge = KBSimBridge(store)

# Baseline run
result1 = bridge.run_with_kb(model)

# Inject disruption mid-run
store.remove(TriplePattern(
    NamedNode("http://ex.org/disruption"),
    NamedNode("http://ex.org/active"),
    Literal(False)
))
store.add(Triple(
    NamedNode("http://ex.org/disruption"),
    NamedNode("http://ex.org/active"),
    Literal(True)
), graph="meta")

# Run again with disruption
result2 = bridge.run_with_kb(model)

# Compare
print(f"Baseline final: {result1.values['Inventory'][-1]:.0f}")
print(f"Disruption final: {result2.values['Inventory'][-1]:.0f}")
```

---

## Thumbnail
- Left: Storm icon over supply chain
- Right: Before/after curves
- Bottom: "Live disruption"

---

## Description
```
Inject real-time events into DynaFX simulations via knowledge graph updates.

Docs: https://achref-yak.github.io/DynaFX/api/bridge/KBSimBridge/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Challenge #9: Add a disruption flag to your model. Run baseline, then inject disruption.
What is the inventory difference?
```
