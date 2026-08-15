# Real-World Project: Hospital Emergency Department

**What it does:** Model a hospital ED with patients, staff, and resource constraints.

**Duration:** 20-25 minutes

**Hook:** "ER overcrowding kills. Let's model it."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "ER overcrowding is a national crisis. Simulation can help."

### 0:30 - Problem (2 min)
- Patients arrive unpredictably
- Staff and beds are limited
- Wait times increase mortality
- "How many beds do we need?"

### 2:30 - Step 1: Patient Agents (5 min)
```python
model = SysdModel(dt=0.25, t_span=(0, 24))

with model.agent("Patient", 10) as a:
    a.prop("severity", "random()", min_val=0, max_val=1)
    a.prop("wait_time", 0.0)
    a.rule("arrive", "always", ["wait_time = 0"])
    a.rule("treat", "severity < 0.5 AND beds_available > 0", ["severity = 0", "beds_available -= 1"])
    a.rule("escalate", "severity >= 0.8", ["severity = 1.0"])
```

### 7:30 - Step 2: Queue System (3 min)
```python
model.queue("triage", capacity=50, service_time="5", servers=2)
model.queue("treatment", capacity=20, service_time="30 + random() * 60", servers=5)
model.event("arrival", rate="8", target_queue="triage")
```

### 10:30 - Step 3: Resources (2 min)
```python
model.resource("doctor", capacity=5, cost_per_unit=100)
model.resource("nurse", capacity=10, cost_per_unit=50)
model.resource("bed", capacity=20, cost_per_unit=200)
```

### 12:30 - Step 4: Knowledge Graph (3 min)
```python
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:ed ex:beds 20 ; ex:doctors 5 ; ex:nurses 10 .
    ex:ed ex:arrival_rate 8 .
""")
```

### 15:30 - Step 5: Closed Loop (3 min)
```python
bridge = KBSimBridge(store)
result = bridge.run_with_kb(model)
```

### 18:30 - Analysis (3 min)
```python
result.plot("ed_simulation.png")
stats = result.des_engine.get_all_stats()
print(f"Avg wait time: {stats['triage']['avg_wait']:.1f} min")
print(f"Bed utilization: {stats['treatment']['utilization']:.1%}")
```

### 21:30 - Optimization (2 min)
- "What if we add 2 more beds?"
- "What if we add 1 more doctor?"
- Run scenarios and compare.

### 23:30 - Outro (1 min)
- "This is how hospitals use simulation."

---

## Full Code

```python
from dynafx.dynamics import SysdModel
from dynafx.knowledge import TripleStore, parse_turtle
from dynafx.bridge import KBSimBridge

# Knowledge graph
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:ed ex:beds 20 ; ex:doctors 5 ; ex:nurses 10 .
    ex:ed ex:arrival_rate 8 .
""")

# Model
model = SysdModel(dt=0.25, t_span=(0, 24))

# Queues
model.queue("triage", capacity=50, service_time="5", servers=2)
model.queue("treatment", capacity=20, service_time="30 + random() * 60", servers=5)
model.event("arrival", rate="8", target_queue="triage")

# Resources
model.resource("doctor", capacity=5, cost_per_unit=100)
model.resource("nurse", capacity=10, cost_per_unit=50)
model.resource("bed", capacity=20, cost_per_unit=200)

# Agents
with model.agent("Patient", 10) as a:
    a.prop("severity", "random()", min_val=0, max_val=1)
    a.prop("wait_time", 0.0)
    a.rule("treat", "severity < 0.5 AND beds_available > 0", ["severity = 0"])
    a.rule("escalate", "severity >= 0.8", ["severity = 1.0"])

# Run
bridge = KBSimBridge(store)
result = bridge.run_with_kb(model)

# Analyze
result.plot("ed_simulation.png")
stats = result.des_engine.get_all_stats()
for queue, data in stats.items():
    print(f"{queue}: avg_wait={data['avg_wait']:.1f}, utilization={data['utilization']:.1%}")
```

---

## Thumbnail
- Left: Hospital floor plan
- Right: Wait time chart
- Bottom: "ER simulation"

---

## Description
```
Model a hospital emergency department in DynaFX.

Docs: https://achref-yak.github.io/DynaFX/
GitHub: https://github.com/Achref-Yak/DynaFX
```
