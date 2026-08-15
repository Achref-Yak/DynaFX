# Video 8: The Closed Loop - When Knowledge Steers Simulation

**What it does:** Connect KB to simulation - facts drive dynamics, results become evidence.

**Duration:** 18-22 minutes

**Hook:** "This is the video that changes how you think about simulation. Watch the knowledge graph steer the model in real time."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Most simulations are dumb calculators. This one reads facts, reasons about them, and writes results back."

### 0:30 - The Loop (2 min)
- KB facts -> simulation params
- Simulation runs
- Results -> evidence triples back to KB
- "A closed learning loop."

### 2:30 - KBSimBridge Setup (2 min)
```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore

store = TripleStore()
bridge = KBSimBridge(store)
```

### 4:30 - Pre-flight: KB to Params (4 min)
```python
claim_map = [
    (NamedNode("http://ex.org/portfolio"),
     NamedNode("http://ex.org/reliability"),
     None, "reliability"),
]
params = bridge.params_from_kb(claim_map, default=0.5)
```
- Extract facts as parameters
- Default values for missing facts

### 8:30 - Mid-flight: KB_QUERY in Expressions (4 min)
```python
model = SysdModel(dt=1.0, t_span=(0, 100))
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "500 * KB_QUERY('PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }')")
    s.outflow("demand", "300")
```
- Live queries each timestep
- "The KB steers the dynamics numerically."

### 12:30 - Post-flight: Evidence Triples (4 min)
```python
def score(initial, final):
    return final[-1] / max(initial[0], 1.0)

evidence_map = [
    ("Inventory", NamedNode("http://ex.org/portfolio"),
     NamedNode("http://ex.org/health"), score),
]
triples = bridge.evidence_from_result(result, evidence_map)
for t in triples:
    store.add(t, graph="evidence")
```

### 16:30 - Full Round-Trip (2 min)
```python
result, evidence = bridge.full_roundtrip(model, claim_map, evidence_map)
```

### 18:30 - Challenge (1 min)
- "Build a closed-loop model. What happens when you change the KB fact mid-run?"

### 19:30 - Outro (30 sec)
- "Next time: live disruption - changing the world mid-simulation."

---

## Code Samples

### Basic Closed Loop
```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, Triple
from dynafx.dynamics import SysdModel

# 1. Set up knowledge graph
store = TripleStore()
store.add(Triple(
    NamedNode("http://ex.org/portfolio"),
    NamedNode("http://ex.org/reliability"),
    Literal(0.85)
), graph="enterprise")

# 2. Build model
model = SysdModel(dt=1.0, t_span=(0, 100))
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "500 * reliability")
    s.outflow("demand", "300")
model.aux("reliability", "0.85")

# 3. Connect KB to model
bridge = KBSimBridge(store)
params = bridge.params_from_kb([
    (NamedNode("http://ex.org/portfolio"),
     NamedNode("http://ex.org/reliability"),
     None, "reliability"),
])

# 4. Run with KB
result = model.simulate(params=params, kb=store)
```

### KB_QUERY in Expressions
```python
model = SysdModel(dt=1.0, t_span=(0, 100))
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "500 * KB_QUERY('PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }')")
    s.outflow("demand", "300")

result = bridge.run_with_kb(model)
```

### Evidence Round-Trip
```python
def score(initial, final):
    return final[-1] / max(initial[0], 1.0)

evidence_map = [
    ("Inventory", NamedNode("http://ex.org/portfolio"),
     NamedNode("http://ex.org/health"), score),
]

triples = bridge.evidence_from_result(result, evidence_map)
for t in triples:
    store.add(t, graph="evidence")
```

### Full Round-Trip
```python
result, evidence = bridge.full_roundtrip(
    model=model,
    claim_map=[
        (NamedNode("http://ex.org/portfolio"),
         NamedNode("http://ex.org/reliability"),
         None, "reliability"),
    ],
    evidence_map=[
        ("Inventory", NamedNode("http://ex.org/portfolio"),
         NamedNode("http://ex.org/health"), score),
    ],
)
```

---

## Thumbnail
- Left: KB -> Sim -> Evidence arrow diagram
- Right: Live updating curve
- Bottom: "The closed loop"

---

## Description
```
Connect knowledge graphs to simulation in DynaFX - the closed loop.

Docs: https://achref-yak.github.io/DynaFX/api/bridge/KBSimBridge/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Challenge #8: Build a closed-loop model with:
- KB fact: supplier reliability (0.85)
- Model: inventory with supply gated by reliability
- Evidence: inventory health score
Run it, then change reliability to 0.5. What happens?
```
