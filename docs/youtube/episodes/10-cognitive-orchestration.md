# Video 10: Cognitive Orchestration - Rules That Fire Themselves

**What it does:** Automate simulation decisions with cognitive rules, thresholds, and actions.

**Duration:** 15-18 minutes

**Hook:** "What if your simulation could make its own decisions? Not hard-coded logic - adaptive, rule-based decisions."

---

## Script Outline

### 0:00 - Hook (30 sec)
- "Most simulations run on autopilot. This one has a brain."

### 0:30 - What is CognitiveOrchestration? (2 min)
- Automated decision-making
- Rules that fire based on conditions
- "Not if-then in code - rules in the knowledge graph."

### 2:30 - CognitiveOrchestrator (4 min)
```python
from dynafx.bridge import CognitiveOrchestrator

orchestrator = CognitiveOrchestrator(store)
orchestrator.add_rule(
    name="reduce_production",
    condition="KB_QUERY('PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }') < 0.5",
    actions=[
        "SET_KB('http://ex.org/production_rate', 200)",
    ],
    priority=10
)
```
- Named rules
- Conditions as SPARQL or expressions
- Actions that modify the KB

### 6:30 - Rule Priority (2 min)
```python
orchestrator.add_rule("emergency", "reliability < 0.3", ["SET_KB('http://ex.org/emergency', true)"], priority=100)
orchestrator.add_rule("slow_down", "reliability < 0.5", ["SET_KB('http://ex.org/production_rate', 300)"], priority=50)
```
- Multiple rules can fire
- Priority determines order
- "Emergency overrides slow_down."

### 8:30 - Integration with Simulation (4 min)
```python
model = SysdModel(dt=1.0, t_span=(0, 100))
model.aux("reliability", "KB_QUERY('PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }')")
model.aux("production_rate", "KB_QUERY('PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:production_rate ?v }')")

with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "production_rate")
    s.outflow("demand", "300")
```

- Orchestrator evaluates rules between timesteps
- Modified KB values feed into next step
- "The simulation adapts in real time."

### 12:30 - ClosedLoopReasoner (2 min)
```python
from dynafx.bridge import ClosedLoopReasoner

reasoner = ClosedLoopReasoner(bridge=bridge, orchestrator=orchestrator)
result = reasoner.run(model)
```
- Combines bridge + orchestrator
- "One command, full loop."

### 14:30 - Challenge (1 min)
- "Add a rule: if inventory < 200, increase production by 50%."

### 15:30 - Outro (30 sec)
- "Next time: provenance - tracking every fact."

---

## Code Samples

### Basic Cognitive Orchestration
```python
from dynafx.bridge import CognitiveOrchestrator
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, Triple

store = TripleStore()
store.add(Triple(
    NamedNode("http://ex.org/portfolio"),
    NamedNode("http://ex.org/reliability"),
    Literal(0.65)
), graph="enterprise")

orchestrator = CognitiveOrchestrator(store)

# Rule: reduce production when reliability is low
orchestrator.add_rule(
    name="reduce_production",
    condition="reliability < 0.5",
    actions=["SET_KB('http://ex.org/production_rate', 200)"],
    priority=10
)

# Rule: emergency protocol
orchestrator.add_rule(
    name="emergency",
    condition="reliability < 0.3",
    actions=[
        "SET_KB('http://ex.org/emergency', true)",
        "SET_KB('http://ex.org/production_rate', 100)",
    ],
    priority=100
)
```

### With Simulation
```python
from dynafx.dynamics import SysdModel
from dynafx.bridge import KBSimBridge, ClosedLoopReasoner

model = SysdModel(dt=1.0, t_span=(0, 100))
model.aux("reliability", "KB_QUERY('PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }')")
model.aux("production_rate", "KB_QUERY('PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:production_rate ?v }')")

with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "production_rate")
    s.outflow("demand", "300")

bridge = KBSimBridge(store)
orchestrator = CognitiveOrchestrator(store)

reasoner = ClosedLoopReasoner(bridge=bridge, orchestrator=orchestrator)
result = reasoner.run(model)
```

---

## Thumbnail
- Left: Rule flowchart
- Right: Adaptive curve
- Bottom: "Decisions that fire themselves"

---

## Description
```
Automate simulation decisions with cognitive rules in DynaFX.

Docs: https://achref-yak.github.io/DynaFX/api/bridge/CognitiveOrchestrator/
GitHub: https://github.com/Achref-Yak/DynaFX
```

---

## Pin Comment
```
Challenge #10: Add a rule: if inventory < 200, increase production by 50%.
What happens when demand spikes?
```
