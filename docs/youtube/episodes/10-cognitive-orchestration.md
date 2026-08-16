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
from dynafx.knowledge.model import NamedNode, Literal
from dynafx.knowledge.production import (
    ProductionRule, SparqlCondition, ComparisonCondition, TripleAction,
)

orchestrator = CognitiveOrchestrator(store)

# SparqlCondition binds ?v from the query; ComparisonCondition applies the threshold.
rule = ProductionRule(
    name="reduce_production",
    body=[
        SparqlCondition(
            query="PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }",
        ),
        ComparisonCondition(left="?v", op="<", right=0.5),
    ],
    head=[
        TripleAction(
            subject=NamedNode("http://ex.org/portfolio"),
            predicate=NamedNode("http://ex.org/production_rate"),
            object_=Literal(200),
        ),
    ],
    priority=10,
)
orchestrator.add_rule(rule)
```
- Named rules
- Conditions via SPARQL + comparisons
- Actions that modify the KB

### 6:30 - Rule Priority (2 min)
```python
orchestrator.add_rule(ProductionRule(
    name="emergency",
    body=[
        SparqlCondition(
            query="PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }",
        ),
        ComparisonCondition(left="?v", op="<", right=0.3),
    ],
    head=[TripleAction(
        subject=NamedNode("http://ex.org/portfolio"),
        predicate=NamedNode("http://ex.org/emergency"),
        object_=Literal(True),
    )],
    priority=100,
))
orchestrator.add_rule(ProductionRule(
    name="slow_down",
    body=[
        SparqlCondition(
            query="PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }",
        ),
        ComparisonCondition(left="?v", op="<", right=0.5),
    ],
    head=[TripleAction(
        subject=NamedNode("http://ex.org/portfolio"),
        predicate=NamedNode("http://ex.org/production_rate"),
        object_=Literal(300),
    )],
    priority=50,
))
```
- Multiple rules can fire
- Priority determines order
- "Emergency overrides slow_down."

### 8:30 - Integration with Simulation (4 min)
```python
from dynafx.dynamics import SysdModel
from dynafx.bridge import KBSimBridge

bridge = KBSimBridge(store)

model = SysdModel(dt=1.0, t_span=(0, 100))
model.aux("reliability", "KB_QUERY(reliability_query)")
model.aux("production_rate", "KB_QUERY(prod_rate_query)")

with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "production_rate")
    s.outflow("demand", "300")

# SPARQL queries are passed by name via params — KB_QUERY takes a param name,
# not an inline query string.
result = bridge.run_with_kb(model, params={
    "reliability_query": "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }",
    "prod_rate_query": "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:production_rate ?v }",
})
```

- Orchestrator evaluates rules between timesteps
- Modified KB values feed into next step
- "The simulation adapts in real time."

### 12:30 - ClosedLoopReasoner (2 min)
```python
from dynafx.bridge import ClosedLoopReasoner, ReasoningPass

passes = [
    ReasoningPass(
        name="baseline",
        claim_map=[],  # (s, p, o, param) tuples pulled into params
        evidence_map=[],  # (stock, subject, predicate, score_fn) written back to KB
        # KB_QUERY(params) read their SPARQL from these named params:
        params_override={
            "reliability_query": "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }",
            "prod_rate_query": "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:production_rate ?v }",
        },
    ),
]
reasoner = ClosedLoopReasoner(bridge, model, passes=passes)
result = reasoner.run()
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
from dynafx.knowledge.production import (
    ProductionRule, SparqlCondition, ComparisonCondition, TripleAction,
)

store = TripleStore()
store.add(Triple(
    NamedNode("http://ex.org/portfolio"),
    NamedNode("http://ex.org/reliability"),
    Literal(0.65)
), graph="enterprise")

orchestrator = CognitiveOrchestrator(store)

# Rule: reduce production when reliability is low
orchestrator.add_rule(ProductionRule(
    name="reduce_production",
    body=[
        SparqlCondition(
            query="PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }",
        ),
        ComparisonCondition(left="?v", op="<", right=0.5),
    ],
    head=[TripleAction(
        subject=NamedNode("http://ex.org/portfolio"),
        predicate=NamedNode("http://ex.org/production_rate"),
        object_=Literal(200),
    )],
    priority=10,
))

# Rule: emergency protocol
orchestrator.add_rule(ProductionRule(
    name="emergency",
    body=[
        SparqlCondition(
            query="PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }",
        ),
        ComparisonCondition(left="?v", op="<", right=0.3),
    ],
    head=[
        TripleAction(
            subject=NamedNode("http://ex.org/portfolio"),
            predicate=NamedNode("http://ex.org/emergency"),
            object_=Literal(True),
        ),
        TripleAction(
            subject=NamedNode("http://ex.org/portfolio"),
            predicate=NamedNode("http://ex.org/production_rate"),
            object_=Literal(100),
        ),
    ],
    priority=100,
))
```

### With Simulation
```python
from dynafx.dynamics import SysdModel
from dynafx.bridge import KBSimBridge

model = SysdModel(dt=1.0, t_span=(0, 100))
model.aux("reliability", "KB_QUERY(reliability_query)")
model.aux("production_rate", "KB_QUERY(prod_rate_query)")

with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "production_rate")
    s.outflow("demand", "300")

bridge = KBSimBridge(store)
result = bridge.run_with_kb(model, params={
    "reliability_query": "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:reliability ?v }",
    "prod_rate_query": "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:production_rate ?v }",
})
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
