# dynafx.bridge

**What it is:** The connection layer between knowledge graphs and simulations — pull parameters from KB facts, push results back as evidence, and run closed-loop reasoning cycles.

**When to use it:** When you need to connect a `TripleStore` to a `SysdModel` — either to steer simulation with KB facts or to record results as evidence.

**Quick start:**
```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore

store = TripleStore()
bridge = KBSimBridge(store)

# Pull KB facts into params
params = bridge.params_from_kb(claim_map)

# Run simulation with KB
result = model.simulate(params=params, kb=store)

# Push results back as evidence
triples = bridge.evidence_from_result(result, evidence_map)
```

---

## Import

```python
from dynafx.bridge import KBSimBridge, ClosedLoopReasoner, CognitiveOrchestrator
```

## Core Classes

| Class | What it does | When to use it |
|-------|--------------|----------------|
| [`KBSimBridge`](KBSimBridge.md) | Two-way bridge between KB and simulation | Always — this is your starting point |
| [`ClosedLoopReasoner`](ClosedLoopReasoner.md) | Multi-pass simulate → grade → nudge → re-simulate | When you need iterative refinement |
| [`CognitiveOrchestrator`](CognitiveOrchestrator.md) | Event-driven rule → action orchestration | When you need event-driven reasoning |

## Quick Example

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
    s.inflow("supply", "desired * reliability")
model.aux("desired", "500")

# 3. Connect KB to model
bridge = KBSimBridge(store)
params = bridge.params_from_kb([
    (NamedNode("http://ex.org/portfolio"),
     NamedNode("http://ex.org/reliability"),
     None, "reliability"),
])

# 4. Run with KB
result = model.simulate(params=params, kb=store)

# 5. Push results back
def score(initial, final):
    return final[-1] / max(initial[0], 1.0)

evidence_map = [
    ("Inventory", NamedNode("http://ex.org/portfolio"),
     NamedNode("http://ex.org/inventory_score"), score),
]
triples = bridge.evidence_from_result(result, evidence_map)
```

---

## See Also

- [`KBSimBridge`](KBSimBridge.md) — Full API reference
- [`ClosedLoopReasoner`](ClosedLoopReasoner.md) — Multi-pass reasoning
- [`CognitiveOrchestrator`](CognitiveOrchestrator.md) — Event-driven orchestration
- [Tutorial 7: Closed-Loop Simulation](../../tutorials/07-closed-loop-simulation.md) — Full walkthrough
