# DynaFX

Multi-paradigm simulation (**SD + ABM + DES**) with cognitive reasoning — knowledge graphs, confidence grading, and argumentation for simulation-driven decisions.

---

## Why DynaFX Exists

Modern enterprises are not just physical systems; they are *knowledge systems*. Decisions are made against a backdrop of contracts, suppliers, risk facts, policies, and obligations — most of which live in documents and databases, not in differential equations. Classical simulation toolkits model the physics of a system but cannot read its knowledge. Classical knowledge engines can reason about facts but cannot *run the system forward in time*.

DynaFX exists to close that gap. It treats the knowledge base and the simulation as **one living system**: enterprise facts are ingested into a knowledge graph, a multi-paradigm simulation is steered by those facts, and the results flow back into the knowledge graph as evidence — which then triggers rules, optimization, and the next round of reasoning.

The result is a digital twin that does not merely mirror an asset. It **senses** its enterprise context, **learns** from every run, **foresees** the future through scenario analysis, and **acts** through policies and optimization. In the vocabulary of the Digital Twin Consortium, this is a *cognitive digital twin*: a twin that learns at run time, foresees the future, and acts accordingly.

---

## What Makes DynaFX Different

- **Model-based reasoning, not just data-driven ML.** Most "cognitive" digital twin work couples a twin to machine-learned anomaly/RUL models. DynaFX instead reasons with *explicit symbolic models*: RDF/OWL semantics, SPARQL queries, production rules, and causal structure — all auditable and reproducible. This is complementary to, and composable with, data-driven methods.
- **Three simulation paradigms under one roof.** System Dynamics (aggregate stocks and flows), Agent-Based Modeling (heterogeneous actors and strategies), and Discrete Event Simulation (queues, resources, schedules) share a single state namespace and run together in one model file. Each paradigm is right for a different kind of question; DynaFX lets you ask them all at once.
- **The knowledge graph is live, not decorative.** Simulation models execute `KB_QUERY` / `KB_ASSERT` builtins *mid-flight*: they read knowledge-graph facts each timestep and write observations back. The twin's dynamics are numerically steered by its ontology.
- **A closed learning loop.** `KBSimBridge` extracts KB facts into parameters, the simulation runs, and results return as evidence triples. `ClosedLoopReasoner` drives simulate → grade → nudge → re-simulate cycles until targets are met.
- **A research platform, not an appliance.** Everything is a Python API with extension points — custom builtins, a plugin registry, model factories, and a fully self-contained RDF/SPARQL engine with zero external dependencies.

---

## Architecture at a Glance

```mermaid
graph LR
    KB[Knowledge Graph<br/>RDF/OWL + SPARQL + rules] --> B[KBSimBridge]
    B --> SIM[Simulation<br/>SD · ABM · DES]
    SIM --> EV[Evidence triples]
    EV --> KB
```

### Simulate a model

```python
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, XSD_DOUBLE, Triple
from dynafx.dynamics import SysdModel, StockDef, FlowDef, AuxDef
from dynafx import KBSimBridge

epc = lambda x: NamedNode("http://epc.org/" + x)

# 1. Knowledge: supplier reliability is a KB fact.
store = TripleStore()
store.add(Triple(epc("Portfolio"), epc("aggregateSupplierReliability"),
                 Literal("0.82", datatype=XSD_DOUBLE)), "enterprise")

# 2. Simulation: an inventory model gated by that reliability.
model = SysdModel(
    stocks=[StockDef(name="Inventory", initial=1000, flows=[
        FlowDef(name="supply", direction="+", expr="desired * reliability"),
    ])],
    aux_vars=[AuxDef(name="desired", expr="500"),
              AuxDef(name="reliability", expr="kb_reliability")],
    dt=1.0,
)

# 3. Bridge: pull the KB fact into a parameter.
bridge = KBSimBridge(store)
params = bridge.params_from_kb([
    (epc("Portfolio"), epc("aggregateSupplierReliability"), None, "kb_reliability"),
])
result = model.simulate(params=params, method="euler", dt=1.0)
print(result.values["Inventory"][-1])
```

### Query a knowledge graph with confidence

```python
from dynafx import TripleStore, parse_turtle, cumulative_fusion, grade_query

store = TripleStore()
for t in parse_turtle(source_a).triples():
    store.add(t, graph="alpha")
for t in parse_turtle(source_b).triples():
    store.add(t, graph="bravo")

fused = cumulative_fusion(store, ["alpha", "bravo"])
result = grade_query(fused, "SELECT ?revenue WHERE { ?s :revenue ?revenue }")
print(f"Confidence: {result.confidence:.2f}")
```

## Features

| Paradigm | Description |
|----------|-------------|
| **System Dynamics** | Stock/flow models, Vensim-style DSL, RK4/Euler integration, submodels, unit checking |
| **Cognitive Reasoning** | RDF triple store, SPARQL, RDFS/OWL inference, KBT source trust, argumentation, SL fusion |
| **Agent-Based** | Strategies, rules, message passing, strategy switching with cooldown |
| **Discrete Event** | Queues, resources, event-driven simulation, DES clock |
| **Knowledge Bridge** | `KB_QUERY` / `KB_ASSERT` in simulation — models query and update the knowledge graph at runtime |

## Installation

```bash
git clone https://github.com/Achref-Yak/DynaFX.git
cd DynaFX
uv pip install -e ".[all]"       # or: pip install -e ".[all]"
```

Python 3.12+. Full documentation is hosted on GitHub Pages: <https://achref-yak.github.io/DynaFX/>.

---

## Getting Oriented

- `multi_paradigm_student.py` — **Full cognitive pipeline:** KG → trust → argumentation → fusion → SD+ABM+DES → feedback loop
- `cognitive_twin_demo.py` — **Self-healing digital twin:** ABM agents update KB mid-simulation
- `knowledge_fusion_showcase.py` — **End-to-end epistemics:** KBT → argumentation → fusion → SPARQL grading
- `decision_toy.py` — **KB-driven scenario ranking:** 4 scenarios, constraint filtering, goal grading
- `full_showcase.py` — **Feature tour:** 14 simulation capabilities in one script
