# DynaFX

Multi-paradigm simulation (**SD + ABM + DES**) with cognitive reasoning — knowledge graphs, confidence grading, and argumentation for simulation-driven decisions.

## Quick Start

```bash
pip install dynafx
```

### Simulate a model

```python
from dynafx import SysdModel, parse_sysd_file

model = parse_sysd_file("model.sysd")
result = model.simulate(t_start=0, t_end=100, dt=0.25, params={"k": 0.5})
print(result.values["MyStock"][-1])
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

## Package Architecture

```
dynafx/
  core/       # Foundational: Graph, Node, Edge, Opinion, SystemDecomposer
  dynamics/   # SD simulation, ABM engine, DES engine, DSL, causal/feedback analysis
  knowledge/  # RDF triple store, Turtle/SPARQL, inference, production rules, CSV ingest
  epistemics/ # SL fusion, KBT, argumentation, evidence matrices
  patterns/   # Reusable recipes: SignalChain, DisruptionCascade
```

## Examples

All 46 examples are in `examples/` with descriptive docstrings. Key ones:

- `multi_paradigm_student.py` — **Full cognitive pipeline:** KG → trust → argumentation → fusion → SD+ABM+DES → feedback loop
- `cognitive_twin_demo.py` — **Self-healing digital twin:** ABM agents update KB mid-simulation
- `knowledge_fusion_showcase.py` — **End-to-end epistemics:** KBT → argumentation → fusion → SPARQL grading
- `decision_toy.py` — **KB-driven scenario ranking:** 4 scenarios, constraint filtering, goal grading
- `full_showcase.py` — **Feature tour:** 14 simulation capabilities in one script
