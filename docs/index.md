# DynaFX

Multi-paradigm system dynamics framework (SD + ABM + DES) with RDF/OWL/SPARQL cognitive reasoning engine and Subjective Logic confidence grading.

## Quick Start

```bash
pip install dynafx
```

```python
from dynafx import SysdModel, parse_sysd_file

model = parse_sysd_file("model.sysd")
result = model.simulate(t_start=0, t_end=100, dt=0.25, params={"k": 0.5})
print(result.values["MyStock"][-1])
```

## Features

| Paradigm | Description |
|----------|-------------|
| **System Dynamics** | Stock/flow models, Vensim-style DSL, RK4/Euler integration, submodels, unit checking |
| **Agent-Based** | Strategies, rules, message passing, strategy switching with cooldown |
| **Discrete Event** | Queues, resources, event-driven simulation, DES clock |
| **Knowledge Base** | RDF triple store, Turtle parser, SPARQL query, RDFS/OWL RL inference, production rules |
| **Epistemics** | Subjective Logic fusion, KBT source scoring, argumentation frameworks, evidence matrices |

## Package Architecture

```
dynafx/
  core/       # Foundational: Graph, Node, Edge, Opinion, SystemDecomposer
  dynamics/   # SD simulation, ABM engine, DES engine, DSL, causal/feedback analysis
  knowledge/  # RDF triple store, Turtle/SPARQL, inference, production rules, CSV ingest
  epistemics/ # SL fusion, KBT, argumentation, evidence matrices
  patterns/   # Reusable recipes: SignalChain, DisruptionCascade
```
