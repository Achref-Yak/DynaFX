# DynaFX

Multi-paradigm system dynamics framework (SD + ABM + DES) with an RDF/OWL/SPARQL knowledge engine and a KB↔simulation bridge. Build closed-loop digital twins where enterprise facts drive the simulation and results return as evidence triples.

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

Connect a model to a knowledge graph via `KB_QUERY`:

```python
from dynafx.dynamics import parse_sysd_file
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, XSD_DOUBLE, Triple

epc = lambda x: NamedNode("http://epc.org/" + x)
store = TripleStore()
store.add(Triple(epc("Portfolio"), epc("aggregateSupplierReliability"),
                 Literal("0.82", datatype=XSD_DOUBLE)), "meta")

model = parse_sysd_file("data/models/global_solar_epc.sysd")
result = model.simulate(
    params={"supplier_q": "PREFIX epc: <http://epc.org/> SELECT ?v "
                          "WHERE { epc:Portfolio epc:aggregateSupplierReliability ?v }"},
    kb=store, method="euler", dt=1.0,
)
```

## Features

| Paradigm | Description |
|----------|-------------|
| **System Dynamics** | Stock/flow models, Vensim-style DSL, RK4/Euler integration, submodels, units checking, causal tracing, feedback loops |
| **Agent-Based** | Strategies, rules, message passing, strategy switching with cooldown |
| **Discrete Event** | Queues, resources, event-driven simulation, DES clock |
| **Knowledge Base** | RDF triple store, Turtle parser, SPARQL query, RDFS/OWL RL inference, TBox, production rules, CSV ingestion |
| **Bridge** | `KBSimBridge` — KB→params, mid-flight `KB_QUERY`, post-flight evidence triples, closed-loop reasoning |

## Package Architecture

```
dynafx/
  core/       # Foundational: Graph, Node, Edge, Entity, BFO, SystemDecomposer
  dynamics/   # SD simulation, ABM engine, DES engine, DSL, causal/feedback, optimization
  knowledge/  # RDF triple store, Turtle/SPARQL, inference, TBox, production rules, CSV ingest
  patterns/   # Reusable recipes: SignalChain, DisruptionCascade
  bridge.py   # KBSimBridge — KB↔simulation glue
```

## Further Reading

- [Architecture](architecture.md) — pillars, module dependencies, design decisions
- [Knowledge Base](knowledge.md) — the RDF/OWL/SPARQL engine in detail
- [Digital Twin](digital-twin.md) — the Global Solar EPC supply-chain twin (L1–L5)
- [Examples](examples.md) — the twin, the `.sysd` model library, and patterns
- [Development](development.md) — setup, tests, linting, CI, and docs deployment
