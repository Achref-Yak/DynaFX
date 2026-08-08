# API Reference

Welcome to the DynaFX API reference. This guide is designed for **Python developers new to DynaFX** — every page includes copy-pasteable examples, clear parameter tables, and troubleshooting tips.

## What is DynaFX?

DynaFX is a multi-paradigm simulation framework that combines:

- **System Dynamics** — stocks, flows, and feedback loops
- **Agent-Based Modeling** — heterogeneous actors with behaviors
- **Discrete Event Simulation** — queues, resources, and schedules
- **Knowledge Graphs** — RDF/OWL/SPARQL for reasoning

All four paradigms share one model and run together in a single simulation.

## Quick Start

```python
from dynafx.dynamics import SysdModel

# 1. Create a model
model = SysdModel(name="hello", dt=0.1, t_span=(0, 50))

# 2. Define stocks and flows
with model.stock("Population", 1000) as s:
    s.inflow("births", "population * 0.02")
    s.outflow("deaths", "population * 0.01")

# 3. Run the simulation
result = model.simulate()

# 4. See the results
print(result.values["Population"][-1])  # final population
result.plot("population.png")
```

## Which Module Do I Need?

| I want to... | Use this module |
|--------------|-----------------|
| Build a simulation model | [`dynafx.dynamics`](dynamics/index.md) |
| Store and query facts as RDF | [`dynafx.knowledge`](knowledge/index.md) |
| Connect a knowledge graph to a simulation | [`dynafx.bridge`](bridge/index.md) |
| Use pre-built simulation patterns | [`dynafx.patterns`](patterns/index.md) |

## Module Overview

### [`dynafx.dynamics`](dynamics/index.md) — Simulation

The core simulation engine. Build models with stocks, flows, agents, and queues.

**Start here:** [`SysdModel`](dynamics/SysdModel.md)

```python
from dynafx.dynamics import SysdModel

model = SysdModel(dt=1.0, t_span=(0, 100))
with model.stock("Inventory", 500) as s:
    s.inflow("supply", "200")
    s.outflow("demand", "150")
result = model.simulate()
```

### [`dynafx.knowledge`](knowledge/index.md) — Knowledge Graphs

An in-memory RDF triple store with SPARQL queries, OWL inference, and production rules.

**Start here:** [`TripleStore`](knowledge/TripleStore.md)

```python
from dynafx.knowledge import TripleStore, parse_turtle

store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:portfolio ex:revenue 950.0 .
    ex:portfolio ex:cost 600.0 .
""")
# Query with SPARQL, add rules, run inference
```

### [`dynafx.bridge`](bridge/index.md) — KB ↔ Simulation Bridge

Connect your knowledge graph to your simulation. Pull parameters from KB facts, push results back as evidence.

**Start here:** [`KBSimBridge`](bridge/KBSimBridge.md)

```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore

store = TripleStore()
# ... add triples ...
bridge = KBSimBridge(store)

# Pull KB facts into simulation params
params = bridge.params_from_kb([
    (subject, predicate, None, "param_name"),
])

# Run simulation with KB-connected builtins
result = model.simulate(params=params, kb=store)

# Push results back as evidence triples
triples = bridge.evidence_from_result(result, evidence_map)
```

### [`dynafx.patterns`](patterns/index.md) — Pre-built Patterns

Ready-made simulation patterns for common scenarios.

**Start here:** [`SignalChain`](patterns/SignalChain.md)

```python
from dynafx.patterns import SignalChain

model = SignalChain.build(
    indicators=["leading_1", "leading_2"],
    outcome="revenue",
    delay=3,
)
```

## Common Workflows

### Workflow 1: Simple SD Model

```python
from dynafx.dynamics import SysdModel

model = SysdModel(dt=0.5, t_span=(0, 50))
with model.stock("Cash", 10000) as s:
    s.inflow("revenue", "sales * price")
    s.outflow("costs", "units * unit_cost")
model.aux("sales", "100")
model.aux("price", "10")
model.aux("units", "80")
model.aux("unit_cost", "6")

result = model.simulate()
result.plot("cash_flow.png", stocks=["Cash"])
```

### Workflow 2: KB-Connected Model

```python
from dynafx.dynamics import SysdModel
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, Triple
from dynafx.bridge import KBSimBridge

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
evidence_map = [("Inventory", subject, predicate, lambda i, f: f[-1])]
triples = bridge.evidence_from_result(result, evidence_map)
```

### Workflow 3: Multi-Paradigm Model

```python
from dynafx.dynamics import SysdModel

model = SysdModel(dt=0.5, t_span=(0, 100))

# System Dynamics layer
with model.stock("Inventory", 500) as s:
    s.inflow("production", "production_rate")
    s.outflow("sales", "demand")

# Agent layer
with model.agent("Customer", 50) as a:
    a.prop("budget", 100, min_val=0)
    a.rule("buy", "budget > price", ["budget -= price"])

# DES layer
model.queue("shipping", capacity=100, service_time="2 + random()")

result = model.simulate()
```

## Installation

```bash
git clone https://github.com/Achref-Yak/DynaFX.git
cd DynaFX
uv pip install -e ".[all]"
```

Python 3.12+ required.

## Next Steps

- **New to DynaFX?** Start with [Tutorial 1: Hello World](../tutorials/01-hello-world.md)
- **Building a simulation?** Read [`SysdModel`](dynamics/SysdModel.md)
- **Using knowledge graphs?** Read [`TripleStore`](knowledge/TripleStore.md)
- **Connecting KB to simulation?** Read [`KBSimBridge`](bridge/KBSimBridge.md)
