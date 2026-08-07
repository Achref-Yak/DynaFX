# DynaFX

[![CI](https://github.com/Achref-Yak/DynaFX/actions/workflows/ci.yml/badge.svg)](https://github.com/Achref-Yak/DynaFX/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000)](https://docs.astral.sh/ruff/)
[![Pyright](https://img.shields.io/badge/types-pyright-6A1B4D)](https://github.com/microsoft/pyright)

Multi-paradigm simulation (**SD + ABM + DES**) with a knowledge-graph engine — RDF/OWL/SPARQL semantics, production rules, and closed-loop KB↔simulation reasoning for decision support.

---

## Why DynaFX?

Most simulation tools stop at modeling. DynaFX goes further — your models can **query knowledge graphs at runtime**, **let KB facts steer the dynamics**, and **write simulation results back as evidence** that rules and optimization can act on. It is open-source, Python-native, and designed so that simulation and reasoning are not separate tools but a single connected system.

---

## System Dynamics

DynaFX provides a `.sysd` DSL for building stock-and-flow models with full arithmetic, lookup tables, and comparisons. The engine supports RK4 and Euler integration, automatic topological sorting of auxiliary variables, and higher-order delays (SMOOTH, SMOOTHI, DELAY3, DELAYN, DELAY_FIXED, CONVEY_BATCH). Time functions like PULSE, STEP, RAMP, and NOISE are built in.

## Knowledge Engine

The knowledge engine is built on a full RDF stack: a triple data model (NamedNode, BlankNode, Literal, Triple), a `TripleStore` with SPO/POS/OSP indices and named graphs, and a Turtle/N-Triples parser and serializer. SPARQL queries can be evaluated directly against the store. RDFS inference (7 rules) and OWL RL inference (4 rules) run as forward-chaining passes.

## Agent-Based Modeling

DynaFX supports agent-based modeling with typed properties, rule-based behavior, and a perceive-decide-act cycle. Agents evaluate conditions (comparisons against aux/stock values, or `always`), apply effects (`+=`, `-=`, `*=`, `/=`, absolute `=`), and clamp properties to valid ranges.

Rules are scoped to strategies, and agents can switch strategies mid-simulation with a configurable cooldown. Meta-rules allow behavior that activates before or after the current strategy's rules. Agents communicate via topic-based message passing (`SEND`), and a perceived inbox aggregates messages per step. The 4-step cycle (Deliver → Decide → Cleanup → Aggregate) ensures deterministic execution order. Aggregated metrics are collected per step for analysis.

## Discrete Event Simulation

The DES engine provides queues with capacity limits and service time expressions, multi-server departure processing, and resource pools with capacity constraints. Queue and resource utilization statistics (`QueueStats`, `ResourceStats`) are tracked automatically. Per-step DES metrics are merged into the shared aux namespace, so SD and ABM components can read queue lengths, utilization, and other DES state directly.

## Cross-Paradigm Integration

SD, ABM, and DES share a unified state dictionary — all three paradigms read and write to the same namespace. A single `.sysd` file can contain stocks, flows, agents, queues, and resources. DES queues can read ABM agent properties and SD aux values.

The `KBSimBridge` connects the knowledge graph to the simulation: it extracts parameters from the KB, injects them into the model, and after simulation writes evidence triples back. `KB_QUERY` can be used inside `.sysd` auxiliary expressions and ABM agent rules to read from the knowledge graph at runtime. `KB_ASSERT` allows agents to update the KB mid-simulation. The `ClosedLoopReasoner` orchestrates multi-pass reasoning-simulation cycles where each pass informs the next.

---

## Quick Start

### Install

Download the wheel from [GitHub Releases](https://github.com/Achref-Yak/DynaFX/releases/tag/v0.2.0):

```bash
pip install dynafx-0.2.0-py3-none-any.whl
```

Or install from source:

```bash
git clone https://github.com/Achref-Yak/DynaFX.git
cd DynaFX
uv pip install -e ".[all]"
```

### System Dynamics

```python
from dynafx import parse_sysd_file

model = parse_sysd_file("data/models/global_solar_epc.sysd")
result = model.simulate()

print(result.values["Global_Panel_Supply"][-1])
result.plot("out.png", stocks=["Global_Panel_Supply"])
```

### Knowledge Graph Pipeline

```python
from dynafx import parse_turtle, grade_queries

store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:portfolio ex:revenue 950.0 .
""")

query = "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:revenue ?v }"
grades = grade_queries([(query, "v", 0.5, 0.0)], store)
print(grades)   # {'0': 1.0} — score in [0, 1]
```

## Tests

```bash
pytest tests/ -q
```

Models can **query the knowledge graph at runtime** via `KB_QUERY` and **update it** via `KB_ASSERT` — the simulation and knowledge layers are bidirectionally connected.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and pull request guidelines. All contributions are welcome.

---

## License

MIT
