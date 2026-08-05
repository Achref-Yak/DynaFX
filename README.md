# DynaFX

[![CI](https://github.com/Achref-Yak/DynaFX/actions/workflows/ci.yml/badge.svg)](https://github.com/Achref-Yak/DynaFX/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000)](https://docs.astral.sh/ruff/)
[![Pyright](https://img.shields.io/badge/types-pyright-6A1B4D)](https://github.com/microsoft/pyright)

Multi-paradigm simulation (**SD + ABM + DES**) with cognitive reasoning — knowledge graphs, confidence grading, and argumentation for simulation-driven decisions.

---

## Why DynaFX?

Most simulation tools stop at modeling. DynaFX goes further — your models can **query knowledge graphs at runtime**, **fuse uncertain evidence from conflicting sources**, and **grade source trust automatically**. It is open-source, Python-native, and designed so that simulation and reasoning are not separate tools but a single connected system.

---

## System Dynamics

DynaFX provides a Vensim-style `.sysd` DSL for building stock-and-flow models with full arithmetic, lookup tables, and comparisons. The engine supports RK4 and Euler integration, automatic topological sorting of auxiliary variables, and higher-order delays (SMOOTH, SMOOTHI, DELAY3, DELAYN, DELAY_FIXED, CONVEY_BATCH). Time functions like PULSE, STEP, RAMP, and NOISE are built in.

## Cognitive Reasoning

The knowledge engine is built on a full RDF stack: a triple data model (NamedNode, BlankNode, Literal, Triple), a `TripleStore` with SPO/POS/OSP indices and named graphs, and a Turtle/N-Triples parser and serializer. SPARQL queries can be evaluated directly against the store. RDFS inference (7 rules) and OWL RL inference (4 rules) run as forward-chaining passes.

## Agent-Based Modeling

DynaFX supports agent-based modeling with typed properties, rule-based behavior, and a perceive-decide-act cycle. Agents evaluate conditions (comparisons against aux/stock values, or `always`), apply effects (`+=`, `-=`, `*=`, `/=`, absolute `=`), and clamp properties to valid ranges.

Rules are scoped to strategies, and agents can switch strategies mid-simulation with a configurable cooldown. Meta-rules allow behavior that activates before or after the current strategy's rules. Agents communicate via topic-based message passing (`SEND`), and a perceived inbox aggregates messages per step. The 4-step cycle (Deliver → Decide → Cleanup → Aggregate) ensures deterministic execution order. Aggregated metrics are collected per step for analysis.

## Discrete Event Simulation

The DES engine provides queues with capacity limits and service time expressions, multi-server departure processing, and resource pools with capacity constraints. Queue and resource utilization statistics (`QueueStats`, `ResourceStats`) are tracked automatically. Per-step DES metrics are merged into the shared aux namespace, so SD and ABM components can read queue lengths, utilization, and other DES state directly.

## Cross-Paradigm Integration

SD, ABM, and DES share a unified state dictionary — all three paradigms read and write to the same namespace. A single `.sysd` file can contain stocks, flows, agents, queues, and resources. DES queues can read ABM agent properties and SD aux values. The CLI provides `--paradigm` and `--stats` flags to control which engines are active.

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

model = parse_sysd_file("models/student_math.sysd")
result = model.simulate(params={
    "KG_anxiety_belief": 0.8,
    "KG_attention_belief": 0.85,
})

print(f"Final performance: {result.values['Math_Performance'][-1]:.2f}")
result.plot("math_outcome.png", stocks=["Math_Performance"])
```

### Knowledge Graph Pipeline

```python
from dynafx import TripleStore, parse_turtle, cumulative_fusion, grade_queries

store = TripleStore()
for t in parse_turtle(source_a).triples():
    store.add(t, graph="alpha")
for t in parse_turtle(source_b).triples():
    store.add(t, graph="bravo")

fused = cumulative_fusion(store, ["alpha", "bravo"])
result = grade_queries(fused, "SELECT ?revenue WHERE { ?s :revenue ?revenue }")
print(f"Confidence: {result.confidence:.2f}")
```

### CLI

```bash
# Simulate a model
dynafx simulate models/student_math.sysd

# Simulate with ABM and DES stats
dynafx simulate models/pandemic_response.sysd --paradigm all --stats

# Validate a model
dynafx validate models/pandemic_seirvh.sysd

# List available models
dynafx list
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
