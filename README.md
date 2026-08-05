# DynaFX

[![CI](https://github.com/Achref-Yak/reasoning_engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Achref-Yak/reasoning_engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000)](https://docs.astral.sh/ruff/)
[![Pyright](https://img.shields.io/badge/types-pyright-6A1B4D)](https://github.com/microsoft/pyright)

Multi-paradigm simulation (**SD + ABM + DES**) with cognitive reasoning — knowledge graphs, confidence grading, and argumentation for simulation-driven decisions.

---

## Why DynaFX?

Most simulation tools stop at modeling. DynaFX goes further — your models can **query knowledge graphs at runtime**, **fuse uncertain evidence from conflicting sources**, and **grade source trust automatically**.

| Capability | Vensim | AnyLogic | DynaFX |
|---|---|---|---|
| System Dynamics | ✅ | ✅ | ✅ |
| Agent-Based Modeling | ❌ | ✅ | ✅ |
| Discrete Event Simulation | ⚠️ | ✅ | ✅ |
| Knowledge Graph (RDF/OWL/SPARQL) | ❌ | ❌ | ✅ |
| Source trust scoring (KBT) | ❌ | ❌ | ✅ |
| Argumentation & evidence fusion | ❌ | ❌ | ✅ |
| Causal tracing & feedback loops | ✅ | ❌ | ✅ |
| Open source | ❌ | ❌ | ✅ |
| Python-native | ❌ | ❌ | ✅ |

---

## Features

### System Dynamics (SD) Engine

| Feature | Status |
|---------|--------|
| Vensim-like `.sysd` DSL with full arithmetic (MIN, MAX, IF, SMOOTH, lookup tables, comparisons) | Stable |
| RK4 / Euler integration | Stable |
| Aux variables with automatic topo-sort | Stable |
| Higher-order delays (SMOOTH, SMOOTHI, DELAY3, DELAYN, DELAY_FIXED, CONVEY_BATCH) | Stable |
| Time functions (PULSE, STEP, RAMP, NOISE) | Stable |
| Causal tracing (`causes_tree`, `effects_tree`, `causal_trace`) | Stable |
| Feedback loop detection | Stable |
| Linear programming optimization (scipy.optimize.linprog) | Stable |
| Sensitivity analysis (uniform / normal / lognormal ensembles) | Stable |
| Units checking (`~Unit~` syntax) | Stable |
| Submodels / module include system | Stable |
| CSV import / export (interpolated lookups) | Stable |
| Scenario comparison (tornado, deviation, summary tables) | Stable |
| Vensim `.mdl` import | Stable |
| Stock / flow ontology (MATERIAL / INFORMATION / FINANCIAL) | Stable |
| Model validation (name resolution, flow conservation, bounds) | Stable |
| Plotting API (`.plot()`, `.plot_with_bands()`) | Stable |

### Cognitive Reasoning Engine (KB)

| Feature | Status |
|---------|--------|
| RDF data model (NamedNode, BlankNode, Literal, Triple) | Stable |
| TripleStore with SPO/POS/OSP indices, named graphs | Stable |
| Turtle / N-Triples parser and serializer | Stable |
| SPARQL query parser and evaluator | Stable |
| RDFS inference (7 rules) | Stable |
| OWL RL inference (4 rules) | Stable |
| SL confidence layer (`fuse_graphs`, `grade_query`) | Stable |
| Evidence Matrix (L1-distance consensus) | Stable |
| KBT (Knowledge-Based Trust) — source reliability scoring via EM | Stable |
| Dung argumentation framework (grounded / preferred semantics) | Stable |
| Argumentation filter in fusion pipeline | Stable |

### Agent-Based Modeling (ABM)

| Feature | Status |
|---------|--------|
| Agent definitions with typed properties | Stable |
| Rule-based behavior (perceive → decide → act) | Stable |
| Condition evaluation (`always`, comparisons, aux/stocks) | Stable |
| Effects (`+=`, `-=`, `*=`, `/=`, absolute `=`) | Stable |
| Property clamping | Stable |
| Aggregated metrics per step | Stable |

### Discrete Event Simulation (DES)

| Feature | Status |
|---------|--------|
| Queues with capacity, service time expressions | Stable |
| Resource pools with capacity constraints | Stable |
| Utilization statistics | Stable |
| Single-server departure processing | Stable |

### Cross-Paradigm

| Feature | Status |
|---------|--------|
| Unified state dict (SD + ABM + DES write to same state) | Stable |
| SD + ABM + DES in a single `.sysd` file | Stable |
| DES queues read ABM agent properties / SD aux values | Stable |
| CLI with `--paradigm` and `--stats` flags | Stable |

---

## Quick Start

### Install

```bash
pip install dynafx    # once published
```

Or from source:

```bash
git clone https://github.com/Achref-Yak/reasoning_engine.git
cd reasoning_engine
uv pip install -e ".[all]"
```

### System Dynamics

```python
from dynafx.system.dsl import parse_sysd_file

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
from dynafx.knowledge import parse_turtle, TripleStore
from dynafx.epistemics import compute_kbt, build_framework

store = TripleStore()
for t in parse_turtle(turtle_string).triples():
    store.add(t, graph="source_a")

kbt = compute_kbt(store, ["source_a", "source_b"])
af = build_framework(store, ["source_a", "source_b"])
accepted = af.compute_grounded()
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

---

## Examples

| Example | What it shows |
|---------|---------------|
| `examples/multi_paradigm_student.py` | **Full cognitive pipeline:** KG → trust scoring → argumentation → fusion → SD+ABM+DES → feedback loop |
| `examples/cognitive_twin_demo.py` | **Self-healing digital twin:** ABM agents update KB mid-simulation, SD reads it via `KB_QUERY` |
| `examples/decision_toy.py` | **KB-driven scenario ranking:** 4 scenarios, constraint filtering, goal grading |
| `examples/knowledge_fusion_showcase.py` | **End-to-end epistemics:** KBT → argumentation → evidence fusion → SPARQL grading |
| `examples/full_showcase.py` | **Feature tour:** 14 simulation capabilities in one script |
| `examples/supply_chain_bridge.py` | **Enterprise bridge:** KB ↔ simulation with closed-loop reasoning |

---

## Tests

```bash
pytest tests/ -q
```

1332+ tests covering the SD engine, KB engine, argumentation, KBT, and SL confidence layer.

---

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────┐
│  Simulation Engine      │◄───────►│  Cognitive Reasoning     │
│  SD + ABM + DES         │  KB_QUERY / KB_ASSERT           │
│  .sysd DSL              │         │  RDF/OWL/SPARQL          │
│  Causal tracing         │         │  KBT source scoring      │
│  Feedback loops         │         │  Dung argumentation      │
│  Sensitivity analysis   │         │  SL fusion + grading     │
│  Optimization (LP)      │         │  Evidence matrix         │
│  Scenario comparison    │         │  RDFS/OWL inference      │
└─────────────────────────┘         └──────────────────────────┘
              │                                │
              └────────────┬───────────────────┘
                           │
              ┌────────────┴───────────────┐
              │  Shared: Opinion,          │
              │  cumulative_fusion,        │
              │  EvidenceMatrix            │
              └────────────────────────────┘
```

Models can **query the knowledge graph at runtime** via `KB_QUERY` and **update it** via `KB_ASSERT` — the simulation and knowledge layers are bidirectionally connected.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and pull request guidelines. All contributions are welcome.

---

## License

MIT
