# DynaFX

[![CI](https://github.com/Achref-Yak/reasoning_engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Achref-Yak/reasoning_engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000)](https://docs.astral.sh/ruff/)
[![Pyright](https://img.shields.io/badge/types-pyright-6A1B4D)](https://github.com/microsoft/pyright)

Multi-paradigm system dynamics framework (**SD + ABM + DES**) with an RDF/OWL/SPARQL cognitive reasoning engine and Subjective Logic (SL) confidence grading.

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
| `examples/multi_paradigm_student.py` | 3-pass KG→KBT→Argumentation→bridge→SD+ABM+DES pipeline |
| `examples/knowledge_fusion_showcase.py` | End-to-end KG→KBT→Argumentation→SL fusion→SPARQL |
| `examples/argumentation_showcase.py` | Turtle→named graphs→RDFS inference→argumentation→fusion |
| `examples/supply_chain_demo.py` | 3-echelon supply chain with DELAY3/SMOOTH/SIN/PULSE |
| `examples/supply_chain_paradigm.py` | 7-stock supply chain with DES escalations queue |
| `examples/signal_showcase.py` | 9 leading-indicator domains built with SignalChain |
| `examples/saas_churn_signal.py` | SaaS churn with 43-day leading indicator, 5 scenarios |
| `examples/full_showcase.py` | 14-section feature tour (~12s runtime) |
| `examples/pandemic_response.py` | SD+ABM+DES pandemic model with cohort analysis |

---

## Tests

```bash
pytest tests/ -q
```

1332+ tests covering the SD engine, KB engine, argumentation, KBT, and SL confidence layer.

---

## Architecture

```
┌─────────────────────────┐    ┌──────────────────────────┐
│  System Dynamics (SD)   │    │  Cognitive Reasoning (KB) │
│  .sysd DSL              │    │  RDF/OWL/SPARQL          │
│  SD + ABM + DES         │    │  Named graphs per source  │
│  Causal tracing         │    │  KBT source scoring      │
│  Feedback loops         │    │  Dung argumentation      │
│  Sensitivity analysis   │    │  SL fusion + grading     │
│  Units checking         │    │  Evidence matrix         │
│  Submodels / includes   │    │  RDFS/OWL inference     │
│  Optimization (LP)      │    └──────────────────────────┘
│  Scenario comparison    │
└──────────┬──────────────┘
           │ share: Opinion, cumulative_fusion, EvidenceMatrix
           └──────────────────────────────────────┐
                                  ┌────────────────┴───────────────┐
                                  │  reason/ (SL + Argumentation) │
                                  │  sl/ (opinion algebra)        │
                                  └────────────────────────────────┘
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and pull request guidelines. All contributions are welcome.

---

## License

MIT
