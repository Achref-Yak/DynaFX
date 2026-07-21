# DynaFX

[![CI](https://github.com/Achref-Yak/reasoning_engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Achref-Yak/reasoning_engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000)](https://docs.astral.sh/ruff/)
[![Pyright](https://img.shields.io/badge/types-pyright-6A1B4D)](https://github.com/microsoft/pyright)
[![pytest](https://img.shields.io/badge/tests-1354-passing-2ea44f)](https://github.com/Achref-Yak/reasoning_engine/actions)

System dynamics, agent-based, and discrete-event simulation framework with an RDF/OWL/SPARQL cognitive reasoning layer. Build SD + ABM + DES models in Python or a single `.sysd` file, connect them to knowledge graphs via `KB_QUERY`, and generate self-contained Plotly dashboards.

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
| Causal tracing (`causes_tree`, `effects_tree`, `causal_trace`, `causes_strip`) | Stable |
| Feedback loop detection (`detect_feedback_loops`, `loops_for_variable`) | Stable |
| Linear programming optimization (scipy.optimize.linprog) | Stable |
| Sensitivity analysis (uniform / normal / lognormal ensembles) | Stable |
| Units checking (`~Unit~` syntax) | Stable |
| Submodels / module include system | Stable |
| CSV import / export (interpolated lookups) | Stable |
| Scenario comparison (tornado, deviation, summary tables) | Stable |
| CompiledSystem caching (~25x speedup via pre-compiled code objects) | Stable |
| Stock / flow ontology (MATERIAL / INFORMATION / FINANCIAL) | Stable |
| Model validation (name resolution, flow conservation, bounds) | Stable |
| Python API model construction (`SysdModel`, `StockDef`, `FlowDef`, `AuxDef`) | Stable |
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
| Topic-based message passing (`SEND`) | Stable |
| Strategy switching with cooldown (`SWITCH_STRATEGY`) | Stable |
| Meta-rules (rules that apply before/after strategy rules) | Stable |
| Perceived inbox (messages aggregated per step) | Stable |
| 4-phase step cycle (Deliver → Decide → Cleanup → Aggregate) | Stable |
| Strategy-scoped rule evaluation | Stable |

### Discrete Event Simulation (DES)

| Feature | Status |
|---------|--------|
| Queues with capacity, service time expressions | Stable |
| Multi-server departure processing | Stable |
| Resource pools with capacity constraints | Stable |
| Utilization tracking (`QueueStats`, `ResourceStats`) | Stable |
| Per-step DES metrics merged into aux namespace | Stable |

### Cross-Paradigm

| Feature | Status |
|---------|--------|
| Unified state dict (SD + ABM + DES write to same namespace) | Stable |
| SD + ABM + DES in a single `.sysd` file | Stable |
| DES queues read ABM agent properties / SD aux values | Stable |
| CLI with `--paradigm` and `--stats` flags | Stable |
| `KBSimBridge` — KB-to-simulation parameter extraction + mid-flight `KB_QUERY` + post-flight evidence triples | Stable |
| `ClosedLoopReasoner` — multi-pass reasoning-simulation cycles | Stable |

### Knowledge Graph Engine (KB)

| Feature | Status |
|---------|--------|
| RDF data model (NamedNode, BlankNode, Literal, Triple) | Stable |
| TripleStore with SPO/POS/OSP indices, named graphs | Stable |
| Turtle / N-Triples parser and serializer | Stable |
| SPARQL query parser and evaluator (SELECT, FILTER, DISTINCT, LIMIT, OFFSET) | Stable |
| RDFS inference (7 rules: subClassOf, subPropertyOf, domain, range, etc.) | Stable |
| OWL RL inference (4 rules: equivalentClass, equivalentProperty, inverseOf, TransitiveProperty) | Stable |
| Production rules (5 condition types, 5 action types, fire-once, priority) | Stable |
| CSV ingestion (`ingest_csv`) with YAML mapping files | Stable |
| Transaction log (append-only temporal store) | Stable |
| Execution provenance tracking | Stable |

### Dashboarding

| Feature | Status |
|---------|--------|
| Self-contained single-file Plotly HTML (no server) | Stable |
| 16-tab solar EPC supply chain dashboard | Stable |
| 12-tab IoT capacity planning dashboard | Stable |
| 11-tab broadband ISP dashboard | Stable |
| Scenario comparison, OAT sensitivity, causal strip, feedback loop charts | Stable |

### Patterns

| Feature | Status |
|---------|--------|
| `SignalChain` — leading-indicator → outcome factory (SaaS churn, telecom SINR, 9 domains) | Stable |
| `DisruptionCascade` — supply chain disruption modeling | Stable |

---

## Quick Start

### Install

```bash
git clone https://github.com/Achref-Yak/reasoning_engine.git
cd reasoning_engine
uv pip install -e ".[all]"
```

### System Dynamics

Build models in Python (primary path):

```python
from dynafx.dynamics import SysdModel, StockDef, FlowDef, AuxDef

model = SysdModel(
    stocks=[StockDef(name="Inventory", initial=1000)],
    flows=[FlowDef(name="production", expr="desired - Inventory / adj")],
    auxes=[
        AuxDef(name="desired", expr="target * demand"),
        AuxDef(name="adj", expr="4"),
    ],
    dt=0.25,
)
result = model.simulate(params={"target": 10})
print(result.values["Inventory"][-1])
```

Or load from a `.sysd` file:

```python
from dynafx.dynamics import parse_sysd_file

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
| `examples/global_solar_epc_dashboard.py` | 16-tab supply chain dashboard: KB→inference→SD+ABM+DES→6 scenarios→OAT→LP optimization |
| `examples/novatel_iot_capacity_dashboard.py` | 12-tab IoT capacity planning: SD+ABM integration, 5 scenarios, 399K devices |
| `examples/atlas_broadband_dashboard.py` | 11-tab ISP dashboard: per-region churn drivers, 3 DES queues, 40 ABM agents |
| `examples/ev_battery_supply_chain.py` | 8-page FPDF report: 6-echelon battery supply chain, 7 scenarios, LP optimization |
| `examples/signal_showcase.py` | 9 leading-indicator domains built with SignalChain (18–111 day lead times) |
| `examples/saas_churn_signal.py` | SaaS churn with 43-day leading indicator, 5 scenarios, 8-param sensitivity |
| `examples/telecom_signal_study.py` | Telecom SINR churn study: 11-page FPDF report, 5 scenarios, causal tracing |
| `examples/argumentation_showcase.py` | Turtle→named graphs→RDFS inference→argumentation→SL fusion→query grading |
| `examples/multi_paradigm_student.py` | KG→KBT→Argumentation→bridge→SD+ABM+DES pipeline |
| `examples/supply_chain_demo.py` | 3-echelon supply chain with DELAY3/SMOOTH/SIN/PULSE |
| `examples/pandemic_response.py` | SD+ABM+DES pandemic model with cohort analysis |

---

## Tests

```bash
pytest tests/ -q
```

1354 tests covering the SD engine, ABM engine, DES engine, KB engine, epistemics (KBT, argumentation, evidence matrix), CSV ingestion, sensitivity, optimization, and production rules.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         dynafx (top-level)                       │
│  KBSimBridge, ClosedLoopReasoner, grade_queries                 │
└──────┬──────────┬──────────────┬──────────────┬─────────────────┘
       │          │              │              │
       ▼          ▼              ▼              ▼
┌────────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐
│  dynamics/ │ │knowledge│ │epistemics│ │  patterns/ │
│  SD + ABM  │ │ RDF    │ │ SL       │ │SignalChain │
│  + DES     │ │ SPARQL │ │ KBT      │ │Disruption  │
│  Causal    │ │ Turtle │ │Argument  │ │Cascade     │
│  Feedback  │ │ Inf    │ │Evidence  │ │            │
│  Opt/LP    │ │ Prod   │ │Matrix    │ │            │
│  Scenario  │ │ Tx/Exec│ │          │ │            │
└──────┬─────┘ └───┬────┘ └────┬─────┘ └──────┬─────┘
       │            │          │               │
       └────────────┴──────────┴───────────────┘
                        │
                   ┌────▼────┐
                   │  core/  │
                   │ Models  │
                   │ Graph   │
                   │ Opinion │
                   └─────────┘
```

- **`core/`** — foundational data models: `Opinion`, `Graph`, `Node`, `Edge`, `SystemDecomposer`
- **`dynamics/`** — SD + ABM + DES simulation engine, causal tracing, feedback detection, sensitivity analysis, LP optimization, scenario comparison, units checking, equation compiler with `CompiledSystem` caching
- **`knowledge/`** — RDF triple store, Turtle parser, SPARQL evaluator, RDFS/OWL RL inference, production rules, CSV ingestion via YAML mappings, transaction log, execution provenance
- **`epistemics/`** — Subjective Logic operators, KBT source reliability (EM), Dung argumentation (grounded/preferred), evidence matrix
- **`patterns/`** — reusable model factories: `SignalChain`, `DisruptionCascade`
- **`bridge.py`** — `KBSimBridge` connects all pillars: KB→param extraction, mid-flight `KB_QUERY`, post-flight evidence triples

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and pull request guidelines. All contributions are welcome.

---

## License

MIT
