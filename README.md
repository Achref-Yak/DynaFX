# DynaFX

[![CI](https://github.com/Achref-Yak/DynaFX/actions/workflows/ci.yml/badge.svg)](https://github.com/Achref-Yak/DynaFX/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000)](https://docs.astral.sh/ruff/)
[![Pyright](https://img.shields.io/badge/types-pyright-6A1B4D)](https://github.com/microsoft/pyright)
[![pytest](https://img.shields.io/badge/tests-1028-passing-2ea44f)](https://github.com/Achref-Yak/DynaFX/actions)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://achref-yak.github.io/DynaFX/)

DynaFX is a semantic simulation platform for building cognitive digital twins.

It unifies multi-paradigm simulation (System Dynamics, Agent-Based Modeling, and Discrete Event Simulation) with symbolic knowledge representation (RDF/OWL/SPARQL). Build SD + ABM + DES models in Python or a single `.sysd` file, connect them to knowledge graphs via `KB_QUERY`, and close the loop when simulation results come back as evidence triples.

Enabling digital twins that reason over knowledge and continuously adapt through feedback.

The KB and the simulation are one living system: knowledge graph → parameters → multi-paradigm simulation → evidence triples → rules/optimization — closed-loop digital twins from visibility (L1) to autonomy (L5).

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
| Pareto optimization (`pareto_optimize`) | Stable |
| Sensitivity analysis (uniform / normal / lognormal ensembles) | Stable |
| Units checking (`~Unit~` syntax) | Stable |
| Submodels / module include system | Stable |
| CSV import / export (interpolated lookups) | Stable |
| Scenario comparison (tornado, deviation, summary tables) | Stable |
| CompiledSystem caching (~25x speedup via pre-compiled code objects) | Stable |
| BFO-based stock / flow ontology (MATERIAL / INFORMATION / FINANCIAL) | Stable |
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
| `KBSimBridge` — KB-to-simulation parameter extraction + mid-flight `KB_QUERY` + post-flight evidence triples | Stable |
| `ClosedLoopReasoner` — multi-pass reasoning-simulation cycles | Stable |
| KB→Sim→Evidence loop — live KB mutation mid-run + evidence round-trip (L1–L5) | Stable |

### Knowledge Graph Engine (KB)

| Feature | Status |
|---------|--------|
| RDF data model (NamedNode, BlankNode, Literal, Triple) | Stable |
| TripleStore with SPO/POS/OSP indices, named graphs | Stable |
| Turtle / N-Triples parser and serializer | Stable |
| SPARQL query parser and evaluator (SELECT, FILTER, DISTINCT, LIMIT, OFFSET) | Stable |
| RDFS inference (7 rules: subClassOf, subPropertyOf, domain, range, etc.) | Stable |
| OWL RL inference (4 rules: equivalentClass, equivalentProperty, inverseOf, TransitiveProperty) | Stable |
| TBox / OWL2-style type hierarchy (`TypeHierarchy`, `load_tbox`) | Stable |
| Production rules (7 condition types, 5 action types, fire-once, priority) | Stable |
| CSV ingestion (`ingest_csv`) with YAML mapping files | Stable |
| Transaction log (append-only temporal store) | Stable |
| Execution provenance tracking | Stable |

### Patterns

| Feature | Status |
|---------|--------|
| `SignalChain` — leading-indicator → outcome factory | Stable |
| `DisruptionCascade` — supply chain disruption modeling | Stable |

---

## Quick Start

### Install

```bash
git clone https://github.com/Achref-Yak/DynaFX.git
cd DynaFX
uv pip install -e ".[all]"
```

### System Dynamics

Build models in Python (primary path):

```python
from dynafx.dynamics import SysdModel, StockDef, FlowDef, AuxDef

model = SysdModel(
    stocks=[
        StockDef(name="Inventory", initial=1000, flows=[
            FlowDef(name="production", direction="+", expr="desired - Inventory / adj"),
        ]),
    ],
    aux_vars=[
        AuxDef(name="desired", expr="target * demand"),
        AuxDef(name="adj", expr="4"),
    ],
    dt=0.25,
)
result = model.simulate(params={"target": 10})
print(result.values["Inventory"][-1])
```

Or load a `.sysd` model and connect it to a knowledge graph via `KB_QUERY`:

```python
from dynafx.dynamics import parse_sysd_file
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, XSD_DOUBLE, XSD_BOOLEAN, Triple

epc = lambda x: NamedNode("http://epc.org/" + x)
store = TripleStore()
store.add(Triple(epc("Portfolio"), epc("aggregateSupplierReliability"), Literal("0.82", datatype=XSD_DOUBLE)), "meta")
store.add(Triple(epc("GlobalDisruption"), epc("active"), Literal("false", datatype=XSD_BOOLEAN)), "meta")

model = parse_sysd_file("data/models/global_solar_epc.sysd")
result = model.simulate(
    params={
        "disruption_q": "PREFIX epc: <http://epc.org/> ASK { epc:GlobalDisruption epc:active true }",
        "supplier_q":   "PREFIX epc: <http://epc.org/> SELECT ?v WHERE { epc:Portfolio epc:aggregateSupplierReliability ?v }",
        "projects_q":   "PREFIX epc: <http://epc.org/> SELECT ?v WHERE { epc:Portfolio epc:projectsAtRisk ?v }",
    },
    kb=store, method="euler", dt=1.0,
)
profit = result.values["Portfolio_Revenue"][-1] - result.values["Portfolio_Cost"][-1]
print(f"Baseline profit: ${profit:,.0f}K")   # $931,425K
```

### Knowledge Graph Pipeline

```python
from dynafx.knowledge import parse_turtle, RuleEngine, rdfs_rules

# parse_turtle returns a populated TripleStore directly
store = parse_turtle(turtle_string, base_iri="http://example.org/")

# RDFS inference derives new type facts from the ontology
RuleEngine(rdfs_rules()).apply(store)

# ASK / SELECT queries
from dynafx.knowledge import sparql_evaluate, parse_sparql
qr = sparql_evaluate(parse_sparql("SELECT ?v WHERE { ?s <http://epc.org/reliability> ?v }"), store)
```

## Tutorials

Ten verified, runnable walkthroughs — every code block executed against the installed package. Start with [Tutorial 1: Hello World](https://achref-yak.github.io/DynaFX/tutorials/01-hello-world/).

[Tutorials](https://achref-yak.github.io/DynaFX/tutorials/) · [System Dynamics](https://achref-yak.github.io/DynaFX/tutorials/02-system-dynamics/) · [Closed-Loop Twin](https://achref-yak.github.io/DynaFX/tutorials/07-closed-loop-twin/) · [Publishing Results](https://achref-yak.github.io/DynaFX/tutorials/10-publishing-results/)

## Examples

### Supply Chain Digital Twin

`examples/global_solar_epc_twin.py` is a living digital twin of a solar EPC enterprise reasoning through a typhoon-induced port closure disruption. It spans the full L1→L5 decision spectrum in one runnable script:

- **L1 Sense** — ingest 7 EPC enterprise CSVs into named graphs + RDFS inference
- **L2 Assemble** — map KB facts (supplier reliability, projects at risk, capacity) to simulation params
- **L3 Model** — the SD + ABM + DES supply chain twin (`data/models/global_solar_epc.sysd`)
- **L4 Live** — baseline run, inject the typhoon via a KB disruption flag, ABM agents write live KB triples
- **L5 Decide** — evidence round-trip, scenario grading/ranking/filtering, production rules, LP mitigation allocation, causal trace, feedback loops, and provenance

```bash
python examples/global_solar_epc_twin.py
```

The run verifies the closed loop end-to-end: baseline profit **$931,425K**, a 30-day typhoon port closure costs **−$2,795K** (supplier reliability 0.82, 22 projects at risk), and the LP allocates a mitigation budget over the chokepoint port.

All demo resources live under `data/`: datasets (`data/epc_*.csv`), ontology (`data/epc-ontology.ttl`), ingestion mappings (`data/mappings/*.yaml`), and `.sysd` models (`data/models/*.sysd`). Data is regenerable via `scripts/generate_epc_csvs.py` (seed=42).

---

## Tests

```bash
pytest tests/ -q
```

1028 tests covering the SD engine, ABM engine, DES engine, KB engine (RDF model, TripleStore, Turtle, SPARQL, inference, TBox, production rules, transactions, CSV ingestion), sensitivity, optimization, and the cross-paradigm bridge.

---

## Architecture

The `dynafx` package is organized as a shared `core/` substrate, two pillars (`dynamics/`, `knowledge/`), a `bridge.py` connecting them, and `patterns/` on top:

- **`core/`** — foundational data models: `Graph`, `Node`, `Edge`, `Entity`, `WorldRelation`, BFO categories, `SystemDecomposer`
- **`dynamics/`** — SD + ABM + DES simulation engine, causal tracing, feedback detection, sensitivity analysis, LP/Pareto optimization, scenario comparison, units checking, equation compiler with `CompiledSystem` caching
- **`knowledge/`** — RDF triple store, Turtle parser, SPARQL evaluator, RDFS/OWL RL inference, TBox/type hierarchy, production rules, CSV ingestion via YAML mappings, transaction log, execution provenance
- **`patterns/`** — reusable model factories: `SignalChain`, `DisruptionCascade`
- **`bridge.py`** — `KBSimBridge` connects the pillars: KB→param extraction, mid-flight `KB_QUERY`, post-flight evidence triples

---

## Citation

If DynaFX contributes to your research, please cite it:

```bibtex
@software{Yak_DynaFX,
  author       = {Achref Yakdhane},
  title        = {DynaFX: {A} semantic simulation platform for cognitive digital twins},
  year         = {2026},
  url          = {https://github.com/Achref-Yak/DynaFX},
  version      = {0.2.0},
  license      = {MIT},
}
```

Plain text: *Achref Yakdhane. (2026). DynaFX: A semantic simulation platform for cognitive digital twins (Version 0.2.0). https://github.com/Achref-Yak/DynaFX*

A persistent DOI (via Zenodo) will be added here once a release is published. See [citation](docs/citation.md) for details.

---

## Documentation

Full documentation is hosted on GitHub Pages: [achref-yak.github.io/DynaFX](https://achref-yak.github.io/DynaFX/)

- **Tutorials** — 10 verified walkthroughs
- **Concepts** — the mental model, no code
- **Scientific Foundations** — design rationale
- **Open Research Problems** — collaboration opportunities
- **Digital Twin** — flagship closed-loop twin walkthrough

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and pull request guidelines. All contributions are welcome.

---

## License

MIT
