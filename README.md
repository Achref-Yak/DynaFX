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

Models can be validated automatically for name resolution, flow conservation, and bounds. Units can be annotated using the `~Unit~` syntax and checked at compile time. A submodel and module include system allows reusable model components to be composed into larger systems.

For analysis, DynaFX provides causal tracing (upstream causes and downstream effects), feedback loop detection, sensitivity analysis with uniform/normal/lognormal ensembles, and scenario comparison with tornado diagrams and deviation summaries. Linear programming optimization and Pareto optimization are available for resource allocation and multi-objective problems. CSV import and export support interpolated lookups for driving models with external data.

The Python API (`SysdModel`, `StockDef`, `FlowDef`, `AuxDef`) allows programmatic model construction with loops, conditionals, and dynamic parameter injection. A `CompiledSystem` cache provides ~25x speedup via pre-compiled code objects. Models can also be imported from Vensim `.mdl` files. A BFO-based stock/flow ontology classifies flows as MATERIAL, INFORMATION, or FINANCIAL. Plotting is available through `.plot()` and `.plot_with_bands()`.

## Cognitive Reasoning

The knowledge engine is built on a full RDF stack: a triple data model (NamedNode, BlankNode, Literal, Triple), a `TripleStore` with SPO/POS/OSP indices and named graphs, and a Turtle/N-Triples parser and serializer. SPARQL queries can be evaluated directly against the store. RDFS inference (7 rules) and OWL RL inference (4 rules) run as forward-chaining passes.

On top of this sits a confidence and trust layer. Subjective Logic provides opinion algebra for fusing beliefs from multiple sources. The `EvidenceMatrix` computes structured consensus using L1-distance analysis. KBT (Knowledge-Based Trust) automatically scores source reliability using expectation-maximization. A Dung argumentation framework with grounded and preferred semantics resolves conflicts between claims, and an argumentation filter can be applied during fusion to defeat unreliable evidence before it propagates.

This means your simulation models can query the knowledge graph, and the knowledge graph can be populated, updated, and graded — all within the same framework.

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
