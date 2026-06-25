# Cognitive Engine

Multi-paradigm system dynamics framework (SD + ABM + DES) with RDF/OWL/SPARQL cognitive reasoning engine and SL confidence grading.

## Architecture

Two pillars sharing a common Subjective Logic (SL) substrate:

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

## Quick start

```python
from cognitive_engine.system.dsl import parse_sysd_file

model = parse_sysd_file("models/student_math.sysd")
result = model.simulate(params={
    "KG_anxiety_belief": 0.8,
    "KG_attention_belief": 0.85,
    "KG_intervention_active": 0.0,
})
print(result.values["Math_Performance"][-1])
```

Or the KB pipeline:

```python
from cognitive_engine.kb.turtle import parse_turtle
from cognitive_engine.kb.store import TripleStore

store = TripleStore()
for t in parse_turtle(source_turtle).triples():
    store.add(t, graph="source_a")

from cognitive_engine.reason.kbt import compute_kbt
from cognitive_engine.reason.argumentation import build_framework

kbt = compute_kbt(store, ["source_a", "source_b"])
af = build_framework(store, ["source_a", "source_b"])
accepted = af.compute_grounded()
```

## CLI

```bash
python -m cognitive_engine.system simulate models/student_math.sysd
python -m cognitive_engine.system validate models/student_math.sysd
python -m cognitive_engine.system list
```

## Key examples

| Example | What it shows |
|---------|---------------|
| `examples/multi_paradigm_student.py` | KG → KBT → Argumentation → bridge → SD+ABM+DES (3-pass pipeline) |
| `examples/knowledge_fusion_showcase.py` | KG → KBT → Argumentation → SL fusion → SPARQL |
| `examples/argumentation_showcase.py` | Turtle → named graphs → RDFS inference → argumentation → fusion |
| `examples/supply_chain_demo.py` | 3-echelon supply chain with DELAY3/SMOOTH |
| `examples/signal_showcase.py` | 9 leading indicator domains using SignalChain |

## Tests

```bash
pytest tests/ -q
```

1192+ tests covering SD engine, KB engine, argumentation, KBT, and SL confidence.

## License

MIT
