# Tutorial 7 — Closed-Loop Simulation

This is the defining pattern of DynaFX. Facts flow **KB → simulation**, and
simulation results flow **simulation → KB** as evidence — a closed loop. The
class that orchestrates this is `KBSimBridge`.

## Step 1 — KB facts become simulation parameters

`params_from_kb` turns matching triples into a parameter dict. Each claim is a
`(subject, predicate, object, param_name)` tuple — pass `None` as the object to
match any value.

```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore, NamedNode, Literal, Triple

kb = TripleStore()
kb.add(Triple(
    NamedNode("http://ex.org/Portfolio"),
    NamedNode("http://ex.org/supplierReliability"),
    Literal(0.1),          # 10% attrition rate
), graph="enterprise")

bridge = KBSimBridge(kb)

claim_map = [
    (NamedNode("http://ex.org/Portfolio"),
     NamedNode("http://ex.org/supplierReliability"),
     None,                      # match any object
     "kb_reliability"),         # parameter name
]
params = bridge.params_from_kb(claim_map, default=0.5, exclude_graphs=set())
print(params)   # {'kb_reliability': 0.1}
```

By default the `schema` and `meta` graphs are excluded (they hold ontology
and derived triples, not source facts).

## Step 2 — simulate with the KB attached

Pass the store as `kb=` and use the parameter where the model needs it:

```python
from dynafx.dynamics import parse_sysd

model = parse_sysd("""
T
dt 1.0
from 0 to 10

stock X: 100
  - Out: X * kb_reliability
""")

result = bridge.run_with_kb(model, params=params)
print(round(result.values["X"][-1], 1))   # ~36.8 (10% attrition over 10 steps)
```

## Step 3 — simulation results return as evidence

`evidence_from_result` converts a trajectory into triples via a scoring
function `(initial_values, final_values) -> number`:

```python
def score(initial, final):
    return min(1.0, max(0.0, final[-1] / 100.0))

evidence_map = [
    ("X",                              # stock name
     NamedNode("http://ex.org/Portfolio"),
     NamedNode("http://ex.org/health"),
     score),
]
triples = bridge.evidence_from_result(result, evidence_map, graph="evidence")
for t in triples:
    kb.add(t, graph="evidence")

print(triples[0].object_.value)   # scored value
print(len(list(kb.triples_in_graph("evidence"))))
```

## Step 4 — the full roundtrip in one call

`full_roundtrip` does claims → params → simulate → evidence:

```python
result, triples = bridge.full_roundtrip(
    model,
    claim_map,
    evidence_map,
    params={},        # extra overrides beyond KB
    kb=kb,            # store to query (and graph for evidence)
)
```

## Step 5 — provenance

`record_provenance` writes an RDF record of the run — parameters, time bounds,
stock start/final values — so every result is auditable:

```python
run = bridge.record_provenance(
    result,
    params=params,
    graph="provenance",
    extra_annotations=[
        Triple(NamedNode("http://ex.org/Portfolio"),
               NamedNode("http://ex.org/health"),
               Literal(result.values["X"][-1] / 100.0)),
    ],
)
print(run.iri)   # e.g. .../run/<uuid>
```

## The full loop

The flagship `examples/global_solar_epc.py` wires all of this together across
the full reasoning loop:

| Stage | Bridge role |
|-------|-------------|
| Sense | CSVs → named-graph KB + RDFS inference |
| Assemble | `params_from_kb` → simulation params |
| Model | SD + ABM + DES model |
| Live | KB disruption flag + agents write triples mid-run |
| Decide | evidence round-trip, scenarios, rules, LP, provenance |

```bash
uv run python examples/global_solar_epc.py
```

## What's next

- Compare alternative futures of the same model in [Scenarios & Sensitivity](08-scenarios-and-sensitivity.md).
- Learn how the KB derives more facts than it was told in [Custom Ontology](09-custom-ontology.md).
