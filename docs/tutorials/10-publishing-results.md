# Tutorial 10 — Publishing Results

A research result is reproducible numbers plus an audit trail. DynaFX gives
you optimization, causal anatomy, feedback detection, grading, provenance, and
plotting — the six pieces a citable twin study needs.

## Optimization with linear programming

`lp_minimize`/`lp_maximize` wrap `scipy.optimize.linprog`. Minimize cost
`c·x` subject to `A_ub x ≤ b_ub` and `bounds`:

```python
from dynafx.dynamics import lp_minimize, lp_maximize

# Two allocation decisions, x0 and x1
result = lp_minimize(
    c=[2.0, 3.0],              # minimize 2·x0 + 3·x1
    A_ub=[[-1.0, -1.0], [1.0, 1.0]],  # x0 + x1 >= 6 (as -x0 - x1 <= -6), and x0 + x1 <= 10
    b_ub=[-6.0, 10.0],
    bounds=[(0, None), (0, None)],
)
print(result.x, result.objective_value)   # [6.0, 0.0], 12.0 — cheapest mix
```

`kb_lp_minimize` pulls coefficients straight from the knowledge base — each
row is a SPARQL query returning a list of floats:

```python
from dynafx.dynamics import kb_lp_minimize

LP_NS = "http://ex.org/lp/"

def _lp(name): return NamedNode(f"{LP_NS}{name}")

kb = TripleStore()
# objective: minimize 2.0*x0 ; bounds: 0 <= x0 <= 10
kb.add(Triple(_lp("obj_x0"), _lp("coeff"), Literal(2.0)), graph="lp")
kb.add(Triple(_lp("obj_x0"), _lp("rowIndex"), Literal(0)), graph="lp")
kb.add(Triple(_lp("b_x0"), _lp("lo"), Literal(0)), graph="lp")
kb.add(Triple(_lp("b_x0"), _lp("hi"), Literal(10)), graph="lp")

result = kb_lp_minimize(
    kb,
    c_query="SELECT ?v WHERE { ?o <http://ex.org/lp/coeff> ?v . ?o <http://ex.org/lp/rowIndex> ?i . } ORDER BY ?i",
    bounds_query="SELECT ?lo ?hi WHERE { ?b <http://ex.org/lp/lo> ?lo . ?b <http://ex.org/lp/hi> ?hi . } ORDER BY ?lo",
    var_count=1,
)
print(result.x, result.objective_value)   # coefficients read from the KB
```

## Causal anatomy

`causes_strip` decomposes a variable's value into contributing factors at a
given state; `causal_trace` bundles causes, effects, and the strip:

```python
from dynafx.dynamics import causal_trace, causes_strip

model = parse_sysd("""
DemandPull
dt 0.5
from 0 to 20

stock "Inventory": 100
  + "orders": pull * 10
  - "usage": demand

aux "demand": 8
aux "pull": 1.0
""")

state = {s.name: s.initial for s in model.stocks}
strip = causes_strip(model, "Inventory", state)     # CausalStrip object
print([f["name"] for f in strip.factors])

trace = causal_trace(model, "Inventory", state)     # dict of to_dict() trees
print(list(trace["causes"].keys()))
```

`detect_feedback_loops` finds reinforcing/balancing cycles:

```python
from dynafx.dynamics import detect_feedback_loops

loops = detect_feedback_loops(model)
for loop in loops.loops:
    print(loop.name, loop.polarity)     # e.g. ('balancing' | 'reinforcing')
```

## Provenance

`record_provenance` turns a completed run into an RDF audit entity with
immutable triples — the reproducible artifact:

```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore, NamedNode, Literal, Triple

store = TripleStore()
bridge = KBSimBridge(store)

result = model.simulate()
run = bridge.record_provenance(
    result, params={"pull": 1.0}, graph="provenance",
    extra_annotations=[
        Triple(NamedNode("http://ex.org/run"), NamedNode("http://ex.org/note"),
               Literal("baseline")),
    ],
)
print(run.iri)   # a fresh NamedNode for this run
```

## Grading against queries

`grade_queries` scores a store against ASK queries with
`(query, var, threshold, weight)` spec tuples and returns per-query verdicts:

```python
from dynafx.bridge import grade_queries

grade_specs = [
    ("SELECT ?v WHERE { <http://ex.org/Portfolio> <http://ex.org/completionPct> ?v }",
     "v", 0.75, 0.6),
]
grades = grade_queries(grade_specs, store)
print(grades)   # {"0": 0.8} — score in [0, 1]
```

## Export and plotting

`SysdModelResult.plot()` writes a matplotlib figure; `to_json` serializes the
`Graph` model for the appendix:

```python
result.plot("output.png", stocks=["Inventory"], subplots=False)
decomposer = model.to_decomposer()          # SystemDecomposer
graph_json = decomposer.graph.to_json()     # reproducible model artifact
```

## The reproducibility checklist

1. **Seed every stochastic stage** — `seed=` on sensitivity/sobol/prcc.
2. **Save the run entity** — `record_provenance` output is your audit trail.
3. **Save the model JSON** — `model.graph.to_json()`.
4. **State the version** — run your pipeline with a pinned commit; cite it.
5. **Ship the data + mappings** — CSVs and YAMLs under `data/` (see Tutorial 5).

## Wrap-up

You now have the full loop: model (Tutorials 1–4) → knowledge (5–6) → twin
(7) → scenarios (8) → ontology (9) → reproducible publication (10). See
[Examples](../examples.md) for the complete flagship twin wired this way.
