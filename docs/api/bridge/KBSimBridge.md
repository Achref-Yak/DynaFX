# KBSimBridge

**What it is:** The two-way bridge between a knowledge graph (TripleStore) and a simulation (SysdModel). Pull parameters from KB facts, push results back as evidence.

**When to use it:** When you need to connect a knowledge graph to a simulation — either to steer the simulation with KB facts or to record simulation results as evidence.

**Quick start:**
```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, Triple

store = TripleStore()
store.add(Triple(
    NamedNode("http://ex.org/portfolio"),
    NamedNode("http://ex.org/reliability"),
    Literal(0.85)
), graph="enterprise")

bridge = KBSimBridge(store)

# Pull KB fact into simulation param
params = bridge.params_from_kb([
    (NamedNode("http://ex.org/portfolio"),
     NamedNode("http://ex.org/reliability"),
     None, "reliability"),
])
# params = {"reliability": 0.85}

result = model.simulate(params=params, kb=store)
```

---

## Import

```python
from dynafx.bridge import KBSimBridge
```

## Constructor

### `KBSimBridge(store, ns_base="http://dynafx.org/")`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `TripleStore` | — | The knowledge graph (required) |
| `ns_base` | `str` | `"http://dynafx.org/"` | Base namespace for generated IRIs |

**Returns:** `KBSimBridge` instance

**Example:**
```python
from dynafx.bridge import KBSimBridge
from dynafx.knowledge import TripleStore

store = TripleStore()
bridge = KBSimBridge(store)
```

---

## Methods

### `params_from_kb(claim_map, default=0.5, exclude_graphs=None, type_coerce=None)`

Extract KB facts and convert them to simulation parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `claim_map` | `list[tuple[NamedNode, NamedNode, object, str]]` | — | List of (subject, predicate, object, param_name) tuples |
| `default` | `float` | `0.5` | Default value when no matching triple exists |
| `exclude_graphs` | `set[str] \| None` | `None` | Graphs to skip (default: `{"schema", "meta"}`) |
| `type_coerce` | `dict[str, str] \| None` | `None` | Type coercion: `{"param_name": "float"}` |

**Returns:** `dict[str, Any]` — parameter name → value mapping

**Example:**
```python
from dynafx.knowledge.model import NamedNode

EX = "http://ex.org/"

params = bridge.params_from_kb([
    (NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}reliability"), None, "reliability"),
    (NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}demand"), None, "demand"),
], default=0.5)

print(params)  # {"reliability": 0.85, "demand": 200.0}
```

---

### `params_for_class(class_iri, subject_filter=None, naming="class_prefix", exclude_predicates=None, default=0.5)`

Introspect KB to extract numeric params for all instances of a class.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_iri` | `str` | — | Full IRI of the RDF class (required) |
| `subject_filter` | `dict[str, Any] \| None` | `None` | Filter: `{predicate_iri: value}` |
| `naming` | `str` | `"class_prefix"` | Key naming: `"class_prefix"` or `"subject_prefix"` |
| `exclude_predicates` | `set[str] \| None` | `None` | Predicates to skip |
| `default` | `float` | `0.5` | Default value for missing predicates |

**Returns:** `dict[str, float]` — parameter name → value mapping

**Example:**
```python
params = bridge.params_for_class(
    class_iri="http://ex.org/Portfolio",
    subject_filter={"http://ex.org/region": "EU"},
)
# {"Portfolio_reliability": 0.85, "Portfolio_demand": 200.0}
```

---

### `evidence_from_result(result, evidence_map, graph="simulation")`

Convert simulation results to KB triples.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `SysdModelResult` | — | Simulation result (required) |
| `evidence_map` | `list[tuple[str, NamedNode, NamedNode, Callable]]` | — | List of (stock_name, subject, predicate, scoring_fn) |
| `graph` | `str` | `"simulation"` | Graph to store evidence triples |

**Returns:** `list[Triple]` — evidence triples

**Example:**
```python
def score(initial, final):
    return final[-1] / max(initial[0], 1.0)

evidence_map = [
    ("Inventory", NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}inventory_score"), score),
]
triples = bridge.evidence_from_result(result, evidence_map)
for t in triples:
    store.add(t, graph="simulation")
```

---

### `evidence_for_stock(stock_name, subject, predicate, result, method="percentile", threshold=0.5)`

Convert a single simulation stock to a KB triple.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stock_name` | `str` | — | Stock name (required) |
| `subject` | `NamedNode` | — | Subject IRI (required) |
| `predicate` | `NamedNode` | — | Predicate IRI (required) |
| `result` | `SysdModelResult` | — | Simulation result (required) |
| `method` | `str` | `"percentile"` | Scoring: `"percentile"`, `"delta"`, or `"threshold"` |
| `threshold` | `float` | `0.5` | Threshold for `"threshold"` method |

**Returns:** `Triple` — evidence triple

**Example:**
```python
triple = bridge.evidence_for_stock(
    stock_name="Inventory",
    subject=NamedNode(f"{EX}portfolio"),
    predicate=NamedNode(f"{EX}inventory_score"),
    result=result,
    method="percentile",
)
store.add(triple, graph="simulation")
```

---

### `run_with_kb(model, params=None, **sim_kwargs)`

Run a simulation with KB-connected builtins.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `SysdModel` | — | The simulation model (required) |
| `params` | `dict[str, Any] \| None` | `None` | Parameter overrides |
| `**sim_kwargs` | `Any` | — | Additional kwargs for `model.simulate()` |

**Returns:** `SysdModelResult`

**Example:**
```python
result = bridge.run_with_kb(model, params={"reliability": 0.85})
```

---

### `full_roundtrip(model, claim_map, evidence_map, params=None, evidence_graph="simulation", param_default=0.5, **sim_kwargs)`

Full KB → Sim → KB round-trip in one call.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `SysdModel` | — | The simulation model (required) |
| `claim_map` | `list[tuple]` | — | For `params_from_kb()` |
| `evidence_map` | `list[tuple]` | — | For `evidence_from_result()` |
| `params` | `dict[str, Any] \| None` | `None` | Additional parameter overrides |
| `evidence_graph` | `str` | `"simulation"` | Graph for evidence triples |
| `param_default` | `float` | `0.5` | Default for missing KB facts |
| `**sim_kwargs` | `Any` | — | Additional kwargs for simulate |

**Returns:** `tuple[SysdModelResult, list[Triple]]` — (result, evidence_triples)

**Example:**
```python
result, evidence = bridge.full_roundtrip(
    model=model,
    claim_map=[
        (NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}reliability"), None, "reliability"),
    ],
    evidence_map=[
        ("Inventory", NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}inventory"), score),
    ],
)
```

---

### `record_provenance(result, params=None, run_id=None, graph="provenance", extra_annotations=None)`

Store a simulation run's provenance as RDF.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `SysdModelResult` | — | Simulation result (required) |
| `params` | `dict[str, Any] \| None` | `None` | Parameters used for this run |
| `run_id` | `str \| None` | `None` | Unique ID (auto-generated if None) |
| `graph` | `str` | `"provenance"` | Graph to store provenance |
| `extra_annotations` | `list[Triple] \| None` | `None` | Additional triples to add |

**Returns:** `NamedNode` — IRI of the run entity

**Example:**
```python
run_node = bridge.record_provenance(result, params=params)
print(run_node.iri)  # http://dynafx.org/run/abc123
```

---

### `compare_runs(store, provenance_graph="provenance")`

Query provenance graph for run comparison.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `TripleStore` | — | Store with provenance data (required) |
| `provenance_graph` | `str` | `"provenance"` | Graph containing provenance |

**Returns:** `dict` — `{"runs": [...], "stock_deltas": {...}}`

**Example:**
```python
comparison = KBSimBridge.compare_runs(store)
for run in comparison["runs"]:
    print(f"Run: {run['iri']}, params: {run['params']}")
```

---

### `load_queries(yaml_path)` (static)

Load SPARQL queries from a YAML file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `yaml_path` | `str` | — | Path to YAML file (required) |

**Returns:** `dict[str, str]` — query name → SPARQL string

**Example:**
```python
queries = KBSimBridge.load_queries("queries/grade.yaml")
```

---

## Common Workflows

### Workflow 1: KB → Simulation

```python
bridge = KBSimBridge(store)

# Define claim map
claim_map = [
    (NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}reliability"), None, "reliability"),
    (NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}demand"), None, "demand"),
]

# Extract params
params = bridge.params_from_kb(claim_map)

# Run simulation
result = model.simulate(params=params, kb=store)
```

### Workflow 2: Simulation → KB

```python
# Define evidence map
def score(initial, final):
    return final[-1] / max(initial[0], 1.0)

evidence_map = [
    ("Inventory", NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}inventory_score"), score),
]

# Convert result to evidence
triples = bridge.evidence_from_result(result, evidence_map)

# Add to store
for t in triples:
    store.add(t, graph="simulation")
```

### Workflow 3: Full Round-Trip

```python
result, evidence = bridge.full_roundtrip(
    model=model,
    claim_map=claim_map,
    evidence_map=evidence_map,
)
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Default value used | No matching triple in KB | Check `exclude_graphs` and KB data |
| Wrong type | `type_coerce` mismatch | Add `type_coerce={"param": "float"}` |
| Empty evidence | Stock has no values | Check simulation ran successfully |

---

## See Also

- [`SysdModel`](../dynamics/SysdModel.md) — The simulation model
- [`TripleStore`](../knowledge/TripleStore.md) — The knowledge graph
- [`ClosedLoopReasoner`](ClosedLoopReasoner.md) — Multi-pass closed-loop reasoning
- [Tutorial 7: Closed-Loop Simulation](../../tutorials/07-closed-loop-simulation.md) — Full walkthrough
