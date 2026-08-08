# ClosedLoopReasoner

**What it is:** Orchestrates multi-pass simulate → grade → nudge → re-simulate cycles. Each pass extracts KB facts, runs simulation, records evidence, and grades results.

**When to use it:** When you need iterative refinement — running simulation multiple times, grading each run, and adjusting parameters until targets are met.

**Quick start:**
```python
from dynafx.bridge import ClosedLoopReasoner, KBSimBridge, ReasoningPass

bridge = KBSimBridge(store)
passes = [
    ReasoningPass(
        name="baseline",
        claim_map=[...],
        evidence_map=[...],
        grade_queries=[("SELECT ?v WHERE { ... }", "v", 0.5, 0.1)],
    ),
]
reasoner = ClosedLoopReasoner(bridge, model, passes)
result = reasoner.run()
```

---

## Import

```python
from dynafx.bridge import ClosedLoopReasoner, ReasoningPass, ClosedLoopResult
```

## Classes

### ReasoningPass

A single pass in a closed-loop pipeline.

#### Constructor

```python
ReasoningPass(
    name: str,                              # Pass name
    claim_map: list[tuple],                 # KB → sim params
    evidence_map: list[tuple],              # sim → KB evidence
    params_override: dict = {},             # Override params
    grade_queries: list[tuple] = [],        # SPARQL grading queries
    param_nudges: dict = {},                # Adjustments between passes
    grade_update: Callable = None,          # Callback after grading
    max_belief_graph: str = None,           # Graph to read from
)
```

#### Grade Queries Format

Each query is: `(sparql_string, variable_name, threshold, penalty_if_fail)`

```python
grade_queries=[
    ("PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:score ?v }", "v", 0.5, 0.1),
]
```

---

### ClosedLoopResult

Result of a closed-loop run.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `passes` | `list[ReasoningPass]` | The passes that were run |
| `results` | `list[SysdModelResult]` | Results from each pass |
| `grades` | `list[dict[str, float]]` | Grades from each pass |
| `run_nodes` | `list[NamedNode]` | Provenance run entities |
| `final_params` | `dict[str, Any]` | Final parameter values |
| `evidence_added` | `int` | Total evidence triples added |

#### Methods

```python
result.summary()  # Returns dict with summary info
```

---

## ClosedLoopReasoner

### Constructor

```python
ClosedLoopReasoner(
    bridge: KBSimBridge,        # The KB-sim bridge
    model: SysdModel,           # The simulation model
    passes: list[ReasoningPass], # List of passes to run
    evidence_graph: str = "simulation",
    provenance_graph: str = "provenance",
    param_default: float = 0.5,
)
```

### Methods

#### `run()`

Execute the closed-loop pipeline.

**Returns:** `ClosedLoopResult`

**Example:**
```python
reasoner = ClosedLoopReasoner(bridge, model, passes)
result = reasoner.run()

print(f"Completed {len(result.passes)} passes")
print(f"Added {result.evidence_added} evidence triples")
print(f"Final params: {result.final_params}")
```

---

## Common Patterns

### Pattern 1: Basic Closed Loop

```python
from dynafx.bridge import ClosedLoopReasoner, KBSimBridge, ReasoningPass

bridge = KBSimBridge(store)

passes = [
    ReasoningPass(
        name="baseline",
        claim_map=[
            (NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}reliability"), None, "reliability"),
        ],
        evidence_map=[
            ("Inventory", NamedNode(f"{EX}portfolio"), NamedNode(f"{EX}score"), score_fn),
        ],
        grade_queries=[
            ("PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:score ?v }", "v", 0.5, 0.1),
        ],
    ),
]

reasoner = ClosedLoopReasoner(bridge, model, passes)
result = reasoner.run()
```

### Pattern 2: Multi-Pass with Nudges

```python
passes = [
    ReasoningPass(
        name="initial",
        claim_map=[...],
        evidence_map=[...],
        grade_queries=[...],
    ),
    ReasoningPass(
        name="refined",
        claim_map=[...],
        evidence_map=[...],
        param_nudges={"reliability": 0.1},  # Nudge after first pass
    ),
]
```

### Pattern 3: With Grade Update Callback

```python
def update_params(grades, store):
    # If grade is low, adjust params
    if grades.get("grade_0", 0) < 0.3:
        return {"reliability": 0.9}
    return {}

passes = [
    ReasoningPass(
        name="adaptive",
        claim_map=[...],
        evidence_map=[...],
        grade_update=update_params,
    ),
]
```

---

## See Also

- [`KBSimBridge`](KBSimBridge.md) — The underlying bridge
- [`CognitiveOrchestrator`](CognitiveOrchestrator.md) — Event-driven alternative
- [Tutorial 7: Closed-Loop Simulation](../../tutorials/07-closed-loop-simulation.md) — Full walkthrough
