# Causal Analysis

**What it is:** Trace causes and effects through model structure — understand *why* something happens.

**When to use it:** When you need to understand the causal relationships in your model.

**Quick start:**
```python
from dynafx.dynamics import causal_trace, causes_strip, detect_feedback_loops

# Trace causes and effects
trace = causal_trace(model, "Inventory", result)
print(trace.cause_tree)
print(trace.effect_tree)

# Decompose a variable into factors
strip = causes_strip(model, "Inventory", result)
print(strip.factors)

# Find feedback loops
loops = detect_feedback_loops(model)
print(loops)
```

---

## Import

```python
from dynafx.dynamics import causal_trace, causes_strip, causes_tree, effects_tree, detect_feedback_loops, loops_for_variable
```

## Functions

### `causal_trace(model, var, state)`

Combined cause/effect analysis.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `SysdModel` | — | The model (required) |
| `var` | `str` | — | Variable name (required) |
| `state` | `SysdModelResult` | — | Simulation result (required) |

**Returns:** `CausalTrace` with `.cause_tree` and `.effect_tree`

---

### `causes_strip(model, var, state)`

Decompose a variable's value into contributing factors.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `SysdModel` | — | The model (required) |
| `var` | `str` | — | Variable name (required) |
| `state` | `SysdModelResult` | — | Simulation result (required) |

**Returns:** `CausalStrip` with `.factors`

---

### `causes_tree(model, var, state)`

Walk upstream dependencies recursively.

---

### `effects_tree(model, var, state)`

Walk downstream to find all affected variables.

---

### `detect_feedback_loops(model)`

Find reinforcing/balancing loops.

**Returns:** `list[FeedbackLoop]`

---

### `loops_for_variable(analysis, var)`

Loops touching one variable.

---

### `get_dependencies(model)`

Analyze a model and return the dependency graph — which variables reference which.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `SysdModel` | — | The model to analyze (required) |

**Returns:** `dict[str, tuple[str, set[str]]]` — maps each variable name to `(expression_string, set_of_referenced_variable_names)`.

- Stocks map to their flow names (no expression string).
- Flows and auxiliaries map to the variables referenced in their expressions.

**Example:**
```python
from dynafx.dynamics import SysdModel, get_dependencies

model = SysdModel(dt=1.0, t_span=(0, 100))
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "200")
    s.outflow("demand", "inventory * 0.1")
model.aux("turnover", "demand / inventory")

deps = get_dependencies(model)
# {"Inventory": ("", {"supply", "demand"}),
#  "supply":    ("200", set()),
#  "demand":    ("inventory * 0.1", {"inventory"}),
#  "turnover":  ("demand / inventory", {"demand", "inventory"})}
```

Used internally by `SysdModel.to_dict()` to build the edge list.

---

## See Also

- [`SysdModel`](SysdModel.md) — The model to analyze
- [Tutorial 10: Publishing Results](../../tutorials/10-publishing-results.md) — Analysis deep dive
