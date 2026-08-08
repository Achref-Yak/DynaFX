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

## See Also

- [`SysdModel`](SysdModel.md) — The model to analyze
- [Tutorial 10: Publishing Results](../../tutorials/10-publishing-results.md) — Analysis deep dive
