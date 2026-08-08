# SignalChain

**What it is:** Pre-built model for leading-indicator → outcome relationships.

**When to use it:** When you have leading indicators that predict an outcome with a delay.

**Quick start:**
```python
from dynafx.patterns import SignalChain

model = SignalChain.build(
    indicators=["leading_1", "leading_2"],
    outcome="revenue",
    delay=3,
)
result = model.simulate()
```

---

## Import

```python
from dynafx.patterns import SignalChain
```

## Factory Method

### `SignalChain.build(indicators, outcome, delay=1)`

Build a signal chain model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `indicators` | `list[str]` | — | Leading indicator names (required) |
| `outcome` | `str` | — | Outcome variable name (required) |
| `delay` | `int` | `1` | Delay in time steps |

**Returns:** `SysdModel` — ready to simulate

**Example:**
```python
model = SignalChain.build(
    indicators=["sentiment", "orders", "inventory"],
    outcome="revenue",
    delay=5,
)
result = model.simulate()
```

---

## See Also

- [`SysdModel`](../dynamics/SysdModel.md) — The underlying model
- [`DisruptionCascade`](DisruptionCascade.md) — Another pattern
