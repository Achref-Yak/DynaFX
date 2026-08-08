# dynafx.patterns

**What it is:** Pre-built simulation patterns for common scenarios — factory functions that create ready-to-run models for specific use cases.

**When to use it:** When your simulation fits a common pattern (signal chains, disruption cascades) and you want a quick start without building from scratch.

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
from dynafx.patterns import SignalChain, DisruptionCascade
```

## Available Patterns

| Pattern | What it does | When to use it |
|---------|--------------|----------------|
| [`SignalChain`](SignalChain.md) | Leading-indicator → outcome model | When you have leading indicators that predict an outcome |
| [`DisruptionCascade`](DisruptionCascade.md) | Supply-chain disruption propagation | When disruptions cascade through a supply chain |

---

## See Also

- [`SignalChain`](SignalChain.md) — Full API reference
- [`DisruptionCascade`](DisruptionCascade.md) — Full API reference
- [Examples](../../examples.md) — Sample usage
