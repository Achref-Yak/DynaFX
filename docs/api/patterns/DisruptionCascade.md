# DisruptionCascade

**What it is:** Pre-built model for supply-chain disruption propagation.

**When to use it:** When modeling how disruptions cascade through a supply chain.

**Quick start:**
```python
from dynafx.patterns import DisruptionCascade

model = DisruptionCascade.build(
    stages=["supplier", "manufacturer", "distributor", "retailer"],
    initial_disruption=0.5,
)
result = model.simulate()
```

---

## Import

```python
from dynafx.patterns import DisruptionCascade
```

## Factory Method

### `DisruptionCascade.build(stages, initial_disruption=0.1, propagation_rate=0.8)`

Build a disruption cascade model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stages` | `list[str]` | — | Supply chain stage names (required) |
| `initial_disruption` | `float` | `0.1` | Initial disruption level |
| `propagation_rate` | `float` | `0.8` | How fast disruption spreads |

**Returns:** `SysdModel` — ready to simulate

**Example:**
```python
model = DisruptionCascade.build(
    stages=["raw_material", "assembly", "distribution", "retail"],
    initial_disruption=0.3,
    propagation_rate=0.7,
)
result = model.simulate()
```

---

## See Also

- [`SysdModel`](../dynamics/SysdModel.md) — The underlying model
- [`SignalChain`](SignalChain.md) — Another pattern
