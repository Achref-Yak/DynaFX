# CognitiveOrchestrator

**What it is:** Event-driven orchestration — ingests external events, triggers production rules, executes actions, and records provenance automatically.

**When to use it:** When you need a continuous event-driven reasoning loop — events come in, rules fire, actions execute, and everything is tracked.

**Quick start:**
```python
from dynafx.bridge import CognitiveOrchestrator

orb = CognitiveOrchestrator(store)
orb.add_rule(my_production_rule)
orb.start()

# Ingest an event — everything else fires automatically
orb.ingest_event("ContainerDelayed", {
    "container_id": "C-123",
    "delay_days": 6,
})
```

---

## Import

```python
from dynafx.bridge import CognitiveOrchestrator
```

## Constructor

### `CognitiveOrchestrator(store, bridge=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `TripleStore` | — | The knowledge graph (required) |
| `bridge` | `KBSimBridge \| None` | `None` | Optional bridge for simulation |

**Returns:** `CognitiveOrchestrator` instance

---

## Methods

### `add_rule(rule)`

Register a ProductionRule with execution recording.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rule` | `ProductionRule` | — | The rule to add (required) |

**Returns:** `None`

---

### `remove_rule(name)`

Remove a rule by name.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Rule name (required) |

**Returns:** `None`

---

### `start()`

Start the event-driven loop.

**Returns:** `None`

---

### `stop()`

Stop the event-driven loop.

**Returns:** `None`

---

### `ingest_event(event_type, payload, source="external", confidence=1.0, timestamp=None)`

Ingest an external event — triggers the full pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event_type` | `str` | — | Event category (required) |
| `payload` | `dict` | — | Key-value data (required) |
| `source` | `str` | `"external"` | Origin identifier |
| `confidence` | `float` | `1.0` | Accuracy confidence [0, 1] |
| `timestamp` | `float \| None` | `None` | Unix timestamp (default: now) |

**Returns:** `Transaction` object

**Example:**
```python
orb.ingest_event("ContainerDelayed", {
    "container_id": "C-123",
    "delay_days": 6,
    "port": "Rotterdam",
})
```

---

### `get_causal_chain(action_id)`

Trace an action back to its rule and recent events.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_id` | `str` | — | Action ID (required) |

**Returns:** `list[dict]` — Chain of events with keys: type, timestamp, detail

---

### `get_rule_status()`

Return status of all registered rules.

**Returns:** `list[dict]` — Rule status with keys: name, enabled, fired, last_fired, last_status

---

## Common Patterns

### Pattern 1: Basic Event Processing

```python
from dynafx.bridge import CognitiveOrchestrator
from dynafx.knowledge import ProductionRule, TripleCondition, TripleAction
from dynafx.knowledge.model import NamedNode, Literal
from dynafx.knowledge.inference import InferencePattern

rule = ProductionRule(
    name="delay_alert",
    body=[
        TripleCondition(InferencePattern(
            predicate=NamedNode("http://ex.org/delay_days"),
            object_=Literal(5),
        )),
    ],
    head=[
        TripleAction(
            subject=NamedNode("http://ex.org/system"),
            predicate=NamedNode("http://ex.org/alert"),
            object_=Literal(True),
        ),
    ],
)

orb = CognitiveOrchestrator(store)
orb.add_rule(rule)
orb.start()

# Events trigger rules automatically
orb.ingest_event("ContainerDelayed", {"container_id": "C-123", "delay_days": 6})
```

### Pattern 2: With Simulation Bridge

```python
bridge = KBSimBridge(store)
orb = CognitiveOrchestrator(store, bridge=bridge)

# Rules can trigger simulations
orb.add_rule(simulate_rule)
orb.start()

orb.ingest_event("DemandChanged", {"level": "high"})
```

---

## See Also

- [`KBSimBridge`](KBSimBridge.md) — For simulation integration
- [`ClosedLoopReasoner`](ClosedLoopReasoner.md) — For multi-pass reasoning
- [`ProductionRule`](../knowledge/production.md) — How to define rules
- [Tutorial 9: Custom Ontology](../../tutorials/09-custom-ontology.md) — Production rules
