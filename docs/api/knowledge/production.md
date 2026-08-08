# Production Rules

**What it is:** Event-driven rules with conditions and actions — fire-once semantics with priorities.

**When to use it:** When you need reactive logic that fires when conditions are met.

**Quick start:**
```python
from dynafx.knowledge import ProductionRule, ProductionRuleEngine, TripleCondition, TripleAction

rule = ProductionRule(
    name="alert",
    condition=TripleCondition(predicate="ex:revenue", object_=Literal(1000)),
    actions=[TripleAction(predicate="ex:alert", object_=Literal(True))],
)
engine = ProductionRuleEngine(store)
engine.add_rule(rule)
engine.start()
```

---

## Import

```python
from dynafx.knowledge import (
    ProductionRule, ProductionRuleEngine,
    TripleCondition, ComparisonCondition, SparqlCondition,
    AndCondition, OrCondition, NotCondition, AggregationCondition,
    TripleAction, LogAction, RetractAction, BridgeAction, SimulateAction,
)
```

## ProductionRule

### Constructor

```python
ProductionRule(
    name: str,                          # Rule name
    description: str = "",              # Description
    condition: Condition = None,        # When to fire
    actions: list[Action] = [],         # What to do
    priority: int = 0,                  # Evaluation priority
    enabled: bool = True,               # Whether active
    fire_once: bool = False,            # Fire only once
)
```

---

## Condition Types

### TripleCondition

Match a triple pattern.

```python
TripleCondition(
    subject: Any = None,
    predicate: Any = None,
    object_: Any = None,
)
```

### ComparisonCondition

Compare a value.

```python
ComparisonCondition(
    left: Any,
    operator: str,  # ">", "<", ">=", "<=", "==", "!="
    right: Any,
)
```

### SparqlCondition

Run a SPARQL query.

```python
SparqlCondition(query: str, variable: str, threshold: float)
```

### Logical Conditions

```python
AndCondition(conditions=[...])
OrCondition(conditions=[...])
NotCondition(condition=...)
AggregationCondition(...)
```

---

## Action Types

### TripleAction

Assert a new triple.

```python
TripleAction(
    subject: Any,
    predicate: Any,
    object_: Any,
)
```

### LogAction

Log a message.

```python
LogAction(message: str)
```

### RetractAction

Remove matching triples.

```python
RetractAction(pattern: TriplePattern)
```

### BridgeAction

Trigger a bridge action.

```python
BridgeAction(bridge_method: str, args: dict)
```

### SimulateAction

Run a simulation.

```python
SimulateAction(model: SysdModel, params: dict)
```

---

## ProductionRuleEngine

### Constructor

```python
ProductionRuleEngine(store: TripleStore)
```

### Methods

```python
engine.add_rule(rule)
engine.remove_rule(name)
engine.start()  # Start event-driven loop
engine.stop()   # Stop loop
```

---

## See Also

- [`Inference`](inference.md) — Forward-chaining rules
- [`CognitiveOrchestrator`](../bridge/CognitiveOrchestrator.md) — Event-driven orchestration
- [Tutorial 9: Custom Ontology](../../tutorials/09-custom-ontology.md) — Rules deep dive
