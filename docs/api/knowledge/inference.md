# Inference

**What it is:** Forward-chaining rule engine with built-in RDFS and OWL-RL rule sets.

**When to use it:** When you need to infer new facts from existing ones using semantic rules.

**Quick start:**
```python
from dynafx.knowledge import RuleEngine, rdfs_rules

engine = RuleEngine(store)
engine.add_rules(rdfs_rules())
engine.run()  # Applies inference to fixpoint
```

---

## Import

```python
from dynafx.knowledge import Rule, RuleEngine, Var, InferencePattern, rdfs_rules, owl_rl_rules
```

## Classes

### Rule

An inference rule with body (premises) and head (conclusions).

```python
Rule(
    name: str,                      # Rule name
    head: list[InferencePattern],   # Patterns to infer
    body: list[InferencePattern],   # Patterns to match
)
```

### InferencePattern

Triple pattern where positions can be `RDFNode`, `Var`, or `None`.

```python
InferencePattern(
    subject: Any = None,
    predicate: Any = None,
    object_: Any = None,
)
```

### Var

A rule variable.

```python
Var("x")  # Represents ?x in rules
```

---

## RuleEngine

### Constructor

```python
RuleEngine(store: TripleStore)
```

### Methods

#### `add_rules(rules)`

Add rules to the engine.

```python
engine.add_rules(rdfs_rules())
engine.add_rules([my_custom_rule])
```

#### `run(max_iterations=100)`

Apply rules to fixpoint.

```python
engine.run(max_iterations=100)
```

---

## Built-in Rule Sets

### `rdfs_rules()`

Returns 7 RDFS inference rules.

```python
engine.add_rules(rdfs_rules())
```

### `owl_rl_rules()`

Returns 4 OWL-RL inference rules.

```python
engine.add_rules(owl_rl_rules())
```

---

## Custom Rules

```python
from dynafx.knowledge import Rule, InferencePattern, Var
from dynafx.knowledge.model import NamedNode

EX = "http://ex.org/"

# Rule: If ?x worksAt ?y and ?y hasParent ?z, then ?x worksFor ?z
rule = Rule(
    name="works_for",
    head=[InferencePattern(
        subject=Var("x"),
        predicate=NamedNode(f"{EX}worksFor"),
        object_=Var("z"),
    )],
    body=[
        InferencePattern(Var("x"), NamedNode(f"{EX}worksAt"), Var("y")),
        InferencePattern(Var("y"), NamedNode(f"{EX}hasParent"), Var("z")),
    ],
)

engine = RuleEngine(store)
engine.add_rule(rule)
engine.run()
```

---

## See Also

- [`TripleStore`](TripleStore.md) — Where rules are applied
- [`ProductionRule`](production.md) — Event-driven rules
- [Tutorial 9: Custom Ontology](../../tutorials/09-custom-ontology.md) — Rules deep dive
