# Tutorial 9 — Custom Ontology

An ontology is how you make the KB *derive* knowledge it wasn't explicitly
told. DynaFX ships two rule engines: `RuleEngine` (forward-chaining RDFS/OWL
RL inference) and `ProductionRuleEngine` (action-oriented business rules). This
tutorial covers custom inference plus the type hierarchy.

## Custom inference rules

A `Rule` has a name, a `body` (premises), and a `head` (conclusions). Patterns
use `Var("x")` for variables and `None` for wildcards:

```python
from dynafx.knowledge import (
    TripleStore, NamedNode, Literal, parse_turtle,
)
from dynafx.knowledge.inference import (
    Rule, RuleEngine, InferencePattern, Var, rdfs_rules,
)

AGE = NamedNode("http://ex.org/age")
STATUS = NamedNode("http://ex.org/status")

adult_rule = Rule(
    name="adult",
    head=[InferencePattern(subject=Var("x"), predicate=STATUS, object_=Literal("adult"))],
    body=[InferencePattern(subject=Var("x"), predicate=AGE, object_=Var("a"))],
)

store = parse_turtle("""
@prefix ex: <http://ex.org/> .
ex:alice ex:age 30 .
ex:child ex:age 5 .
""")

engine = RuleEngine([adult_rule])
added = engine.apply(store)
print(added)   # 2 — both alice and child match
```

!!! note "Simple rules, real machinery"
    Rules are structural patterns, not arbitrary Python. For age-based
    thresholds you'd add a second pattern or use `ProductionRuleEngine` (which
    supports `ComparisonCondition`).

## Filtering with the production rule engine

When you need **conditions on values** (not just structure), use production
rules. They evaluate against the store and can log or add/retract triples:

```python
from dynafx.knowledge import TripleStore, NamedNode, Literal, Triple
from dynafx.knowledge.inference import InferencePattern
from dynafx.knowledge.production import (
    ProductionRule, ProductionRuleEngine,
    TripleCondition, ComparisonCondition, LogAction, TripleAction,
)

COMPLETION = NamedNode("http://ex.org/completionPct")
MITIGATE = NamedNode("http://ex.org/requiresMitigation")

snapshot = TripleStore()
snapshot.add(Triple(
    NamedNode("http://ex.org/Portfolio"), COMPLETION, Literal(0.6),
), graph="evidence")

rule = ProductionRule(
    name="portfolio-at-risk",
    body=[
        TripleCondition(InferencePattern(
            subject=NamedNode("http://ex.org/Portfolio"),
            predicate=COMPLETION,
            object_="?v",
        )),
        ComparisonCondition("?v", "<", 0.75),
    ],
    head=[
        LogAction("Portfolio below 75% — mitigation required"),
        TripleAction(NamedNode("http://ex.org/Portfolio"), MITIGATE, Literal(1.0)),
    ],
)

engine = ProductionRuleEngine(snapshot)
engine.add_rule(rule)
results = engine.evaluate()

for r in results:
    if r.success:
        print(r.action_type)   # 'log', 'triple_add'
```

The `InferencePattern` here uses the **string** `"?v"` as the object — a named
variable — and `ComparisonCondition` binds it to a threshold.

## The type hierarchy

For OWL2-style subtyping without full inference, use `TypeHierarchy`:

```python
from dynafx.knowledge import TypeHierarchy, TypeNode

h = TypeHierarchy()
h.add_type("AGENT")
h.add_type("PERSON", parent="AGENT")
h.add_type("CUSTOMER", parent="PERSON")

print(h.is_subtype("CUSTOMER", "AGENT"))   # True
print(h.get_ancestors("CUSTOMER"))         # ['CUSTOMER', 'PERSON', 'AGENT']
```

`MDM_TYPE_HIERARCHY` and `GENERAL_TBOX` ship pre-built lattices; `load_tbox`
loads the built-in general ontology:

```python
from dynafx.knowledge import load_tbox
from dynafx.knowledge.loader import validate_against_tbox

tbox = load_tbox("general")
print(validate_against_tbox("FACT", "SUPPORTS", tbox))   # True — valid pair
print(validate_against_tbox("STOCK", "flow_in", tbox))   # False — not in general TBox
```

## Bundling custom rules for reuse

Package your rules as a function so the closed-loop pipeline can attach them:

```python
def my_rules() -> list[Rule]:
    return [adult_rule]

engine = RuleEngine(my_rules() + rdfs_rules())
```

## What's next

- Turn a scenario study into a reproducible, citable result in [Publishing Results](10-publishing-results.md).
