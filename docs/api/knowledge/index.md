# dynafx.knowledge

**What it is:** An in-memory RDF triple store with SPARQL queries, OWL inference, and production rules — a complete knowledge graph engine with zero external dependencies.

**When to use it:** When you need to store structured facts, query them with SPARQL, run inference, or connect knowledge to simulations.

**Quick start:**
```python
from dynafx.knowledge import TripleStore, parse_turtle

store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:portfolio ex:revenue 950.0 .
    ex:portfolio ex:cost 600.0 .
""")
```

---

## Import

```python
from dynafx.knowledge import TripleStore, parse_turtle, parse_sparql, sparql_evaluate
```

## Core Classes

| Class | What it does | When to use it |
|-------|--------------|----------------|
| [`TripleStore`](TripleStore.md) | In-memory RDF store with named graphs | Always — this is your starting point |
| [`NamedNode`](NamedNode.md) | An RDF node identified by IRI | Creating subjects and predicates |
| [`Literal`](NamedNode.md#literal) | An RDF literal value | Creating objects with values |
| [`Triple`](Triple.md) | A (subject, predicate, object) fact | Adding data to the store |
| [`TriplePattern`](Triple.md#triplepattern) | A query pattern with wildcards | Querying the store |

## Data Loading

| Function | What it does | When to use it |
|----------|--------------|----------------|
| [`parse_turtle`](turtle.md) | Parse Turtle text into a store | Loading data from strings |
| [`serialize_turtle`](turtle.md#serialize_turtle) | Convert triples to Turtle text | Exporting data |
| [`ingest_csv`](ingest_csv.md) | CSV → RDF ingestion | Loading tabular data |

```python
from dynafx.knowledge import parse_turtle, ingest_csv, MappingDef

# Load from Turtle
store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:portfolio ex:revenue 950.0 .
""")

# Load from CSV
mapping = MappingDef(
    target_graph="enterprise",
    entity_class="http://ex.org/Portfolio",
    columns=[...]
)
report = ingest_csv(mapping, "data/portfolio.csv", store)
```

## Querying

| Function | What it does | When to use it |
|----------|--------------|----------------|
| [`parse_sparql`](sparql.md) | Parse SPARQL query string | Before evaluating |
| [`sparql_evaluate`](sparql.md#sparql_evaluate) | Evaluate parsed query | Running queries |

```python
from dynafx.knowledge import parse_sparql, sparql_evaluate

query = "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:revenue ?v }"
ast = parse_sparql(query)
result = sparql_evaluate(ast, store)
print(result.bindings)  # [{'v': Literal(950.0)}]
```

## Inference

| Function | What it does | When to use it |
|----------|--------------|----------------|
| [`Rule`](inference.md) | Forward-chaining inference rule | Custom inference |
| [`RuleEngine`](inference.md#ruleengine) | Applies rules to fixpoint | Running inference |
| [`rdfs_rules()`](inference.md#rdfs_rules) | Built-in RDFS rules (7 rules) | RDFS inference |
| [`owl_rl_rules()`](inference.md#owl_rl_rules) | Built-in OWL-RL rules (4 rules) | OWL inference |

```python
from dynafx.knowledge import RuleEngine, rdfs_rules

engine = RuleEngine(store)
engine.add_rules(rdfs_rules())
engine.run()  # Applies inference to fixpoint
```

## Production Rules

| Class | What it does | When to use it |
|-------|--------------|----------------|
| [`ProductionRule`](production.md) | Rule with conditions and actions | Event-driven logic |
| [`ProductionRuleEngine`](production.md#productionruleengine) | Evaluates rules against store | Running rules |

```python
from dynafx.knowledge import ProductionRule, ProductionRuleEngine, TripleCondition, TripleAction

rule = ProductionRule(
    name="alert",
    condition=TripleCondition(predicate="ex:revenue", object_=Literal(1000)),
    actions=[TripleAction(predicate="ex:alert", object_=Literal(True))],
)
engine = ProductionRuleEngine(store)
engine.add_rule(rule)
```

---

## Common Workflows

### Workflow 1: Store and Query Facts

```python
from dynafx.knowledge import TripleStore, parse_sparql, sparql_evaluate
from dynafx.knowledge.model import NamedNode, Literal, Triple

store = TripleStore()
store.add(Triple(
    NamedNode("http://ex.org/portfolio"),
    NamedNode("http://ex.org/revenue"),
    Literal(950.0)
), graph="enterprise")

# Query
query = "PREFIX ex: <http://ex.org/> SELECT ?v WHERE { ex:portfolio ex:revenue ?v }"
ast = parse_sparql(query)
result = sparql_evaluate(ast, store)
print(result.bindings[0]["v"].value)  # 950.0
```

### Workflow 2: Load from Turtle

```python
from dynafx.knowledge import parse_turtle

store = parse_turtle("""
    @prefix ex: <http://ex.org/> .
    ex:portfolio ex:revenue 950.0 .
    ex:portfolio ex:cost 600.0 .
    ex:supplier ex:reliability 0.85 .
""")
```

### Workflow 3: With KB-Connected Simulation

```python
from dynafx.dynamics import SysdModel
from dynafx.knowledge import TripleStore
from dynafx.bridge import KBSimBridge

store = TripleStore()
# ... add triples ...

model = SysdModel(dt=1.0)
with model.stock("Inventory", 1000) as s:
    s.inflow("supply", "KB_QUERY('SELECT ?v WHERE { ... }')")

bridge = KBSimBridge(store)
params = bridge.params_from_kb([...])
result = model.simulate(params=params, kb=store)
```

---

## Available Expressions in KB Queries

When using `KB_QUERY` in simulation expressions:

```python
# Simple query
"KB_QUERY('SELECT ?v WHERE { ex:reliability ?v }')"

# Query with variable subject
"KB_QUERY_TEMPLATE('SELECT ?v WHERE { {subject} ex:reliability ?v }', entity_iri)"
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `SyntaxError` in Turtle | Invalid Turtle syntax | Check prefix declarations and syntax |
| Empty SPARQL results | No matching triples | Verify data exists in the store |
| `ImportError` | Missing optional dependency | `pip install pyyaml` for YAML support |

---

## See Also

- [`TripleStore`](TripleStore.md) — Full API reference for the store
- [`NamedNode`](NamedNode.md) — RDF node types
- [`Triple`](Triple.md) — Triple class
- [Tutorial 5: Knowledge Graphs](../../tutorials/05-knowledge-graph.md) — Getting started with knowledge
- [Tutorial 6: Semantic Queries](../../tutorials/06-semantic-queries.md) — SPARQL deep dive
