# Knowledge Base

The knowledge pillar provides a self-contained RDF/OWL/SPARQL engine with no external dependencies. It powers the KB↔simulation bridge: enterprise data is ingested into named graphs, inference derives new facts, and simulation models read (and write) the store mid-run via `KB_QUERY`/`KB_ASSERT`.

## RDF Data Model

`dynafx.knowledge.model` defines the core types:

- `RDFNode` — base class
- `NamedNode` — an IRI
- `BlankNode` — a blank node (auto-generated id)
- `Literal` — a typed literal (string, float, integer, boolean, etc.)
- `Triple` — `(subject, predicate, object_)`
- `TriplePattern` — a triple with any position wildcarded (used for matching)
- `xsd` — XSD type constants (`XSD_DOUBLE`, `XSD_INTEGER`, `XSD_BOOLEAN`, …)

## TripleStore

`dynafx.knowledge.store.TripleStore` is an in-memory triple store with:

- **Nested indices** — SPO, POS, and OSP for fast pattern matching (all 8 patterns).
- **Named graphs** — triples are stored per graph; graphs can be isolated, copied, and removed.
- **Graph queries** — `triples(pattern, graph=...)` filters by graph.

```python
from dynafx.knowledge import TripleStore
from dynafx.knowledge.model import NamedNode, Literal, XSD_DOUBLE, Triple

store = TripleStore()
s = NamedNode("http://epc.org/Portfolio")
store.add(Triple(s, NamedNode("http://epc.org/reliability"),
                 Literal("0.82", datatype=XSD_DOUBLE)), graph="meta")
```

## Turtle Parsing & Serialization

`dynafx.knowledge.turtle` provides a Turtle/N-Triples tokenizer, recursive-descent parser, and serializer. It supports `@prefix`/`@base`, `a` for `rdf:type`, all literal types, blank nodes, `;` and `,` grouping, comments, and base-IRI resolution.

```python
from dynafx.knowledge import parse_turtle, serialize_turtle

triples = parse_turtle(turtle_text).triples()
```

## SPARQL

`dynafx.knowledge.sparql` provides a SPARQL 1.1 parser and evaluator. The public entry point is `dynafx.knowledge.sparql_evaluate(ast, store)`:

- **SELECT** with FILTER, DISTINCT, LIMIT, OFFSET
- **ASK** — returns `QueryResult` with `.cardinality` (1 or 0)
- **DESCRIBE**
- BGP matching across named graphs

```python
from dynafx.knowledge import sparql_evaluate, parse_sparql

ast = parse_sparql("SELECT ?v WHERE { <http://epc.org/Portfolio> <http://epc.org/reliability> ?v }")
result = sparql_evaluate(ast, store)
```

> **Note:** the evaluator does not support GROUP BY / aggregates. Pre-compute aggregate values and store them as explicit triples.

## Inference

`dynafx.knowledge.inference` provides a forward-chaining rule engine:

- `Rule`, `Var`, `InferencePattern` — declarative rules
- `RuleEngine` — applies rules until fixpoint
- `rdfs_rules()` — 7 rules (subClassOf, subPropertyOf, domain, range, type propagation)
- `owl_rl_rules()` — 4 rules (equivalentClass, equivalentProperty, inverseOf, TransitiveProperty)

```python
from dynafx.knowledge import TripleStore, RuleEngine, rdfs_rules

engine = RuleEngine(rdfs_rules())
engine.apply(store)
```

## TBox / Type Hierarchy

`dynafx.knowledge.hierarchy` and `dynafx.knowledge.loader` provide an OWL2-style type system:

- `TypeNode`, `TypeHierarchy`, `MDM_TYPE_HIERARCHY`
- `TBox`, `GENERAL_TBOX`, `load_tbox(name)` — load a TBox by name
- `validate_against_tbox(node_type, edge_type, tbox)` — validate roles/edges against the TBox

## Production Rules

`dynafx.knowledge.production` provides a rule engine for KB-driven actions:

- 5 condition types: `TripleCondition`, `ComparisonCondition`, `AndCondition`, `OrCondition`, `SparqlCondition`
- 5 action types: `LogAction`, `TripleAction`, `RetractAction`, `SimulateAction`, `BridgeAction`
- Fire-once, max-fires, priority ordering

```python
from dynafx.knowledge import ProductionRule, ProductionRuleEngine, TripleCondition, TripleAction
from dynafx.knowledge.inference import InferencePattern

rule = ProductionRule(
    name="portfolio-at-risk",
    body=[TripleCondition(InferencePattern(subject=..., predicate=..., object_="?v"))],
    head=[TripleAction(...)],
)
engine = ProductionRuleEngine(store)
engine.add_rule(rule)
engine.evaluate()
```

## CSV Ingestion

`dynafx.knowledge.ingest_csv` provides declarative CSV→RDF ingestion driven by YAML mapping files:

- `MappingDef` — loaded from YAML with IRI prefix expansion
- `ColumnMapping` — CSV column → predicate + type + iri_prefix
- `ingest_csv(md, path, store, strict=False)` — returns `IngestReport`
- `load_all_mappings(directory)` — load all YAML mappings

Type coercion: string, float, integer, boolean, iri. Foreign-key columns use `type: iri` with `iri_prefix:` to create entity relationships.

## Transactions & Execution

- `TransactionStore` — append-only temporal log backed by RDF (`Transaction`, `TransactionQuery`).
- `ExecutionStore` — provenance-tracked action records (`ExecutionRecord`).
