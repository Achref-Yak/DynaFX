# API Reference

A curated index of the public `dynafx` API. Everything here is imported from a
top-level namespace — no `_private` modules. The tutorials show each symbol
in working code; the links point to the relevant one.

This page is hand-maintained. If a symbol is missing, it isn't part of the
public API contract (yet).

---

## Flagship types

The three types you will use in almost every script. Full details in
Tutorials [7](tutorials/07-closed-loop-twin.md), [1](tutorials/01-hello-world.md),
and [5](tutorials/05-knowledge-graph.md).

### `dynafx.dynamics.SysdModel`

A multi-paradigm model (SD + ABM + DES). Build it from a `.sysd` string with
`parse_sysd`, from a file with `parse_sysd_file`, or imperatively in Python
(`model.stock("X", 100).inflow(...)`, `model.agent(...)`, `model.queue(...)`).

```python
model = SysdModel(
    stocks=[StockDef(name="Inventory", initial=1000, flows=[
        FlowDef(name="supply", direction="+", expr="desired * reliability"),
    ])],
    aux_vars=[AuxDef(name="desired", expr="500")],
    dt=1.0,
)

result = model.simulate(
    method="rk4",               # or "euler"
    t_span=None,                # override the model's time range
    dt=None,                    # override the step size
    params={"reliability": 0.9},# parameter overrides
    kb=store,                   # enable KB_QUERY / KB_ASSERT builtins
)
```

### `dynafx.dynamics.SysdModelResult`

Returned by `simulate()`. Holds every trajectory as lists indexed by time:

| Attribute | Contents |
|-----------|----------|
| `values` | `{stock_name: [float, ...]}` — per-stock trajectories |
| `aux_values` | `{aux_name: [float, ...]}` — derived quantities per step |
| `times` | `[float, ...]` — the time axis |
| `stocks` | list of stock names |
| `method`, `steps`, `model_name` | run metadata |
| `abm_engine`, `des_engine` | the ABM/DES engines if present (or `None`) |

Also supports dict-style access (`result["values"]`) for backward compat, and
`result.plot("out.png", stocks=["Inventory"])`.

### `dynafx.knowledge.TripleStore`

An RDF store with named graphs and SPO/POS/OSP indices. The KB half of the
twin.

```python
store = TripleStore()
store.add(Triple(NamedNode("http://ex.org/a"),
                 NamedNode("http://ex.org/b"),
                 Literal(1.0)), graph="enterprise")

for t in store.triples(TriplePattern(subject=None, predicate=p, object_=None)):
    print(t)

store.graphs()                 # named graphs present
store.remove(TriplePattern(...))
```

Key methods: `add`, `remove`, `triples`, `triples_in_graph`, `graphs`,
`all_triples`, `copy_graph`, `remove_graph`, `on_add`/`on_remove` callbacks,
`__contains__` and `__len__`.

### `dynafx.bridge.KBSimBridge`

The two-way connector: KB facts → simulation params → evidence triples back
into the KB. See [Tutorial 7](tutorials/07-closed-loop-twin.md).

| Method | Direction | Purpose |
|--------|-----------|---------|
| `params_from_kb(claim_map, default=..., exclude_graphs=...)` | KB → sim | Extract triples as model parameters |
| `params_for_class(...)` | KB → sim | Parameter extraction scoped to an entity class |
| `evidence_from_result(result, evidence_map)` | sim → KB | Turn trajectories into evidence triples |
| `evidence_for_stock(...)` | sim → KB | Evidence for a single stock |
| `run_with_kb(model, params, kb=...)` | both | Simulate with KB-connected builtins |
| `full_roundtrip(...)` | both | params → run → evidence in one call |
| `record_provenance(result, params, graph=...)` | sim → KB | Record the run as PROV triples |
| `compare_runs(...)` | both | Compare two runs and their evidence |

---

## `dynafx.dynamics`

### DSL & simulation

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `SysdModel` | Multi-paradigm model container (see flagship) | [1](tutorials/01-hello-world.md) |
| `SysdModelResult` | Simulation trajectory (see flagship) | [1](tutorials/01-hello-world.md) |
| `parse_sysd(source)` | Parse a `.sysd` string into a `SysdModel` | [1](tutorials/01-hello-world.md) |
| `parse_sysd_file(path)` | Parse a `.sysd` file into a `SysdModel` | [7](tutorials/07-closed-loop-twin.md) |
| `StockDef` | Stock declaration (`name`, `initial`, `flows`) | [2](tutorials/02-system-dynamics.md) |
| `FlowDef` | Flow declaration (`name`, `direction` `+`/`-`, `expr`) | [2](tutorials/02-system-dynamics.md) |
| `AuxDef` | Derived quantity (recomputed each step) | [2](tutorials/02-system-dynamics.md) |
| `TableDef` | Lookup table declaration | [2](tutorials/02-system-dynamics.md) |
| `ValidationResult`, `ValidationIssue` | Model validation output (`model.validate()`) | — |

### Agent-Based Modeling

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `AgentDef` | Agent type definition | [3](tutorials/03-agent-based-modeling.md) |
| `AgentInstance` | A runtime agent with `state` and inbox | [3](tutorials/03-agent-based-modeling.md) |
| `ABMEngine` | Runs agent step cycles; `get_metrics()` aggregates | [3](tutorials/03-agent-based-modeling.md) |

### Discrete Event Simulation

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `QueueDef` | Queue declaration (capacity, service time, servers) | [4](tutorials/04-discrete-event-simulation.md) |
| `ResourceDef` | Resource pool declaration | [4](tutorials/04-discrete-event-simulation.md) |
| `EventDef` | Scheduled event | [4](tutorials/04-discrete-event-simulation.md) |
| `DESEngine` | Event-queue driver; `get_all_stats()` | [4](tutorials/04-discrete-event-simulation.md) |
| `Queue`, `Resource` | Runtime queue / resource entities | [4](tutorials/04-discrete-event-simulation.md) |
| `QueueStats`, `ResourceStats` | Utilization statistics | [4](tutorials/04-discrete-event-simulation.md) |

### Structural analysis

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `causal_trace(model, var, state)` | Causes + effects + factor strip | [10](tutorials/10-publishing-results.md) |
| `causes_strip(model, var, state)` | Decompose a variable into contributing factors | [10](tutorials/10-publishing-results.md) |
| `causes_tree`, `effects_tree` | Upstream / downstream dependency trees | — |
| `detect_feedback_loops(model)` | Find reinforcing/balancing loops | [10](tutorials/10-publishing-results.md) |
| `loops_for_variable(analysis, var)` | Loops touching one variable | — |

### Scenario comparison & sensitivity

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `ScenarioComparison`, `ScenarioDef`, `ScenarioResult` | Compare parameterized runs; summary/deviation/rank | [8](tutorials/08-scenarios-and-sensitivity.md) |
| `SensitivityAnalyzer` | Sobol / Morris / PRCC / OAT influence analysis | [8](tutorials/08-scenarios-and-sensitivity.md) |
| `SensitivityResult` | Sensitivity outputs (`.first_order`, `.mu_star`, ...) | [8](tutorials/08-scenarios-and-sensitivity.md) |

### Optimization

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `lp_minimize`, `lp_maximize` | scipy `linprog` wrappers | [10](tutorials/10-publishing-results.md) |
| `kb_lp_minimize`, `kb_lp_maximize` | LP with coefficients read from SPARQL | [10](tutorials/10-publishing-results.md) |
| `calibrate`, `kb_calibrate` | Parameter calibration | — |
| `optimize`, `kb_optimize` | Bounded parameter optimization | — |
| `pareto_optimize` | Multi-objective Pareto search | — |
| `LPResult`, `CalibrationResult`, `OptimizationResult`, `ParetoResult` | Result types (`.x`, `.objective_value`, `.success`) | — |

### Units & submodels

| Symbol | Purpose |
|--------|---------|
| `UnitChecker` | Verify `stock = flow × time` dimensional consistency |
| `Unit`, `UnitRegistry`, `UnitCheckResult`, `UnitViolation` | Unit model + check outputs |
| `SubmodelDef`, `IncludeDef` | Include submodels with name-prefixed scoping |

### Emergent properties

| Symbol | Purpose |
|--------|---------|
| `EmergentProperty`, `Condition`, `Effect` | Declarative emergent-behavior checks |
| `ComparisonOp`, `EffectType` | Condition/effect enums |
| `run_consistency_checks` | Run emergent checks over a result |
| `ConsistencyResult`, `ConsistencyViolation` | Consistency outputs |

---

## `dynafx.knowledge`

### RDF model

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `NamedNode` | IRI node (subjects and predicates) | [5](tutorials/05-knowledge-graph.md) |
| `Literal` | Value node (`Literal(0.9)`, `.value`) | [5](tutorials/05-knowledge-graph.md) |
| `BlankNode` | Anonymous node | — |
| `Triple` | `(subject, predicate, object)` fact | [5](tutorials/05-knowledge-graph.md) |
| `TriplePattern` | Match template (`None` = wildcard) | [5](tutorials/05-knowledge-graph.md) |
| `RDFNode` | Base class for `NamedNode`/`BlankNode`/`Literal` | — |
| `xsd` | XSD datatype namespace constants | — |

### Store, Turtle, SPARQL

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `TripleStore` | RDF store with named graphs (see flagship) | [5](tutorials/05-knowledge-graph.md) |
| `parse_turtle(text, base_iri=...)` | Parse Turtle into a `TripleStore` | [5](tutorials/05-knowledge-graph.md) |
| `serialize_turtle(triples, prefixes=...)` | Serialize triples to Turtle text | — |
| `parse_sparql(query)` | Parse a SPARQL query string | [6](tutorials/06-semantic-queries.md) |
| `sparql_evaluate(ast, store)` | Evaluate a parsed query against a store | [6](tutorials/06-semantic-queries.md) |
| `QueryResult` | `.bindings` (list of var→node dicts), `.cardinality` | [6](tutorials/06-semantic-queries.md) |

### Inference

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `Rule` | A forward-chaining rule (`body` premises → `head` conclusions) | [9](tutorials/09-custom-ontology.md) |
| `RuleEngine` | Applies rules to a store to a fixpoint | [9](tutorials/09-custom-ontology.md) |
| `InferencePattern` | Pattern in a rule body/head (`Var` or `None`) | [9](tutorials/09-custom-ontology.md) |
| `Var("x")` | Rule variable | [9](tutorials/09-custom-ontology.md) |
| `rdfs_rules()`, `owl_rl_rules()` | Built-in RDFS (7) and OWL-RL (4) rule sets | [5](tutorials/05-knowledge-graph.md) |

### CSV ingestion

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `ingest_csv(mapping, path, store, strict=...)` | Declarative CSV→RDF ingestion | [5](tutorials/05-knowledge-graph.md) |
| `MappingDef` | Mapping spec (target graph, entity class, columns) | [5](tutorials/05-knowledge-graph.md) |
| `ColumnMapping` | Column → predicate + type coercion | [5](tutorials/05-knowledge-graph.md) |
| `IngestReport` | `.rows_parsed`, `.triples_added`, `.errors` | [5](tutorials/05-knowledge-graph.md) |
| `load_all_mappings(dir)` | Load all YAML mapping files | — |

### TBox & type hierarchy

| Symbol | Purpose |
|--------|---------|
| `TypeHierarchy` | OWL2-style subtype lattice (`is_subtype`, `get_ancestors`) |
| `TypeNode` | A node in the hierarchy |
| `TBox`, `load_tbox(name)` | Load a named TBox |
| `GENERAL_TBOX`, `MDM_TYPE_HIERARCHY` | Built-in hierarchies |

### Production rules & transactions

| Symbol | Purpose | Tutorial |
|--------|---------|----------|
| `ProductionRule` | Rule with conditions + actions (fire-once, priority) | [9](tutorials/09-custom-ontology.md) |
| `ProductionRuleEngine` | Evaluates rules against a store | [9](tutorials/09-custom-ontology.md) |
| `TripleCondition`, `ComparisonCondition`, `SparqlCondition`, `AndCondition`, `OrCondition`, `NotCondition`, `AggregationCondition` | Condition types | [9](tutorials/09-custom-ontology.md) |
| `TripleAction`, `LogAction`, `RetractAction`, `BridgeAction`, `SimulateAction` | Action types | [9](tutorials/09-custom-ontology.md) |
| `Transaction`, `TransactionStore`, `TransactionQuery` | Append-only temporal log | — |
| `ExecutionRecord`, `ExecutionStore` | Provenance-tracked action records | — |

---

## `dynafx.bridge`

| Symbol | Purpose |
|--------|---------|
| `KBSimBridge` | The KB↔sim bridge (see flagship) |
| `ReasoningPass` | One simulate → grade cycle definition |
| `grade_queries(specs, store, prefix=...)` | Score a store against `(query, var, threshold, penalty)` specs |
| `ClosedLoopReasoner` | Multi-pass sense → grade → nudge → re-simulate loop |
| `ClosedLoopResult` | Closed-loop run output |
| `CognitiveOrchestrator` | Rule + action orchestration with provenance |

---

## `dynafx.patterns`

| Symbol | Purpose |
|--------|---------|
| `SignalChain.build(...)` | Factory: leading-indicator → outcome model |
| `DisruptionCascade.build(...)` | Factory: supply-chain disruption propagation |

## `dynafx.core`

| Symbol | Purpose |
|--------|---------|
| `Graph`, `Node`, `Edge` | Foundational structural model |
| `Entity`, `WorldRelation` | Domain entities and relations |
| `NodeType`, `EdgeType`, `ReasoningMode` | Type enums |
| `SystemDecomposer` | Manual node/edge graph construction (`add_node`, `add_edge`) |
