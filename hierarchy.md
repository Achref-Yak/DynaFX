# DynaFX — Complete Feature Hierarchy

## System Analysis — Three Pillars + Bridge + Patterns

---

### LEVEL 0: USER INTERFACE (Presentation)

```
├── 0.1  Python API (top-level)
│   ├── SysdModel, parse_sysd_file, parse_sysd
│   ├── TripleStore, parse_turtle
│   ├── KBSimBridge, ClosedLoopReasoner, grade_queries
│   ├── ScenarioComparison, causal_trace, detect_feedback_loops
│   └── lp_minimize / pareto_optimize / kb_lp_minimize
│
├── 0.2  Plugin Registry (registry.py)
│   └── custom builtins and DES hook registration
│
├── 0.3  Example Scripts (1 script)
│   └── Supply chain twin: global_solar_epc_twin (L1-L5, SD+ABM+DES+KB)
│
└── 0.5  Model Library (5 .sysd models)
    ├── Supply chain twin: global_solar_epc
    └── Library fixtures: vmi, reverse_logistics, cold_chain, supply_chain_demo
```

---

### LEVEL 1: DYNAMICS PILLAR (dynafx.dynamics)

```
├── 1.1  DSL Parser (dsl.py, _parser.py)
│   ├── 1.1.1  Tokenizer — numbers, identifiers, operators, keywords
│   ├── 1.1.2  Recursive descent expression parser
│   │   ├── Arithmetic: +, -, *, /, ^ (pow)
│   │   ├── Comparison: >, >=, <, <=, ==, !=
│   │   ├── Functions: ABS, EXP, LN, SQRT, SIN, COS, MIN, MAX, IF
│   │   ├── Time functions: PULSE, STEP, RAMP, NOISE (seeded)
│   │   ├── Delays: SMOOTH, SMOOTHI, DELAY3, DELAYN, DELAY_FIXED, CONVEY_BATCH
│   │   └── Lookup tables: WITH LOOKUP / table keyword
│   ├── 1.1.3  Line-oriented indent-based block parser
│   │   ├── stock keyword — state variables
│   │   ├── flow keyword — rate equations (inflow/outflow)
│   │   ├── aux keyword — algebraic intermediates
│   │   ├── table keyword — lookup/interpolation tables
│   │   ├── model/end — scoping blocks
│   │   └── submodel / include — modular composition
│   ├── 1.1.4  KB builtins
│   │   ├── KB_QUERY — ASK→1.0/0.0, SELECT→float, SPARQL strings via params
│   │   └── KB_ASSERT — write triples mid-run
│   └── 1.1.5  Validation (dsl.py: validate())
│       ├── Name resolution (all refs exist)
│       ├── Flow conservation (inflow/outflow balance)
│       └── Non-negativity bounds
│
├── 1.2  Compilation & Execution
│   ├── 1.2.1  Equations (equations.py)
│   │   ├── rk4_step — Runge-Kutta 4th order
│   │   ├── euler_step — first-order Euler
│   │   └── compile_equations — pre-compiled code objects
│   ├── 1.2.2  CompiledSystem caching (dsl.py)
│   │   ├── ASTs, code objects, topo-sort cached on first simulate()
│   │   └── ~25x speedup
│   ├── 1.2.3  Smooth/delay expressions
│   │   ├── smooth_delay_exprs / smooth_init_exprs
│   │   └── SMOOTH/DELAY accept parameter names (ExprNode)
│   └── 1.2.4  LookupTable injection into _ns (not _s)
│
├── 1.3  Agent-Based Modeling (agent.py)
│   ├── AgentInstance — typed properties
│   ├── ABMEngine — 4-phase step (Deliver → Decide → Cleanup → Aggregate)
│   ├── Rules — perceive → decide → act
│   ├── Message — topic-based SEND
│   ├── Strategies / SWITCH_STRATEGY with cooldown
│   ├── Meta-rules (before/after strategy rules)
│   └── Aggregated metrics per step
│
├── 1.4  Discrete Event Simulation (des.py)
│   ├── DESClock
│   ├── EventQueue
│   ├── Queue — capacity, service time expressions, multi-server departure
│   ├── Resource — pools with capacity constraints
│   ├── DESEngine — event-driven stepping
│   └── QueueStats / ResourceStats — utilization tracking
│
├── 1.5  Units Checking (units.py)
│   ├── Unit / UnitRegistry — ~Unit~ annotations
│   ├── UnitChecker — propagation through expressions
│   └── stock = flow × time consistency
│
├── 1.6  Causal Analysis (causal.py)
│   ├── causes_tree / effects_tree
│   ├── causal_trace
│   └── causes_strip — factor list with total_value
│
├── 1.7  Feedback Loops (feedback.py)
│   ├── detect_feedback_loops — LoopAnalysis
│   └── loops_for_variable — FeedbackLoop (name, nodes, polarity)
│
├── 1.8  Optimization (optimization.py)
│   ├── lp_minimize / lp_maximize — scipy.optimize.linprog wrapper
│   ├── pareto_optimize — Pareto front
│   ├── calibrate — parameter calibration
│   └── KB-constrained: kb_lp_minimize, kb_lp_maximize, kb_calibrate, kb_optimize
│
├── 1.9  Scenario Comparison (scenario.py)
│   ├── ScenarioDef / ScenarioResult
│   ├── ScenarioComparison — comparison / deviation / tornado / summary
│   └── rank() with grade specs (KB-graded ranking)
│
├── 1.10 Sensitivity Analysis (sensitivity.py)
│   ├── uniform / normal / lognormal ensembles
│   └── SensitivityAnalyzer / SensitivityResult
│
├── 1.11 Emergent Properties (emergent.py)
│   ├── EmergentProperty / Condition / Effect
│   └── run_consistency_checks
│
└── 1.12 BFO Alignment (bfo.py)
    ├── get_bfo_alignment / validate_bfo_alignment
    ├── BfoContinuantCategory / ContinuantSubcategory / OccurrentSubcategory
    └── get_bfo_summary
```

---

### LEVEL 2: KNOWLEDGE PILLAR (dynafx.knowledge)

```
├── 2.1  RDF Data Model (model.py)
│   ├── RDFNode, NamedNode, BlankNode, Literal
│   ├── Triple, TriplePattern
│   ├── XSD types (double, integer, boolean, …)
│   └── RDF / RDFS / OWL namespace constants
│
├── 2.2  TripleStore (store.py)
│   ├── SPO / POS / OSP nested indices
│   ├── Named graphs — isolation, copy, removal
│   ├── Pattern matching — all 8 patterns
│   └── Triple identity = (s, p, o)
│
├── 2.3  Turtle / N-Triples (turtle.py)
│   ├── Tokenizer, recursive-descent parser, serializer
│   ├── @prefix / @base, a, all literal types, blank nodes
│   ├── ; and , grouping, comments
│   └── Base-IRI resolution, empty-prefix PNAME_LN
│
├── 2.4  SPARQL (sparql.py, _sparql_parser.py)
│   ├── SELECT / ASK / DESCRIBE
│   ├── FILTER, DISTINCT, LIMIT, OFFSET
│   ├── QueryResult — cardinality, bindings
│   └── No GROUP BY — aggregates pre-computed as explicit triples
│
├── 2.5  Inference (inference.py)
│   ├── Rule / Var / InferencePattern
│   ├── RuleEngine — forward-chaining
│   ├── rdfs_rules — 7 rules (subClassOf, subPropertyOf, domain, range, …)
│   └── owl_rl_rules — 4 rules (equivalentClass, equivalentProperty, inverseOf, TransitiveProperty)
│
├── 2.6  TBox / Type Hierarchy (hierarchy.py, loader.py)
│   ├── TypeNode / TypeHierarchy / MDM_TYPE_HIERARCHY
│   ├── TBox / GENERAL_TBOX / BUILTIN_TBOXES
│   ├── load_tbox(name)
│   └── validate_against_tbox
│
├── 2.7  Production Rules (production.py)
│   ├── ProductionRuleEngine — fire_once / max_fires / priority
│   ├── Conditions: TripleCondition, ComparisonCondition, AndCondition, OrCondition, SparqlCondition
│   └── Actions: LogAction, TripleAction, RetractAction, SimulateAction, BridgeAction
│
├── 2.8  CSV Ingestion (ingest_csv.py)
│   ├── MappingDef — from_yaml with IRI prefix expansion
│   ├── ColumnMapping — predicate + type + iri_prefix
│   ├── IngestReport — rows_parsed / skipped / triples_added / errors
│   ├── Type coercion: string, float, integer, boolean, iri
│   └── Lenient by default, strict opt-in
│
├── 2.9  Transactions (transactions.py)
│   ├── Transaction / TransactionQuery / TransactionStore
│   └── Append-only temporal log backed by RDF
│
└── 2.10 Execution (execution.py)
    ├── ExecutionRecord / ExecutionStore
    └── Provenance-tracked action records
```

---

### LEVEL 3: CORE INFRASTRUCTURE (dynafx.core)

```
├── 3.1  Foundational Models (models.py)
│   ├── BfoCategory — BFO top-level categories
│   ├── NodeType / EdgeType — typed graph elements
│   ├── ReasoningMode / Severity enums
│   ├── Node / Edge / Entity / WorldRelation / TypedEdge
│   ├── Parameter / Payload / TimeInfo / Span
│   ├── Interpretation / ConversationTree
│   ├── EmergentProperty / FeedbackLoop
│   ├── Graph — node/edge container with lookup
│   └── Context / Annotation / Trace / EvidenceCounts / Violation / ReviewResult
│
└── 3.2  SystemDecomposer (decomposer.py)
│   ├── add_node / add_edge — manual graph construction
│   └── detect — structural EmergentProperty detection
```
---

### LEVEL 4: BRIDGE (bridge.py) + PATTERNS (patterns/)

```
├── 4.1  KBSimBridge
│   ├── params_from_kb(claim_map, default, exclude_graphs)
│   ├── evidence_from_result(result, evidence_map, graph)
│   ├── record_provenance(run, params, graph, extra_annotations)
│   └── params_for_class
│
├── 4.2  ClosedLoopReasoner
│   ├── simulate → grade → nudge → re-simulate
│   └── ReasoningPass
│
├── 4.3  grade_queries
│   └── grade SPARQL query results
│
├── 4.4  Patterns (patterns/)
│   ├── SignalChain — leading-indicator → outcome factory
│   └── DisruptionCascade — supply-chain disruption propagation
```

---

### CROSS-CUTTING CONCERNS

```
├── Unified State Dict
│   └── SD stocks + ABM metrics + DES metrics → single dict
│
├── Naming Convention Restructuring
│   └── dynamics/ (was system/), knowledge/ (was kb/), core/
│
├── GitHub Pages docs
│   └── mkdocs + Material, deployed via .github/workflows/deploy.yml
│
└── Makefile
    ├── test — pytest
    └── test-v — verbose pytest
```

---

### Test Coverage Summary (1028 tests, 35 files)

| Area | Test Files | Tests |
|------|-----------|-------|
| DES | test_des.py | 105 |
| ABM additions (messages/strategies) | test_abm_additions.py | 64 |
| KB — inference | test_kb_inference.py | 59 |
| KB — production rules | test_kb_production.py | 53 |
| DSL | test_dsl.py | 53 |
| ABM | test_abm.py | 42 |
| Units checking | test_units.py | 40 |
| Sensitivity | test_sensitivity.py | 40 |
| KB — TripleStore | test_kb_store.py | 40 |
| KB — RDF model | test_kb_model.py | 39 |
| KB — Turtle | test_kb_turtle.py | 38 |
| Delays | test_delays.py | 37 |
| KB — SPARQL | test_kb_sparql.py | 36 |
| Graph / core | test_graph.py | 32 |
| Emergent properties | test_emergent.py | 31 |
| KB — transactions | test_kb_transactions.py | 25 |
| Math functions | test_math_functions.py | 24 |
| KB ↔ Sim bridge | test_kb_sim_bridge.py | 24 |
| KB ↔ Sim bridge (ext) | test_kb_sim_bridge_ext.py | 23 |
| Scenario comparison | test_scenario.py | 22 |
| CSV I/O | test_csv_io.py | 22 |
| Plugins | test_plugins.py | 21 |
| Registry | test_registry.py | 18 |
| TBox | test_tbox.py | 17 |
| Optimization | test_optimization.py | 17 |
| Solver | test_solver.py | 15 |
| DisruptionCascade pattern | test_disruption_cascade.py | 15 |
| Causal tracing | test_causal.py | 15 |
| Submodels | test_submodels.py | 11 |
| Validation | test_validation.py | 10 |
| Cross-paradigm | test_crossparadigm.py | 10 |
| Reference models | test_reference_models.py | 8 |
| Feedback loops | test_feedback.py | 8 |
| Accuracy | test_accuracy.py | 8 |
| CTA (conversation tree) | test_cta.py | 6 |
| **Total** | **35 test files** | **1028** |
