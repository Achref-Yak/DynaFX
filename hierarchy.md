# DynaFX — Complete Feature Hierarchy

## System Analysis — Four-Pillar Architecture

---

### LEVEL 0: USER INTERFACE (Presentation)

```
├── 0.1  Python API (top-level)
│   ├── SysdModel, parse_sysd_file, parse_sysd
│   ├── TripleStore, parse_turtle
│   ├── cumulative_fusion, EvidenceMatrix
│   └── ScenarioComparison, causalcross_trace, detect_feedback_loops
│
├── 0.2  Domain Configuration (domain.py)
│   └── reasoning parameter tuning via contextvars
│
├── 0.3  Plugin Registry (registry.py)
│   └── custom builtins and DES hook registration
│
├── 0.4  Example Scripts (1 script)
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
│   │   ├── Delays: SMOOTH, SMOOTHI, DELAY3, DELAYN, DELAY_FIXED
│   │   └── Lookup tables: WITH LOOKUP / table keyword
│   ├── 1.1.3  Line-oriented indent-based block parser
│   │   ├── stock keyword — state variables
│   │   ├── flow keyword — rate equations (inflow/outflow)
│   │   ├── aux keyword — algebraic intermediates
│   │   ├── table keyword — lookup/interpolation tables
│   │   ├── model/endomodel/end — scoping blocks
│   │   └── submodel / include — modular composition
│   ├── 1.1.4  Vensim .mdl import (vensim.py)
│   │   ├── INTEG stock parsing
│   │   ├── WITH LOOKUP table mapping
│   │   ├── TIME → t mapping
│   │   ├── SMOOTH/DELAY1 mapping
│   │   └── Continuation line joining
│   └── 1.1.5  Validation (dsl.py: validate())
│       ├── Name resolution (all refs exist)
│       ├── Flow conservation (inflow/outflow balance)
│       ├── Non-negativity bounds
│       └── Zero-divisor detection
│
├── 1.2  Simulation Engine
│   ├── 1.2.1  ODE Solvers (equations.py)
│   │   ├── RK4 (4th order Runge-Kutta)
│   │   └── Euler (1st order forward)
│   ├── 1.2.2  CompiledSystem Cache (dsl.py)
│   │   ├── Code object pre-compilation (compile ... eval)
│   │   ├── Topological sort for aux evaluation order
│   │   ├── SMOOTH/DELAY ODE parameter resolution
│   │   └── Pipeline delay buffer management
│   ├── 1.2.3  Units Checking (units.py)
│   │   ├── Unit dataclass (dimensions, exponents)
│   │   ├── UnitRegistry (compatible units, conversions)
│   │   ├── UnitChecker (~Unit~ annotation parsing)
│   │   ├── Expression-level propagation
│   │   └── Stock = flow × time consistency
│   ├── 1.2.4  Submodels / Modularity (dsl.py)
│   │   ├── SubmodelDef — nested scoped blocks
│   │   ├── IncludeDef — multi-file composition with params
│   │   ├── Word-boundary name prefixing
│   │   ├── _FUNC_NAMES exclusion from replacement
│   │   └── Longest-first replacement sorting
│   ├── 1.2.5  CSV I/O (dsl.py)
│   │   ├── import_data — load external time series
│   │   ├── get_imported_interpolator — linear interpolation
│   │   ├── export_results — save simulation to CSV
│   │   └── Clamp at boundary (no extrapolation)
│   ├── 1.2.6  Parameter Override Protocol
│   │   ├── params dict merged into _s state vector
│   │   ├── Aux numeric expressions overridable via params
│   │   └── Numerical params injected into eval namespace
│   └── 1.2.7  Unified State Dict Paradigm
│       ├── _s: stocks + params merged
│       ├── _a: aux values (computed, topo-sorted)
│       ├── _ns: eval namespace (functions, constants)
│       └── Cross-paradigm sharing (DES/ABM metrics in _s)
│
├── 1.3  ABM — Agent-Based Modeling (agent.py)
│   ├── 1.3.1  AgentDef — agent type definition
│   │   ├── count (population size)
│   │   └── stochastic initialization
│   ├── 1.3.2  AgentPropDef — numeric-only properties
│   │   ├── name, initial value
│   │   ├── min/max clamping bounds
│   │   └── per-agent stochastic seeding
│   ├── 1.3.3  AgentRuleDef — per-step behavior rules
│   │   ├── Conditions: expressions, "always" keyword
│   │   ├── Effects: +=, -=, *=, /=, = (absolute set)
│   │   ├── Condition evaluation via safe eval
│   │   └── Multi-rule precedence (indent-based)
│   ├── 1.3.4  AgentInstance — per-agent runtime
│   │   ├── perceive — read shared state
│   │   ├── decide — evaluate condition tree
│   │   └── act — apply effects with deltas
│   ├── 1.3.5  ABMEngine — population management
│   │   ├── step() — all agents perceive/decide/act
│   │   └── get_metrics() — aggregated statistics
│   └── 1.3.6  Aggregated Metrics
│       ├── {type}_{prop}_avg, _sum, _min, _max, _var
│       └── {type}_count
│
├── 1.4  DES — Discrete Event Simulation (des.py)
│   ├── 1.4.1  Event Queue (DESClock, EventQueue)
│   │   ├── Time-sliced (step-based) processing
│   │   └── Future event list
│   ├── 1.4.2  Queue (single-server + multi-server)
│   │   ├── capacity (max entities)
│   │   ├── service_time expression (compiled per step)
│   │   ├── arrival_rate expression (reads state)
│   │   ├── fill_servers (multi-server with _in_service tracking)
│   │   ├── enqueue / dequeue / drop
│   │   └── per-step departure processing
│   ├── 1.4.3  Resource Pool
│   │   ├── capacity
│   │   ├── acquire / release
│   │   └── utilization tracking
│   ├── 1.4.4  DESEngine — orchestration
│   │   ├── step() — process arrivals → service → departures
│   │   └── get_metrics() — queue lengths, utilization
│   └── 1.4.5  Statistics
│       ├── QueueStats: length_history, total_arrivals, total_departures, total_dropped
│       └── ResourceStats: utilization over time
│
├── 1.5  Structural Analysis
│   ├── 1.5.1  Causal Tracing (causal.py)
│   │   ├── causes_tree — forward causes of a variable
│   │   ├── effects_tree — downstream effects
│   │   ├── causes_strip — direct causes only
│   │   └── causal_trace — full bidirectional trace
│   ├── 1.5.2  Feedback Loop Detection (feedback.py)
│   │   ├── detect_feedback_loops — DFS cycle detection
│   │   ├── loops_for_variable — loops containing a variable
│   │   └── Polarity analysis (reinforcing/balancing)
│   ├── 1.5.3  SD Ontology (ontology.py)
│   │   ├── Stock/flow subtype inference
│   │   ├── MATERIAL / INFORMATION / FINANCIAL classification
│   │   └── Cross-type flow validation
│   └── 1.5.4  BFO Alignment (bfo.py)
│       └── Basic Formal Ontology continuant category mapping
│
├── 1.6  Emergent Properties (emergent.py)
│   ├── EmergentProperty — Condition + Effect pairs
│   ├── Condition — expression + ComparisonOp + threshold
│   ├── Effect — EffectType + expression
│   ├── ConsistencyResult / ConsistencyViolation
│   └── run_consistency_checks — 4 checker functions
│
├── 1.7  Optimization (optimization.py)
│   ├── lp_minimize — Linear Programming (scipy.optimize.linprog)
│   ├── calibrate — parameter calibration against targets
│   └── optimize — constrained optimization (Nelder-Mead + bounds)
│
├── 1.8  Sensitivity Analysis (sensitivity.py)
│   ├── Method: OAT (One-At-a-Time)
│   ├── Method: Morris (elementary effects)
│   ├── Method: Sobol (Saltelli estimator, first + total order)
│   ├── Method: FAST (Fourier Amplitude Sensitivity Test)
│   ├── Method: RBD (Random Balance Design)
│   ├── Ensemble simulation (simulate_ensemble)
│   ├── Tornado plots
│   └── SensitivityAnalyzer / SensitivityResult
│
├── 1.9  Scenario Comparison (scenario.py)
│   ├── ScenarioDef — name + param overrides
│   ├── ScenarioResult — simulation result per scenario
│   ├── ScenarioComparison — aggregate analysis
│   │   ├── plot_comparison — overlay stocks across scenarios
│   │   ├── plot_deviation — deviation from baseline
│   │   ├── tornado — tornado chart of scenario impact
│   │   ├── summary — numeric comparison table
│   │   └── deviation_table — detailed deviation report
│   └── Factory from compare_dicts
│
├── 1.10  Templates / Signal Chains (signal_chain.py)
│   ├── SignalChain.build() — leading-indicator to outcome model
│   ├── Parameters: trace_expr, detection_delay, decision_lag
│   ├── outcome_threshold, outcome_sensitivity, threshold_direction
│   ├── has_feedback, has_tracking flags
│   └── 9 domain configs (social, financial, operational, etc.)
│
└── 1.11  Simulation Control (controller.py)
    ├── SimulationController — step-by-step control
    ├── pause/resume/set_param at runtime
    └── Gaming mode primitives
```

---

### LEVEL 2: KNOWLEDGE PILLAR (dynafx.knowledge)

```
├── 2.1  RDF Data Model (model.py)
│   ├── RDFNode (abstract base)
│   │   ├── NamedNode(iri) — IRI reference
│   │   ├── BlankNode(id) — anonymous resource
│   │   └── Literal(value, datatype, lang_tag)
│   │       ├── XSD type shortcuts (string, integer, decimal, boolean, double)
│   │       └── Language-tagged strings
│   ├── Triple(s, p, o) — subject, predicate, object
│   │   ├── Equality/hashing ignores opinion
│   │   └── Named graph membership
│   ├── TriplePattern(s?, p?, o?) — pattern matching wildcards
│   ├── Namespace constants: RDF, RDFS, OWL, XSD, PROV
│   └── Immutable frozen dataclass design
│
├── 2.2  TripleStore (store.py)
│   ├── 2.2.1  Storage
│   │   ├── SPO / POS / OSP nested-index strategy
│   │   ├── Named graphs: dict[str, set[Triple]]
│   │   └── Opinion dedup: max-belief for identical (s,p,o)
│   ├── 2.2.2  Pattern Matching (8 patterns)
│   │   ├── All patterns: (s,p,o), (s,p,?), (s,?,o), ...
│   │   └── O(1) prefix lookup via nested dicts
│   ├── 2.2.3  Graph Operations
│   │   ├── add / remove / query triples
│   │   ├── create_graph / remove_graph / copy_graph
│   │   ├── all_triples() iterator
│   │   └── Graph isolation and freeze
│   ├── 2.2.4  Named Graph (Source) Management
│   │   ├── Per-source isolation
│   │   ├── Cross-graph fusion entry points
│   │   └── Meta graph (for KBT reliability scores)
│   └── 2.2.5  SPARQL Protocol integration
│       └── Bindings → QueryResult
│
├── 2.3  Turtle / N-Triples (turtle.py)
│   ├── 2.3.1  Tokenizer
│   │   ├── IRI tokens (< >, prefixed names)
│   │   ├── Literals (string, integer, decimal, boolean, typed, lang)
│   │   ├── Blank nodes ([] and _:id)
│   │   ├── Keywords: a (rdf:type), true, false
│   │   └── Grouping: ; (predicate), , (object)
│   ├── 2.3.2  Recursive Descent Parser
│   │   ├── @prefix / @base directives
│   │   ├── Base IRI resolution
│   │   ├── Empty-prefix PNAME_LN (:s)
│   │   └── Comments (#)
│   ├── 2.3.3  Serializer
│   │   ├── Turtle format (prefix + ; grouping)
│   │   ├── N-Triples format
│   │   └── Proper indentation and line breaks
│   └── 2.3.4  NTriples Parser
│       └── Strict line-per-triple format
│
├── 2.4  SPARQL (sparql.py)
│   ├── 2.4.1  Lexer
│   │   ├── Case-insensitive keywords
│   │   ├── SELECT, WHERE, FILTER, prefixes
│   │   └── Variable tokens (?var, $var)
│   ├── 2.4.2  Recursive Descent Parser
│   │   ├── SELECT clause (projection vars)
│   │   ├── WHERE clause (triple patterns)
│   │   ├── FILTER expressions (=, !=, <, >, &&, ||, !)
│   │   └── PREFIX declarations
│   ├── 2.4.3  Algebra Tree
│   │   ├── BGP (Basic Graph Pattern)
│   │   ├── Filter node
│   │   └── Project node
│   ├── 2.4.4  Evaluator
│   │   ├── TripleStore query over named graphs
│   │   ├── Filter evaluation with binding
│   │   ├── DISTINCT, LIMIT, OFFSET
│   │   └── QueryResult (bindings + metadata)
│   └── 2.4.5  Query Grading (bridge to confidence)
│       └── grade_query — SL confidence per result row
│
├── 2.5  Inference Engine (inference.py)
│   ├── 2.5.1  Rule System
│   │   ├── Rule(head, body) — forward-chaining production rules
│   │   ├── Var — logic variable with unification
│   │   ├── InferencePattern — triple pattern with vars
│   │   └── RuleEngine — forward-chaining until fixpoint
│   ├── 2.5.2  RDFS Rule Set (7 rules)
│   │   ├── rdfs:subClassOf → rdf:type transitivity
│   │   ├── rdfs:subPropertyOf transitivity
│   │   ├── rdfs:domain / rdfs:range inference
│   │   └── Type propagation
│   ├── 2.5.3  OWL RL Rule Set (4 rules)
│   │   ├── owl:equivalentClass → mutual subclass
│   │   ├── owl:equivalentProperty → mutual subproperty
│   │   ├── owl:inverseOf
│   │   └── owl:TransitiveProperty
│   └── 2.5.4  Opinion Propagation
│       ├── min — conservative (most skeptical)
│       ├── product — multiplicative
│       └── average — balanced
│
├── 2.6  Confidence Layer (confidence.py)
│   ├── fuse_graphs — merge named graphs via SL
│   │   ├── Per-triple cumulative_fusion
│   │   ├── Conflict detection
│   │   └── Confidence-weighted output
│   ├── grade_query — SL confidence per SPARQL result
│   │   ├── Per-binding confidence score
│   │   └── Overall query confidence
│   └── argumentative_filter — filter triples through framework
│       ├── Accept/reject based on grounded semantics
│       └── Source reliability threshold
│
├── 2.7  Type Hierarchy (hierarchy.py)
│   ├── TypeNode — hierarchical type graph
│   ├── TypeHierarchy — OWL2-style class hierarchy
│   │   ├── subsumption testing
│   │   ├── LCA (Least Common Ancestor)
│   │   └── Ancestor/descendant traversal
│   └── MDM_TYPE_HIERARCHY — multi-domain MDM ontology
│
└── 2.8  TBox Loader (loader.py)
    ├── TBox — terminology box (schema-level triples)
    ├── load_tbox — load from file
    └── GENERAL_TBOX — domain-agnostic default
```

---

### LEVEL 3: EPISTEMICS PILLAR (dynafx.epistemics)

```
├── 3.1  Subjective Logic Algebra
│   ├── 3.1.1  SL Operators (sl_operators.py)
│   │   ├── Opinion(b, d, u, a) — belief/disbelief/uncertainty/base rate
│   │   ├── cumulative_fusion — ⊕ operator for belief merging
│   │   ├── consensus_compromise — weighted average
│   │   ├── discounting — trust-weighted opinion transfer
│   │   └── product logic bridge
│   └── 3.1.2  SL Parameters (sl_params.py)
│       ├── bind_parameters — configure SL tuning params
│       └── get_parameter_summary — diagnostic report
│
├── 3.2  Product Logic (product_logic.py)
│   ├── ProductOperator — fuzzy logic connectives
│   ├── ProductAtom — atomic product expressions
│   └── Bridge to SL (Opinion → truth value)
│
├── 3.3  Evidence Matrix (evidence.py)
│   ├── ClaimAssessment — opinion per claim per source
│   ├── PairwiseAgreement — L1-distance agreement scores
│   ├── ConsensusLevel — unanimous / strong / weak / none / conflict
│   └── EvidenceMatrix — multi-source, multi-claim analysis
│       ├── Pairwise agreement computation
│       ├── Cumulative fusion per claim
│       ├── Consensus classification
│       └── EvidenceMatrixResult
│
├── 3.4  Fusion (fusion.py)
│   ├── cumulative_fusion — sequential ⊕ fusion
│   ├── consensus_compromise — averaging fusion
│   ├── classify_fusion_situation — conflict/agreement/unknown
│   └── consensus_to_fusion_situation — bridge from EvidenceMatrix
│
├── 3.5  Argumentation Framework (argumentation.py)
│   ├── Argument — claim + premises + opinion
│   ├── Attack — relation between conflicting arguments
│   │   ├── AttackType: rebut (contradictory), undermine (low belief),
│   │   │             undercut (exception)
│   │   └── SupportType: entail, justify, strengthen
│   └── ArgumentationFramework — Dung semantics
│       ├── Grounded semantics (unique, skeptical)
│       ├── Preferred semantics (multiple, credulous)
│       ├── Complete semantics
│       └── build_framework — factory from triple store
│           ├── Rebut attacks from contradictory claims
│           ├── Undermine attacks from low-belief triples
│           └── Source reliability attacks from KBT scores
│
├── 3.6  Knowledge-Based Trust (kbt.py)
│   ├── KBTResult — source reliability scores
│   └── compute_kbt — EM algorithm
│       ├── E-step: infer likely true values (weighted vote)
│       ├── M-step: recompute source trust from accuracy
│       ├── Convergence check
│       └── Write prov:reliability triples to meta graph
│
├── 3.7  SL Validation (sl_validation.py)
│   ├── ArgumentType — support categories
│   ├── ValidationArgument — structured argument
│   ├── ValidationAttack — structured attack
│   ├── ValidationResult — overall validation outcome
│   ├── ValidationResultDetail — per-claim details
│   ├── validate_system_internal — full system walk
│   └── get_validation_summary — human-readable summary
│
├── 3.8  Reasoning Modes (modes.py, mode_operators.py)
│   ├── ReasoningMode — enum of reasoning styles
│   ├── Mode operators — composition and transformation
│   └── Mode-specific SL parameter profiles
│
├── 3.9  Graph Operations (graph_ops.py)
│   ├── list_nodes / get_node / get_edge
│   ├── query_contested — find disputed claims
│   ├── query_by_role — filter by epistemic role
│   ├── get_trace_history — audit trail
│   ├── create_node / create_edge / set_role / set_parameter
│   ├── merge_nodes — entity resolution
│   └── retract — remove with cascade
│
├── 3.10  Validation Helpers (validators.py)
│   └── Standalone validation utilities
│
└── 3.11  Visualizer (visualizer.py)
    └── Argumentation graph visualization
```

---

### LEVEL 4: CORE INFRASTRUCTURE (dynafx.core)

```
├── 4.1  Foundational Data Models (models.py)
│   ├── Opinion(b, d, u, a) — SL opinion (shared across pillars)
│   ├── Graph — directed labeled graph
│   │   ├── Node (type: CONCEPT, RELATION, ROLE, RULE)
│   │   └── Edge (type: CAUSAL, HIERARCHICAL, TEMPORAL, SEMANTIC)
│   ├── NodeType — CONCEPT, RELATION, ROLE, RULE, HIDDEN
│   ├── EdgeType — CAUSAL, HIERARCHICAL, TEMPORAL, SEMANTIC, FEEDBACK
│   ├── Entity — base for typed entities
│   ├── EmergentProperty — threshold-based emergence detection
│   ├── FusionSituation — conflict/agreement classification
│   ├── ReasoningMode — epistemic mode enum
│   └── Span / WorldRelation — spatiotemporal grounding
│
├── 4.2  System Decomposer (decomposer.py)
│   ├── add_node / add_edge — pure graph construction
│   └── No extraction noise — manual API only
│
├── 4.3  Math Utilities (math.py)
│   └── Numerical helper functions
│
└── 4.4  Configuration (config.py)
    └── System-wide defaults
```

---

### CROSS-CUTTING CONCERNS

```
├── Unified State Dict
│   └── SD stocks + ABM metrics + DES metrics → single dict
│
├── Naming Convention Restructuring
│   └── dynamics/ (was system/), knowledge/ (was kb/),
│       epistemics/ (was reason/ + sl/ + tbox/)
│       Deprecation warning shims for legacy names
│
├── Framework vs Model Library
│   └── 29 pre-built .sysd models as reference
│
└── Makefile
    ├── test — pytest
    └── typecheck — pyright
```

---

### Test Coverage Summary

| Area | Test Files | Approx. Tests |
|------|-----------|---------------|
| DSL / parsing / simulation | test_dsl.py | ~160 |
| ABM | test_abm.py | ~30 |
| DES | test_des.py | ~105 |
| Cross-paradigm | test_crossparadigm.py | ~11 |
| Causal tracing | test_causal.py | ~15 |
| Feedback loops | test_feedback.py | ~8 |
| Scenario comparison | test_scenario.py | ~20 |
| Sensitivity | test_sensitivity.py | ~40 |
| Optimization | test_optimization.py | ~12 |
| Units checking | test_units.py | ~40 |
| Submodels | test_submodules.py | ~11 |
| CSV I/O | test_csv_io.py | ~12 |
| Math functions / delays | test_math_functions.py, test_delays.py | ~35 |
| Accuracy | test_accuracy.py | ~8 |
| Reference models | test_reference_models.py | ~10 |
| Vensim import | test_vensim.py | ~6 |
| Emergent properties | test_emergent.py | ~8 |
| **KB — RDF model** | test_kb_model.py | ~41 |
| **KB — TripleStore** | test_kb_store.py | ~40 |
| **KB — Turtle** | test_kb_turtle.py | ~40 |
| **KB — SPARQL** | test_kb_sparql.py | ~35 |
| **KB — Inference** | test_kb_inference.py | ~53 |
| **KB — Confidence** | test_kb_confidence.py | ~30 |
| **Epistemics — Evidence** | test_evidence.py, test_evidence_matrix.py | ~35 |
| **Epistemics — Fusion** | test_fusion.py | ~15 |
| **Epistemics — Argumentation** | test_argumentation.py | ~27 |
| **Epistemics — KBT** | test_kbt.py | ~14 |
| **Epistemics — SL validation** | test_validation.py | ~20 |
| **Epistemics — modes** | test_reasoning_modes.py | ~8 |
| **Core** | test_math.py, test_graph.py, test_plugins.py, test_registry.py | ~45 |
| Controller | test_controller.py | ~10 |
| Complexity | test_complexity_metrics.py | ~5 |
| **Total** | **42 test files** | **~1,146** |
