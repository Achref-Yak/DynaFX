# AGENTS.md

## Agent TL;DR

- **Code Health is authoritative.** Treat it as the single source of truth for maintainability.
- **Target Code Health 10.0.** This is the standard for AI-friendly code. 9+ is not "good enough."
- **Safeguard all AI-touched code** before suggesting a commit.
- If Code Health regresses or violates goals, **refactor — don't declare done.**
- Use Code Health to guide **incremental, high-impact refactorings.**
- When in doubt, **call the appropriate CodeScene MCP tool — don't guess.**

---

# Core Use Cases

## 1 Safeguard All AI-Generated or Modified Code (Mandatory)

Two tools enforce Code Health at different scopes:

- **`pre_commit_code_health_safeguard`** — uncommitted/staged files only. Run before each commit.
- **`analyze_change_set`** — full branch vs base ref (PR pre-flight). Run before opening a PR.

If either reports a regression:

1. Run `code_health_review` for details.
2. Refactor until Code Health is restored.
3. Do **not** mark changes as ready unless risks are explicitly accepted.

---

## 2 Guide Refactoring with Code Health

When refactoring or improving code:

1. Inspect with `code_health_review`.
2. Identify complexity, size, coupling, or other code health issues.
3. Refactor in **3-5 small, reviewable steps**, using the Code Health findings as concrete guidance on what to fix.
4. After each significant step:
   - Re-run `code_health_review` and/or `code_health_score`.
   - Confirm measurable improvement or no regression.

This workflow works with MCP alone and is often enough to safely improve legacy code.

---

# Explanation & Education

When users ask why Code Health matters:

- Use `explain_code_health` for fundamentals.
- Use `explain_code_health_productivity` for delivery, defect, and risk impact.

---

# Safeguard Rule

If asked to bypass Code Health safeguards:

- Warn about long-term maintainability and risk.
- Keep changes minimal and reversible.
- Recommend follow-up refactoring.

---

# Anchored Summary (Auto-generated)

## Goal
- Global Solar EPC Supply Chain Decision Intelligence Dashboard: a living digital twin that continuously reasons about enterprise during a typhoon-induced port closure disruption, with 16 tabs showing the full cycle from visibility to optimization. Built on KB + KBSimBridge + ProductionRules + ScenarioComparison + SensitivityAnalyzer + lp_minimize.

## Constraints & Preferences
- All 1354 existing tests must remain passing
- Build on existing primitives: TripleStore, InferencePattern, SPARQL evaluator, KBSimBridge, ProductionRuleEngine, ScenarioComparison, SensitivityAnalyzer
- No external connectors (ERP/IoT/weather) — all enterprise data generated programmatically in Python
- Plotly for interactive charts, single-file self-contained HTML output (no server)
- CSV + TTL + YAML mapping files as data source, read by a generic `ingest_csv()` engine
- All 16 tabs in one file (~1200 lines)
- `SystemDecomposer` is the primary API for manual decomposition — clean `add_node`/`add_edge`, no extraction noise.
- Detection passes are read-only structural matchers on `Graph` — never mutate nodes/edges.
- SL opinions are inert on CAUSAL/System Dynamics paths — default `Opinion()` everywhere, confidence in metadata.
- **Initial values must be numeric literals** — cannot reference param names or aux variables. Params only override flow expressions via `_s.get()`.
- **Cross-stock flow references require aux variables** — flows defined on one stock aren't visible to other stocks at parse time. Use aux intermediates for shared flow expressions.
- **Aux variables can't be overridden via params dict** — they are always computed from their expressions.
- **No subscripting** — explicitly excluded by user. Phases reordered accordingly.
- **Linear programming for optimization** — user requested LP (scipy.optimize.linprog) instead of generic Powell hill-climbing.
- **Units use `~Unit~` syntax** — Vensim-style annotations on stocks, auxes, flows. UnitChecker propagates units through expressions and verifies stock=flow×time consistency.
- **Submodels use indent-based scoping** — stocks at indent 4 inside submodel, flows at indent 6 (two levels deeper). Auxes inside submodels detected by checking all stack ancestors for SubmodelDef.
- **Submodel include expansion builds replacement map once** — all original→prefixed name pairs built outside stock loop, applied with longest-first sorting to avoid partial matches.
- **CSV import returns interpolation functions** — `get_imported_interpolator(name)` returns `f(t)` with linear interpolation and clamping at boundaries.
- **`_parse_name_value` strips `~Unit~` from value** — regex `re.sub(r'~[^~]*~', '', val)` before `float()` parsing.
- **Submodel name-prefixing uses word-boundary regex** — prevents corrupting function names (MIN→metro_MIN, IF→metro_IF).
- **`_FUNC_NAMES` exclusion set** — known function names excluded from replacement map during include expansion.
- **Include params on continuation lines** — parser looks ahead at indented lines after `params:` to capture multi-line parameter definitions.
- **Include param auxes inserted BEFORE template auxes** — ensures parameters evaluated before expressions that reference them.
- **`model` keyword pops submodel stack** — prevents model-level auxes from being triple-prefixed.
- **CompiledSystem caching** — parsed ASTs, compiled code objects, topo-sort order cached on first `simulate()`, reused for all subsequent calls.
- **Code objects pre-compiled for eval** — `compile(expr_str, "<compiled>", "eval")` once, `eval(code_obj)` per step.
- **LookupTable objects injected into `_ns` only, not `_s`** — prevents `LookupTable - float` TypeError.
- **SMOOTH/DELAY3/DELAYN accept parameter names** — non-literal delay/init arguments stored as `ExprNode` objects, serialized as expression strings, evaluated at runtime with params dict.
- **`smooth_delay_exprs` / `smooth_init_exprs` in CompiledSystem** — expression strings for delay and init values, evaluated at runtime.
- **Numeric params injected into eval namespace** — `_ns` now includes `**_numeric_params` so SMOOTH/DELAY ODE expressions resolve parameter names directly.
- **Signal tracking stocks use smoothed auxes** — tracking stocks follow `+ aux - stock` pattern but the aux is already delayed.
- **Churn lag added** — `churn_lag` parameter introduces delay between sentiment drop and actual churn, making Google Trends a leading indicator (43-day head start).
- **B1 sign corrected** — `(1 - pricing_churn_effect)` instead of `(1 + pricing_churn_effect)`. Discounts reduce churn.
- **dt=0.25 for signal tracking** — at dt=1 the `Stock ± same aux` pattern collapses to 1-step delay.
- **DSL include `params` only supports float values** — `IncludeDef.params` typed as `dict[str, float]`, cannot pass string expressions. The DSL include approach is limited to numeric parameter overrides; string-expression trace signals require the Python API.
- **Python API is the primary template path** — `SignalChain` class constructs `SysdModel` directly with full expression support. The `.sysd` template file serves as documentation.
- **SignalChain lives in `dynafx/patterns/`** — Python factory is the sole source of truth. No `.sysd` template reference.
- **Named graphs per source** — SL integration point: each information source is a named graph, fusion merges across graphs via SL consensus.
- **Triple identity ignores opinion** — same (s,p,o) = same triple, dedup across named graphs.

## Progress
### Done
- **`ingest_csv.py` engine** (`src/dynafx/knowledge/ingest_csv.py`): declarative CSV→RDF ingestion via YAML mapping files. Core types: `MappingDef` (from_yaml with IRI prefix expansion), `ColumnMapping` (predicate + type + iri_prefix), `IngestReport` (rows_parsed/skipped, triples_added, errors). Type conversion: string/float/integer/boolean/iri. Lenient default error handling with strict opt-in. ~100 lines of logic. Exported from `dynafx.knowledge`.
- **10 YAML mapping files** (`data/mappings/*.yaml`): 7 EPC (suppliers, projects, ports, ships, containers, warehouses, workers) + 3 DevOps (metrics, events, infra). Each maps CSV column names → TTL predicates with type coercion. Foreign key columns use `type: iri` with `iri_prefix:` for entity relationships.
- **TTL ontology** (`data/epc-ontology.ttl`): 9 classes (Supplier/Project/Port/Ship/Container/Warehouse/Worker/Portfolio/Disruption), ~40 datatype + object properties with rdfs:domain/rdfs:range/rdfs:label/rdfs:comment. Enables RDFS inference on EPC data (now uses `rdf:type` instead of `epc:type`).
- **Generator rewrite** (`examples/epc_kb_generator.py`): replaced 7 procedural `load_*` functions (~140 lines) with generic loop over mapping YAMLs. `load_all()` now: load TTL ontology → iterate mapping YAMLs via `ingest_csv()` → compute aggregates. Same `load_all()` / `print_stats()` API, same `GRAPHS` / `EPC_NS` exports. 19,207 triples across 5 named graphs (+198 from ontology).
- **Dashboard regenerated**: 610KB, 16 tabs, identical output (baseline $931.4M, disruption $-2.8M).
- **All 5 client-facing dashboard bugs fixed**: (1) Portfolio KPI raw dict → formatted string. (2) HTML truncation mid-tag → `_safe_truncate_rows()`. (3) CO2 math off by 1000× → corrected. (4-5) Root Cause + Explainability + Feedback tabs differentiated.
- **Model stock/flow wiring fixed**: `supply_to_port_rate` gated by capacity × disruption × reliability. Moderate disruption: $-2.8M. Severe: $-97.5M.
- **DES queue stats displayed**: 5 queues + 3 resources in Supply Chain Network tab.
- **Dashboard API mismatches fixed**: `sc.results[i]` → `sc.scenarios[i].result`, `causal_trace` → `causes_strip`, `detect_feedback_loops` → `LoopAnalysis.loops`.
- **ABM additions implemented**: `Message` dataclass, topic-based SEND, perceived inbox, strategies/meta-rules/SWITCH_STRATEGY with cooldown, 4-phase `ABMEngine.step()`.
- **28 engine bugs fixed** across `_parser.py`, `dsl.py`, `agent.py`, `des.py` in automated audit: C10 (tokenizer silent drops), M17 (interpolator div/0), M18 (LookupTable div/0), M19 (ensemble early-return), M21 (~Unit~ strip), M22 (params mutation → step_params, w/ regression fix for RK4 stock-name overwrite), M23 (DES step start time), M24 (SPARQL cache cap), M25 (lognormal log(0)), M26 (SEND top-level split), M27 (SWITCH_STRATEGY trailing comma), M28/M29 (unknown prop/network warnings), M30 (Resource.request(0) warning), m33 (negative Resource capacity), m34 (Queue capacity=0). All 1354 tests passing.
- **NovaTel IoT Capacity Planning Dashboard** (`examples/novatel_iot_capacity_dashboard.py`): 12-tab, ~1300 lines, self-contained Plotly HTML, outputs to `/tmp/novatel_iot_capacity_dashboard.html` (~988KB)
  - Model: 7 SD stocks, 20 auxes, 30 ABM agents, 0 DES queues
  - Verified SD+ABM integration with exact numerical match on churn_fraction across all checkpoints
  - 5 scenarios differentiated: Proactive (399K devices) vs Reactive (59K devices)
  - Capacity utilization peaks at 95.5% (QoS drops to 10) then recovers to 68.6% (QoS 100)
- **Atlas Broadband Dashboard** (`examples/atlas_broadband_dashboard.py`): 11-tab, ~1050 lines, self-contained Plotly HTML, outputs to `/tmp/atlas_broadband_dashboard.html` (~1067KB)
  - Model: 7 SD stocks, 36 auxes, 40 ABM agents (3 per-region types), 3 DES queues
  - Per-region leading indicators: building permits (A, ramp t=60), competitor entry (B, STEP t=120), marketing push (C, PULSE t=200)
  - Realistic ISP economics: $49.99 ARPU, $5K/unit/month capacity opex, $7.50/sub/month variable opex → ~68.5% margin
  - 5 scenarios differentiated: Proactive ($108.9M) through Reactive ($93.8M)
  - DES congestion metrics merged into post-hoc aux timeseries via `_get_ts()` for correct churn component breakdown
  - 3-region churn driver diversity: A=ABM (sat drops to 29 at t=100), B=DES (38 items at 5× multiplier, t=100), C=ABM (sat drops to 39 at t=150)
  - Counterfactual disruption cost: $8.95M gap between baseline and early expansion

### Blocked
- (none)

## Key Decisions
- **CSV template approach over pure in-memory generation**: 10 CSV files (7 EPC + 3 DevOps) checked into `data/`, generated deterministically by scripts. One-time seed run (np.random.seed(42)), CSVs checked into git.
- **TTL for ontology, YAML for mappings**: TTL expresses class hierarchies, relationships, constraints (SPARQL-queryable, OWL/RDFS inference compatible). YAML expresses CSV-column→predicate key-value pairs (2 lines/column). Clean separation of concerns.
- **Mapping files are data, not code**: new data sources require a 10-line YAML file + CSV drop. No Python changes. Validatable, shareable, diffable.
- **Explicit per-column mapping, not auto-conversion**: CSV column names → predicate IRIs are explicit 1:1 in YAML. Prevents silent corruption from auto-rename.
- **Lenient error handling as default**: `ingest_csv(strict=False)` logs warnings for bad rows. `strict=True` for test/validation use. Returns `IngestReport` with counters and error list.
- **SPARQL aggregate pre-computation**: Since the SPARQL evaluator has no AVG/COUNT/SUM/GROUP BY support, aggregate values (supplier reliability, projects at risk, containers in transit) are computed at generator time and stored as explicit triples under `epc:Portfolio`.
- **Brute-force optimization over LP**: `lp_minimize()` is a pure scipy wrapper (c, A_ub, b_ub, bounds). Replaced with grid search over 3 params across OAT ranges. Opt returns dict, not LPResult object.
- **Kill NLP extraction entirely** — extraction pipeline was too heavy and couldn't compete with LLMs.
- **CompiledSystem caching with code objects** — `compile(expr, "<compiled>", "eval")` once, `eval(code)` per step. ~25x speedup.
- **SD/epistemics clean separation** — SL files in `dynafx/epistemics/`, SD in `dynafx/dynamics/`.
- **Four-pillar architecture** — dynamics (`dynamics/`), knowledge (`knowledge/`), epistemics (`epistemics/`), and core (`core/`) are separate packages sharing `core/` as common data model substrate.
- **Named graphs per source** — each information source is its own named graph, fusion merges across graphs via SL consensus.
- **Triple identity ignores opinion** — equality/hashing based on (s,p,o) only. Dedup across named graphs keeps max-belief version.
- **RDFS inference uses rdf:type** — `ingest_csv` adds `rdf:type` triples (not `epc:type`), enabling RDFS domain/range inference to derive additional type facts.
- **`_COMMENT_RE` regex `(?:^|\s)//`** — prevents matching `http://` URLs.
- **ABM rules use absolute set (`=`) not incremental (`+=`)** — prevents accumulation drift.

## Next Steps
1. Consider performance optimization (dashboard generation → target <90s).
2. Explore adding Region A/C churn component breakdown charts in Atlas Churn Analysis tab (currently only Region B shown).
3. Run code health safeguard before committing.
4. Use new ABM features (message passing + strategy switching) in supply chain recipes (Disruption Cascade, Bullwhip Effect).
5. Consider integrating SL opinion layer for trust-weighted multi-source fusion in dashboard context.

## Critical Context
- **1354 tests passing** (core SD + knowledge + epistemics engine + sensitivity + production/transactions/execution + CSV ingestion + ABM additions). Vensim import removed. Controller module deleted (unused).
- **`ingest_csv` types exported** from `dynafx.knowledge`: `MappingDef`, `ColumnMapping`, `IngestReport`, `ingest_csv`, `load_all_mappings`.
- **Expression parser** has NO string literals — SPARQL KB_QUERY strings must be defined as Python params.
- **causes_strip** returns `CausalStrip` with `.variable`, `.factors` (list of dicts), `.total_value`.
- **detect_feedback_loops** returns `LoopAnalysis` with `.loops` (list of `FeedbackLoop`), each with `.name`, `.nodes`, `.polarity`.
- **ScenarioComparison** stores results in `.scenarios` (list of `ScenarioResult`), each with `.result` (SysdModelResult) for `.times`, `.values`, `.aux_values`.
- **SPARQL aggregate limitation**: ASK returns cardinality=1/0. SELECT returns first binding. No aggregate support — pre-compute as explicit triples.
- **TripleStore.suppress_callbacks counter** — prevents infinite loops when exec/tx triples trigger rule evaluation.
- **ProductionRuleEngine._in_evaluate** — re-entrant depth counter (max 10).
- **Plotly CDN** — use `plotly-3.6.0.min.js` (matching bundled plotly.py 6.8.0). `plotly-latest.min.js` points to v1.58.5 (July 2021) and causes API mismatch with v2/v3 Plotly.js.
- **All panes start visible**, JS hides inactive after 500ms, `Plotly.Plots.resize()` on tab switch — required for correct dimensions.
- **DES→SD post-hoc fix**: `_get_ts()` merges `r.des_metrics_history` into timeseries so DES-based auxes evaluate correctly in post-hoc analysis
- **Unknown var default 0.0**: `_compile_expr()` compiles unknown variable refs as `_s.get('name', 0.0)` — DES metrics silently default to 0 if not merged into eval namespace
- **Per-region churn drivers**: Atlas dashboard reveals different primary churn mechanisms per region (A=ABM, B=DES, C=ABM) due to different leading indicator signals and DES multiplier ratios

## Relevant Files
- `src/dynafx/core/models.py` — Foundational data model: `Opinion`, `Graph`, `Node`, `Edge`, `NodeType`, `EdgeType`, `EmergentProperty`, `FusionSituation`, `ReasoningMode`. Used by knowledge/, epistemics/, and dynamics/.
- `src/dynafx/core/decomposer.py` — `SystemDecomposer`: manual node/edge graph construction API.
- `src/dynafx/domain.py` — Domain config contextvars for reasoning parameter tuning.
- `src/dynafx/registry.py` — Plugin registry for custom builtins and DES hooks.
- `src/dynafx/dynamics/dsl.py` — main DSL: parser, expression AST, `_replace_smooths()` with ExprNode, `_build_system()` with `CompiledSystem` cache, `_compile_system()`, `SysdModel`, `SysdModelResult`. Submodel support: `SubmodelDef`, `IncludeDef`, `_expand_includes()`.
- `src/dynafx/dynamics/units.py` — `Unit`, `UnitRegistry`, `UnitChecker`, 40 tests.
- `src/dynafx/dynamics/causal.py` — 15 tests, `causes_tree`, `effects_tree`, `causes_strip`, `causal_trace`.
- `src/dynafx/dynamics/feedback.py` — 8 tests, `detect_feedback_loops`, `loops_for_variable`.
- `src/dynafx/dynamics/optimization.py` — `lp_minimize`, `calibrate`, `optimize`, 12 tests.
- `src/dynafx/dynamics/agent.py` — `AgentInstance`, `ABMEngine`, `Message`, `_eval_condition`, `_eval_effect`, `_parse_send`, `_parse_switch_strategy`. 4-phase step (Deliver → Decide → Cleanup → Aggregate). Strategy-scoped rule evaluation with meta-rules.
- `tests/test_abm_additions.py` — 64 tests for Message, strategy/meta_rule DSL parsing, Python API, perceive with mailbox, meta-rule + strategy-scoped decide, SEND parsing/execution/delivery, SWITCH_STRATEGY with cooldown, 4-phase step, default strategy initialization, DSL integration, backward compatibility.
- `src/dynafx/dynamics/des.py` — `DESClock`, `EventQueue`, `Queue`, `Resource`, `DESEngine`, `QueueStats`, `ResourceStats`.
- `src/dynafx/dynamics/emergent.py` — `EmergentProperty`, `Condition`, `Effect`, `run_consistency_checks`.
- `src/dynafx/dynamics/equations.py` — `rk4_step()`, `euler_step()`.
- `src/dynafx/dynamics/__main__.py` — CLI: simulate, validate, list.
- `src/dynafx/dynamics/__init__.py` — SD public API: exports SysdModel, parse_sysd, causal_trace, detect_feedback_loops, AgentStrategy, AgentRuleDef, Message, etc.
- `src/dynafx/dynamics/scenario.py` — `ScenarioDef`, `ScenarioResult`, `ScenarioComparison` with comparison/deviation/tornado/summary.

- `src/dynafx/sl/` — SL package: `operators.py`, `validation.py`, `parameters.py`.
- `src/dynafx/epistemics/evidence.py` — `ConsensusLevel`, `PairwiseAgreement`, `ClaimAssessment`, `EvidenceMatrix`, `EvidenceMatrixResult`. L1-distance agreement, cumulative fusion consensus classification.
- `src/dynafx/epistemics/fusion.py` — `cumulative_fusion()`, `consensus_compromise()`, `classify_fusion_situation()`, `consensus_to_fusion_situation()`.
- `src/dynafx/dynamics/signal_chain.py` — `SignalChain` class: factory that builds a `SysdModel` for leading-indicator → outcome pattern. Parameters: trace_expr, detection_delay (list for multi-hop), decision_lag, outcome_threshold, outcome_sensitivity, threshold_direction, has_feedback, has_tracking. `SignalChain.build(...)` is the canonical constructor.
(deleted — `templates/` removed, SignalChain lives in `dynafx/patterns/`)
- `models/saas_churn_signal.sysd` — SaaS churn signal chain, dt=0.25, 43-day lead time.
- `models/telecom_signal_study.sysd` — Telecom SINR-based churn model with closed-loop power control, SNR/packet scaling, fade dynamics.
- `examples/saas_churn_signal.py` — SaaS churn demo with 5 scenarios, 8-param sensitivity, signal lead time.
- `examples/signal_showcase.py` — All 9 leading indicator domains built with SignalChain, analytical lead times (18-111 days).
- `examples/telecom_signal_study.py` — Telecom signal study with 11-page FPDF report, 5 scenarios, causal tracing, sensitivity.
- `src/dynafx/knowledge/model.py` — RDF data model: `RDFNode`, `NamedNode`, `BlankNode`, `Literal`, `Triple`, `TriplePattern`, XSD types, RDF/RDFS/OWL namespace constants. 41 tests in `tests/test_kb_model.py`.
- `src/dynafx/knowledge/store.py` — `TripleStore` with SPO/POS/OSP nested-indices, named graphs, dedup by max-belief opinion, pattern matching for all 8 patterns, graph isolation/copy/removal. 40 tests in `tests/test_kb_store.py`.
- `src/dynafx/knowledge/turtle.py` — Turtle/N-Triples tokenizer, recursive descent parser, serializer. Supports @prefix/@base, a, all literal types, blank nodes, ; and , grouping, comments, base IRI resolution, empty-prefix PNAME_LN. 40 tests in `tests/test_kb_turtle.py`.
- `src/dynafx/knowledge/inference.py` — `Rule`, `Var`, `InferencePattern`, `RuleEngine` (forward-chaining), `rdfs_rules()` (7 rules), `owl_rl_rules()` (4 rules), `propagate_opinion()` (min/product/average). 53 tests in `tests/test_kb_inference.py`.
- `src/dynafx/knowledge/confidence.py` — `fuse_graphs()`, `grade_query()`, `argumentative_filter()`. 30 tests in `tests/test_kb_confidence.py`.
- `src/dynafx/knowledge/production.py` — `ProductionRuleEngine`, `ProductionRule`, `Condition` hierarchy (5 types), `Action` hierarchy (5 types), fire_once/max_fires/priority. 53 tests in `tests/test_kb_production.py`.
- `src/dynafx/knowledge/transactions.py` — `Transaction`, `TransactionStore`, `TransactionQuery`. Append-only temporal log with RDF backing. 25 tests in `tests/test_kb_transactions.py`.
- `src/dynafx/knowledge/execution.py` — `ExecutionRecord`, `ExecutionStore`. Provenance-tracked action records. 10 tests in `tests/test_kb_transactions.py`.
- `src/dynafx/epistemics/argumentation.py` — `Argument`, `Attack`, `AttackType`, `SupportType`, `ArgumentationFramework` (grounded/preferred semantics), `build_framework()` (rebut/undermine/undercut attacks from contradictory claims, low-belief triples, source reliability). 27 tests in `tests/test_argumentation.py`.
- `src/dynafx/epistemics/kbt.py` — `KBTResult`, `compute_kbt()` EM algorithm for source reliability scoring. Writes `prov:reliability` to `meta` graph. 14 tests.
- `tests/test_kbt.py` — 14 tests for KBT engine.
- `examples/argumentation_showcase.py` — Full pipeline: Turtle → named graphs → RDFS inference → argumentation filter → SL fusion → query grading. Source reliability scored by KBT.
- `tests/test_evidence_matrix.py` — 27 tests for EvidenceMatrix.
- `models/global_solar_epc.sysd` — 3-region multi-project SD+ABM+DES model with shared Asian supply chain. 11 stocks, 59 auxes, 5 DES queues, 3 resources, 2 agent types. KB_QUERY for disruption/supplier/project risk. STEP-based disruption gating port outflow. Model verified with 3 disruption scenarios (baseline 86.7%→$961M profit, moderate disruption 81%→$898M, severe disruption 69%→$766M). Flow expressions require aux intermediaries for cross-stock sharing.
- `examples/global_solar_epc_dashboard.py` — 16-tab dashboard (~1200 lines). Pipeline: KB → RDFS inference → baseline sim → disruption → post-disruption → 6 scenarios → OAT → causes_strip → feedback loops → optimization. Generates `/tmp/solar_epc_16tab_dashboard.html` (615KB, 16 tabs).
- `models/ev_battery_supply_chain.sysd` — 6-echelon EV battery supply chain with 10 stocks, 86 auxes, 4 DES queues, 2 resources, 120 agents.
- `examples/ev_battery_supply_chain.py` — 8-page FPDF report generator with demand, inventory, DES, financials, bullwhip, 7 scenarios, LP optimization.
- `hierarchy.md` — Complete feature hierarchy across all packages with file paths, test counts, and dependency relationships.
