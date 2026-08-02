# AGENTS.md

## Agent TL;DR

- **All 1028 existing tests must keep passing.** Run `uv run pytest` before declaring work done.
- **Docs must stay accurate.** `uv run mkdocs build --strict` must build clean; the GitHub Pages site auto-rebuilds on `docs/**` + `mkdocs.yml` pushes.
- **No epistemics / Subjective Logic references.** The SL layer was removed; docs must not reference it.
- **Refactor with the codebase's own primitives** (TripleStore, InferencePattern, SPARQL evaluator, KBSimBridge, ProductionRuleEngine, ScenarioComparison, SensitivityAnalyzer). Don't reinvent.
- **Verify before commit** — run the verification suite, then commit in small, reviewable steps.

---

# Anchored Summary (Auto-generated)

## Goal
- Global Solar EPC Supply Chain Digital Twin: a living digital twin that continuously reasons about the enterprise during a typhoon-induced port closure disruption, spanning the full L1–L5 spectrum from sense to optimize. Built on KB + KBSimBridge + ProductionRules + ScenarioComparison + SensitivityAnalyzer + lp_minimize.
- SL / epistemics (Subjective Logic, KBT, argumentation, evidence matrix) has been **removed from the codebase**. Docs must not reference it.

## Constraints & Preferences
- All 1028 existing tests must remain passing
- Build on existing primitives: TripleStore, InferencePattern, SPARQL evaluator, KBSimBridge, ProductionRuleEngine, ScenarioComparison, SensitivityAnalyzer
- No external connectors (ERP/IoT/weather) — all enterprise data generated programmatically in Python
- Plotly for interactive charts, single-file self-contained HTML output (no server)
- CSV + TTL + YAML mapping files as data source, read by a generic `ingest_csv()` engine
- `SystemDecomposer` is the primary API for manual decomposition — clean `add_node`/`add_edge`, no extraction noise.
- Detection passes are read-only structural matchers on `Graph` — never mutate nodes/edges.
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
- **Named graphs per source** — each information source is a named graph; queries read across the union.
- **Triple identity is (s,p,o)** — same (s,p,o) = same triple, dedup across named graphs keeps max-belief version.

## Progress
### Done
- **`ingest_csv.py` engine** (`src/dynafx/knowledge/ingest_csv.py`): declarative CSV→RDF ingestion via YAML mapping files. Core types: `MappingDef` (from_yaml with IRI prefix expansion), `ColumnMapping` (predicate + type + iri_prefix), `IngestReport` (rows_parsed/skipped, triples_added, errors). Type conversion: string/float/integer/boolean/iri. Lenient default error handling with strict opt-in. ~100 lines of logic. Exported from `dynafx.knowledge`.
- **7 YAML mapping files** (`data/mappings/*.yaml`): EPC (suppliers, projects, ports, ships, containers, warehouses, workers). Each maps CSV column names → TTL predicates with type coercion. Foreign key columns use `type: iri` with `iri_prefix:` for entity relationships.
- **TTL ontology** (`data/epc-ontology.ttl`): 9 classes (Supplier/Project/Port/Ship/Container/Warehouse/Worker/Portfolio/Disruption), ~40 datatype + object properties with rdfs:domain/rdfs:range/rdfs:label/rdfs:comment. Enables RDFS inference on EPC data (now uses `rdf:type` instead of `epc:type`).
- **Supply chain twin example built** (`examples/global_solar_epc_twin.py`): 12-section L1→L5 pipeline — 7 EPC CSVs → named-graph KB + RDFS inference → `params_from_kb` → baseline/typhoon runs (KB disruption flag) → evidence round-trip → scenario grading/ranking/FILTER → production rules (portfolio-at-risk) → `kb_lp_minimize` → `causes_strip`/feedback loops → `record_provenance` → maturity ladder. Verifies baseline $931.4M / disruption −$2.8M.
- **Model stock/flow wiring fixed**: `supply_to_port_rate` gated by capacity × disruption × reliability. Moderate disruption: $-2.8M. Severe: $-97.5M.
- **ABM additions implemented**: `Message` dataclass, topic-based SEND, perceived inbox, strategies/meta-rules/SWITCH_STRATEGY with cooldown, 4-phase `ABMEngine.step()`.
- **28 engine bugs fixed** across `_parser.py`, `dsl.py`, `agent.py`, `des.py` in automated audit: C10 (tokenizer silent drops), M17 (interpolator div/0), M18 (LookupTable div/0), M19 (ensemble early-return), M21 (~Unit~ strip), M22 (params mutation → step_params, w/ regression fix for RK4 stock-name overwrite), M23 (DES step start time), M24 (SPARQL cache cap), M25 (lognormal log(0)), M26 (SEND top-level split), M27 (SWITCH_STRATEGY trailing comma), M28/M29 (unknown prop/network warnings), M30 (Resource.request(0) warning), m33 (negative Resource capacity), m34 (Queue capacity=0). All 1028 tests passing.
- **DES metrics invisible to aux replay (library bug fixed)** — `src/dynafx/dynamics/dsl.py` (~line 900): DES metrics were merged into per-step `step_params` but never into `params`, so `params_history` (used by post-hoc aux replay) lacked them and any aux referencing `install_{z}_length`/`Orders_length` replayed as 0.0. Added `params.update(des_metrics)` mirroring the ABM merge. Regression test: `tests/test_crossparadigm.py::test_des_metrics_visible_in_aux_replay` (queue arrival_rate=10 → `aux_values["watch"] > 0`).
- **`kb_lp_minimize`/`kb_lp_maximize` numeric-literal unwrap** — `optimization.py::_eval_q` now unwraps SPARQL results (`val = v.value if hasattr(v, "value") else v`). Regression test: `tests/test_kb_sim_bridge_ext.py::test_kb_lp_minimize_reads_numeric_literals`.
- **SL / epistemics removed from codebase** — `dynafx/epistemics/`, `dynafx/sl/`, `knowledge/confidence.py`, `Opinion`/`FusionSituation`, and all epistemics tests deleted. Docs rewritten project-wide (README, docs/, AGENTS.md, hierarchy.md, CHANGELOG.md) to remove every reference.
- **Dead SL-era code removed** — `core/config.py` + `domain.py` (mutually-referencing, zero consumers/tests) deleted; `dynafx/epistemics/` bytecode purged (so `import dynafx.epistemics` now fails); `default_priors.json` package-data entry dropped. `[tool.setuptools.packages.find] where=["src"]` added (stops `tests/` leaking into sdist). Docs fixed: git install URL, `cd DynaFX`, working quick-start/`parse_turtle`/KB_QUERY samples, duplicate `test_delay3_converges_to_input` removed, mypy/pre-commit doc claims dropped, twin LP made deterministic via `rowIndex` + `ORDER BY ?i` (objective $1.67K stable across runs).
- **CLI removed** — `src/dynafx/__main__.py`, `src/dynafx/dynamics/__main__.py`, `[project.scripts]` entrypoint, `TestCLIIntegration` (2 tests), Makefile `run` target, and all doc references deleted.
- **Docs rewrite** — README + docs/ (index, architecture, development, knowledge, digital-twin, examples) rewritten from scratch; GitHub Pages deploy workflow added; `mkdocs.yml` description/nav updated.
- **CI lint fixed (ruff 0.16.1)** — all 347 pre-existing ruff errors in `src/` fixed so `uv run ruff check src/` exits clean: UP045 (Optional→`X | None`), E402 import order, F841/B007 unused vars, SIM108/105/102/101/103 simplifications, C408 `dict()`→literal (unsafe-fix in `disruption_cascade.py`), RUF005 list-concat→star, UP028 `yield from`, RUF034, RUF059, RUF012 `ClassVar` on `_OPS`, B023 closures bound via default args (`_replace_refs` in dsl.py, `_walk_refs` in scenario.py, `_add` in ingest_csv.py). Restored `tokenize` re-export in `sparql.py` (`# noqa: F401`) that auto-fix had removed (test_kb_sparql regression). Local .venv on 3.12 (3.13 interpreter missing `zlib`).

### Blocked
- (none)

## Key Decisions
- **`data/` is the single home for all demo resources, committed to git**: CSVs (`epc_*.csv`), TTL ontology (`epc-ontology.ttl`), YAML mappings (`data/mappings/`), and `.sysd` models (`data/models/`). Generated deterministically by `scripts/generate_epc_csvs.py` (seed=42). A fresh clone can run the demo without generating data.
- **TTL for ontology, YAML for mappings**: TTL expresses class hierarchies, relationships, constraints (SPARQL-queryable, OWL/RDFS inference compatible). YAML expresses CSV-column→predicate key-value pairs (2 lines/column). Clean separation of concerns.
- **Mapping files are data, not code**: new data sources require a 10-line YAML file + CSV drop. No Python changes. Validatable, shareable, diffable.
- **Explicit per-column mapping, not auto-conversion**: CSV column names → predicate IRIs are explicit 1:1 in YAML. Prevents silent corruption from auto-rename.
- **Lenient error handling as default**: `ingest_csv(strict=False)` logs warnings for bad rows. `strict=True` for test/validation use. Returns `IngestReport` with counters and error list.
- **SPARQL aggregate pre-computation**: Since the SPARQL evaluator has no AVG/COUNT/SUM/GROUP BY support, aggregate values (supplier reliability, projects at risk, containers in transit) are computed at generator time and stored as explicit triples under `epc:Portfolio`.
- **Brute-force optimization over LP**: `lp_minimize()` is a pure scipy wrapper (c, A_ub, b_ub, bounds). Replaced with grid search over 3 params across OAT ranges. Opt returns dict, not LPResult object.
- **Kill NLP extraction entirely** — extraction pipeline was too heavy and couldn't compete with LLMs.
- **CompiledSystem caching with code objects** — `compile(expr, "<compiled>", "eval")` once, `eval(code)` per step. ~25x speedup.
- **Three-pillar + bridge architecture** — dynamics (`dynamics/`), knowledge (`knowledge/`), and core (`core/`) are separate packages sharing `core/` as common data model substrate; `bridge.py` (`KBSimBridge`) connects KB and simulation; `patterns/` holds reusable model factories.
- **Named graphs per source** — each information source is its own named graph; queries read across the union.
- **Triple identity is (s,p,o)** — equality/hashing based on (s,p,o) only. Dedup across named graphs keeps max-belief version.
- **RDFS inference uses rdf:type** — `ingest_csv` adds `rdf:type` triples (not `epc:type`), enabling RDFS domain/range inference to derive additional type facts.
- **`_COMMENT_RE` regex `(?:^|\s)//`** — prevents matching `http://` URLs.
- **ABM rules use absolute set (`=`) not incremental (`+=`)** — prevents accumulation drift.
- **GitHub Pages for docs** — mkdocs + Material theme, deployed via `.github/workflows/deploy.yml` (GitHub Actions Pages source), auto-rebuild on push to `main`.

## Next Steps
1. Use new ABM features (message passing + strategy switching) in supply chain recipes (Disruption Cascade, Bullwhip Effect).
2. Keep docs accurate as features evolve; the Pages site rebuilds automatically on `docs/**` + `mkdocs.yml` changes.

## Critical Context
- **1028 tests passing** (core SD + knowledge + sensitivity + production/transactions/execution + CSV ingestion + ABM additions). Vensim import removed. Controller module deleted (unused). SL/epistemics removed.
- **`ingest_csv` types exported** from `dynafx.knowledge`: `MappingDef`, `ColumnMapping`, `IngestReport`, `ingest_csv`, `load_all_mappings`.
- **Expression parser** has NO string literals — SPARQL KB_QUERY strings must be defined as Python params.
- **causes_strip** returns `CausalStrip` with `.variable`, `.factors` (list of dicts), `.total_value`.
- **detect_feedback_loops** returns `LoopAnalysis` with `.loops` (list of `FeedbackLoop`), each with `.name`, `.nodes`, `.polarity`.
- **ScenarioComparison** stores results in `.scenarios` (list of `ScenarioResult`), each with `.result` (SysdModelResult) for `.times`, `.values`, `.aux_values`.
- **SPARQL aggregate limitation**: ASK returns cardinality=1/0. SELECT returns first binding. No aggregate support — pre-compute as explicit triples.
- **TripleStore.suppress_callbacks counter** — prevents infinite loops when exec/tx triples trigger rule evaluation.
- **ProductionRuleEngine._in_evaluate** — re-entrant depth counter (max 10).
- **DES→SD post-hoc fix**: `_get_ts()` merges `r.des_metrics_history` into timeseries so DES-based auxes evaluate correctly in post-hoc analysis
- **Unknown var default 0.0**: `_compile_expr()` compiles unknown variable refs as `_s.get('name', 0.0)` — DES metrics silently default to 0 if not merged into eval namespace

## Relevant Files
- `src/dynafx/core/models.py` — Foundational data model: `Graph`, `Node`, `Edge`, `NodeType`, `EdgeType`, `Entity`, `WorldRelation`, `EmergentProperty`, `ReasoningMode`. Used by knowledge/, dynamics/, and core/.
- `src/dynafx/core/decomposer.py` — `SystemDecomposer`: manual node/edge graph construction API.
- `src/dynafx/registry.py` — Plugin registry for custom builtins and DES hooks.
- `src/dynafx/dynamics/dsl.py` — main DSL: parser, expression AST, `_replace_smooths()` with ExprNode, `_build_system()` with `CompiledSystem` cache, `_compile_system()`, `SysdModel`, `SysdModelResult`. Submodel support: `SubmodelDef`, `IncludeDef`, `_expand_includes()`. KB_QUERY/KB_ASSERT builtins via `_make_kb_builtins`.
- `src/dynafx/dynamics/units.py` — `Unit`, `UnitRegistry`, `UnitChecker`, 40 tests.
- `src/dynafx/dynamics/causal.py` — 15 tests, `causes_tree`, `effects_tree`, `causes_strip`, `causal_trace`.
- `src/dynafx/dynamics/feedback.py` — 8 tests, `detect_feedback_loops`, `loops_for_variable`.
- `src/dynafx/dynamics/optimization.py` — `lp_minimize`, `lp_maximize`, `pareto_optimize`, `calibrate`, `kb_lp_minimize`/`kb_lp_maximize`/`kb_calibrate`/`kb_optimize`, 17 tests.
- `src/dynafx/dynamics/agent.py` — `AgentInstance`, `ABMEngine`, `Message`, `_eval_condition`, `_eval_effect`, `_parse_send`, `_parse_switch_strategy`. 4-phase step (Deliver → Decide → Cleanup → Aggregate). Strategy-scoped rule evaluation with meta-rules.
- `tests/test_abm_additions.py` — 64 tests for Message, strategy/meta_rule DSL parsing, Python API, perceive with mailbox, meta-rule + strategy-scoped decide, SEND parsing/execution/delivery, SWITCH_STRATEGY with cooldown, 4-phase step, default strategy initialization, DSL integration, backward compatibility.
- `src/dynafx/dynamics/des.py` — `DESClock`, `EventQueue`, `Queue`, `Resource`, `DESEngine`, `QueueStats`, `ResourceStats`.
- `src/dynafx/dynamics/emergent.py` — `EmergentProperty`, `Condition`, `Effect`, `run_consistency_checks`.
- `src/dynafx/dynamics/equations.py` — `rk4_step()`, `euler_step()`.
- `src/dynafx/dynamics/bfo.py` — BFO alignment helpers (`get_bfo_alignment`, `validate_bfo_alignment`, `get_bfo_summary`).
- `src/dynafx/dynamics/__init__.py` — SD public API: exports SysdModel, parse_sysd, causal_trace, detect_feedback_loops, AgentStrategy, AgentRuleDef, Message, etc.
- `src/dynafx/dynamics/scenario.py` — `ScenarioDef`, `ScenarioResult`, `ScenarioComparison` with comparison/deviation/tornado/summary.
- `src/dynafx/patterns/signal_chain.py` — `SignalChain` class: factory that builds a `SysdModel` for leading-indicator → outcome pattern. Parameters: trace_expr, detection_delay (list for multi-hop), decision_lag, outcome_threshold, outcome_sensitivity, threshold_direction, has_feedback, has_tracking. `SignalChain.build(...)` is the canonical constructor.
- `src/dynafx/patterns/disruption_cascade.py` — `DisruptionCascade`: supply-chain disruption propagation model factory.
- `src/dynafx/bridge.py` — `KBSimBridge` (params_from_kb, evidence_from_result, record_provenance), `ClosedLoopReasoner`, `grade_queries`, `ReasoningPass`.
- `src/dynafx/knowledge/model.py` — RDF data model: `RDFNode`, `NamedNode`, `BlankNode`, `Literal`, `Triple`, `TriplePattern`, XSD types, RDF/RDFS/OWL namespace constants. 39 tests in `tests/test_kb_model.py`.
- `src/dynafx/knowledge/store.py` — `TripleStore` with SPO/POS/OSP nested-indices, named graphs, pattern matching for all 8 patterns, graph isolation/copy/removal. 40 tests in `tests/test_kb_store.py`.
- `src/dynafx/knowledge/turtle.py` — Turtle/N-Triples tokenizer, recursive descent parser, serializer. Supports @prefix/@base, a, all literal types, blank nodes, ; and , grouping, comments, base IRI resolution, empty-prefix PNAME_LN. 38 tests in `tests/test_kb_turtle.py`.
- `src/dynafx/knowledge/inference.py` — `Rule`, `Var`, `InferencePattern`, `RuleEngine` (forward-chaining), `rdfs_rules()` (7 rules), `owl_rl_rules()` (4 rules). 59 tests in `tests/test_kb_inference.py`.
- `src/dynafx/knowledge/hierarchy.py` + `loader.py` — `TypeNode`, `TypeHierarchy`, `MDM_TYPE_HIERARCHY`, `TBox`, `GENERAL_TBOX`, `load_tbox`, `validate_against_tbox`. 17 tests in `tests/test_tbox.py`.
- `src/dynafx/knowledge/production.py` — `ProductionRuleEngine`, `ProductionRule`, `Condition` hierarchy (5 types), `Action` hierarchy (5 types), fire_once/max_fires/priority. 53 tests in `tests/test_kb_production.py`.
- `src/dynafx/knowledge/transactions.py` — `Transaction`, `TransactionStore`, `TransactionQuery`. Append-only temporal log with RDF backing. 25 tests in `tests/test_kb_transactions.py`.
- `src/dynafx/knowledge/execution.py` — `ExecutionRecord`, `ExecutionStore`. Provenance-tracked action records. 10 tests.
- `examples/global_solar_epc_twin.py` — supply chain digital twin example: 7 EPC CSVs → named-graph KB → params_from_kb → baseline/typhoon runs → evidence round-trip → scenario grading/ranking/filtering → production rules → kb_lp_minimize → causal/feedback → provenance → L1–L5 maturity ladder. Verifies baseline $931.4M / disruption −$2.8M.
- `data/models/global_solar_epc.sysd` — 3-region multi-project SD+ABM+DES model with shared Asian supply chain. 11 stocks, 59 auxes, 5 DES queues, 3 resources, 2 agent types. KB_QUERY for disruption/supplier/project risk. STEP-based disruption gating port outflow. Model verified with 3 disruption scenarios (baseline 86.7%→$961M profit, moderate disruption 81%→$898M, severe disruption 69%→$766M). Flow expressions require aux intermediaries for cross-stock sharing.
- `data/models/{vmi,reverse_logistics,cold_chain,supply_chain_demo}.sysd` — library test fixtures used by `tests/test_reference_models.py` and `tests/test_delays.py`.
- `hierarchy.md` — Complete feature hierarchy across all packages with file paths, test counts, and dependency relationships.
- `docs/` — mkdocs site (index, architecture, knowledge, digital-twin, examples, development), deployed to GitHub Pages via `.github/workflows/deploy.yml`.

