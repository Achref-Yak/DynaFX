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
- `SystemDecomposer` is the primary API for manual decomposition — clean `add_node`/`add_edge`, no extraction noise.
- Detection passes are read-only structural matchers on `Graph` — never mutate nodes/edges.
- SL opinions are inert on CAUSAL/System Dynamics paths — default `Opinion()` everywhere, confidence in metadata.
- `SystemDecomposer` is the primary API for manual decomposition — clean `add_node`/`add_edge`, no extraction noise.
- **Full Vensim parity target** — subscripting, optimization, causal tracing, time functions, delays, stochastic distributions, units checking, submodels, gaming mode.
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
- **Template `.sysd` file in `templates/` directory** — `templates/signal_chain.sysd` as reference, not the primary mechanism.
- **Named graphs per source** — SL integration point: each information source is a named graph, fusion merges across graphs via SL consensus.
- **Triple identity ignores opinion** — same (s,p,o) = same triple, dedup across named graphs.

## Progress
### Done
- **C2 (CONVEY_BATCH) implemented and fixed**: CONVEY_BATCH(input, delay, batch_size) accumulates input as a smooth ODE, emits batches to a pipeline buffer when accumulator ≥ batch_size, and outputs matured batch value to the output slot. Includes support for variable batch sizes (aux/param references). Fixed early-return bug in `_build_system` that skipped pipeline processing when only CONVEY_BATCH entries existed (no DELAY_FIXED entries). Fixed batch_size serialization: `str(ExprRef(...))` → `_serialize_expr(...)`. 12 tests all passing.
- Removed entire NLP extraction pipeline: `nlp/` (7 files, 1,432 lines), `extract/` (13 files, 3,083 lines), `perception/` (1 file, 180 lines), `operators/extract.py` (656 lines). Deleted 14 NLP-dependent test files, rewrote 2 more.
- Fixed `_compile_expr` to resolve `t` correctly in table lookups (was always 0.0).
- Added scientific notation support in tokenizer (`3e-7` now parses as one number token).
- Added stock name normalization: spaces → underscores (e.g., `ICU Beds` → `ICU_Beds`).
- Added comparison operators (`>`, `>=`, `<`, `<=`, `==`, `!=`) to tokenizer and parser.
- **Aux variables (P0)**: `AuxDef` dataclass, `aux_vars` field, `aux` keyword parsing, `_compile_expr` extended with `aux_names`, `_build_system` compiles/topo-sorts/evaluates auxes before stock equations.
- **Fixed params resolving**: parameters passed at simulate time merge into state vector `_s` so expression references resolve.
- **Validation (P1)**: `SysdModel.validate(params=set)` with name resolution, flow conservation, non-negativity bounds, zero-divisor detection.
- **CLI entry point (P1)**: `src/dynafx/system/__main__.py` with `simulate`, `validate`, `list` subcommands. Extended with `--paradigm` flag (sd/abm/des/all). (v0.3.0: removed dead `--stats`/`--dir` flags, fixed paradigm mutation via `copy.deepcopy`.)
- **Plotting API (P1)**: `SysdModelResult` class with `.plot(path, stocks, subplots, title)` and `.plot_with_bands(path, mean, std, p5, p95)`.
- **Sensitivity (P1)**: `SysdModel.simulate_ensemble(params, fixed_params, n, method, seed)` with uniform/normal/lognormal distributions.
- **Vensim `.mdl` import (P1)**: `src/dynafx/system/vensim.py` parses INTEG stocks, WITH LOOKUP tables, TIME→t mapping, SMOOTH/DELAY1 mapping, continuation line joining.
- **Model library (P1)**: 12+ curated `.sysd` models in `models/` plus `pandemic_seirvh.sysd`.
- **SD ontology (P2)**: `src/dynafx/system/ontology.py` with stock/flow subtype inference (MATERIAL/INFORMATION/FINANCIAL), cross-type flow validation.
- **Pandemic model bug fixes**: recovery_fraction dynamic compensation, ICU_Fatality time-normalization, vaccination inflow to Recovered, healthcare_stress_avg consumed by IF-gate and Expansion flow.
- **Deleted `rules/engine.py`** — dead code in SD context. Removed entire `rules/` package.
- **EmergentProperty dataclass** (`emergent.py`): `Condition`, `Effect`, `ComparisonOp`, `EffectType`, `ConsistencyResult`, `ConsistencyViolation`, `run_consistency_checks`, 4 checker functions.
- **SD/SL clean separation completed**: Removed `Opinion`/`Parameter` from `Equation.confidence`. Removed `LoopClassification`. Removed `Parameter` import from dsl.py. Removed `opinion: Opinion` from `EmergentProperty`. Removed backward compat `Parameter` handling in `equations.py`.
- **SL files moved to `dynafx/sl/`**: `operators.py`, `validation.py`, `parameters.py` moved out of `system/`.
- **Accuracy test suite**: `tests/test_accuracy.py` — 8 tests against analytical solutions.
- **ABM engine complete**: `AgentDef`, `AgentPropDef`, `AgentRuleDef` dataclasses. `_parse_agent_property()`, `_parse_agent_rule()` helpers. `AgentInstance` (perceive/decide/act), `ABMEngine` (step/get_metrics). Condition evaluation via safe eval with `always` keyword. Effects: `+=`, `-=`, `*=`, `/=`, `=` (absolute set returning delta). Property clamping. Aggregated metrics. Integrated into `SysdModel.simulate()`.
- **ABM bug fixes**: `_eval_condition` now supports `always` keyword. Multi-rule parsing fixed — stack pop moved before parent check so same-indent rules don't merge into one. `shared_state.update(params)` so ABM agents can see parameters. CLI fixed: `_agents` → `instances`, `agent_id` → `id`.
- **DES engine complete** (`system/des.py`): `DESClock`, `EventQueue`, `Queue`, `Resource`, `DESEngine`, `QueueStats`, `ResourceStats`.
- **DES integrated into `SysdModel.simulate()`**: DESEngine created when queues/resources/events defined, `step()` called after each RK4/Euler step.
- **DES parser**: `queue "Name": capacity N, service_time expr` and `resource "Name": capacity N`.
- **Cross-paradigm tests** (`tests/test_crossparadigm.py`): 11 tests covering SD+ABM, SD+DES, ABM+DES, all three together, pure SD, CLI integration.
- **Bug fixes**: `_parse_name_value` now handles `=` as separator. Queue/Resource capacity parsing extracts `capacity N` from comma-separated args. `always` keyword in ABM eval namespace. Multi-rule stack fix. Agent field name fixes (`id` not `agent_id`). DES `peek()` method added.
- **Example models**: `models/pandemic_response.sysd` (SD+ABM+DES, 6 stocks, 750 agents, 2 queues, 5 resources), `models/call_center_abm_des.sysd`, `models/predator_prey_abm.sysd`.
- **Example script**: `examples/pandemic_response.py` — demonstrates parsing, parameterized simulation, ABM cohort analysis, DES stats, sensitivity analysis.
- **Phase 1 complete: Expression system + math functions + time functions + SMOOTHI**: ABS, EXP, LN, SQRT, SIN, COS, PI, SMOOTHI, PULSE, STEP, RAMP, NOISE with reproducible seeding.
- **Phase 1 tests**: `tests/test_math_functions.py` — 22 tests all passing.
- **Phase 1 example models**: `models/seasonal_epidemic_example.sysd`, `models/smooth_demo.sysd`.
- **Bug fix: DES service_time** — `_compiled_service_time` callable on Queue, `_get_service_time()` method, `enqueue()` accepts `event_queue` param, `_process_queue_departures()` each step.
- **Bug fix: Vensim importer inflow/outflow split** — `_split_inflow_outflow()` handles nested parentheses and function calls.
- **Phase 1 (Delays): Higher-order delays complete** — DELAY3, DELAYN, DELAY_FIXED with 13 tests.
- **Phase 2: Causal tracing complete** — 15 tests, `system/causal.py` with `causes_tree`, `effects_tree`, `causes_strip`, `causal_trace`.
- **Phase 3: Feedback loop detection complete** — 8 tests, `system/feedback.py` with `detect_feedback_loops`, `loops_for_variable`.
- **Phase 4: Linear programming + calibration + optimization complete** — 12 tests, `system/optimization.py`, dependency `scipy>=1.10`.
- **Supply chain SD-only demo** — `examples/supply_chain_demo.py` + `models/supply_chain_demo.sysd` with 3-echelon, DELAY3/DELAY_FIXED/SMOOTH/SIN/PULSE/NOISE/MIN/MAX.
- **Supply chain SD+DES paradigm demo** — `examples/supply_chain_paradigm.py` with 7 stocks + Escalations queue + SupportStaff resource. DES queue arrival rate reads aux values (`_ns_arrival` includes `_a_abm`). Feedback: `escalation_penalty = MAX(0.9, 1.0 - 0.01 * queue_length)` gates `wh_to_retail`. Tuned with spike at day 60 to avoid inventory buffer buildup. Generates 8-page PDF report with 5 scenarios.
- **DES arrival_rate namespace fix** — `dsl.py` `_ns_arrival` now includes `_a_abm` (aux dict) so queue arrival rates can reference aux variables (demand, retail_sales, etc.).
- **Phase 5: Units checking complete** — 40 tests, `system/units.py` with `Unit`, `UnitRegistry`, `UnitChecker`.
- **Phase 6: Submodels / modules complete** — 11 tests, `SubmodelDef`, `IncludeDef`, `_expand_includes()`.
- **Phase 7: CSV import/export complete** — 12 tests, `SysdModel.import_data()`, `get_imported_interpolator()`, `export_results()`.
- **Full showcase demo** — `examples/full_showcase.py` + `models/full_showcase.sysd` with 14 sections all working, ~12s runtime.
- **Submodel expansion bugs fixed** — `_FUNC_NAMES` exclusion, word-boundary regex, continuation-line params, eval order fix, model keyword pops submodel stack.
- **Performance optimization: CompiledSystem caching** — ~25x speedup (300s+ → 12s), code object pre-compilation, pre-filtered numeric params.
- **SaaS churn signal model — all 8 review fixes applied**: dt 0.25, B1 sign corrected, 43-day lead time, signal tracking stocks, churn lag added.
- **SaaS churn signal example script complete** — `examples/saas_churn_signal.py` with 5 scenarios, 8-param sensitivity, feedback loops, causal trace, signal lead time.
- **SMOOTH/DELAY dynamic parameter resolution** — non-literal delay args stored as `ExprNode`, evaluated at runtime with params dict, `_ns` includes `**_numeric_params`.
- **Signal tracking stocks restructured** — track smoothed auxes, not raw inputs.
- **EvidenceMatrix complete** — `src/dynafx/reason/evidence.py` with `ConsensusLevel`, `PairwiseAgreement`, `ClaimAssessment`, `EvidenceMatrix`, `EvidenceMatrixResult`. L1-distance agreement scoring. 27 tests in `tests/test_evidence_matrix.py`.
- **EvidenceMatrix consensus → FusionSituation integration** — `consensus_to_fusion_situation()` in `fusion.py`, `classify_fusion_situations()` method on `EvidenceMatrixResult`.
- **EvidenceMatrix exported** — `__init__.py` exports `ClaimAssessment`, `ConsensusLevel`, `EvidenceMatrix`, `EvidenceMatrixResult`, `PairwiseAgreement`.
- **`src/dynafx/templates/` package created** — `SignalChain` class in `signal_chain.py` constructs a `SysdModel` for the leading-indicator → outcome pattern.
- **Template `.sysd` reference file** — `templates/signal_chain.sysd` documents the structure for DSL-side use.
- **Signal showcase example** — `examples/signal_showcase.py` builds and simulates all 9 leading indicator domains with comparison table.
- **`SignalChain` updated with `threshold_direction` parameter** — supports "below" direction for decline signals (foot traffic, card payments).
- **Showcase lead times fixed** — analytical computation from model parameters (pipeline delay + threshold sensitivity), all 9 domains produce lead times (18-111 days).
- **Cognitive reasoning engine plan written** — `upcoming2.plan` with 7 source + 6 test files, 130 tests, ~1610 source lines.
- **Phase 1: RDF data model** — `src/dynafx/kb/model.py` with `RDFNode`, `NamedNode`, `BlankNode`, `Literal`, `Triple`, `TriplePattern`, XSD shortcuts, namespace constants (RDF/RDFS/OWL). 41 tests all passing.
- **Phase 2: TripleStore** — `src/dynafx/kb/store.py` with `TripleStore` class, SPO/POS/OSP nested-index prefix strategy, named graph per source, dedup by max-belief opinion, graph isolation/copy/removal. 40 tests all passing.
- **Phase 3: Turtle/N-Triples** — `src/dynafx/kb/turtle.py` with tokenizer, recursive descent parser, serializer, N-Triples support, base IRI resolution. 40 tests all passing.
- **Phase 4: SPARQL** — `src/dynafx/kb/sparql.py` with lexer (case-insensitive keywords), recursive descent parser, algebra tree, filter expressions, evaluator producing `QueryResult`. 35 tests all passing.
- **Phase 5: Inference engine** — `src/dynafx/kb/inference.py` with `Rule`, `Var`, `InferencePattern`, `RuleEngine` (forward-chaining), RDFS (7 rules) and OWL RL (4 rules) rule sets, opinion propagation (min/product/average). 53 tests all passing.
- **Phase 6: Confidence layer** — `src/dynafx/kb/confidence.py` with `fuse_graphs()`, `grade_query()`. 30 tests all passing.
- **Phase 7: `kb/__init__.py`** — public exports, integration wiring to `reason/` and `tbox/`.
- **Phase 8: Scenario comparison** — `system/scenario.py` with `ScenarioDef`, `ScenarioResult`, `ScenarioComparison` (plot_comparison, plot_deviation, tornado, summary, deviation_table). 20 tests all passing.
- **Argumentation engine** — `reason/argumentation.py` with `Argument`, `Attack`, `AttackType`, `SupportType`, `ArgumentationFramework` (Dung grounded/preferred semantics), `build_framework()` (rebut, undermine, source reliability attacks). Integrated into `kb/confidence.py` as `argumentative_filter()`. 27 tests all passing.
- **`TripleStore.all_triples()`** — added method to iterate all triples across all graphs.
- **KBT (Knowledge-Based Trust) engine** — `reason/kbt.py` with `KBTResult`, `compute_kbt()` EM algorithm that scores source reliability without ground truth. Iterates E-step (infer likely true values by weighted vote) and M-step (recompute source trust from accuracy). Outputs `prov:reliability` triples to `meta` graph. 14 tests all passing.
- **Argumentation showcase** — `examples/argumentation_showcase.py` demonstrates full pipeline: Turtle parse → named graphs → RDFS inference → SL consensus → argumentation filter → fusion. Source reliability scored by KBT.
- **Knowledge Fusion Showcase** — `examples/knowledge_fusion_showcase.py` demonstrates KG → KBT → EvidenceMatrix → Argumentation → filter → SL fusion → SPARQL grading in a single pipeline.
- **Student Math SD model** — `models/student_math.sysd`: 3 stocks (Math_Anxiety, Math_Performance, Self_Efficacy), 6 flows, 10 auxes. KG_* params injected from DynaFX bridge. IF uses function-call syntax `IF(cond, a, b)` not keyword syntax `IF...THEN...ELSE`.
- **Multi-Paradigm Student Pipeline** — `examples/multi_paradigm_student.py`: 3-pass orchestration (pre-diagnosis → intervention → follow-up). Each pass: Turtle → RDFS inference → KBT → argumentation → filter → bridge → simulate SD+ABM+DES → extract evidence → feed to next pass. Bridge reads from original store (not filtered) to bypass grounded semantics skepticism. Pass 1: anxiety 25→59, performance 55→10, reinforcing loop dominates. Pass 2: intervention reverses, anxiety to 0.5, performance recovers to 26. Pass 3: sustained recovery to 34.
- **KBT tests fixed** — `tests/test_kbt.py` 14 tests handle TripleStore dedup (max-belief version stored for identical (s,p,o) across graphs). All passing.
- **Stale NLP remnants purged** — Deleted `agents/`, `api/`, `scripts/`, `demo/`, `memory/`, `kernel/`, `domains/`, `docs/`, `tests/test_agents.py` + 6 memory/kernel test files (~1424 lines dead scripts, 10 stale docs, ~109 obsolete tests). Removed `perception.hypothesis_generator` dead lazy import from top-level `__init__.py`. Removed `_load_dotenv()` function (unused).
- **Circular import fixed** — `kb/__init__` → `kb.confidence` → `reason.argumentation` → `kb.model` cycle broken by deferring `build_framework` import inside `argumentative_filter()`.
- **`pyproject.toml` cleaned** — Removed `torch`, `transformers`, `spacy`, `nltk`, `fastapi`, `uvicorn`, `websockets`, `leidenalg` from dependencies (all from deleted NLP pipeline). Core deps now only `networkx`, `numpy`, `scipy`, `pydantic`.
- **`sl/__init__.py`** and **`tbox/__init__.py`** populated with proper exports.
- **`Makefile`** fixed — `run` target now points to `dynafx.system` instead of deleted `dynafx.cli`.
- **README.md rewritten** from scratch — describes current two-pillar architecture with quick start examples and example table.
- **Telecom signal study model** — `models/telecom_signal_study.sysd`: SINR-based churn with closed-loop power control (Power_Ramp_Up/Down), SNR scaling (noise_floor in denominator), packet scaling (bandwidth/packet_size/traffic_load), fade depth 0.03 for 97% fade excursion. Example script `examples/telecom_signal_study.py` generates 11-page FPDF report with signal chain plot, fade zoom, 5-scenario comparison, causal tracing, 7 feedback loops, n=20 sensitivity.
- **API refactoring Phase 1**: Public entry points — `system/__init__.py` exports SysdModel, parse_sysd, causal_trace, detect_feedback_loops, etc. Top-level `dynafx/__init__.py` cleaned of 44 stale NLP exports, replaced with core names (SysdModel, TripleStore, parse_turtle, etc.). 1130 tests all passing, pyright 0 errors.
- **API refactoring Phase 2**: Naming collisions — `sl.Argument`→`sl.ValidationArgument`, `sl.Attack`→`sl.ValidationAttack`, `kb.evaluate`→`kb.sparql_evaluate`, `SignalChain.__new__`→`SignalChain.build()`. Backward-compat shims with DeprecationWarning for all.
- **API refactoring Phase 3**: SL consolidation deferred (risk of legacy `core.models` dependency in `sl/`). Docstring fix: `sl/__init__.py` directs users to `dynafx.reason` for core SL algebra.
- **API refactoring Phase 4**: CLI cleanup — removed `--stats` flag (parsed but never checked), removed `--dir` flag on `list` (ignored by `_find_library_models()`), added `copy.deepcopy(model)` before paradigm filtering to prevent mutating the parsed model.
- **API refactoring Phase 5**: Docstrings — `SysdModel`, `SysdModel.simulate()`, `SysdModelResult`, `FlowDef`, `StockDef`, `AuxDef`.
- **Legacy code deletion**: Removed `mp/`, `mdm/`, `analysis.py` (dead), `operators/` + `policy/` + `schemas/` (cognitive operator framework, 20+ files, only used by 2 detect_emergence calls), and orphaned `core/` submodules (state, embeddings, concept, trace, events, diff, workflow, loom, schema, higraph, pipeline, operator). 11 test files removed (~250 tests). Kept `registry.py` (used by system/), `domain.py` (used by reason/), and `core/models.py`/`math.py`/`decomposer.py`/`config.py` (still depended on by core pillars). 1081 tests passing, pyright 0 errors.
- **DELAY_FIXED/RK4 buffer corruption fix**: Moved pipeline buffer management outside of `f()` into separate `_process_pipeline_delays()`, called exactly once per step from `simulate()` loop. DELAY_FIXED variables get `"0.0"` ODE so RK4 intermediate calls don't corrupt the FIFO buffer. Added 4 regression tests (`TestPipelineDelayFix`): RK4/Euler match, no premature emission, supply chain fill rate match, retailer never depletes. Retailer stays at min 1278+ with 100% fill rate.
- **Sobol formula corrected**: Saltelli estimator uses only A, B, AB_i matrices (N(k+2) evaluations). First-order via `E[yB * yAB_i]` (shared column i), total-order via `1 - E[yA * yAB_i]` (shared X_{-i}). No BA_i matrices needed.
- **Sensitivity tests expanded**: 6 → 40 tests. 8 test classes covering all 5 methods + plots + edge cases. All 40 passing.
- **ALLOCATE_FRACTION builtin**: Proportional multi-outflow allocation function added to `dsl.py`. Formula: `demand_i * min(1, available / total_demand)`. Prevents inventory over-drafting when multiple outflows compete for one stock. Used by Chem_Inventory in EV battery model to split output between chemical processing and downstream shipping.
- **Multi-outflow validation warning**: `validate()` now warns when a stock has multiple outflows that could independently over-draft — directs modeler to use `ALLOCATE_FRACTION` or `MIN(stock/dt, ...)` guards.
- **Bullwhip analysis corrected**: Changed from max-ratio (misleading) to CV (coefficient of variation) ratios. Z-score plotting for unit-comparable overlay across echelons. Proper formula: `CV = std(rate) / mean(rate)`, ratio > 1.0 means amplification.
- **EV Battery Supply Chain model** (`models/ev_battery_supply_chain.sysd`): 6-echelon SD+ABM+DES model spanning Lithium Mine → Chemical Processing → Cell Factory → Pack Assembly → Warehouse → Customers. 10 stocks, ~86 auxes, 4 DES queues, 2 resources, 120 agents (100 automakers, 20 suppliers). Behavioral rules: 9 per automaker, 5 per supplier. Dynamic pricing, scarcity premiums, finite reserve depletion at day 125.
- **EV Battery Supply Chain report** (`examples/ev_battery_supply_chain.py`): 8-page FPDF report with demand overview, inventory, DES queue dynamics, financials (cost breakdown with scarcity premium), bullwhip CV analysis, 7 scenario comparison, 56-variable multi-echelon LP optimization.
- **Complete feature hierarchy** (`hierarchy.md`): Full vertical slice of every component across all 4 packages (dynamics, epistemics, knowledge, core). 50+ entries with file paths, test counts, and dependency relationships.
- **Structural deep analysis of 8 critique issues**: (1) Profit gap — end-state vs average margin ($11,782 vs $16,331). (2) Impossible unit math — scarcity premium missing from reviewer's cost assumptions. (3) Margin crash — chart is linear but margin drops at day 125. (4) Phantom shipments — 20K pack material leak (WH outflow ≠ fulfillment). (5) Unrecorded surplus — same root cause as #4. (6) Hidden WIP — Chem_Inventory double-drain bug. (7) Time-step aliasing — dt=0.25 vs DES event-driven times 0.05/0.08. (8) 7 vs 9 rules documentation mismatch.
- **Model fixes applied to EV battery chain**: (a) Chem_Inventory: ALLOCATE_FRACTION replaces double independent outflows — proportional allocation prevents over-drafting. (b) Warehouse: outflow changed from `wh_shipping_rate` to `fulfillment_rate` — closes 20K pack material leak, mass balance reconciled (gap=13 out of 58K, 0.02%). (c) fulfillment_rate gates against `MAX(0, Warehouse_Inventory)/dt` — prevents negative inventory. (d) Chem_Inventory min=0.0 post-fix (was -89).
- **Fixes revealed supply chain overproduction**: Warehouse builds 20,088 packs (188 days of inventory) because DES inflow (58K packs) >> fulfillment outflow (39K packs). Dynamic pricing premium collapses from $57K to $50K. Cash drops from $690M to $391M. Bullwhip effect is root cause — order-up-to policies with SMOOTH forecasting amplify variability 14.3x from WH to Mine.
- **ProductionRuleEngine complete** — 5 condition types (TripleCondition, SparqlCondition, ComparisonCondition, AggregationCondition, And/Or/Not), 5 action types (TripleAction, RetractAction, LogAction, BridgeAction, SimulateAction), event-driven via TripleStore.on_add callback, fire_once/max_fires/priority/enabled, signature-based dedup. 36 tests.
- **TransactionStore complete** — append-only temporal log backed by list + RDF triples in "transactions" graph. `record()`, `query()` (type/time/source), `recent()`, `count_by_type/source`. Fires store.on_add for automatic rule triggering. 21 tests.
- **ExecutionStore complete** — provenance-tracked action records in "executions" graph. `record()`, `get()`, `by_rule()`, `by_type()`, `recent()`, `last_execution()`. 10 tests.
- **CognitiveOrchestrator complete** — wires TransactionStore → ProductionRuleEngine → ExecutionStore. `ingest_event()`, `add_rule()` (wraps actions with execution recording), `get_causal_chain()`, `get_rule_status()`. +90 lines in bridge.py.
- **All new code in kb/production.py, kb/transactions.py, kb/execution.py** — ~900 total lines. CognitiveOrchestrator in bridge.py (+90 lines). Exports in kb/__init__.py (+40 lines).
- **Solar EPC demo complete** (`examples/solar_epc_demo.py`): End-to-end 5-layer intelligence report (6-page PDF, 330KB) with Situational Awareness (4 KPI gauges + event timeline), Diagnostics (causal network + 5 feedback loops), Predictive Analytics (4-panel forecasts + risk matrix), Scenario Analysis (4 scenarios + OAT sensitivity + tornado), Decision Intelligence (6 prioritized recommendations + $330K cost vs $1.49M+ benefit). 70 automated actions, 5 production rules, 4 simulation runs, 606 KB triples across 4 named graphs. Fixes: 3-digit hex color parsing, `$%` formatting in exec summary.
- **Solar EPC interactive HTML dashboard** (`examples/solar_epc_dashboard.py`): Self-contained 224KB single-file HTML dashboard with 6 tabbed pages matching the 5-layer framework. Uses Plotly.js (CDN) for interactive gauges, event timelines, causal chain diagrams, forecast subplots (±95% CI bands), risk matrices, tornado charts, OAT sensitivity bars, recommendation accordions, and business impact charts. No server needed — opens in any browser. Tab switching triggers Plotly.Plots.resize() for correct rendering. 12 Plotly figures, 6 tabs, 1273 tests passing.
- **Cross-paradigm dashboard fixes** (`examples/cross_paradigm_dashboard.py`): `crossing_day is None` format error fixed; CSS braces not escaped in `HTML_TEMPLATE.format()` fixed (2 instances). Runtime reduced 365→200 days (390s→210s). DES P50/P90/P99 wait percentiles added via M/M/1 queueing theory.
- **Global Solar EPC model created** (`models/global_solar_epc.sysd`): 3-region multi-project model with shared Asian supply chain. 11 stocks, 59 auxes, 5 DES queues, 3 resources, 2 agent types. KB_QUERY for disruption/supplier/project risk. STEP-based disruption gating port outflow. Model verified with 3 disruption scenarios (baseline 86.7%→$961M profit, moderate disruption 81%→$898M, severe disruption 69%→$766M).

### Done
- **Dashboard pipeline API mismatches fixed**: `ScenarioResult.result.times` access pattern (not direct `.times`), `CausalStrip.factors` iteration (not `dict.items()`), `FeedbackLoop.nodes`/`.polarity` attribute access (not dict subscript). All 16 tabs build successfully.
- **Dashboard generates correctly**: 615KB, 16 tabs, ~120s runtime. Pipeline: KB load → RDFS inference → baseline → disruption → post-disruption → 6 scenarios → OAT → causes_strip → feedback loops → brute-force opt. Output at `/tmp/solar_epc_16tab_dashboard.html`.

### Blocked
- (none)

## Key Decisions
- **Kill NLP extraction entirely** — extraction pipeline was too heavy and couldn't compete with LLMs.
- **Replace text-to-model bridge with a proper DSL** — Vensim-inspired `.sysd` format.
- **SMOOTH implemented as aux state variables** — each `SMOOTH(x, d)` call spawns an auxiliary ODE.
- **SMOOTHI extends SMOOTH** — same ODE mechanism but accepts 3rd argument for initial value.
- **Aux variables evaluated before stock equations** — compiled into lambdas, stored in `_a` dict, topo-sorted by dependency.
- **Parameters merged into state vector** — so expression references to param names resolve via `_s.get()`.
- **Time functions defined inside `f()` closure** — PULSE, STEP, RAMP, NOISE capture `t` from the inner function scope.
- **PI compiled as literal constant** — `_COMPILED_CONSTANTS` dict prevents PI from being looked up in `_s.get()`.
- **SD/SL clean separation** — SL files in `dynafx/sl/`, SD in `dynafx/system/`.
- **Bridge functions deleted** — user explicitly rejected bridging.
- **ABM agent state is numeric only** — no SL opinions on agents.
- **DES with full queuing theory** — resource pools, capacity, utilization stats from the start.
- **Unified state dict for all paradigms** — SD, ABM, DES share one `state: dict[str, float]`.
- **Extend .sysd for ABM/DES** — same file format, new keywords.
- **Full Vensim parity + multi-paradigm** — user chose "Full parity + multi-paradigm". 14-phase plan defined.
- **No visualization for now** — user deferred visualization phase.
- **No subscripting** — user explicitly excluded subscripting.
- **Linear programming for optimization** — user requested LP specifically.
- **DELAY3/DELAYN implemented as chained SMOOTHs** — each stage uses T/N delay.
- **DELAY_FIXED uses time-stamped buffer** — module-level `_delay_fixed_buffers` dict stores (t, val) history.
- **smooth_params uses tagged tuples** — `("smooth", ...)` or `("delay_fixed", ...)`.
- **Causal tracing uses regex-based ref extraction** — simpler than full AST analysis.
- **Feedback loop detection uses DFS with rec_stack** — standard cycle detection, polarity from edge sign analysis.
- **Units checking uses `~Unit~` syntax** — Vensim-style annotations parsed by regex, propagated through expressions.
- **Submodel stock parser pops stack BEFORE parent check** — critical fix for multi-stock submodels.
- **Submodel replacement map built once per include** — prevents double-prefixing.
- **Replacement uses longest-first sorting** — avoids partial name matches.
- **Model keyword pops submodel stack** — prevents model-level auxes from being triple-prefixed.
- **Include expression scanning for param refs** — regex `\b([A-Za-z_]\w*)\b` extracts names not in known_names or _FUNC_NAMES.
- **CompiledSystem caching with code objects** — `compile(expr, "<compiled>", "eval")` once, `eval(code)` per step. ~25x speedup.
- **SMOOTH/DELAY non-literal args stored as ExprNode** — not defaulted to 1.0. Runtime evaluation with params dict.
- **Signal tracking stocks track smoothed auxes** — not raw inputs, prevents τ=1 override of parameterized delays.
- **B1 dominated by R1** — pricing discounts reduce churn but hurt ARPU → quality → net negative.
- **EvidenceMatrix uses L1-distance agreement** — `1 - mean(|b_a-b_b|, |d_a-d_d|)`. Simpler, more intuitive than direction-based scoring.
- **EvidenceMatrix uses cumulative_fusion** — fuses opinions across sources per claim, classifies consensus by conflict/agreement ratios.
- **Python API is the primary template path** — `SignalChain` class constructs `SysdModel` directly. DSL include approach limited to numeric param overrides.
- **`SignalChain.build()` is the canonical API** — `SignalChain.__new__` deprecated in favor of `SignalChain.build()`. `SignalChain(...)` still works with a DeprecationWarning.
- **Template structure: trace → DELAY3 → SMOOTH → IF → impact → core stock outflow** — the trace signal is a leading indicator that modulates an existing process, not the stock's inflow.
- **Top-level `dynafx/__init__.py` exports only core names** — no stale NLP exports. `SysdModel`, `TripleStore`, `parse_turtle`, `parse_sysd` are the primary public exports.
- **Renamed exports keep DeprecationWarning shims** — `sl.Argument`→`sl.ValidationArgument`, `sl.Attack`→`sl.ValidationAttack`, `kb.evaluate`→`kb.sparql_evaluate` all have `__getattr__` shims that warn.
- **`system/__init__.py` is the SD public API** — re-exports `SysdModel`, `parse_sysd`, `parse_sysd_file`, `causal_trace`, `detect_feedback_loops`, `lp_minimize`, `calibrate`, `optimize`, `SysdModelResult`, `ScenarioComparison`, `UnitsError`.
- **Two-pillar architecture** — SD engine (`system/`) and cognitive reasoning engine (`kb/` + `tbox/`) are separate packages sharing `reason/` as common SL substrate. Their data models and reasoning modes are incompatible, so clear separation is necessary.
- **SL as "Confidence Layer"** — deterministic math (`Opinion`, `cumulative_fusion`, `EvidenceMatrix`) grades triples with belief, does not construct the graph itself. KB is standard semantic web machinery; SL makes it uncertainty-aware.
- **Named graphs per source** — each information source is its own named graph, fusion merges across graphs via SL consensus, the primary SL integration point.
- **Triple identity ignores opinion** — equality/hashing based on (s,p,o) only. Dedup across named graphs keeps max-belief version.
- **Turtle tokenizer: DECIMAL before INTEGER** — prevents `3.14` being tokenized as INTEGER+JUNK.
- **Turtle tokenizer: BLANK_NODE before PNAME_LN** — prevents `_:b1` matching as PNAME_LN (prefix `_`).
- **Turtle PNAME_LN allows empty prefix** — `:s` is valid PNAME_LN with empty prefix "".
- **Turtle serializer: no leading `;` on continuation lines** — `;` only at end of non-last predicate-object groups, not at start of continuation.
- **DES is single-server queue** — processes at most 1 departure per queue per step. At dt=0.5, max 2 departures/day. Cannot handle high-throughput order processing (e.g., 100/day). Best suited for low-frequency observational queues (exceptions, escalations).
- **DES arrival_rate eval namespace excludes aux values by default** — `_ns_arrival` in `dsl.py` only includes stocks (`_s`) and numeric params (`_numeric_params`). Supply chain DES fix: added `_a_abm` (aux dict) to arrival rate namespace so `retail_sales`, `demand`, etc. resolve correctly.
- **DES→SD coupling through aux readback** — Supply chain model reads `Escalations_length` into aux `escalation_queue`, then applies `escalation_penalty = MAX(0.9, 1.0 - 0.01 * Escalations_length)` to gate `wh_to_retail`. Penalty must be gentle enough to allow full queue drainage after spike ends.
- **Supply chain DES queue must be low-frequency** — WH_Orders (order processing queue) failed because DES throughput (2/day) < order demand (100+/day). Replaced with Escalations (customer complaint queue) at 0.1 * gap arrival rate, which produces manageable arrival volumes (2-14/day).
- **Supply chain shock timing matters** — spike at day 60 (not 180) prevents massive inventory buffer buildup. With wh_ship_capacity=105 > avg demand=100, excess 5/day × 180 days = 900 extra retail inventory absorbs the entire spike silently.
- **Turtle base IRI resolution** — `_resolve_iri()` method applied consistently to all IRI tokens (subjects, predicates, objects, directive IRIs).
- **KBT replaces hardcoded source reliability** — EM algorithm iterates E-step (weighted vote) and M-step (trust recomputation) without ground truth. Converges in <10 iterations.
- **KBT feeds argumentation** — `prov:reliability` triples written to `meta` graph, consumed by `build_framework()` for undermine attacks.
- **Argumentation sits between inference and fusion** — pipeline: Turtle → named graphs → RDFS inference → argumentation filter → SL fusion → query grading.
- **Dung grounded semantics is default** — preferred semantics available for skeptical reasoning. Source reliability attacks require `prov:reliability` triples with threshold > `min_attack_strength`.
- **Bridge reads unfiltered store for KG→SD params** — argumentation grounded semantics kills mutually-rebutting hasIssue claims (anxiety vs attention vs pacing all OUT). Bridge bypasses filter via max-belief across all original source graphs.
- **ABM rules use absolute set (`=`) not incremental (`+=`)** — prevents accumulation drift over 336 steps. `=` sets property to exact value each step, clamping to [min, max].
- **ALLOCATE_FRACTION uses proportional allocation** — `demand_i * min(1, available / total_demand)`. Splits available inventory proportionally among all outflows. More realistic than independent MIN/MAX guards for multi-outflow material stocks. Cascading priority (process first, ship remainder) considered but proportional chosen for fairness across downstream demand.
- **`_COMMENT_RE` regex `(?:^|\s)//`** — changed from `//.*$` which matched `http://` URLs, corrupting all lines with URLs. New pattern only matches `//` preceded by whitespace or at line start.
- **KB_ASSERT stores string objects as Literal** — `force_literal=True` in `_kb_assert()` for the object position ensures "disrupted" is stored as `Literal("disrupted")` not `NamedNode(iri="disrupted")`, matching SPARQL query literal handling.
- **ABM shared_state includes `t`** — `shared_state["t"] = t0` added so ABM conditions can reference `t` for time-based rules (e.g., `t >= 60`). Previously `t` was only available in the aux eval namespace `_ns`, not in the ABM perceive state.

## Next Steps
1. Add code health safeguard checks before committing.
2. Consider performance optimization (dashboard ~120s runtime → target 60s).
3. Explore interactive controls (sliders/scenario toggles) within the HTML dashboard.

## Critical Context
- **1273 tests passing** (core SD + kb + reason engine + sensitivity + production/transactions/execution).
- **Old NLP pipeline fully removed**: `agents/`, `api/`, `scripts/`, `demo/`, `memory/`, `kernel/`, `domains/`, `docs/` directories deleted. `extract/`, `nlp/`, `perception/` were already removed earlier. Remaining `tests/` all pass with no stale imports.
- **`SysdModel.simulate()` returns `SysdModelResult`** — `stocks` is a list of stock names, `values` is `dict[str, list[float]]`. **Aux values NOT exposed in results** — only stock values.
- **Parameters not in `params` dict return 0.0** — `_s.get('beta', 0.0)` returns 0 if param not provided.
- **DES queues process departures** — `service_time` expressions compiled against current state at each step.
- **Aux vars with numeric expressions can be overridden via params** — `simulate()` merges float-valued AuxDef expressions into params dict.
- **Optimize function clamps Nelder-Mead results to bounds** — Nelder-Mead doesn't respect bounds natively.
- **Full showcase demo runs in ~12s** (was 300s+ before caching).
- **SaaS churn model: 43-day signal lead time**, dt=0.25, B1 sign corrected, signal tracking stocks track smoothed auxes.
- **DSL include `params` only supports float values** — cannot pass string expressions. The Python API (`SignalChain` class) supports full expressions.
- **`SignalChain.build()` is the canonical API** — `model = SignalChain.build(...)` returns a `SysdModel`. `SignalChain(...)` works with a DeprecationWarning.
- **SignalChain showcase generates all 9 domains** — `examples/signal_showcase.py` builds, simulates, and compares all domains. Lead times computed analytically from pipeline delays + threshold sensitivity. All 9 produce lead times (18-111 days).
- **Framework identified as "framework"** (not SaaS/PaaS/library), with capability to model OODA loops (Observe via state dict, Orient via causal tracing, Decide via ABM/IF logic, Act via state updates).
- **Closest to AnyLogic (multi-paradigm)** and **Vensim (expression/causal logic)**, distinct by being code-first/Python-native.
- **Cognitive reasoning engine (`kb/`) separate from SD engine (`system/`)** — two pillars sharing `reason/` as common SL substrate. `tbox/` provides OWL2-style type hierarchy consumed by `kb/`.
- **`kb/` Phase 1-8 complete**: model.py (41 tests), store.py (40 tests), turtle.py (40 tests), sparql.py (35 tests), inference.py (53 tests), confidence.py (30 tests), production.py (36 tests), transactions.py (21 tests), execution.py (10 tests). Full plan in `upcoming2.plan`.
- **RDF node types are frozen dataclasses** — NamedNode(iri), BlankNode(id), Literal(value, datatype, lang_tag). Triple equality/hashing ignores opinion.
- **TripleStore uses SPO/POS/OSP nested-index prefix strategy** — O(1) pattern matching for all 8 pattern types. Named graphs as `dict[str, set[tuple]]` for O(1) membership.
- **Turtle parser is recursive descent with tokenizer** — supports @prefix/@base, a (rdf:type), string/integer/decimal/boolean/typed/lang literals, blank nodes, ; and , grouping, comments, base IRI resolution, empty-prefix PNAME_LN (`:s`).
- **Delta DES not yet paused** — needs both state in ServiceEvent and time-remaining tracking for multi-step operations.
- **ALLOCATE_FRACTION validated on EV battery chain** — Chem_Inventory min=0.0 (was -89 before fix). WH mass balance gap=13 (0.02% error from RK4). Cash=$391M (down from $690M because fixing the warehouse leak also killed the scarcity pricing premium).
- **Compiler-level auto-allocation for multi-outflow stocks**: `_compile_system()` in `dsl.py` now detects multi-outflow stocks where all outflows follow the `MIN(<ref>/dt, demand)` pattern and auto-applies `ALLOCATE_FRACTION` at the AST level. This is a framework-wide fix — every `.sysd` model with multi-outflow MIN-gated stocks gets proportional allocation automatically. 9 dedicated tests. Validate updated with `info` level for auto-allocatable and `warning` for non-auto-allocatable stocks.
- **Fix revealed fundamental supply chain overproduction** — DES ships 58K packs but only 39K are consumed. Warehouse builds 20K end inventory (188 days). Bullwhip is root cause: 14.3x CV amplification from WH to Mine.
- **Compiler auto-allocation confirmed**: Reverted Chem_Inventory from explicit `ALLOCATE_FRACTION` back to native `MIN(…/dt, demand)` pattern — compiler now auto-detects and transforms at the AST level. 1155 tests passing.
- **Expression parser has NO string literals** — tokenizer only handles numbers, identifiers, operators. SPARQL strings for KB_QUERY must be defined as Python params and referenced by name. Inline URL strings (`http://...`) cause tokenizer to fail on `://`.
- **KB_ASSERT fails in aux expressions with URL strings** — `KB_ASSERT("http://...")` in an aux line triggers tokenizer error on `://`. KB_ASSERT works correctly in ABM rule effect lines (different parser path).
- **Flows between stocks require aux intermediaries** — same-named flow on different stocks has independent expression. `- flow_name: expr` on one stock and `+ flow_name` (no expr = default 0) on another stock means unrelated values. Must define `aux flow_name_rate: <expr>` and reference it from both stock flow expressions.
- **Aux param override works for any float-valued aux** — `simulate(params={'aux_name': val})` overrides `aux aux_name: <KB_QUERY(..)>` computation. Enables KB-driven params to be passed without a live TripleStore.
- **causes_strip** returns `CausalStrip` with `.variable`, `.factors` (list of dicts: name, value, contribution), `.total_value`. Iterate `.factors` not `.items()`.
- **detect_feedback_loops** returns `LoopAnalysis` with `.loops` (list of `FeedbackLoop`). Each has `.name`, `.nodes`, `.polarity` ("reinforcing"/"balancing"). Use `.nodes` not `["variables"]`.
- **ScenarioComparison** stores results in `.scenarios` (list of `ScenarioResult`), each with `.result` (SysdModelResult) for `.times`, `.values`, `.aux_values`.

## Relevant Files
- `src/dynafx/core/models.py` — Foundational data model: `Opinion`, `Graph`, `Node`, `Edge`, `NodeType`, `EdgeType`, `EmergentProperty`, `FusionSituation`, `ReasoningMode`. Used by kb/, reason/, sl/, and system/.
- `src/dynafx/core/decomposer.py` — `SystemDecomposer`: manual node/edge graph construction API.
- `src/dynafx/domain.py` — Domain config contextvars for reasoning parameter tuning.
- `src/dynafx/registry.py` — Plugin registry for custom builtins and DES hooks.
- `src/dynafx/system/dsl.py` — main DSL: parser, expression AST, `_replace_smooths()` with ExprNode, `_build_system()` with `CompiledSystem` cache, `_compile_system()`, `SysdModel`, `SysdModelResult`. Submodel support: `SubmodelDef`, `IncludeDef`, `_expand_includes()`.
- `src/dynafx/system/units.py` — `Unit`, `UnitRegistry`, `UnitChecker`, 40 tests.
- `src/dynafx/system/causal.py` — 15 tests, `causes_tree`, `effects_tree`, `causes_strip`, `causal_trace`.
- `src/dynafx/system/feedback.py` — 8 tests, `detect_feedback_loops`, `loops_for_variable`.
- `src/dynafx/system/optimization.py` — `lp_minimize`, `calibrate`, `optimize`, 12 tests.
- `src/dynafx/system/agent.py` — `AgentInstance`, `ABMEngine`, `_eval_condition`.
- `src/dynafx/system/des.py` — `DESClock`, `EventQueue`, `Queue`, `Resource`, `DESEngine`, `QueueStats`, `ResourceStats`.
- `src/dynafx/system/emergent.py` — `EmergentProperty`, `Condition`, `Effect`, `run_consistency_checks`.
- `src/dynafx/system/equations.py` — `rk4_step()`, `euler_step()`.
- `src/dynafx/system/__main__.py` — CLI: simulate, validate, list.
- `src/dynafx/system/__init__.py` — SD public API: exports SysdModel, parse_sysd, causal_trace, detect_feedback_loops, etc.
- `src/dynafx/system/scenario.py` — `ScenarioDef`, `ScenarioResult`, `ScenarioComparison` with comparison/deviation/tornado/summary.

- `src/dynafx/system/vensim.py` — Vensim `.mdl` import.
- `src/dynafx/system/ontology.py` — SD ontology.
- `src/dynafx/sl/` — SL package: `operators.py`, `validation.py`, `parameters.py`.
- `src/dynafx/reason/evidence.py` — `ConsensusLevel`, `PairwiseAgreement`, `ClaimAssessment`, `EvidenceMatrix`, `EvidenceMatrixResult`. L1-distance agreement, cumulative fusion consensus classification.
- `src/dynafx/reason/fusion.py` — `cumulative_fusion()`, `consensus_compromise()`, `classify_fusion_situation()`, `consensus_to_fusion_situation()`.
- `src/dynafx/templates/__init__.py` — exports `SignalChain`.
- `src/dynafx/templates/signal_chain.py` — `SignalChain` class: factory that builds a `SysdModel` for leading-indicator → outcome pattern. Parameters: trace_expr, detection_delay (list for multi-hop), decision_lag, outcome_threshold, outcome_sensitivity, threshold_direction, has_feedback, has_tracking. `SignalChain.build(...)` is the canonical constructor.
- `templates/signal_chain.sysd` — Reference `.sysd` template documenting the signal chain structure for DSL-side use.
- `models/saas_churn_signal.sysd` — SaaS churn signal chain, dt=0.25, 43-day lead time.
- `models/telecom_signal_study.sysd` — Telecom SINR-based churn model with closed-loop power control, SNR/packet scaling, fade dynamics.
- `examples/saas_churn_signal.py` — SaaS churn demo with 5 scenarios, 8-param sensitivity, signal lead time.
- `examples/signal_showcase.py` — All 9 leading indicator domains built with SignalChain, analytical lead times (18-111 days).
- `examples/telecom_signal_study.py` — Telecom signal study with 11-page FPDF report, 5 scenarios, causal tracing, sensitivity.
- `src/dynafx/kb/model.py` — RDF data model: `RDFNode`, `NamedNode`, `BlankNode`, `Literal`, `Triple`, `TriplePattern`, XSD types, RDF/RDFS/OWL namespace constants. 41 tests in `tests/test_kb_model.py`.
- `src/dynafx/kb/store.py` — `TripleStore` with SPO/POS/OSP nested-indices, named graphs, dedup by max-belief opinion, pattern matching for all 8 patterns, graph isolation/copy/removal. 40 tests in `tests/test_kb_store.py`.
- `src/dynafx/kb/turtle.py` — Turtle/N-Triples tokenizer, recursive descent parser, serializer. Supports @prefix/@base, a, all literal types, blank nodes, ; and , grouping, comments, base IRI resolution, empty-prefix PNAME_LN. 40 tests in `tests/test_kb_turtle.py`.
- `src/dynafx/kb/inference.py` — `Rule`, `Var`, `InferencePattern`, `RuleEngine` (forward-chaining), `rdfs_rules()` (7 rules), `owl_rl_rules()` (4 rules), `propagate_opinion()` (min/product/average). 53 tests in `tests/test_kb_inference.py`.
- `src/dynafx/kb/confidence.py` — `fuse_graphs()`, `grade_query()`, `argumentative_filter()`. 30 tests in `tests/test_kb_confidence.py`.
- `src/dynafx/kb/production.py` — `ProductionRuleEngine`, `ProductionRule`, `Condition` hierarchy (5 types), `Action` hierarchy (5 types), fire_once/max_fires/priority. 36 tests in `tests/test_kb_production.py`.
- `src/dynafx/kb/transactions.py` — `Transaction`, `TransactionStore`, `TransactionQuery`. Append-only temporal log with RDF backing. 21 tests in `tests/test_kb_transactions.py`.
- `src/dynafx/kb/execution.py` — `ExecutionRecord`, `ExecutionStore`. Provenance-tracked action records. 10 tests in `tests/test_kb_transactions.py`.
- `src/dynafx/reason/argumentation.py` — `Argument`, `Attack`, `AttackType`, `SupportType`, `ArgumentationFramework` (grounded/preferred semantics), `build_framework()` (rebut/undermine/undercut attacks from contradictory claims, low-belief triples, source reliability). 27 tests in `tests/test_argumentation.py`.
- `src/dynafx/reason/kbt.py` — `KBTResult`, `compute_kbt()` EM algorithm for source reliability scoring. Writes `prov:reliability` to `meta` graph. 14 tests.
- `tests/test_kbt.py` — 14 tests for KBT engine.
- `ex/kb_transactions.py` — 21 tests for TransactionStore + 10 tests for ExecutionStore.
- `examples/argumentation_showcase.py` — Full pipeline: Turtle → named graphs → RDFS inference → argumentation filter → SL fusion → query grading. Source reliability scored by KBT.
- `tests/test_evidence_matrix.py` — 27 tests for EvidenceMatrix.
- `upcoming2.plan` — Full plan for kb/ package: model, store, turtle, sparql, inference, confidence. 1610 source lines, 130 tests across 6 phases.
- `models/global_solar_epc.sysd` — 3-region multi-project SD+ABM+DES model with shared Asian supply chain. 11 stocks, 59 auxes, 5 DES queues, 3 resources, 2 agent types. KB_QUERY for disruption/supplier/project risk. STEP-based disruption gating port outflow. Model verified with 3 disruption scenarios (baseline 86.7%→$961M profit, moderate disruption 81%→$898M, severe disruption 69%→$766M). Flow expressions require aux intermediaries for cross-stock sharing.
- `examples/global_solar_epc_dashboard.py` — 16-tab dashboard (~1200 lines). Pipeline: KB → RDFS inference → baseline sim → disruption → post-disruption → 6 scenarios → OAT → causes_strip → feedback loops → optimization. Generates `/tmp/solar_epc_16tab_dashboard.html` (615KB, 16 tabs).
- `models/ev_battery_supply_chain.sysd` — 6-echelon EV battery supply chain with 10 stocks, 86 auxes, 4 DES queues, 2 resources, 120 agents.
- `examples/ev_battery_supply_chain.py` — 8-page FPDF report generator with demand, inventory, DES, financials, bullwhip, 7 scenarios, LP optimization.
- `hierarchy.md` — Complete feature hierarchy across all packages with file paths, test counts, and dependency relationships.
