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
- Build a Vensim-class system dynamics modeling framework with full parity plus multi-paradigm support (SD + ABM + DES) and a separate RDF/OWL/SPARQL cognitive reasoning engine with SL confidence grading, both general-purpose Python-native.

## Constraints & Preferences
- `SystemDecomposer` is the primary API for manual decomposition — clean `add_node`/`add_edge`, no extraction noise.
- Detection passes are read-only structural matchers on `Graph` — never mutate nodes/edges.
- SL opinions are inert on CAUSAL/System Dynamics paths — default `Opinion()` everywhere, confidence in metadata.
- All structural analysis lives in `detect_emergence.py` as module-level functions — decomposer only delegates `detect()`.
- Custom `.sysd` DSL uses indent-based Vensim-like syntax with full arithmetic + functions (MIN, MAX, IF, SMOOTH, lookup tables, comparison operators).
- DSL is parsed by a hand-written line-oriented parser with recursive descent expression parser — zero dependencies beyond stdlib.
- Model parameters are runtime values passed via `params` dict — merged into state vector `_s` so expression references resolve.
- **SD/SL clean separation**: SD structures carry no SL types. SL files live in `cognitive_engine/sl/` package, not `system/`.
- **ABM agent state is numeric only** — no SL opinions on agents. Pluggable via generic dict.
- **DES includes full queuing theory** — resource pools, capacity constraints, utilization stats.
- **Unified state dict**: SD, ABM, and DES all write to the same `state: dict[str, float]` during simulation, enabling cross-paradigm interactions.
- **No visualization for now** — visualization phase deferred.
- **No domain-specific APIs** — no /orders, /shipments, /customs endpoints. Pure modeling framework.
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
- **CLI entry point (P1)**: `src/cognitive_engine/system/__main__.py` with `simulate`, `validate`, `list` subcommands. Extended with `--paradigm` flag (sd/abm/des/all) and `--stats` flag.
- **Plotting API (P1)**: `SysdModelResult` class with `.plot(path, stocks, subplots, title)` and `.plot_with_bands(path, mean, std, p5, p95)`.
- **Sensitivity (P1)**: `SysdModel.simulate_ensemble(params, fixed_params, n, method, seed)` with uniform/normal/lognormal distributions.
- **Vensim `.mdl` import (P1)**: `src/cognitive_engine/system/vensim.py` parses INTEG stocks, WITH LOOKUP tables, TIME→t mapping, SMOOTH/DELAY1 mapping, continuation line joining.
- **Model library (P1)**: 12+ curated `.sysd` models in `models/` plus `pandemic_seirvh.sysd`.
- **SD ontology (P2)**: `src/cognitive_engine/system/ontology.py` with stock/flow subtype inference (MATERIAL/INFORMATION/FINANCIAL), cross-type flow validation.
- **Pandemic model bug fixes**: recovery_fraction dynamic compensation, ICU_Fatality time-normalization, vaccination inflow to Recovered, healthcare_stress_avg consumed by IF-gate and Expansion flow.
- **Deleted `rules/engine.py`** — dead code in SD context. Removed entire `rules/` package.
- **EmergentProperty dataclass** (`emergent.py`): `Condition`, `Effect`, `ComparisonOp`, `EffectType`, `ConsistencyResult`, `ConsistencyViolation`, `run_consistency_checks`, 4 checker functions.
- **SD/SL clean separation completed**: Removed `Opinion`/`Parameter` from `Equation.confidence`. Removed `LoopClassification`. Removed `Parameter` import from dsl.py. Removed `opinion: Opinion` from `EmergentProperty`. Removed backward compat `Parameter` handling in `equations.py`.
- **SL files moved to `cognitive_engine/sl/`**: `operators.py`, `validation.py`, `parameters.py` moved out of `system/`.
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
- **EvidenceMatrix complete** — `src/cognitive_engine/reason/evidence.py` with `ConsensusLevel`, `PairwiseAgreement`, `ClaimAssessment`, `EvidenceMatrix`, `EvidenceMatrixResult`. L1-distance agreement scoring. 27 tests in `tests/test_evidence_matrix.py`.
- **EvidenceMatrix consensus → FusionSituation integration** — `consensus_to_fusion_situation()` in `fusion.py`, `classify_fusion_situations()` method on `EvidenceMatrixResult`.
- **EvidenceMatrix exported** — `__init__.py` exports `ClaimAssessment`, `ConsensusLevel`, `EvidenceMatrix`, `EvidenceMatrixResult`, `PairwiseAgreement`.
- **`src/cognitive_engine/templates/` package created** — `SignalChain` class in `signal_chain.py` constructs a `SysdModel` for the leading-indicator → outcome pattern.
- **Template `.sysd` reference file** — `templates/signal_chain.sysd` documents the structure for DSL-side use.
- **Signal showcase example** — `examples/signal_showcase.py` builds and simulates all 9 leading indicator domains with comparison table.
- **`SignalChain` updated with `threshold_direction` parameter** — supports "below" direction for decline signals (foot traffic, card payments).
- **Showcase lead times fixed** — analytical computation from model parameters (pipeline delay + threshold sensitivity), all 9 domains produce lead times (18-111 days).
- **Cognitive reasoning engine plan written** — `upcoming2.plan` with 7 source + 6 test files, 130 tests, ~1610 source lines.
- **Phase 1: RDF data model** — `src/cognitive_engine/kb/model.py` with `RDFNode`, `NamedNode`, `BlankNode`, `Literal`, `Triple`, `TriplePattern`, XSD shortcuts, namespace constants (RDF/RDFS/OWL). 41 tests all passing.
- **Phase 2: TripleStore** — `src/cognitive_engine/kb/store.py` with `TripleStore` class, SPO/POS/OSP nested-index prefix strategy, named graph per source, dedup by max-belief opinion, graph isolation/copy/removal. 40 tests all passing.
- **Phase 3: Turtle/N-Triples** — `src/cognitive_engine/kb/turtle.py` with tokenizer, recursive descent parser, serializer, N-Triples support, base IRI resolution. 40 tests all passing.
- **Phase 4: SPARQL** — `src/cognitive_engine/kb/sparql.py` with lexer (case-insensitive keywords), recursive descent parser, algebra tree, filter expressions, evaluator producing `QueryResult`. 35 tests all passing.
- **Phase 5: Inference engine** — `src/cognitive_engine/kb/inference.py` with `Rule`, `Var`, `InferencePattern`, `RuleEngine` (forward-chaining), RDFS (7 rules) and OWL RL (4 rules) rule sets, opinion propagation (min/product/average). 53 tests all passing.
- **Phase 6: Confidence layer** — `src/cognitive_engine/kb/confidence.py` with `fuse_graphs()`, `grade_query()`. 30 tests all passing.
- **Phase 7: `kb/__init__.py`** — public exports, integration wiring to `reason/` and `tbox/`.
- **Phase 8: Scenario comparison** — `system/scenario.py` with `ScenarioDef`, `ScenarioResult`, `ScenarioComparison` (plot_comparison, plot_deviation, tornado, summary, deviation_table). 20 tests all passing.
- **Argumentation engine** — `reason/argumentation.py` with `Argument`, `Attack`, `AttackType`, `SupportType`, `ArgumentationFramework` (Dung grounded/preferred semantics), `build_framework()` (rebut, undermine, source reliability attacks). Integrated into `kb/confidence.py` as `argumentative_filter()`. 27 tests all passing.
- **`TripleStore.all_triples()`** — added method to iterate all triples across all graphs.
- **KBT (Knowledge-Based Trust) engine** — `reason/kbt.py` with `KBTResult`, `compute_kbt()` EM algorithm that scores source reliability without ground truth. Iterates E-step (infer likely true values by weighted vote) and M-step (recompute source trust from accuracy). Outputs `prov:reliability` triples to `meta` graph. 14 tests all passing.
- **Argumentation showcase** — `examples/argumentation_showcase.py` demonstrates full pipeline: Turtle parse → named graphs → RDFS inference → SL consensus → argumentation filter → fusion. Source reliability scored by KBT.
- **Knowledge Fusion Showcase** — `examples/knowledge_fusion_showcase.py` demonstrates KG → KBT → EvidenceMatrix → Argumentation → filter → SL fusion → SPARQL grading in a single pipeline.
- **Student Math SD model** — `models/student_math.sysd`: 3 stocks (Math_Anxiety, Math_Performance, Self_Efficacy), 6 flows, 10 auxes. KG_* params injected from cognitive engine bridge. IF uses function-call syntax `IF(cond, a, b)` not keyword syntax `IF...THEN...ELSE`.
- **Multi-Paradigm Student Pipeline** — `examples/multi_paradigm_student.py`: 3-pass orchestration (pre-diagnosis → intervention → follow-up). Each pass: Turtle → RDFS inference → KBT → argumentation → filter → bridge → simulate SD+ABM+DES → extract evidence → feed to next pass. Bridge reads from original store (not filtered) to bypass grounded semantics skepticism. Pass 1: anxiety 25→59, performance 55→10, reinforcing loop dominates. Pass 2: intervention reverses, anxiety to 0.5, performance recovers to 26. Pass 3: sustained recovery to 34.
- **KBT tests fixed** — `tests/test_kbt.py` 14 tests handle TripleStore dedup (max-belief version stored for identical (s,p,o) across graphs). All passing.
- **Stale NLP remnants purged** — Deleted `agents/`, `api/`, `scripts/`, `demo/`, `memory/`, `kernel/`, `domains/`, `docs/`, `tests/test_agents.py` + 6 memory/kernel test files (~1424 lines dead scripts, 10 stale docs, ~109 obsolete tests). Removed `perception.hypothesis_generator` dead lazy import from top-level `__init__.py`. Removed `_load_dotenv()` function (unused).
- **Circular import fixed** — `kb/__init__` → `kb.confidence` → `reason.argumentation` → `kb.model` cycle broken by deferring `build_framework` import inside `argumentative_filter()`.
- **`pyproject.toml` cleaned** — Removed `torch`, `transformers`, `spacy`, `nltk`, `fastapi`, `uvicorn`, `websockets`, `leidenalg` from dependencies (all from deleted NLP pipeline). Core deps now only `networkx`, `numpy`, `scipy`, `pydantic`.
- **`sl/__init__.py`** and **`tbox/__init__.py`** populated with proper exports.
- **`Makefile`** fixed — `run` target now points to `cognitive_engine.system` instead of deleted `cognitive_engine.cli`.
- **README.md rewritten** from scratch — describes current two-pillar architecture with quick start examples and example table.

### In Progress
- (none)

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
- **SD/SL clean separation** — SL files in `cognitive_engine/sl/`, SD in `cognitive_engine/system/`.
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
- **`SignalChain.__new__` returns a `SysdModel` instance** — `model = SignalChain(...)` gives a ready-to-simulate `SysdModel`, no `.sysd` file needed.
- **Template structure: trace → DELAY3 → SMOOTH → IF → impact → core stock outflow** — the trace signal is a leading indicator that modulates an existing process, not the stock's inflow.
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

## Next Steps
- **ReputationTracker (Beta-distribution)** — `reason/reputation.py` with Beta-based reputation tracking, trust decay, and initial reputation priors. ~12 tests.
- **Source reliability UI / reporting** — expose KBT scores in a structured report format. ~5 tests.
- **Batch argumentation** — run argumentation framework across multiple `EvidenceMatrix` scenarios. ~8 tests.
- **Phase 9: Gaming mode** — interactive parameter adjustment, `GameSession` with pause/resume/set_param. ~10 tests.
- **Phase 10: Step-based control** — `SimulationController`, single-step advance, state injection, event callbacks. ~10 tests.
- **Phase 11: Batch simulation** — `BatchRunner`, multi-scenario parallel execution, summary tables. ~10 tests.
- **Phase 12: CONVEY + transport delays** — pipeline delay as FIFO buffer with timestamp tracking. ~8 tests.
- **Fix supply chain model dynamics** — retailer stockout due to DELAY_FIXED buffer issues with RK4 intermediate calls.

## Critical Context
- **1192 tests passing** (core SD + kb + reason engine).
- **Old NLP pipeline fully removed**: `agents/`, `api/`, `scripts/`, `demo/`, `memory/`, `kernel/`, `domains/`, `docs/` directories deleted. `extract/`, `nlp/`, `perception/` were already removed earlier. Remaining `tests/` all pass with no stale imports.
- **`SysdModel.simulate()` returns `SysdModelResult`** — `stocks` is a list of stock names, `values` is `dict[str, list[float]]`. **Aux values NOT exposed in results** — only stock values.
- **Parameters not in `params` dict return 0.0** — `_s.get('beta', 0.0)` returns 0 if param not provided.
- **DES queues process departures** — `service_time` expressions compiled against current state at each step.
- **Aux vars with numeric expressions can be overridden via params** — `simulate()` merges float-valued AuxDef expressions into params dict.
- **Optimize function clamps Nelder-Mead results to bounds** — Nelder-Mead doesn't respect bounds natively.
- **Full showcase demo runs in ~12s** (was 300s+ before caching).
- **SaaS churn model: 43-day signal lead time**, dt=0.25, B1 sign corrected, signal tracking stocks track smoothed auxes.
- **DSL include `params` only supports float values** — cannot pass string expressions. The Python API (`SignalChain` class) supports full expressions.
- **`SignalChain` builds a `SysdModel` with**: trace expr → multi-hop DELAY3 → SMOOTH → IF threshold → signal_impact → adjusted_outflow → core stock. Base inflow/outflow params included. Feedback loop optional. Tracking stocks for Trace/Detected/Interpreted. `threshold_direction` supports "above"/"below" for decline signals.
- **SignalChain showcase generates all 9 domains** — `examples/signal_showcase.py` builds, simulates, and compares all domains. Lead times computed analytically from pipeline delays + threshold sensitivity. All 9 produce lead times (18-111 days).
- **Framework identified as "framework"** (not SaaS/PaaS/library), with capability to model OODA loops (Observe via state dict, Orient via causal tracing, Decide via ABM/IF logic, Act via state updates).
- **Closest to AnyLogic (multi-paradigm)** and **Vensim (expression/causal logic)**, distinct by being code-first/Python-native.
- **Cognitive reasoning engine (`kb/`) separate from SD engine (`system/`)** — two pillars sharing `reason/` as common SL substrate. `tbox/` provides OWL2-style type hierarchy consumed by `kb/`.
- **`kb/` Phase 1-7 complete**: model.py (41 tests), store.py (40 tests), turtle.py (40 tests), sparql.py (35 tests), inference.py (53 tests), confidence.py (30 tests). Full plan in `upcoming2.plan`.
- **RDF node types are frozen dataclasses** — NamedNode(iri), BlankNode(id), Literal(value, datatype, lang_tag). Triple equality/hashing ignores opinion.
- **TripleStore uses SPO/POS/OSP nested-index prefix strategy** — O(1) pattern matching for all 8 pattern types. Named graphs as `dict[str, set[tuple]]` for O(1) membership.
- **Turtle parser is recursive descent with tokenizer** — supports @prefix/@base, a (rdf:type), string/integer/decimal/boolean/typed/lang literals, blank nodes, ; and , grouping, comments, base IRI resolution, empty-prefix PNAME_LN (`:s`).
- **Delta DES not yet paused** — needs both state in ServiceEvent and time-remaining tracking for multi-step operations.

## Relevant Files
- `src/cognitive_engine/system/dsl.py` — main DSL: parser, expression AST, `_replace_smooths()` with ExprNode, `_build_system()` with `CompiledSystem` cache, `_compile_system()`, `SysdModel`, `SysdModelResult`. Submodel support: `SubmodelDef`, `IncludeDef`, `_expand_includes()`.
- `src/cognitive_engine/system/units.py` — `Unit`, `UnitRegistry`, `UnitChecker`, 40 tests.
- `src/cognitive_engine/system/causal.py` — 15 tests, `causes_tree`, `effects_tree`, `causes_strip`, `causal_trace`.
- `src/cognitive_engine/system/feedback.py` — 8 tests, `detect_feedback_loops`, `loops_for_variable`.
- `src/cognitive_engine/system/optimization.py` — `lp_minimize`, `calibrate`, `optimize`, 12 tests.
- `src/cognitive_engine/system/agent.py` — `AgentInstance`, `ABMEngine`, `_eval_condition`.
- `src/cognitive_engine/system/des.py` — `DESClock`, `EventQueue`, `Queue`, `Resource`, `DESEngine`, `QueueStats`, `ResourceStats`.
- `src/cognitive_engine/system/emergent.py` — `EmergentProperty`, `Condition`, `Effect`, `run_consistency_checks`.
- `src/cognitive_engine/system/equations.py` — `rk4_step()`, `euler_step()`.
- `src/cognitive_engine/system/__main__.py` — CLI: simulate, validate, list.
- `src/cognitive_engine/system/scenario.py` — `ScenarioDef`, `ScenarioResult`, `ScenarioComparison` with comparison/deviation/tornado/summary.

- `src/cognitive_engine/system/vensim.py` — Vensim `.mdl` import.
- `src/cognitive_engine/system/ontology.py` — SD ontology.
- `src/cognitive_engine/sl/` — SL package: `operators.py`, `validation.py`, `parameters.py`.
- `src/cognitive_engine/reason/evidence.py` — `ConsensusLevel`, `PairwiseAgreement`, `ClaimAssessment`, `EvidenceMatrix`, `EvidenceMatrixResult`. L1-distance agreement, cumulative fusion consensus classification.
- `src/cognitive_engine/reason/fusion.py` — `cumulative_fusion()`, `consensus_compromise()`, `classify_fusion_situation()`, `consensus_to_fusion_situation()`.
- `src/cognitive_engine/templates/__init__.py` — exports `SignalChain`.
- `src/cognitive_engine/templates/signal_chain.py` — `SignalChain` class: factory that builds a `SysdModel` for leading-indicator → outcome pattern. Parameters: trace_expr, detection_delay (list for multi-hop), decision_lag, outcome_threshold, outcome_sensitivity, threshold_direction, has_feedback, has_tracking.
- `templates/signal_chain.sysd` — Reference `.sysd` template documenting the signal chain structure for DSL-side use.
- `models/saas_churn_signal.sysd` — SaaS churn signal chain, dt=0.25, 43-day lead time.
- `examples/saas_churn_signal.py` — SaaS churn demo with 5 scenarios, 8-param sensitivity, signal lead time.
- `examples/signal_showcase.py` — All 9 leading indicator domains built with SignalChain, analytical lead times (18-111 days).
- `src/cognitive_engine/kb/model.py` — RDF data model: `RDFNode`, `NamedNode`, `BlankNode`, `Literal`, `Triple`, `TriplePattern`, XSD types, RDF/RDFS/OWL namespace constants. 41 tests in `tests/test_kb_model.py`.
- `src/cognitive_engine/kb/store.py` — `TripleStore` with SPO/POS/OSP nested-indices, named graphs, dedup by max-belief opinion, pattern matching for all 8 patterns, graph isolation/copy/removal. 40 tests in `tests/test_kb_store.py`.
- `src/cognitive_engine/kb/turtle.py` — Turtle/N-Triples tokenizer, recursive descent parser, serializer. Supports @prefix/@base, a, all literal types, blank nodes, ; and , grouping, comments, base IRI resolution, empty-prefix PNAME_LN. 40 tests in `tests/test_kb_turtle.py`.
- `src/cognitive_engine/kb/inference.py` — `Rule`, `Var`, `InferencePattern`, `RuleEngine` (forward-chaining), `rdfs_rules()` (7 rules), `owl_rl_rules()` (4 rules), `propagate_opinion()` (min/product/average). 53 tests in `tests/test_kb_inference.py`.
- `src/cognitive_engine/kb/confidence.py` — `fuse_graphs()`, `grade_query()`, `argumentative_filter()`. 30 tests in `tests/test_kb_confidence.py`.
- `src/cognitive_engine/reason/argumentation.py` — `Argument`, `Attack`, `AttackType`, `SupportType`, `ArgumentationFramework` (grounded/preferred semantics), `build_framework()` (rebut/undermine/undercut attacks from contradictory claims, low-belief triples, source reliability). 27 tests in `tests/test_argumentation.py`.
- `src/cognitive_engine/reason/kbt.py` — `KBTResult`, `compute_kbt()` EM algorithm for source reliability scoring. Writes `prov:reliability` to `meta` graph. 14 tests.
- `tests/test_kbt.py` — 14 tests for KBT engine.
- `examples/argumentation_showcase.py` — Full pipeline: Turtle → named graphs → RDFS inference → argumentation filter → SL fusion → query grading. Source reliability scored by KBT.
- `tests/test_evidence_matrix.py` — 27 tests for EvidenceMatrix.
- `upcoming2.plan` — Full plan for kb/ package: model, store, turtle, sparql, inference, confidence. 1610 source lines, 130 tests across 6 phases.
