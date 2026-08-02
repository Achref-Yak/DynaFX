# Open Research Problems

This is an invitation to collaborate — not just to contribute code. Each problem below is a concrete research question we believe is tractable with DynaFX as infrastructure. For each: **why it matters**, **what a solution might involve**, and **how you could contribute**.

If you are a researcher, pick a problem, open an issue or a discussion, and we will help you scope it. Publications arising from these problems will be credited to their authors.

---

## 1. How should uncertainty propagate between the knowledge graph and the simulation?

**Why it matters.** A digital twin is only as trustworthy as the facts it reasons over. Supplier reliability, project risk, and disruption status are all uncertain — they come from estimates, partial data, or conflicting sources. Today, a KB fact becomes a single numeric parameter (`params_from_kb` → `0.82`). That point-estimate collapse throws away the uncertainty a decision-maker actually faces.

**What a solution might involve.** Propagating distributions rather than points: KB facts carrying an interval or distribution, parameters becoming distributions, and scenario/sensitivity machinery downstream. Alternatively, coupling the KB to a probabilistic semantics (e.g., a probability-of-truth on triples, or probabilistic entailment) so inference itself is uncertainty-aware.

**How to contribute.** A study of uncertainty representation in the KB layer; a case study propagating reliability uncertainty through the supply-chain twin; or a formal argument for where uncertainty should live (fact-level vs. parameter-level vs. simulation-output-level).

---

## 2. How can OWL reasoning scale to large digital twins?

**Why it matters.** The built-in reasoner is forward-chaining to a fixpoint, in-memory, single-process. Fine for enterprise-scale demo graphs (thousands of triples). Real twins — sensor streams, multi-enterprise federations — reach millions of triples, and naive forward chaining does not scale.

**What a solution might involve.** Incremental/rete-style materialization that only recomputes affected entailments on triple insertion; selective reasoning over reachable fragments; or offloading to a production reasoner via export (SPARQL update round-trips) while keeping the same ontology semantics.

**How to contribute.** Benchmarks of reasoning performance across graph sizes; an incremental entailment strategy; or a federated reasoning protocol.

---

## 3. How should reinforcement learning interact with symbolic reasoning?

**Why it matters.** DynaFX's *act* stage is currently symbolic: production rules and LP optimization. RL would let the twin learn policies — procurement strategies, mitigation sequencing — from simulated experience. But a pure black-box policy discards everything the knowledge graph knows. The open question is the interface: how do symbolic facts constrain, reward, or guide the learning agent?

**What a solution might involve.** A Gym-like environment wrapping `SysdModel` + KB; rewards sourced from evidence triples; action spaces that correspond to policies (strategy switches, budget allocations) already representable in the twin; and hybrid schemes where rules provide safety constraints on learned actions.

**How to contribute.** An RL environment for the supply-chain twin; a study comparing learned vs. rule-based mitigation policies; or a hybrid rule+RL policy architecture.

---

## 4. How can distributed DynaFX simulations synchronize knowledge?

**Why it matters.** The flagship twin is a single process. Real enterprises run federated twins — each partner maintains its own knowledge graph and model, and the federation must agree on shared facts (demand, capacity, disruption status) without a central authority.

**What a solution might involve.** An explicit protocol for named-graph exchange and reconciliation: which graphs are shared, what merge semantics apply (triple identity is already `(s,p,o)`), and how conflicts are resolved. Extends naturally from the existing named-graph design.

**How to contribute.** A two-node federated twin demonstrating knowledge synchronization; a merge/reconciliation semantics for conflicting facts; or a consistency model for multi-agent knowledge updates.

---

## 5. How should semantic evidence from simulations be validated?

**Why it matters.** Evidence triples are the twin's learning: "revenue = $928M", "portfolio is at risk". Before a decision-maker acts on evidence, they need to trust it. Today evidence is recorded with provenance (PROV) but there is no validation layer — no check that an evidence value is plausible, consistent across runs, or supported by the assumptions it derives from.

**What a solution might involve.** Validation as a reasoning pass: consistency constraints over evidence (physical bounds, conservation, cross-check against independent facts), sensitivity of conclusions to their evidence, and confidence metadata attached to evidence triples.

**How to contribute.** A validation rule set over evidence; a case study auditing the supply-chain twin's conclusions; or a framework for "evidence quality" as a first-class concept.

---

## 6. What is the right division of labor between symbolic and data-driven models?

**Why it matters.** DynaFX is deliberately model-based. But data-driven components (anomaly detection, RUL, demand forecasting) are complementary. The research question is *where the seam goes*: which inferences are better symbolic (causal structure, policy logic, ontology) and which are better learned (patterns, residuals, forecasts), and how the two compose.

**What a solution might involve.** Composition patterns where a learned model emits a parameter or a prior into the symbolic model, and the symbolic model's evidence constrains the learner. A systematic comparison across twin scenarios.

**How to contribute.** A hybrid case study (e.g., learned demand forecast feeding the supply-chain twin); an evaluation methodology for hybrid twins; or a taxonomy of KB↔ML integration patterns.

---

## 7. How should learned models be integrated into a symbolic twin?

**Why it matters.** DynaFX reasons symbolically by design — explicit triples, rules, and equations you can audit. Yet real twin components are naturally learned: demand forecasts, anomaly detection, RUL, policies. Problem 6 asks *where the seam goes*; this asks the architectural question beneath it: given the platform's existing seams — `params_from_kb` (KB→parameter), `KB_ASSERT` (mid-run evidence), `registry.register_builtin` (expression-level functions), and `calibrate`/surrogates (sim-consumption) — what is the canonical, reproducible pattern for composing a learner in and out?

**What a solution might involve.** A `LearnerParamSource` wrapping any predictor and emitting KB facts through the existing bridge; a registered `FORECAST`-style builtin evaluated per timestep; a surrogate-training path that learns an emulator of a heavy simulation to accelerate sensitivity/Pareto search; and an RL environment wrapping `SysdModel` + KB with evidence-triple rewards (cross-ref Problem 3). The unifying constraint: the symbolic layer stays the audit backbone — the learned component emits parameters/evidence/actions but never replaces the reasoning layer. Ingestion-side ML (entity resolution, extraction) is explicitly out of scope — the import path is declarative CSV→RDF by design.

**How to contribute.** A minimal `LearnerParamSource` proving the composition pattern on the supply-chain twin (learned demand forecast → params → twin → evidence); an evaluation of a registered ML builtin vs. precomputed inputs; or a surrogate that accelerates `SensitivityAnalyzer` by an order of magnitude while preserving conclusion ranking.

---

## 8. How should federated digital twins preserve provenance across trust boundaries?

**Why it matters.** Provenance today is recorded per-run in the twin's own graph. In a federation, evidence crosses organizational boundaries, and each partner's provenance must be trusted by the others.

**What a solution might involve.** Signable provenance records, hash-chained run logs, or a ledger-based record of which twin produced which evidence — all representable in the existing PROV model.

**How to contribute.** A provenance protocol for multi-partner runs; a threat model for federated twin evidence; or a reference implementation of cross-boundary provenance.

---

## How to get involved

- **Pick a problem** — open a GitHub discussion linking to this page.
- **Bring a dataset** — the ingestion pipeline (`ingest_csv` + YAML mappings) makes new domains easy to wire in.
- **Bring a paper** — implement a published algorithm on DynaFX and publish the result here.
- **Ask a question we haven't listed** — open problems are living; we add new ones as the platform grows.
