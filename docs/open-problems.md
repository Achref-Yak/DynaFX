# Open Research Problems

Each problem below is a concrete research question we believe is tractable with DynaFX as infrastructure. For each: **why it matters**, **what a solution might involve**, and **how you could contribute**.

If you are a researcher, pick a problem, open an issue or a discussion, and we will help you scope it. Publications arising from these problems will be credited to their authors.

---

## 1. How can OWL reasoning scale to large digital twins?

**Why it matters.** The built-in reasoner is forward-chaining to a fixpoint, in-memory, single-process. Fine for enterprise-scale demo graphs (thousands of triples). Real twins — sensor streams, multi-enterprise federations — reach millions of triples, and naive forward chaining does not scale.

**What a solution might involve.** Incremental/rete-style materialization that only recomputes affected entailments on triple insertion; selective reasoning over reachable fragments; or offloading to a production reasoner via export (SPARQL update round-trips) while keeping the same ontology semantics.

**How to contribute.** Benchmarks of reasoning performance across graph sizes; an incremental entailment strategy; or a federated reasoning protocol.

---

## 2. How should semantic evidence from simulations be validated?

**Why it matters.** Evidence triples are the twin's learning: "revenue = $928M", "portfolio is at risk". Before a decision-maker acts on evidence, they need to trust it. Today evidence is recorded with provenance (PROV) but there is no validation layer — no check that an evidence value is plausible, consistent across runs, or supported by the assumptions it derives from.

**What a solution might involve.** Validation as a reasoning pass: consistency constraints over evidence (physical bounds, conservation, cross-check against independent facts), sensitivity of conclusions to their evidence, and confidence metadata attached to evidence triples.

**How to contribute.** A validation rule set over evidence; a case study auditing the supply-chain twin's conclusions; or a framework for "evidence quality" as a first-class concept.

---

## How to get involved

- **Pick a problem** — open a GitHub discussion linking to this page.
- **Bring a dataset** — the ingestion pipeline (`ingest_csv` + YAML mappings) makes new domains easy to wire in.
- **Bring a paper** — implement a published algorithm on DynaFX and publish the result here.
- **Ask a question we haven't listed** — open problems are living; we add new ones as the platform grows.
