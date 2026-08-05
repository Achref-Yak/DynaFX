# Scientific Foundations

This page documents the *why* behind DynaFX: the design decisions, their rationale, and their limits. It is intended to be citable in research that builds on the platform.

---

## Why RDF?

RDF (Resource Description Framework) models knowledge as **subject–predicate–object triples** with globally unique IRIs. We chose it for four reasons:

1. **Open, standardized, and interoperable.** RDF is a W3C standard. Knowledge built in DynaFX can be exported, merged, and queried with the wider semantic-web ecosystem (or moved to a production triple store) without a translation layer.
2. **Identity is explicit.** Every entity is an IRI; every property is a named relation. This matches how enterprises actually describe themselves — contracts reference suppliers, projects reference ports — as typed, nameable relationships, not rows in a table.
3. **Schema-flexible.** RDF admits data without a fixed schema and lets the ontology impose structure where needed. A system ingesting heterogeneous sources (CSVs, sensor exports, documents) benefits enormously from this tolerance.
4. **Reasoning-ready.** RDF's model is the substrate for RDFS and OWL, which give us class hierarchies, domain/range constraints, and entailment — inference that a relational schema cannot express.

**Assumption:** the system's facts are representable as entities and binary relations. Multi-valued and n-ary facts must be reified — a known limitation we accept in exchange for the benefits above.

---

## Why OWL and SPARQL?

**OWL** provides the ontology layer: classes (`rdfs:subClassOf`), properties (`rdfs:subPropertyOf`), and constraints (domain/range, inverse, transitivity). We use the **RL profile** — a tractable fragment with polynomial-time reasoning — via a forward-chaining engine. This gives us the expressive tools that matter for enterprise models (hierarchies, property semantics, type propagation) without the undecidability of full OWL.

**SPARQL** is the query language over the resulting graph. It is the standard interface between "what the system knows" and "what the system computes". In DynaFX, SPARQL strings are used in three places:

- *Pre-flight*: `params_from_kb` extracts KB facts into parameters (we support a declarative claim-map form).
- *Mid-flight*: `KB_QUERY` builtins evaluate ASK/SELECT queries every timestep, so simulation dynamics are numerically steered by knowledge.
- *Post-flight / reasoning*: SPARQL backs production-rule conditions and scenario grading.

**Assumption:** the knowledge the system needs is queryable as a graph. Temporal and probabilistic knowledge require additional structure that is out of scope today (see [Open Research Problems](open-problems.md)).

---

## Why Combine SD + ABM + DES?

Each simulation paradigm models a different kind of question, and real systems are rarely single-paradigm:

| Paradigm | Models | Right for |
|----------|--------|-----------|
| **System Dynamics** | Aggregate stocks, flows, feedback | Long-horizon behavior, policy levers, feedback structure |
| **Agent-Based Modeling** | Heterogeneous actors, rules, interaction | Emergence, strategy switching, distributed decisions |
| **Discrete Event Simulation** | Queues, resources, schedules | Congestion, capacity, event timing, logistics |

In the flagship example (see the [Case Study](case-study-solar-epc.md)), the port closure is an SD-level supply-rate disruption, ABM agents switch procurement strategies under crisis conditions, and DES queues model berth and yard congestion — all in one model, all reading the same knowledge graph. A single paradigm could not represent that combination faithfully.

The three paradigms share **one state namespace** (stocks + agent metrics + queue metrics in a single dict). This makes cross-paradigm coupling — a DES queue gated by an SD stock, an aux reading an agent aggregate — a first-class design decision rather than an ad-hoc integration.

**Assumption:** the paradigms' outputs are commensurable (numbers on a shared time axis). Where the disciplines disagree on time semantics (continuous vs. event-driven), DynaFX uses a fixed integration step and merges DES/ABM metrics into the shared per-step state.

---

## Closed-Loop Reasoning

The core intellectual claim of DynaFX is that **knowledge and dynamics should be one loop**:

```
knowledge → parameters → simulation → evidence → knowledge
```

This is a *semantic simulation*: the model's dynamics are not just calibrated against data — they are *steered by* the knowledge graph at run time, and the results *become* knowledge. `ClosedLoopReasoner` operationalizes this as iterative **simulate → grade → nudge → re-simulate** cycles until targets are met.

This is a *closed-loop reasoning* architecture: the system learns at run time (evidence), foresees the future (scenario/sensitivity), and acts (rules + optimization) — with explicit, auditable mechanisms rather than opaque black boxes.

---

## Design Assumptions

- **All enterprise data is representable as RDF triples.** CSV→RDF via YAML mappings is the canonical ingestion path.
- **Aggregates are materialized, not computed on query.** The SPARQL evaluator has no GROUP BY; aggregate facts (reliability, projects-at-risk) are computed at ingest and stored explicitly.
- **Determinism is the default.** Simulations are seeded; reproducibility is a research requirement.
- **Models are explicit and auditable.** No hidden fitted parameters inside the model — learning is structural (evidence and rules), not black-box.
- **Single-process, in-memory.** The triple store and simulation run in one process. Scale-out and federation are future work.

---

## Limitations

We document these deliberately, as they bound the claims one can make with DynaFX today:

1. **No ML/DL training.** The platform reasons with explicit symbolic models. Data-driven components (anomaly detection, RUL models) can be *composed* by producing parameters or evidence, but are not trained inside DynaFX.
2. **No streaming / real-time ingestion.** Data is ingested as CSVs/triples; there is no live IoT or sensor pipeline. Real-time streaming is future work.
3. **Limited SPARQL expressiveness.** No GROUP BY / aggregates, no subqueries, no property paths. Complex analytics must be pre-computed or expressed as explicit triples.
4. **Single-process scale.** The store is in-memory; very large graphs require a production triple store (export is trivial, but the built-in engine is not a database).
5. **No native probabilistic reasoning.** Uncertainty propagation between the knowledge graph and the simulation is an [open research problem](open-problems.md).
6. **Reified n-ary facts.** Multi-valued facts require explicit reification.

---

## Citable Materials

- [Concepts](concepts.md) — the vocabulary (knowledge graph, bridge, evidence, scenario, policy, closed loop).
- [Architecture](architecture.md) — the layered pipeline and extension points.
- [Open Research Problems](open-problems.md) — questions we are seeking collaborators for.
- [Case Study](case-study-solar-epc.md) — the flagship end-to-end example.
