# DynaFX YouTube Series: "Build Thinking Simulations"

**Tagline:** Stop writing simulations that just crunch numbers. Build ones that reason.

**Audience:** Python developers who want to build simulations that connect to knowledge graphs, adapt to real-time events, and reason about outcomes.

**Total:** 12 episodes + 3 bonus episodes + 3 real-world projects

---

## Series Philosophy

Every video is labeled by **what it does**, not by difficulty. Each is self-contained but builds on the previous. Every episode has:

- A hook that explains why this matters
- Live code that works
- A project you can build yourself
- A challenge for the comments

---

## The Series

| # | Title | What It Does | Duration |
|---|-------|--------------|----------|
| 1 | [Your First Simulation in 5 Minutes](episodes/01-first-simulation.md) | Parse and run a model | 8-10 min |
| 2 | [Building Models with Python](episodes/02-building-models.md) | Construct models programmatically | 15-18 min |
| 3 | [What If? Comparing Scenarios](episodes/03-comparing-scenarios.md) | Run multiple parameter sets | 12-15 min |
| 4 | [Which Parameter Matters?](episodes/04-sensitivity-analysis.md) | Find what drives the output | 15-18 min |
| 5 | [Agents with Brains](episodes/05-agent-modeling.md) | Behavioral modeling with strategies | 15-18 min |
| 6 | [Queues and Resources](episodes/06-queueing-modeling.md) | Event-driven simulation | 12-15 min |
| 7 | [Knowledge Graphs: Memory](episodes/07-knowledge-graphs.md) | Store and query facts | 15-18 min |
| 8 | [The Closed Loop](episodes/08-closed-loop.md) | KB ↔ Simulation connection | 18-22 min |
| 9 | [Live Disruption](episodes/09-live-disruption.md) | Mid-simulation changes | 15-18 min |
| 10 | [Cognitive Orchestration](episodes/10-cognitive-orchestration.md) | Event-driven orchestration: rules → actions | 15-18 min |
| 11 | [Signal Cascades](episodes/11-signal-cascades.md) | Leading indicators and feedback | 15-18 min |
| 12 | [The Full Picture](episodes/12-full-case-study.md) | Enterprise reasoning end-to-end | 20-25 min |

---

## Bonus Content

| Title | What It Does | Duration |
|-------|--------------|----------|
| [DynaFX vs Other Tools](bonus/comparison.md) | Compare to Vensim, NetLogo, SimPy | 10-12 min |
| [Common Mistakes](bonus/common-mistakes.md) | Top 10 beginner errors | 10-12 min |

---

## Real-World Projects

| Title | What It Builds | Duration |
|-------|----------------|----------|
| [Hospital Emergency Department](real-projects/hospital-ed.md) | Patient flow with SD + ABM + DES + KB | 25-30 min |
| [Global Supply Chain](real-projects/supply-chain.md) | Disruption modeling with full stack | 25-30 min |
| [Climate Policy Impact](real-projects/climate-policy.md) | Policy analysis with sensitivity | 25-30 min |

---

## Episode Map

| # | Topic | Docs Tutorial | Key APIs |
|---|-------|---------------|----------|
| 1 | Parse & run | 01-hello-world | `parse_sysd`, `simulate` |
| 2 | Python DSL | 02-system-dynamics | `SysdModel`, `stock`, `aux`, `param` |
| 3 | Scenarios | 08-scenarios | `ScenarioComparison`, `ScenarioDef` |
| 4 | Sensitivity | 08-scenarios | `SensitivityAnalyzer`, `sobol` |
| 5 | ABM | 03-agent-based | `agent`, `prop`, `rule`, `strategy` |
| 6 | DES | 04-discrete-event | `queue`, `resource`, `event` |
| 7 | Knowledge | 05-knowledge-graph | `TripleStore`, `parse_turtle` |
| 8 | Bridge | 07-closed-loop | `KBSimBridge`, `KB_QUERY` |
| 9 | Live events | — | `KB_QUERY`, `KB_ASSERT` |
| 10 | Rules | 09-custom-ontology | `ProductionRule`, `CognitiveOrchestrator` |
| 11 | Optimization | 10-publishing | `lp_minimize`, `calibrate` |
| 12 | Case study | case-study | All APIs |

---

## Production Checklist (Per Episode)

- [ ] Script outline (1-page)
- [ ] Code samples (tested, copy-pasteable)
- [ ] Thumbnail (logo + result + hook)
- [ ] Description (links to docs, code, GitHub)
- [ ] Pin comment (community challenge)
- [ ] End screen (next video + subscribe)

---

## Thumbnail & Title Formula

**Title:** `[Action verb] + [what you'll build] + with DynaFX`

Examples:
- "Build a Supply Chain That Thinks — with DynaFX"
- "Find What Matters — Sensitivity Analysis with DynaFX"
- "Real-Time Disruption — Live Simulation with DynaFX"

**Thumbnail:** Left = DynaFX logo/code, Right = visual result, Bottom = hook text.

---

## Community Engagement

- **Pin comment** on each video: link to docs + GitHub + challenge
- **Challenge of the week:** viewers build a model, share in discussions
- **Viewer showcase:** feature community projects in later videos
- **Live coding sessions:** monthly, build viewer-suggested models
