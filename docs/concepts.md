# Concepts

This page is the mental model of DynaFX — the ideas you need before any code. Each concept is defined in plain language, then tied to where it lives in the platform and why it matters.

---

## Knowledge Graph

A **knowledge graph** is a collection of facts about the world represented as subject–predicate–object triples. DynaFX uses the W3C standard RDF data model: every fact is `(subject, predicate, object)`, subjects and predicates are IRIs, and objects are either IRIs (links to other entities) or typed literals (numbers, strings, booleans).

The knowledge graph is the system's *memory and identity*. It holds what the model knows: contracts, supplier reliability, project risk, port status, obligations. In DynaFX it is organized into **named graphs** — one per information source — so provenance is preserved at the source level.

**Where it lives:** `dynafx/knowledge/` (`TripleStore`, `NamedNode`, `Literal`, `Triple`).

## Ontology

An **ontology** is a vocabulary of classes and properties plus the rules relating them. DynaFX ships an RDFS/OWL-RL reasoner that derives new facts from the ontology: if `Supplier rdfs:subClassOf Enterprise` and `X rdf:type Supplier`, inference derives `X rdf:type Enterprise`.

The ontology gives the model *semantics*. It is what lets queries ask about concepts ("all at-risk projects") instead of raw identifiers, and what lets two datasets share meaning.

**Where it lives:** `dynafx/knowledge/inference.py` (`rdfs_rules`, `owl_rl_rules`), TBox hierarchy in `hierarchy.py`.

## Inference

**Inference** is the derivation of new facts from existing ones using the ontology's rules. DynaFX implements forward-chaining: rules are applied repeatedly until no new facts appear (a fixpoint). This is how a small, explicit knowledge base becomes a richer one without manual curation.

Inference matters because real systems combine facts from many sources with different vocabularies; inference reconciles them into a single, queryable picture.

**Where it lives:** `dynafx/knowledge/inference.py` (`Rule`, `RuleEngine`).

## Simulation Model

A **simulation model** is an executable abstraction of a system — a set of equations and rules that, given a start state and parameters, produces a trajectory over time. DynaFX supports three modeling paradigms, each answering different kinds of questions:

- **System Dynamics (SD)** — aggregate stocks, flows, and feedback. "How does inventory behave under a demand shock?"
- **Agent-Based Modeling (ABM)** — heterogeneous actors with individual rules. "How does a strategy switch ripple through 40 buyers?"
- **Discrete Event Simulation (DES)** — queues, resources, and scheduled events. "Where do the queues build up when the port closes?"

The three share one state namespace, so a single model can combine all three.

**Where it lives:** `dynafx/dynamics/` (`SysdModel`, `StockDef`, `FlowDef`, `AuxDef`, `AgentDef`, `QueueDef`, `ResourceDef`).

## Stock

A **stock** is an accumulation — the state of the system at a point in time (inventory, workforce, cash, containers in transit). Stocks change only through their flows. In SD, the fundamental equation is `stock(t+dt) = stock(t) + (inflows − outflows)·dt`.

Stocks are the model's *continuous state*. Everything observable about the aggregate system is ultimately a stock or a function of stocks.

## Flow

A **flow** is the rate at which a stock changes. Inflows add, outflows remove. Flows are driven by auxiliaries, parameters, other stocks, and — uniquely in DynaFX — by knowledge-graph facts queried mid-run.

Flows are the model's *dynamics*: the causal structure of the system lives in how flows are defined.

**Where it lives:** `FlowDef` (direction `+`/`−`, an expression).

## Aux (Auxiliary)

An **auxiliary** (aux) is an algebraic intermediate — a named quantity computed from other quantities each timestep, with no memory. Auxes keep flow equations readable and let the model expose meaningful named quantities (profit, fill rate, risk score).

## Agent

An **agent** is an individual actor with typed properties, a perception of its environment, and rules mapping perceived state to action. Agents can send topic-based messages and switch strategies under conditions. Where SD models the *aggregate*, agents model the *individual and the emergent*.

**Where it lives:** `dynafx/dynamics/agent.py` (`AgentDef`, `AgentRuleDef`, `ABMEngine`, `Message`).

## Event

An **event** is a discrete occurrence at a point in time — an arrival at a queue, a departure, a shipment. DES models time as a sequence of events rather than continuous integration. Events are what make DES the right tool for congestion and scheduling questions.

**Where it lives:** `dynafx/dynamics/des.py` (`DESClock`, `EventQueue`, `DESEngine`).

## Feedback

**Feedback** is a loop of causation where a variable influences itself through a chain of other variables. *Reinforcing* loops amplify (growth, collapse); *balancing* loops stabilize (inventory reaching a target). DynaFX detects feedback loops structurally and reports their polarity.

Feedback is the signature of systemic behavior — the reason simple intuitions fail and simulation matters.

**Where it lives:** `dynafx/dynamics/feedback.py` (`detect_feedback_loops`, `loops_for_variable`).

## Bridge

The **bridge** is the connective tissue between the knowledge graph and the simulation. It does three jobs:

1. **Pre-flight** — extract KB facts into simulation parameters (`params_from_kb`).
2. **Mid-flight** — the simulation reads and writes the KB each timestep (`KB_QUERY` / `KB_ASSERT`).
3. **Post-flight** — simulation results return as evidence triples (`evidence_from_result`).

The bridge is what makes the system *semantic*: knowledge steers dynamics, and dynamics feed knowledge back.

**Where it lives:** `dynafx/bridge.py` (`KBSimBridge`, `ClosedLoopReasoner`).

## Evidence

**Evidence** is a simulation result written back into the knowledge graph as a triple. Where the knowledge graph holds what the system *believes* from data, evidence holds what the system *observed* from its own dynamics — revenue, cost, fill rate, risk.

Evidence closes the loop: it makes simulation outcomes first-class knowledge, queryable and reason-over-able like any other fact.

**Where it lives:** `KBSimBridge.evidence_from_result`, the evidence named graph in the flagship example.

## Scenario

A **scenario** is a coherent alternative future — a set of parameter changes (disruption severity, recovery lead time, policy lever) run through the model. DynaFX runs scenario sets, grades outcomes against targets, ranks them, and filters by constraints.

Scenarios are how the system *foresees*: not one forecast, but a ranked map of possible futures.

**Where it lives:** `dynafx/dynamics/scenario.py` (`ScenarioComparison`, `ScenarioDef`, `ScenarioResult`).

## Policy

A **policy** is a rule or budget that an actor applies to steer a system toward a goal. In DynaFX policies appear in several forms: production rules over the KB ("portfolio-at-risk → requiresMitigation"), agent strategies, and LP optimizations ("allocate this budget to minimize loss").

Policies are how the system *acts* on what it foresees.

**Where it lives:** `dynafx/knowledge/production.py` (`ProductionRuleEngine`), `dynafx/dynamics/optimization.py` (`lp_minimize`, `kb_lp_minimize`), ABM strategies.

## Closed Loop

The **closed loop** is the defining pattern of DynaFX: knowledge steers the simulation, and the simulation's results become knowledge.

1. **Read** — KB facts become parameters and mid-run queries (`params_from_kb`, `KB_QUERY`).
2. **Run** — the multi-paradigm simulation advances in time.
3. **Write** — results return as evidence triples (`evidence_from_result`, `KB_ASSERT`).
4. **Reason** — production rules, scenarios, and optimization act on the updated graph.

`ClosedLoopReasoner` operationalizes this as iterative **simulate → grade → nudge → re-simulate** cycles until targets are met.

The loop is what makes the platform more than a simulator with a database attached: knowledge is not a passive input but a live participant, updated every run and read back on the next.
