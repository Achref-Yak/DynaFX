# Initial Implementation — Cognitive Reasoning Graph Engine

## What Is This?

Imagine you have a messy document — say, a product requirements doc (PRD) with
a bunch of paragraphs, bullet points, and arguments. You want to turn that
mess into a clean **map of ideas** that shows:

- What claims are being made?
- What evidence supports or contradicts them?
- What conditions or assumptions qualify those claims?
- How confident should I be in each claim?

This engine does that automatically. It's like having a super-careful reader
who draws a diagram of every argument in your document, checks the diagram for
logical errors, and fixes them.

---

## The Big Idea: Reasoning Graphs

A **graph** is just a collection of circles connected by arrows. In our world:

- **Nodes** are the circles — each one is a single idea (a claim, a fact, a
  condition).
- **Edges** are the arrows — each one says how two ideas relate (supports,
  contradicts, qualifies, etc.).

Here's a tiny example:

```
  [EVIDENCE] "70% of users clicked"
       │
       │ SUPPORTS
       ▼
  [CLAIM]    "Users want this feature"
       │
       │ QUALIFIES
       ▼
  [CONDITION] "Only if latency < 200ms"
```

That's a reasoning graph. Small graph, big idea: you can now ask questions
like "Does this claim actually have evidence?" or "Is there a contradiction
I'm missing?"

---

## The Three Node Types

Every idea in a document gets classified into one of three roles:

| Type | Meaning | Think of it as |
|---|---|---|
| **CLAIM** | An assertion that needs support | "This feature will save us money" |
| **EVIDENCE** | Data or observation | "We ran an A/B test, conversion went up 5%" |
| **CONDITION** | A qualifier or constraint | "Only if we have at least 3 engineers" |

A claim without evidence is just an opinion. A piece of evidence without a
claim is just a factoid. Conditions stop you from over-promising.

---

## The Five Edge Types

Arrows have types too:

| Type | Meaning |
|---|---|
| **SUPPORTS** | Source helps prove the target |
| **CONTRADICTS** | Source argues against the target |
| **QUALIFIES** | Source sets a boundary on the target |
| **INFERS** | Target logically follows from source |
| **JUSTIFIES** | Source provides the reason for target |

---

## Opinions (Subjective Logic)

Every node and every edge has an **opinion**. An opinion is a tuple of four
numbers: `(belief, disbelief, uncertainty, base_rate)`.

- **belief**: how much I trust this statement (0 to 1)
- **disbelief**: how much I distrust it (0 to 1)
- **uncertainty**: how much I don't know (0 to 1)
- **base_rate**: the prior probability, before seeing any evidence (0 to 1)

The three numbers belief + disbelief + uncertainty always add up to 1.
Think of them as splitting a pie:

- If I'm very sure: `(0.9, 0.05, 0.05)` — 90% belief, 5% doubt, 5% unknown.
- If I know nothing: `(0.0, 0.0, 1.0)` — total ignorance.

The **base_rate** is your prior. If base_rate is 0.5, you're saying "before
I saw any evidence, I thought this was equally likely to be true or false."

### Why Subjective Logic?

Regular logic (true/false) can't handle uncertainty. Subjective Logic is
like regular logic but for opinions — it has operators to combine opinions:

- **conjunction**: "A AND B" — how much I believe both A and B
- **disjunction**: "A OR B" — how much I believe at least one of A or B
- **cumulative_fusion**: combining two independent sources about the same
  thing (reduces uncertainty)
- **conditional_deduction**: if A implies B, and I know my opinion on A,
  what should I think about B?

These are all pure math functions. They take opinion tuples in, give opinion
tuples out.

---

## The Validators (The Math Police)

Before any human or LLM reviews the graph, we run two automatic math checks:

### V1: Product Logic (Category Hierarchy)

Every node has a **category** from 1 to 4:

| # | Name | Example |
|---|---|---|
| 1 | Necessity | "The system must handle 1000 req/s" |
| 2 | Fact | "The current system handles 500 req/s" |
| 3 | Belief | "I think we should use PostgreSQL" |
| 4 | Concept | "Eventual consistency" |

Rule: **A higher category cannot imply a lower one.** You can't prove a
fact (2) from a concept (4). You *can* support a concept with a fact.
The reasoning arrow flows from concrete → abstract, not the reverse.

### V2: Level Mapping (Cycle Detection)

Graphs shouldn't have cycles. If A supports B and B supports A, you have a
circular argument ("It's true because it's true"). We use
[Kahn's algorithm](https://en.wikipedia.org/wiki/Topological_sorting) to
check — if the graph can't be topologically sorted, there's a cycle, and
we flag it.

---

## The Pipeline (How It All Connects)

The engine runs in a loop of up to 3 rounds:

```
  ┌─────────────────────────────────────┐
  │           Round 1                   │
  │                                     │
  │  [Source Text]                      │
  │       │                             │
  │       ▼                             │
  │  [Creator Agent] (LLM)             │
  │  Reads text, outputs JSON graph     │
  │       │                             │
  │       ▼                             │
  │  [Validators V1 + V2] (math)       │
  │  Checks category hierarchy & cycles │
  │       │                             │
  │       ▼                             │
  │  [Reviewer Agent] (LLM)            │
  │  Reads graph + validator results,   │
  │  checks semantic correctness        │
  │       │                             │
  │       ▼                             │
  │  [Gate: accept?]                    │
  │  Yes → done, return graph           │
  │  No  → send feedback back to        │
  │         Creator, try again (round 2)│
  └─────────────────────────────────────┘
```

### The Two Agents

**Creator Agent** (in `extraction.py`):
- The "writer" — reads the source text and builds a first draft of the graph.
- Uses an LLM (Llama 3.3 70B via Groq).
- Prompt tells it to look for spans of text, assign types, categories, and
  relationships.

**Reviewer Agent** (in `reviewers.py`):
- The "editor" — checks the graph for semantic issues.
- Does NOT re-run math checks (those were already done).
- Focuses on: "Do these claims match the source text?" "Is this edge type
  appropriate?" "Is anything missing?"
- Returns accept/reject + feedback.

### Why Two Agents?

Separation of concerns. Creation is generative — it has to invent structure
from text. Review is analytical — it identifies problems. If the same agent
did both, it might overlook its own mistakes. This is the same reason you
have a separate editor for your writing.

---

## The 3-Round Loop (CAF-Gen)

CAF-Gen = Critique-Apply-Feedback Generation.

1. Creator makes a graph.
2. Math checks run (V1 + V2).
3. Reviewer checks semantics, gets pre-computed math results as context.
4. If everything passes → return graph.
5. If anything fails → feedback goes back to Creator → try again.
6. After 3 rounds, if still failing → raise an error.

This is **fail-closed**: the system prefers to reject a bad graph rather
than return a garbage one. You can always tweak the prompt or retry with
a better model.

---

## Project Structure

```
src/cognitive_engine/
    __init__.py       — Exports for easy importing
    models.py         — The data types (Node, Edge, Graph, Opinion, etc.)
    validators.py     — V1 (category hierarchy) + V2 (cycle detection)
    sl_operators.py   — Subjective Logic math functions
    extraction.py     — Creator Agent (LLM) — reads text, outputs graph
    reviewers.py      — Reviewer Agent (LLM) — checks graph semantics
    orchestrator.py   — The 3-round CAF-Gen loop
    cli.py            — Command-line interface

tests/
    test_validators.py    — 12 tests for V1 and V2
    test_sl_operators.py  — 10 tests for SL operators
```

---

## How To Use It

```bash
# Make sure you have a Groq API key
export GROQ_API_KEY="gsk_..."

# Run the pipeline on a text file
PYTHONPATH=src python3.12 -m cognitive_engine.cli my_prd.txt

# Or save the output to a file
PYTHONPATH=src python3.12 -m cognitive_engine.cli my_prd.txt --output result.json

# Use a different model
PYTHONPATH=src python3.12 -m cognitive_engine.cli my_prd.txt --model llama-3.3-70b-versatile
```

You get back a JSON file with all nodes, edges, opinions, and metadata.

---

## What We DIDN'T Build (Yet)

- **7×7 Argumentation Ontology**: The white paper describes a richer set of
  7 node types and 7 edge types. For the PoC, we use 3 node types and 5 edge
  types — the minimum needed to prove the pipeline works.
- **Gold-standard comparison**: No F1/precision/recall metrics yet. We'll add
  those when you hand-annotate a few documents.
- **Web interface**: It's a CLI tool for now.
- **Fine-tuning**: The LLM is used as-is (no custom training).

---

## Glossary (For When Your Brain Hurts)

| Term | Plain English |
|---|---|
| **Node** | A single idea in the graph (claim, evidence, or condition) |
| **Edge** | A relationship between two ideas |
| **Opinion** | A 4-number score: how much I believe, disbelieve, am uncertain, and my prior |
| **Subjective Logic** | Math for combining opinions (like regular logic but handles uncertainty) |
| **V1 (Product Logic)** | Rule: you can't prove concrete facts from abstract concepts |
| **V2 (Level Mapping)** | Rule: no circular arguments |
| **Creator Agent** | The LLM that writes the first draft of the graph |
| **Reviewer Agent** | The LLM that checks the graph for mistakes |
| **CAF-Gen** | The loop: Create → Validate → Review → Accept/Retry |
| **Fail-closed** | If unsure, reject rather than return garbage |
| **Ontology** | The set of allowed node/edge types |
| **Topological sort** | A way to check if a graph has cycles (if you can't order nodes from start to end, there's a loop) |
