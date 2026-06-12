# Formal Theory

## Subjective Logic

Subjective Logic (Jøsang, 2016) is a probabilistic logic that explicitly models **uncertainty** through opinion tuples. Every proposition carries a 4-tuple:

```
ω = (b, d, u, a)
```

where:

| Component | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| Belief | `b` | [0, 1] | Confidence that the proposition is true |
| Disbelief | `d` | [0, 1] | Confidence that the proposition is false |
| Uncertainty | `u` | [0, 1] | Vacuousness / lack of evidence |
| Base rate | `a` | [0, 1] | Prior probability in the absence of evidence |

**Invariant:** `b + d + u = 1`

**Projected probability:** `P = b + a·u`

### Operators

The engine implements the following operators from Jøsang (2016):

| Operator | Function | Use |
|----------|----------|-----|
| Conjunction | `conjunction(ω_x, ω_y)` | AND of two opinions |
| Disjunction | `disjunction(ω_x, ω_y)` | OR of two opinions |
| Cumulative Fusion | `cumulative_fusion(ω_a, ω_b)` | Merging independent evidence |
| Conditional Deduction | `conditional_deduction(ω_p, warrant)` | Deduction along an edge |

### Propagation Strategy

Opinions propagate through the graph in **topological order** using a fusion strategy that selects the operator based on source node types:

- **AXIOM → target**: conjunction (axiom constrains the target)
- **EVIDENCE → target**: cumulative fusion (independent evidence)
- **CONDITION → target**: conjunction (condition restricts applicability)
- **Default**: cumulative fusion

## Category Theory

Every node is assigned a **category** from a 4-level hierarchy:

| Level | Category | Example |
|-------|----------|---------|
| 1 | **Necessity** | "The system *must* handle 10k requests" |
| 2 | **Fact** | "Traffic analysis *shows* that 70% of calls come from mobile" |
| 3 | **Belief** | "The rewrite team *recommends* a microservices approach" |
| 4 | **Concept** | "Security is a continuous process, not a state" |

### Monotonicity Constraint

Along any edge, the source category must be ≤ the target category. This prevents reasoning from specific facts to abstract necessities without justification.

### Algebraic Operations

- **Valuation:** `v(Necessity) = 1, v(Fact) = 2, v(Belief) = 3, v(Concept) = 4`
- **Negation:** `¬(Necessity) = Concept, ¬(Fact) = Belief, ¬(Belief) = Fact, ¬(Concept) = Necessity`
- **Conjunction/Disjunction:** Takes the max/min of valuations

## Demarcation Dimensions

Each node is annotated with five dimensions that classify the nature of the proposition:

### Cognitive vs. Epistemic

| Value | Meaning |
|-------|---------|
| COGNITIVE | Pertains to mental models, reasoning, expectations (e.g., CONDITION nodes) |
| EPISTEMIC | Pertains to knowledge, evidence, justification (e.g., EVIDENCE, JUSTIFICATION) |
| NA | Not applicable (e.g., AXIOM) |

### Epistemic vs. Institutional

| Value | Meaning |
|-------|---------|
| EPISTEMIC | Depends on personal knowledge (modal + stative verb: "must be") |
| INSTITUTIONAL | Depends on rules/regulations (modal + action verb: "must support") |
| NA | No modal present |

### Affect vs. Cognition

| Value | Meaning |
|-------|---------|
| AFFECT | Contains sentiment-laden adjectives (good, poor, dangerous, etc.) |
| COGNITION | Neutral, factual framing |
| NA | No tokens available |

### Constraint vs. Enablement

| Value | Meaning |
|-------|---------|
| CONSTRAINT | Restriction, prohibition (cannot, prevent, block, negated modal) |
| ENABLEMENT | Permission, capability (can, enable, allow, support) |
| NA | No constraint/enablement signal |

### Synchronic vs. Diachronic

| Value | Meaning |
|-------|---------|
| SYNCHRONIC | Present-tense, single point in time |
| DIACHRONIC | Past or future, change over time |
| NA | No tense information |
