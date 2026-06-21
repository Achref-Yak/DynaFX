## System Overview

The cognitive-engine is a formally grounded reasoning system that extracts structured knowledge from text and propagates subjective beliefs through a directed graph using Jøsang's Subjective Logic algebra. Every layer from representation through propagation has proven formal semantics.

---

## Formal Guarantees by Layer

### Layer 1: Graph Data Model

```
Graph = (Nodes, Edges, Entities, Interpretations)
Node  = (id, type, opinion, category, ...)
Edge  = (id, source_id, target_id, type, warrant, ...)
```

**Invariants:**

| Invariant | Check | Enforced At |
|---|---|---|
| `b + d + u == 1` | `check_opinion_invariant()` | Every operator boundary |
| `u >= 0` | Runtime assertion | `count_violations()` |
| Directed acyclic (optional) | `check_cycle_free()` | `extract_max_dag()` before TNA |

**Cycle handling:** `extract_max_dag()` uses greedy MFAS (minimum feedback arc set via Kahn's algorithm) to produce a DAG by dropping back-edges. Dropped edges are logged in `state.metadata["dropped_edges"]`. This is the only approximation in the system — all downstream propagation is exact.

---

### Layer 2: Subjective Logic Algebra (Jøsang 2016)

All SL operators are pure functions in `core/math.py`. Each corresponds to a proven theorem:

| Operator | Function | Theorem | Semantics |
|---|---|---|---|
| Conjunction | `conjunction(ω_x, ω_y)` | §5.3 | `ω_x ∧ ω_y` |
| Disjunction | `disjunction(ω_x, ω_y)` | §5.3 | `ω_x ∨ ω_y` |
| Conditional deduction | `conditional_deduction(ω_p, warrant)` | §5.5 | `ω_q = ω_p ⊗ (ω_{q|p}, ω_{q|¬p})` |
| Cumulative fusion | `cumulative_fusion(ω_a, ω_b)` | §8.2 | Merge independent evidence |
| Consensus compromise | `consensus_compromise(ω_a, ω_b)` | §8.3 | Resolve conflicting views |
| Trust transfer | `trust_transfer(ω_s, ω_r)` | §9.2 | `ω_s ◁ ω_r` |
| Reverse warrant | `reverse_warrant(warrant, ...)` | §5.5 | Bayesian inversion |

**Opinion type:** `(b, d, u, a)` where `b + d + u = 1`, `a ∈ [0,1]` is the prior base rate. Projected probability: `P(ω) = b + a·u`.

---

### Layer 3: Trust Network Analysis Propagation

**Before (heuristic):**

```
B_i^{t+1} = σ( Σ_j W_ji · B_j^t + E_i )
    50 iterations, no convergence guarantee
    scalar float only (b + a·u collapsed)
```

**After (formal, this system):**

```
For node in topological_order:
    support = [conditional_deduction(parent.opinion, support_warrant)
               for parent in support_parents]
    attacks = [conditional_deduction(parent.opinion, attack_warrant)
               for parent in attack_parents]
    ω_support = cumulative_fusion(support)
    ω_attack  = cumulative_fusion(attacks)
    ω_attack_inverted = (d_attack, b_attack, u_attack, 1-a_attack)
    ω_node = cumulative_fusion(ω_support, ω_attack_inverted)
```

**Properties:**

| Property | Before | After |
|---|---|---|
| Convergence proof | None | Guaranteed (one pass, closed form) |
| Belief type | Scalar `float` | Full `[b, d, u, a]` |
| Edge semantics | Scalar weight `× belief` | SL deduction with warrants |
| Evidence fusion | None (weighted sum) | `cumulative_fusion` |
| Cycle handling | Silently iterated | `extract_max_dag()` drops back-edges |
| Soundness | None (heuristic) | Jøsang 2016 theorems 9.1–9.6 |

**Default warrants:**

- Support: `(ω_{q|p} = (0.9, 0.05, 0.05), ω_{q|¬p} = (0.1, 0.85, 0.05))`
- Attack: `(ω_{¬q|p} = (0.05, 0.9, 0.05), ω_{¬q|¬p} = (0.85, 0.1, 0.05))`

Attack semantics: `conditional_deduction` with attack warrant produces an opinion about `¬q`. Inverting this (swap b↔d, a→1−a) and fusing with support evidence via `cumulative_fusion` gives independent-evidence aggregation for q.

---

### Layer 4: Extraction Pipeline

```
Input text
  → mode gate (CAUSAL → build_world_model, else → run_argumentation)
    → FrameNet frame matching (175 verified frames in FRAMENodeType_MAP)
    → type classification (ARGUMENTATION_TYPE_RULES / WORLD_MODEL_TYPE_RULES)
    → edge assignment (ARGUMENTATION_EDGE_LOOKUP / WORLD_MODEL_EDGE_LOOKUP)
    → Graph with typed nodes + edges + opinions
```

Extraction is fully deterministic given the same input text and mode. Type rules are priority-sorted and mutually exclusive. Edge lookups are bidirectional (forward + reverse).

---

## Component Boundaries

```
State = (Graph, ABox, TBox, Metadata, Trace)
                 │
                 ▼
          ┌──────────────┐
          │  Pipeline     │
          │  (O_n ∘ ...) │
          └──────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 Extract    Propagate     Simulate
 (text→G)   (G→G+TNA)    (G'→G'+TNA)
    │            │            │
    ▼            ▼            ▼
 Compress    Merge        Attention
 (G→G')      (G×G→G'')    (G→G_filtered)
    │            │            │
    └────────────┼────────────┘
                 ▼
          ┌──────────────┐
          │   Result      │
          │  (State with  │
          │   updated G)  │
          └──────────────┘
```

Every operator:
1. Receives `State`
2. Reads/writes `state.graph`
3. Appends to `state.trace` (append-only provenance monoid)
4. Returns `State`

No operator has side effects outside the State object. The pipeline is a pure function composition `O_n ∘ ... ∘ O_1`.

---

## Key Files

| File | Role |
|---|---|
| `core/models.py` | `Graph`, `Node`, `Edge`, `Opinion` dataclasses |
| `core/math.py` | All SL operators, TNA propagation, MFAS extraction |
| `core/state.py` | `State` — the carrier data structure |
| `operators/propagate.py` | `PropagateOperator` — TNA entry point |
| `operators/simulate.py` | `SimulateOperator` — what-if TNA re-propagation |
| `operators/extract.py` | `ExtractOperator` — text-to-graph extraction |
| `extract/frame_rules.py` | `FRAMENodeType_MAP` — 175 FrameNet frames |
| `extract/types.py` | Type rules for argumentation and world models |
| `extract/edges.py` | Edge lookups by mode |
| `core/pipeline.py` | `Pipeline` — operator composition |
| `core/workflow.py` | `Workflow` — SL primitive registry |

---

## Limitations

1. **MFAS approximation** — `extract_max_dag()` uses the greedy Kahn-based heuristic, not optimal MFAS (which is NP-hard). For argumentation graphs (typically small, <100 nodes), the approximation is within practical bounds.

2. **Default warrants are static** — `_TNA_SUPPORT_WARRANT` and `_TNA_ATTACK_WARRANT` are hardcoded. Future work: learn warrants from data or derive them from the FrameNet frame hierarchy.

3. **No loopy BP** — Graphs with cycles have edges dropped rather than using loopy belief propagation. For applications requiring true cyclic reasoning (e.g., mutual support networks), a hybrid TNA+loopy approach would be needed.

4. **Opinion initialization** — Nodes with no incoming edges keep their initial opinion from the extraction phase. There is no recursive source grounding (e.g., tying leaf nodes to corpus evidence counts).

5. **Scalar objective** — `state.metadata["objective"]` is the sum of projected probabilities, a simple aggregate. No probabilistic interpretation is assigned to this value.
