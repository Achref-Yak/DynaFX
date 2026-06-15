# Formal Theory

## Subjective Logic

Subjective Logic (Jøsang, 2016) models **uncertainty** through opinion tuples:

```
ω = (b, d, u, a)
```

| Component | Range | Meaning |
|-----------|-------|---------|
| Belief `b` | [0, 1] | Confidence proposition is true |
| Disbelief `d` | [0, 1] | Confidence proposition is false |
| Uncertainty `u` | [0, 1] | Vacuousness / lack of evidence |
| Base rate `a` | [0, 1] | Prior probability |

**Invariant:** `b + d + u = 1`  
**Projected probability:** `P = b + a·u`

### Operators

| Operator | Formula | Use |
|----------|---------|-----|
| Conjunction | `AND(ω_x, ω_y)` | Joint truth of two propositions |
| Disjunction | `OR(ω_x, ω_y)` | At least one true |
| Cumulative Fusion | `ω_a ⊕ ω_b` | Merging independent evidence |
| Consensus Compromise | `ω_a ⋈ ω_b` | Averaging conflicting sources |
| Conditional Deduction | `ω_p ⊗ ω_{p→q}` | Deduction along a warrant |

All implemented in `core/math.py` — see `conjunction()`, `disjunction()`, `cumulative_fusion()`, `consensus_compromise()`, `conditional_deduction()`.

## Category Theory

Four-level category hierarchy:

| Level | Category | Example |
|-------|----------|---------|
| 1 | **Necessity** | "The system *must* handle 10k requests" |
| 2 | **Fact** | "Traffic *shows* 70% of calls from mobile" |
| 3 | **Belief** | "Team *recommends* microservices" |
| 4 | **Concept** | "Security is a process, not a state" |

**Monotonicity:** Along any edge, `source_cat ≤ target_cat`. Prevents reasoning from specific facts to abstract necessities without justification.

**Algebraic operations:** `valuation(Necessity)=1`…`valuation(Concept)=4`. Negation flips: `¬(1)=4, ¬(2)=3, ¬(3)=2, ¬(4)=1`. Conjunction = max, disjunction = min.

## Dung Argumentation Semantics

Nodes have two roles: **support** and **attack**, propagated via:

```
S(x) = Σ_{y→x, type=SUPPORTS} P(y) · strength
A(x) = Σ_{y→x, type=ATTACKS} P(y) · strength
acceptability(x) = S(x) - A(x)
```

Implemented in `dung_semantics()` in `core/math.py`. The stable extension is computed via iterative refinement.

## Belief Propagation

Propagate beliefs via fixed-point iteration:

```
B_i^{t+1} = σ( Σ_j W_ji · B_j^t + E_i )

where:
  W_ji = edge weight from j to i
  E_i = evidence for node i
  σ = sigmoid
```

Converges when `‖B^{t+1} - B^t‖_2 < ε`. Implemented as `propagate_step()` + `fixed_point_iteration()`.

## Master Equation

The final belief R(x) for each node is a weighted sum of four components:

```
R(x) = α · P(x) + β · G(x) + γ · L(x) - δ · A(x) - λ · violations(x)

where:
  P(x) = Bayesian probability (from propagate_step)
  G(x) = Graph propagation belief (from BP fixed point)
  L(x) = Logic consistency score
  A(x) = Attack strength (from argumentation)
  α, β, γ, δ, λ = coefficients
```

Implemented as `master_equation()` and `master_equation_all()` in `core/math.py`.

## Convergence Norm

The state delta norm across cycles:

```
‖Δs‖ = G_dist + A_dist + H_dist + op_change

where:
  G_dist = graph_distance(node_sets, edge_sets, belief_deltas)
  A_dist = attention_distance (same as G_dist in current impl)
  H_dist = hidden_state_distance (1.0 in current impl)
  op_change = 0 if same operator, 1 if different
```

Convergence when `‖Δs‖ < ε` or `stall_count ≥ convergence_window`.

## Graph Metrics

- **Betweenness centrality** — fraction of shortest paths passing through a node
- **Leverage score** — ratio of out-degree to in-degree
- **Feedback loop classification** — categorization by negation count (positive/negative/balanced)
- **Graph diff score** — normalized Jaccard distance between two graphs
