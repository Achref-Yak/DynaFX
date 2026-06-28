"""Formal mathematical primitives for DynaFX.

Central collection of all formulas used across the system.
Organized by domain: Subjective Logic, Category Theory, Graph
Propagation, Argumentation, Cognitive Architecture, Bayesian,
GNN, Neuro-Symbolic, Convergence, Embedding, Memory, Systems,
and math utilities.

Every function is pure (no side effects, no I/O).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# MATH UTILITIES
# ═══════════════════════════════════════════════════════════════

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid: σ(x) = 1/(1+e^{-x})."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def sigmoid_array(values: list[float]) -> list[float]:
    """Vectorized sigmoid over a list."""
    return [sigmoid(v) for v in values]


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    """Temperature-scaled softmax: P(i) = e^{v_i/τ} / Σ_k e^{v_k/τ}."""
    if not values:
        return []
    max_v = max(values)
    exp_vals = [math.exp((v - max_v) / temperature) for v in values]
    total = sum(exp_vals)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exp_vals]


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def normalize_sum(b: float, d: float, u: float) -> tuple[float, float, float]:
    """Normalize b+d+u to 1. If total is 0, set u=1."""
    total = b + d + u
    if abs(total - 1.0) < 1e-9:
        return (b, d, u)
    if total == 0:
        return (0.0, 0.0, 1.0)
    return (b / total, d / total, u / total)


def l2_norm(delta: dict[UUID, float]) -> float:
    """L2 norm: sqrt((1/N) * Σ_i Δ_i²)."""
    if not delta:
        return 0.0
    n = len(delta)
    return math.sqrt(sum(v * v for v in delta.values()) / n)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(a,b) = dot(a,b) / (||a|| · ||b||)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cosine_distance(a: list[float], b: list[float]) -> float:
    return 1.0 - cosine_similarity(a, b)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """|A ∩ B| / |A ∪ B|."""
    if not set_a and not set_b:
        return 1.0
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    return len(set_a & set_b) / union


def jaccard_distance(set_a: set, set_b: set) -> float:
    return 1.0 - jaccard_similarity(set_a, set_b)


def count_proximity(n_q: int, n_p: int) -> float:
    """1 - |n_q - n_p| / max(n_q, n_p, 1)."""
    denom = max(n_q, n_p, 1)
    return 1.0 - abs(n_q - n_p) / denom


# ═══════════════════════════════════════════════════════════════
# SUBJECTIVE LOGIC
# ═══════════════════════════════════════════════════════════════

def projected_probability(b: float, u: float, a: float) -> float:
    """P(ω) = b + a·u."""
    return b + a * u


def dirichlet_strength(b: float, d: float, u: float) -> float:
    """W = (b+d)/u (infinite if u=0)."""
    if u == 0:
        return float('inf')
    return (b + d) / u


def opinion_from_counts(
    positive: int, negative: int,
    pseudocount: float = 2.0,
    base_rate: float = 0.5,
) -> tuple[float, float, float, float]:
    """Convert Dirichlet evidence counts to SL opinion."""
    total = positive + negative + pseudocount
    b = positive / total
    d = negative / total
    u = pseudocount / total
    return (b, d, u, base_rate)


def mean_opinion(
    opinions: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Arithmetic mean of opinion tuples, then re-normalized."""
    if not opinions:
        return (0.0, 0.0, 1.0, 0.5)
    n = len(opinions)
    b = sum(o[0] for o in opinions) / n
    d = sum(o[1] for o in opinions) / n
    u = sum(o[2] for o in opinions) / n
    a = sum(o[3] for o in opinions) / n
    b, d, u = normalize_sum(b, d, u)
    return (b, d, u, a)


def trust_weight(b: float, u: float, alpha: float = 0.5) -> float:
    """w = b + α·u."""
    return b + alpha * u


def compute_trust_weights(
    opinions: list[tuple[float, float, float, float]],
    alpha: float = 0.5,
) -> list[float]:
    weights = [trust_weight(o[0], o[2], alpha) for o in opinions]
    total = sum(weights)
    if total == 0:
        return [1.0 / len(opinions)] * len(opinions)
    return [w / total for w in weights]


def conjunction(
    omega_x: tuple[float, float, float, float],
    omega_y: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Subjective Logic conjunction ω_x ∧ ω_y (Jøsang 2016 §5.3)."""
    b_x, d_x, u_x, a_x = omega_x
    b_y, d_y, u_y, a_y = omega_y
    denom = 1.0 - a_x * a_y
    if denom == 0:
        return (0.0, 0.0, 1.0, a_x * a_y)
    b = b_x * b_y + (a_x * b_y * (1 - a_y) * u_x + a_y * b_x * (1 - a_x) * u_y) / denom
    d = d_x + d_y - d_x * d_y
    u = u_x * u_y + (b_x * (1 - a_y) * u_y + b_y * (1 - a_x) * u_x) / denom
    return (b, d, u, a_x * a_y)


def disjunction(
    omega_x: tuple[float, float, float, float],
    omega_y: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Subjective Logic disjunction ω_x ∨ ω_y."""
    b_x, d_x, u_x, a_x = omega_x
    b_y, d_y, u_y, a_y = omega_y
    denom = 1.0 - a_x * a_y
    if denom == 0:
        return (1.0, 0.0, 0.0, a_x * a_y)
    b = b_x + b_y - b_x * b_y
    d = d_x * d_y + (a_x * d_y * (1 - a_y) * u_x + a_y * d_x * (1 - a_x) * u_y) / denom
    u = u_x * u_y + (d_x * (1 - a_y) * u_y + d_y * (1 - a_x) * u_x) / denom
    return (b, d, u, a_x * a_y)


def conditional_deduction(
    omega_p: tuple[float, float, float, float],
    warrant: tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Conditional deduction ω_c|p,ω_c|¬p applied to ω_p.
    
    Returns ω_c = ω_p ⊗ (ω_c|p, ω_c|¬p).
    """
    b_p, d_p, u_p, _ = omega_p
    omega_c_given_p, omega_c_given_not_p = warrant
    b_c_given_p, d_c_given_p, u_c_given_p, a_c = omega_c_given_p
    b_c_given_not_p, d_c_given_not_p, u_c_given_not_p, _ = omega_c_given_not_p
    b = b_p * b_c_given_p + d_p * b_c_given_not_p + u_p * a_c
    d = b_p * d_c_given_p + d_p * d_c_given_not_p + u_p * (1 - a_c)
    u = u_p + b_p * u_c_given_p + d_p * u_c_given_not_p
    return (b, d, u, a_c)


def cumulative_fusion(
    omega_a: tuple[float, float, float, float],
    omega_b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Cumulative fusion (independent evidence, Jøsang 2016 §8.2)."""
    b_a, d_a, u_a, a_a = omega_a
    b_b, d_b, u_b, a_b = omega_b
    kappa = u_a + u_b - u_a * u_b
    a = (a_a + a_b) / 2.0
    if kappa == 0:
        return ((b_a + b_b) / 2.0, (d_a + d_b) / 2.0, 0.0, a)
    b = (b_a * u_b + b_b * u_a) / kappa
    d = (d_a * u_b + d_b * u_a) / kappa
    u = (u_a * u_b) / kappa
    return (b, d, u, a)


def consensus_compromise(
    omega_a: tuple[float, float, float, float],
    omega_b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Consensus & compromise fusion (Jøsang 2016 §8.3)."""
    b_a, d_a, u_a, a_a = omega_a
    b_b, d_b, u_b, a_b = omega_b
    conflict = b_a * d_b + d_a * b_b
    denom = 1.0 - conflict
    if denom <= 1e-12:
        return (0.0, 0.0, 1.0, (a_a + a_b) / 2.0)
    b = (b_a * b_b + b_a * u_b + b_b * u_a) / denom
    d = (d_a * d_b + d_a * u_b + d_b * u_a) / denom
    u = (u_a * u_b) / denom
    a = (a_a + a_b) / 2.0
    return (b, d, u, a)


def weighted_belief_fusion(
    omega_a: tuple[float, float, float, float],
    omega_b: tuple[float, float, float, float],
    w_a: float, w_b: float,
) -> tuple[float, float, float, float]:
    """Weighted belief fusion (Jøsang 2016 §8.4), w_a + w_b = 1."""
    b = w_a * omega_a[0] + w_b * omega_b[0]
    d = w_a * omega_a[1] + w_b * omega_b[1]
    u = w_a * omega_a[2] + w_b * omega_b[2]
    a = w_a * omega_a[3] + w_b * omega_b[3]
    return (b, d, u, a)


def trust_transfer(
    omega_source: tuple[float, float, float, float],
    omega_recommender: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Trust transfer (Jøsang 2016 §9.2)."""
    b_s, _, u_s, _ = omega_source
    b_r, d_r, u_r, a_r = omega_recommender
    b = b_s * b_r
    d = b_s * d_r
    u = 1.0 - b_s + b_s * u_r
    return (b, d, u, a_r)


def opinion_conflict(
    omega_a: tuple[float, float, float, float],
    omega_b: tuple[float, float, float, float],
    threshold: float = 0.6,
) -> bool:
    b_a, d_a, _, _ = omega_a
    b_b, d_b, _, _ = omega_b
    return (b_a > threshold and d_b > threshold) or (b_b > threshold and d_a > threshold)


def reverse_warrant(
    forward_warrant: tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
    base_rate_source: float,
    base_rate_target: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Bayesian inversion of a conditional warrant.
    
    Given ω_t|s and ω_t|¬s, compute ω_s|t and ω_s|¬t.
    """
    omega_t_s, omega_t_ns = forward_warrant
    b_t_s, d_t_s, u_t_s, _ = omega_t_s
    b_t_ns, d_t_ns, u_t_ns, _ = omega_t_ns
    p_t_s = projected_probability(b_t_s, u_t_s, base_rate_target)
    p_t_ns = projected_probability(b_t_ns, u_t_ns, base_rate_target)
    p_s = base_rate_source
    p_t = p_s * p_t_s + (1 - p_s) * p_t_ns
    if p_t == 0:
        return ((0.0, 0.0, 1.0, base_rate_source),)*2
    p_s_t = p_t_s * p_s / p_t
    p_s_nt = p_t_ns * p_s / (1 - p_t) if p_t < 1 else 0.0
    b_s_t = p_s_t * (1 - u_t_s)
    d_s_t = (1 - p_s_t) * (1 - u_t_s)
    u_s_t = u_t_s
    b_s_nt = p_s_nt * (1 - u_t_ns)
    d_s_nt = (1 - p_s_nt) * (1 - u_t_ns)
    u_s_nt = u_t_ns
    return (
        (b_s_t, d_s_t, u_s_t, base_rate_source),
        (b_s_nt, d_s_nt, u_s_nt, base_rate_source),
    )


def subjective_abduction(
    omega_effect: tuple[float, float, float, float],
    warrant: tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Subjective abduction: cause from effect + warrant."""
    b_e, d_e, u_e, _ = omega_effect
    omega_cause_given_effect, omega_cause_given_not_effect = warrant
    b_c_e, d_c_e, u_c_e, a_c = omega_cause_given_effect
    b_c_ne, d_c_ne, u_c_ne, _ = omega_cause_given_not_effect
    b = b_e * b_c_e + d_e * b_c_ne + u_e * a_c
    d = b_e * d_c_e + d_e * d_c_ne + u_e * (1 - a_c)
    u = b_e * u_c_e + d_e * u_c_ne + u_e
    return (b, d, u, a_c)


def analogy_warrant_transform(
    opinion: tuple[float, float, float, float],
    delta: float = 0.2,
) -> tuple[float, float, float, float]:
    """Reduce belief by delta, shifting to uncertainty."""
    b, d, u, a = opinion
    reduction = b * delta
    return (b - reduction, d, u + reduction, a)


# ═══════════════════════════════════════════════════════════════
# CATEGORY THEORY (4-level hierarchy)
# ═══════════════════════════════════════════════════════════════

NECESSITY = 1
FACT = 2
BELIEF = 3
CONCEPT = 4

CATEGORY_LEVELS: dict[str, int] = {
    "NECESSITY": 1, "AXIOM": 1,
    "FACT": 2, "EVIDENCE": 2, "OBSERVATION": 2, "DOCUMENT": 2,
    "BELIEF": 3, "CLAIM": 3, "CONDITION": 3, "JUSTIFICATION": 3,
    "COUNTERCLAIM": 3, "HYPOTHESIS": 3, "RULE": 3, "DECISION": 3, "ACTION": 3,
    "CONCEPT": 4, "ENTITY": 4, "EVENT": 4, "FALLACY": 4,
}


def category_level(node_type: str) -> int:
    return CATEGORY_LEVELS.get(node_type.upper(), 2)


def category_conjunction(a: int, b: int) -> int:
    return max(a, b)


def category_disjunction(a: int, b: int) -> int:
    return min(a, b)


def category_negation(cat: int) -> int:
    return cat


def implication_valid(src_cat: int, tgt_cat: int) -> bool:
    return src_cat <= tgt_cat


def equivalence_valid(a: int, b: int) -> bool:
    return a == b


def category_valuate(category: int, truth: bool) -> int:
    return category if truth else 0


# ═══════════════════════════════════════════════════════════════
# SYMBOLIC LOGIC (Level 0)
# ═══════════════════════════════════════════════════════════════

def modus_ponens_strength(
    rule_strength: float,
    antecedent_strengths: list[float],
) -> float:
    """Strength = min(rule.strength, min(antecedent strengths))."""
    return min(rule_strength, *antecedent_strengths)


def inference_closure(
    facts: dict[str, bool],
    rules: list[tuple[list[str], str, float]],
) -> tuple[dict[str, bool], list[str]]:
    """Apply Modus Ponens until fixpoint. Returns (all_facts, inferred_order)."""
    inferred_order: list[str] = []
    changed = True
    while changed:
        changed = False
        for antecedents, consequent, strength in rules:
            if all(facts.get(a) for a in antecedents) and consequent not in facts:
                facts[consequent] = True
                inferred_order.append(consequent)
                changed = True
    return facts, inferred_order


# ═══════════════════════════════════════════════════════════════
# COGNITIVE ARCHITECTURE (Level 1 — ACT-R / SOAR)
# ═══════════════════════════════════════════════════════════════

def base_level_activation(
    base: float,
    access_count: int,
    time_since_creation: float,
    decay_rate: float = 0.05,
) -> float:
    """B = B_0 + log(access_count / t^decay_rate)."""
    if time_since_creation > 0 and access_count > 0:
        return base + math.log(access_count / (time_since_creation ** decay_rate))
    return base


def activation_spreading(
    base_activation: float,
    associations: list[tuple[float, float]],
) -> float:
    """A_i = B_i + Σ_j W_j · S_ij."""
    total = base_activation + sum(w * s for w, s in associations)
    return total


def belief_from_activation(activation: float) -> float:
    """Convert activation to [0,1] via sigmoid."""
    return sigmoid(activation)


def softmax_retrieval(
    activations: dict[str, float],
    temperature: float = 1.0,
) -> dict[str, float]:
    """P(retrieve i) = e^{A_i/τ} / Σ_k e^{A_k/τ}."""
    names = list(activations.keys())
    values = [activations[n] for n in names]
    probs = softmax(values, temperature)
    return dict(zip(names, probs))


# ═══════════════════════════════════════════════════════════════
# BAYESIAN / PROBABILISTIC (Level 2)
# ═══════════════════════════════════════════════════════════════

def bayes_rule(
    likelihood: float,
    prior: float,
    evidence: float,
) -> float:
    """P(H|E) = P(E|H) · P(H) / P(E)."""
    if evidence == 0:
        return prior
    return likelihood * prior / evidence


def joint_probability(
    config: dict[str, Any],
    variables: dict[str, dict],
    topo_order: list[str],
) -> float:
    """P(config) = Π_i P(X_i | Parents(X_i)).
    
    variables: name → {parents: list[str], cpt: dict}
    """
    prob = 1.0
    for name in topo_order:
        var = variables.get(name)
        if var is None:
            continue
        value = config.get(name)
        parent_values = tuple(config.get(p) for p in var.get("parents", []))
        cpt = var.get("cpt", {})
        key = parent_values + (value,)
        if key in cpt:
            prob *= cpt[key]
        elif (value,) in cpt:
            prob *= cpt[(value,)]
        else:
            prob *= 1.0 / max(len(var.get("domain", [True, False])), 1)
    return prob


def expectation(probs: dict[Any, float]) -> float:
    """E[X] = Σ_x x · P(x)."""
    return sum(x * p for x, p in probs.items())


# ═══════════════════════════════════════════════════════════════
# GRAPH PROPAGATION (Level 4)
# ═══════════════════════════════════════════════════════════════

EDGE_WEIGHTS: dict[str, float] = {
    "INFERS": 0.9, "SUPPORTS": 0.85, "DIRECT": 0.95, "JUSTIFIES": 0.8,
    "CIRCUMSTANTIAL": 0.6, "QUALIFIES": 0.5, "REBUTS": 0.6, "HEARSAY": 0.4,
    "CONTRADICTS": 0.85, "ATTACKS": 0.8, "CAUSES": 0.8, "SUPPORT": 0.75,
    "ENABLES": 0.7, "DEPENDS": 0.6, "TEMPORAL": 0.5, "SIMILAR": 0.5,
    "EVIDENCE": 0.8, "PART_OF": 0.6, "CITES": 0.6, "FLOWS_TO": 0.6,
}

NODE_PRIORS: dict[str, float] = {
    "AXIOM": 0.9, "EVIDENCE": 0.8, "JUSTIFICATION": 0.7, "CONDITION": 0.5,
    "CLAIM": 0.6, "COUNTERCLAIM": 0.4, "FALLACY": 0.2, "ENTITY": 0.5,
    "EVENT": 0.5, "CONCEPT": 0.5, "RULE": 0.6, "HYPOTHESIS": 0.4,
    "OBSERVATION": 0.7, "DECISION": 0.5, "ACTION": 0.5, "DOCUMENT": 0.6,
}


def propagate_step(
    beliefs: dict[UUID, float],
    adjacency: dict[UUID, list[tuple[UUID, float]]],
    evidence: dict[UUID, float],
) -> dict[UUID, float]:
    """B_i^{t+1} = σ( Σ_j W_ji · B_j^t + E_i )."""
    new_beliefs = {}
    for nid in beliefs:
        weighted_sum = 0.0
        for source_id, weight in adjacency.get(nid, []):
            weighted_sum += weight * beliefs.get(source_id, 0.5)
        external = evidence.get(nid, 0.5)
        new_beliefs[nid] = sigmoid(weighted_sum + external)
    return new_beliefs


def build_adjacency(
    node_ids: set[UUID],
    edges: list,
    edge_weight_fn=None,
) -> dict[UUID, list[tuple[UUID, float]]]:
    """Build adjacency: target → [(source, weight)]."""
    if edge_weight_fn is None:
        def edge_weight_fn(e):
            return EDGE_WEIGHTS.get(e.type.name if hasattr(e, 'type') else str(e.type), 0.5)
    adjacency = {nid: [] for nid in node_ids}
    for edge in edges:
        source = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        target = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        if source in node_ids and target in node_ids:
            adjacency[target].append((source, edge_weight_fn(edge)))
    return adjacency


def initialize_beliefs(
    node_ids: set[UUID],
    get_node_type_fn,
    get_opinion_fn=None,
) -> dict[UUID, float]:
    """Initialize beliefs from node type priors or opinions."""
    beliefs = {}
    for nid in node_ids:
        if get_opinion_fn:
            opinion = get_opinion_fn(nid)
            if opinion:
                b, d, u, a = opinion
                beliefs[nid] = projected_probability(b, u, a)
                continue
        node_type = get_node_type_fn(nid)
        beliefs[nid] = NODE_PRIORS.get(node_type.upper() if isinstance(node_type, str) else node_type.name, 0.5)
    return beliefs


def convergence_l2(
    beliefs_a: dict[UUID, float],
    beliefs_b: dict[UUID, float],
) -> float:
    """Δ = sqrt( Σ_i (B_i^{t+1} - B_i^t)^2 )."""
    shared = set(beliefs_a) & set(beliefs_b)
    if not shared:
        return 1.0
    return math.sqrt(sum((beliefs_b[nid] - beliefs_a[nid]) ** 2 for nid in shared))


def similarity_diffusion(
    embeddings: dict[UUID, list[float]],
) -> dict[UUID, float]:
    """Average cosine similarity per node (excluding self)."""
    node_ids = list(embeddings.keys())
    n = len(node_ids)
    if n == 0:
        return {}
    result = {}
    for i, nid in enumerate(node_ids):
        others = [
            cosine_similarity(embeddings[nid], embeddings[jid])
            for j, jid in enumerate(node_ids) if j != i
        ]
        result[nid] = float(sum(others)) / len(others) if others else 0.0
    return result


def extract_max_dag(
    node_ids: set[UUID],
    edges: list,
    edge_weight_fn=None,
) -> tuple[list, list, list[UUID]]:
    """Extract a maximum DAG subgraph by greedily dropping back-edges.

    Uses Kahn's algorithm to compute topological order; edges that go
    against the order (later→earlier) are dropped to break cycles.

    Args:
        node_ids: Set of node UUIDs.
        edges: List of Edge objects (must have source_id, target_id).
        edge_weight_fn: Optional weight function for sorting edge retention
            priority. Higher weight = more likely kept. Defaults to EDGE_WEIGHTS.

    Returns:
        (dag_edges, dropped_edges, topological_order)
    """
    if edge_weight_fn is None:
        edge_weight_fn = lambda e: EDGE_WEIGHTS.get(
            e.type.name if hasattr(e, 'type') else str(e.type), 0.5)

    in_degree: dict[UUID, int] = {nid: 0 for nid in node_ids}
    children: dict[UUID, list[UUID]] = {nid: [] for nid in node_ids}
    edge_map: dict[tuple[UUID, UUID], object] = {}

    for edge in edges:
        src = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        tgt = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        if src in node_ids and tgt in node_ids:
            in_degree[tgt] += 1
            children[src].append(tgt)
            edge_map[(src, tgt)] = edge

    queue = [nid for nid, d in in_degree.items() if d == 0]
    order: list[UUID] = []

    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for child in children.get(nid, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    remaining = [nid for nid in node_ids if nid not in order]
    order.extend(remaining)

    position = {nid: i for i, nid in enumerate(order)}
    dag_edges = []
    dropped = []

    for (src, tgt), edge in edge_map.items():
        if position[src] < position[tgt]:
            dag_edges.append(edge)
        else:
            dropped.append(edge)

    return dag_edges, dropped, order


def topological_sort(
    node_ids: set[UUID],
    edges: list,
) -> list[UUID]:
    """Standard topological sort. Nodes in cycles are appended at the end."""
    in_degree: dict[UUID, int] = {nid: 0 for nid in node_ids}
    children: dict[UUID, list[UUID]] = {nid: [] for nid in node_ids}

    for edge in edges:
        src = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        tgt = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        if src in node_ids and tgt in node_ids:
            in_degree[tgt] += 1
            children[src].append(tgt)

    queue = [nid for nid, d in in_degree.items() if d == 0]
    order: list[UUID] = []

    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for child in children.get(nid, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    order.extend(nid for nid in node_ids if nid not in order)
    return order


# Default warrants for TNA
_TNA_SUPPORT_WARRANT: tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
] = (
    (0.9, 0.05, 0.05, 0.5),
    (0.1, 0.85, 0.05, 0.5),
)

_TNA_ATTACK_WARRANT: tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
] = (
    (0.05, 0.9, 0.05, 0.5),
    (0.85, 0.1, 0.05, 0.5),
)


def _extract_warrant_opinion(
    w: object,
) -> tuple[float, float, float, float]:
    """Normalize a warrant component to (b, d, u, a) tuple."""
    if isinstance(w, tuple):
        return w
    return (w.belief, w.disbelief, w.uncertainty, w.prior)


def tna_propagate(
    node_ids: set[UUID],
    edges: list,
    get_opinion_fn,
    get_edge_warrant_fn=None,
) -> dict[UUID, tuple[float, float, float, float]]:
    """Trust Network Analysis propagation: one topological pass.

    Replaces the heuristic 50-iteration Master Equation with a single
    forward pass using Jøsang's Subjective Logic operators.

    For each node in topological order:
      1. Classify incoming edges as support or attack via SUPPORT_EDGES
         and ATTACK_EDGES constants.
      2. Apply conditional_deduction per edge with edge-specific or
         default warrant.
      3. cumulative_fusion across all support contributions.
      4. cumulative_fusion across attack contributions, invert
         (swap b↔d, a→1-a), then cumulative_fusion with support result.

    Args:
        node_ids: Set of node UUIDs (should be a DAG — call extract_max_dag
            first if cycles are present).
        edges: List of Edge objects (must have source_id, target_id, type).
        get_opinion_fn: Callable(nid) → Opinion, tuple, or None.
        get_edge_warrant_fn: Optional callable(edge) → warrant tuple
            ((b,d,u,a), (b,d,u,a)). If None, falls back to edge.warrant,
            then to _TNA_SUPPORT_WARRANT or _TNA_ATTACK_WARRANT by edge type.

    Returns:
        {nid: (b, d, u, a)} for all nodes.
    """
    parents: dict[UUID, list] = {nid: [] for nid in node_ids}
    children: dict[UUID, list[UUID]] = {nid: [] for nid in node_ids}
    in_degree: dict[UUID, int] = {nid: 0 for nid in node_ids}

    for edge in edges:
        src = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        tgt = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        if src in node_ids and tgt in node_ids:
            parents[tgt].append(edge)
            children[src].append(tgt)
            in_degree[tgt] += 1

    order = topological_sort(node_ids, edges)

    opinions: dict[UUID, tuple[float, float, float, float]] = {}
    for nid in node_ids:
        src = get_opinion_fn(nid)
        if src is not None:
            if isinstance(src, tuple):
                opinions[nid] = src
            else:
                opinions[nid] = (src.belief, src.disbelief, src.uncertainty, src.prior)
        else:
            opinions[nid] = (0.0, 0.0, 1.0, 0.5)

    for nid in order:
        incoming = parents.get(nid, [])
        if not incoming:
            continue

        support_group: list[tuple[float, float, float, float]] = []
        attack_group: list[tuple[float, float, float, float]] = []

        for edge in incoming:
            src_id = edge.source_id if hasattr(edge, 'source_id') else edge[0]
            if src_id not in opinions:
                continue
            parent_op = opinions[src_id]

            type_name = edge.type.name if hasattr(edge.type, 'name') else str(edge.type)

            if get_edge_warrant_fn:
                warrant = get_edge_warrant_fn(edge)
            elif hasattr(edge, 'warrant') and edge.warrant is not None:
                w = edge.warrant
                warrant = (
                    _extract_warrant_opinion(w[0]),
                    _extract_warrant_opinion(w[1]),
                )
            elif type_name in ATTACK_EDGES:
                warrant = _TNA_ATTACK_WARRANT
            else:
                warrant = _TNA_SUPPORT_WARRANT

            deduced = conditional_deduction(parent_op, warrant)

            if type_name in ATTACK_EDGES:
                attack_group.append(deduced)
            else:
                support_group.append(deduced)

        fused: Optional[tuple[float, float, float, float]] = None

        if support_group:
            fused = support_group[0]
            for op in support_group[1:]:
                fused = cumulative_fusion(fused, op)

        if attack_group:
            fused_attack = attack_group[0]
            for op in attack_group[1:]:
                fused_attack = cumulative_fusion(fused_attack, op)
            b_a, d_a, u_a, a_a = fused_attack
            inverted = (d_a, b_a, u_a, 1.0 - a_a)

            if fused is not None:
                fused = cumulative_fusion(fused, inverted)
            else:
                fused = inverted

        if fused is not None:
            opinions[nid] = fused

    return opinions


# ═══════════════════════════════════════════════════════════════
# ARGUMENTATION (Level 5 — Dung-style)
# ═══════════════════════════════════════════════════════════════

SUPPORT_EDGES = {"SUPPORTS", "INFERS", "JUSTIFIES", "DIRECT"}
ATTACK_EDGES = {"ATTACKS", "CONTRADICTS", "REBUTS"}
QUALIFY_EDGES = {"QUALIFIES", "CIRCUMSTANTIAL", "HEARSAY"}


def compute_support(
    node_id: UUID,
    predecessors: list[tuple[UUID, str, float]],
    beliefs: dict[UUID, float],
) -> float:
    """S(v) = Σ_{u∈Supp(v)} w_{uv} · B(u)."""
    total = 0.0
    for src, role, weight in predecessors:
        if role == "support":
            total += weight * beliefs.get(src, 0.5)
    return total


def compute_attack(
    node_id: UUID,
    predecessors: list[tuple[UUID, str, float]],
    beliefs: dict[UUID, float],
) -> float:
    """A(v) = Σ_{u∈Att(v)} w_{uv} · B(u)."""
    total = 0.0
    for src, role, weight in predecessors:
        if role == "attack":
            total += weight * beliefs.get(src, 0.5)
    return total


def argument_acceptability(support: float, attack: float) -> float:
    """Acc(v) = S(v) - A(v)."""
    return support - attack


def argument_strength(acceptability: float) -> float:
    """P(v) = σ(Acc(v))."""
    return sigmoid(acceptability)


def dung_semantics(
    beliefs: dict[UUID, float],
    attack_graph: dict[UUID, list[UUID]],
    weights: Optional[dict[UUID, float]] = None,
    max_iterations: int = 20,
) -> set[UUID]:
    """Find Dung's preferred extension via iteration to fixpoint.

    A node is accepted if all its attackers are rejected.
    Returns set of accepted node IDs.

    When weights are provided, provenance-weighted acceptance:
    - Initial acceptance uses belief * weight > 0.5
    - An attacker is only "effective" if its weight exceeds the target's weight
      (a high-provenance node is harder to defeat)
    """
    if weights is None:
        weights = {}

    def _weight(nid: UUID) -> float:
        return weights.get(nid, 1.0)

    def _initial_accept(nid: UUID) -> bool:
        b = beliefs.get(nid, 0.5)
        w = _weight(nid)
        return b * w > 0.5

    accepted: set[UUID] = set(nid for nid in beliefs if _initial_accept(nid))
    rejected: set[UUID] = set(nid for nid in beliefs if not _initial_accept(nid))

    for _ in range(max_iterations):
        old_acc = set(accepted)
        old_rej = set(rejected)
        for nid in beliefs:
            attackers = set(attack_graph.get(nid, []))
            # An attacker is "effective" if its weight >= target's weight
            effective_attackers = {
                a for a in attackers if _weight(a) >= _weight(nid)
            }
            if not effective_attackers or effective_attackers.issubset(rejected):
                accepted.add(nid)
                rejected.discard(nid)
            else:
                rejected.add(nid)
                accepted.discard(nid)
        if accepted == old_acc and rejected == old_rej:
            break
    return accepted


# ═══════════════════════════════════════════════════════════════
# GNN (Level 3 — Graph Neural Network)
# ═══════════════════════════════════════════════════════════════

def gnn_feature_dim(num_node_types: int = 7) -> int:
    return num_node_types + 1


def gnn_encode_features(
    node_ids: list[UUID],
    get_type_fn,
    get_opinion_fn,
    num_node_types: int = 7,
    type_index_map: Optional[dict] = None,
) -> tuple[list[list[float]], dict[int, UUID]]:
    """Encode nodes as [type_one_hot | projected_probability]."""
    if type_index_map is None:
        type_index_map = {}
    n = len(node_ids)
    in_dim = gnn_feature_dim(num_node_types)
    features = [[0.0] * in_dim for _ in range(n)]
    idx_map = {}
    for i, nid in enumerate(node_ids):
        idx_map[i] = nid
        type_str = get_type_fn(nid)
        type_idx = type_index_map.get(type_str, 0)
        features[i][type_idx] = 1.0
        opinion = get_opinion_fn(nid)
        if opinion:
            b, d, u, a = opinion
            features[i][-1] = projected_probability(b, u, a)
        else:
            features[i][-1] = 0.5
    return features, idx_map


def gnn_build_adjacency_matrix(
    n: int,
    edges: list,
    edge_weight_fn=None,
) -> list[list[float]]:
    """Build adjacency matrix with edge weights + self-loops."""
    if edge_weight_fn is None:
        def edge_weight_fn(e):
            return EDGE_WEIGHTS.get(e.type.name if hasattr(e, 'type') else str(e.type), 0.5)
    adj = [[0.0] * n for _ in range(n)]
    for i in range(n):
        adj[i][i] = 1.0
    return adj


# ═══════════════════════════════════════════════════════════════
# NEURO-SYMBOLIC (Level 6)
# ═══════════════════════════════════════════════════════════════

def neurosymbolic_fuse(
    neural_beliefs: dict[UUID, float],
    symbolic_beliefs: dict[UUID, float],
    lambda_neural: float = 0.5,
) -> dict[UUID, float]:
    """F(x) = λ · f_neural(x) + (1-λ) · f_symbolic(x)."""
    all_keys = set(neural_beliefs) | set(symbolic_beliefs)
    fused = {}
    for key in all_keys:
        n = neural_beliefs.get(key, 0.5)
        s = symbolic_beliefs.get(key, 0.5)
        fused[key] = lambda_neural * n + (1 - lambda_neural) * s
    return fused


def compute_logic_penalty(
    edges: list,
    get_belief_fn,
    source_threshold: float = 0.7,
    target_threshold: float = 0.3,
) -> dict[UUID, int]:
    """Count implication violations (A ∧ ¬B)."""
    violations: dict[UUID, int] = {}
    for edge in edges:
        src = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        tgt = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        if get_belief_fn(src) > source_threshold and get_belief_fn(tgt) < target_threshold:
            violations[tgt] = violations.get(tgt, 0) + 1
    return violations


def total_loss(
    predictions: dict[UUID, float],
    targets: dict[UUID, float],
    logic_penalty_count: int,
    logic_penalty_weight: float = 0.1,
) -> float:
    """L_total = (1/N)·Σ(pred-target)² + α·L_logic."""
    mse = 0.0
    count = 0
    for nid, pred in predictions.items():
        if nid in targets:
            mse += (pred - targets[nid]) ** 2
            count += 1
    if count > 0:
        mse /= count
    return mse + logic_penalty_weight * logic_penalty_count


# ═══════════════════════════════════════════════════════════════
# MASTER EQUATION (Level 7 + unified)
# ═══════════════════════════════════════════════════════════════

def master_equation(
    node_id: UUID,
    probability: float,
    graph_belief: float,
    logic_consistency: float,
    attack_strength: float,
    violations: int,
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.2,
    lambda_v: float = 0.1,
) -> float:
    """R(x) = α·P(x) + β·G(x) + γ·L(x) - δ·A(x) - λ_v·violations(x)."""
    return (alpha * probability + beta * graph_belief + gamma * logic_consistency
            - delta * attack_strength - lambda_v * violations)


def master_equation_all(
    node_ids: list[UUID],
    probabilities: dict[UUID, float],
    graph_beliefs: dict[UUID, float],
    logic_consistencies: dict[UUID, float],
    attack_strengths: dict[UUID, float],
    violations: dict[UUID, int],
    alpha: float = 0.3,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.2,
    lambda_v: float = 0.1,
) -> dict[UUID, float]:
    result = {}
    for nid in node_ids:
        result[nid] = master_equation(
            nid,
            probabilities.get(nid, 0.5),
            graph_beliefs.get(nid, 0.5),
            logic_consistencies.get(nid, 1.0),
            attack_strengths.get(nid, 0.0),
            violations.get(nid, 0),
            alpha, beta, gamma, delta, lambda_v,
        )
    return result


def compute_support_sum(
    node_id: UUID,
    edges: list,
    beliefs: dict[UUID, float],
) -> float:
    """Σ_{j∈Supp(i)} W_ji · B_j."""
    total = 0.0
    for edge in edges:
        target = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        source = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        if target == node_id and source in beliefs:
            type_name = edge.type.name if hasattr(edge.type, 'name') else str(edge.type)
            weight = EDGE_WEIGHTS.get(type_name, 0.5)
            total += weight * beliefs[source]
    return total


def compute_attack_sum(
    node_id: UUID,
    edges: list,
    beliefs: dict[UUID, float],
) -> float:
    """Σ_{k∈Att(i)} B_k."""
    total = 0.0
    attack_types = {"ATTACKS", "CONTRADICTS", "REBUTS"}
    for edge in edges:
        target = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        source = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        if target == node_id and source in beliefs:
            type_name = edge.type.name if hasattr(edge.type, 'name') else str(edge.type)
            if type_name in attack_types:
                total += beliefs[source]
    return total


def global_objective(
    truth_values: dict[UUID, float],
    violations: dict[UUID, int],
    lambda_v: float = 0.1,
) -> float:
    """G = Σ_v T(v) - λ·C_violations."""
    return sum(truth_values.values()) - lambda_v * sum(violations.values())


def count_violations(
    opinions: dict[UUID, tuple[float, float, float, float]],
    edges: list,
    opinion_threshold: float = 0.01,
) -> dict[UUID, int]:
    """Count constraint violations per node.
    
    Checks: opinion invariant |b+d+u-1| > threshold,
    negative uncertainty, edge consistency.
    """
    violations: dict[UUID, int] = {}
    for nid, (b, d, u, a) in opinions.items():
        count = 0
        if abs(b + d + u - 1.0) > opinion_threshold:
            count += 1
        if u < 0:
            count += 1
        if count > 0:
            violations[nid] = count
    for nid in opinions:
        for edge in edges:
            src = edge.source_id if hasattr(edge, 'source_id') else edge[0]
            if src == nid and nid in opinions:
                b, d, u, a = opinions[nid]
                type_name = edge.type.name if hasattr(edge.type, 'name') else str(edge.type)
                if type_name in ("SUPPORTS", "INFERS", "JUSTIFIES", "DIRECT"):
                    if projected_probability(b, u, a) < 0.2:
                        violations[nid] = violations.get(nid, 0) + 1
    return violations


def fixed_point_iteration(
    initial_beliefs: dict[UUID, float],
    update_fn,
    max_iterations: int = 100,
    threshold: float = 1e-6,
) -> dict[UUID, float]:
    """Iterate B^{t+1} = F(B^t) until ‖B^{t+1} - B^t‖_2 < threshold."""
    beliefs = dict(initial_beliefs)
    for _ in range(max_iterations):
        old = dict(beliefs)
        beliefs = update_fn(beliefs)
        change = math.sqrt(sum((beliefs[k] - old.get(k, 0.5)) ** 2 for k in beliefs))
        if change < threshold:
            break
    return beliefs


# ═══════════════════════════════════════════════════════════════
# CONVERGENCE (Runtime)
# ═══════════════════════════════════════════════════════════════

def convergence_norm(
    graph_distance: float,
    attention_distance: float,
    hidden_distance: float,
    operator_change: float,
) -> float:
    """‖Δs‖ = 0.4·δG + 0.3·δA + 0.2·δH + 0.1·δT."""
    return 0.4 * graph_distance + 0.3 * attention_distance + 0.2 * hidden_distance + 0.1 * operator_change


def graph_distance(
    nodes_a: set[UUID],
    nodes_b: set[UUID],
    edges_a: list,
    edges_b: list,
    beliefs_a: dict[UUID, float],
    beliefs_b: dict[UUID, float],
) -> float:
    """0.3·node_jaccard + 0.3·edge_jaccard + 0.4·belief_L2."""
    node_dist = jaccard_distance(nodes_a, nodes_b)
    edge_set_a = {(e.source_id, e.target_id, e.type.name if hasattr(e.type, 'name') else str(e.type))
                  if hasattr(e, 'source_id') else e for e in edges_a}
    edge_set_b = {(e.source_id, e.target_id, e.type.name if hasattr(e.type, 'name') else str(e.type))
                  if hasattr(e, 'source_id') else e for e in edges_b}
    edge_dist = jaccard_distance(edge_set_a, edge_set_b)
    shared = nodes_a & nodes_b
    if shared:
        belief_diff = math.sqrt(sum((beliefs_b.get(nid, 0.5) - beliefs_a.get(nid, 0.5)) ** 2 for nid in shared) / len(shared))
    else:
        belief_diff = 1.0
    return 0.3 * node_dist + 0.3 * edge_dist + 0.4 * belief_diff


def hidden_state_distance(
    v_curr: list[float],
    v_prev: list[float],
) -> float:
    """L2: sqrt((1/N)·Σ(v_i - v_prev_i)²)."""
    if not v_curr or not v_prev:
        return 1.0
    n = min(len(v_curr), len(v_prev))
    return math.sqrt(sum((v_curr[i] - v_prev[i]) ** 2 for i in range(n)) / n)


# ═══════════════════════════════════════════════════════════════
# MEMORY / RETRIEVAL
# ═══════════════════════════════════════════════════════════════

def memory_similarity(
    jaccard: float,
    count_prox: float,
    jaccard_weight: float = 0.6,
) -> float:
    """score = 0.6·Jaccard + 0.4·count_proximity."""
    return jaccard_weight * jaccard + (1 - jaccard_weight) * count_prox


def graph_to_sets(
    node_ids: set[UUID],
    edge_pairs: set[tuple[UUID, UUID]],
) -> tuple[set[UUID], set[tuple[UUID, UUID]]]:
    return (node_ids, edge_pairs)


# ═══════════════════════════════════════════════════════════════
# SYSTEMS / FEEDBACK LOOPS
# ═══════════════════════════════════════════════════════════════

def betweenness_centrality(
    node_ids: set[UUID],
    edges: list,
) -> dict[UUID, float]:
    """Brandes' algorithm for betweenness centrality (unweighted)."""
    adj = {nid: [] for nid in node_ids}
    for edge in edges:
        src = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        tgt = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        if src in adj and tgt in adj:
            adj[src].append(tgt)
    centrality = {nid: 0.0 for nid in node_ids}
    n = len(node_ids)
    for s in node_ids:
        stack = []
        paths = {nid: [] for nid in node_ids}
        sigma = {nid: 0 for nid in node_ids}
        distance = {nid: -1 for nid in node_ids}
        sigma[s] = 1
        distance[s] = 0
        queue = [s]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in adj.get(v, []):
                if distance[w] < 0:
                    queue.append(w)
                    distance[w] = distance[v] + 1
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]
                    paths[w].append(v)
        delta = {nid: 0.0 for nid in node_ids}
        while stack:
            w = stack.pop()
            for v in paths.get(w, []):
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                centrality[w] += delta[w]
    if n > 2:
        factor = (n - 1) * (n - 2)
        if factor > 0:
            for nid in centrality:
                centrality[nid] /= factor
    return centrality


def leverage_score(
    in_degree: int,
    out_degree: int,
    max_in: int,
    max_out: int,
    betweenness: float,
) -> float:
    """score = 0.4·in_norm + 0.3·out_norm + 0.3·betweenness."""
    in_norm = in_degree / max(max_in, 1)
    out_norm = out_degree / max(max_out, 1)
    return 0.4 * in_norm + 0.3 * out_norm + 0.3 * betweenness


def classify_feedback_loop(negation_count: int) -> str:
    """'balancing' if odd negation edges, else 'reinforcing'."""
    return "balancing" if negation_count % 2 == 1 else "reinforcing"


# ═══════════════════════════════════════════════════════════════
# COMPARISON / GRAPH DIFF
# ═══════════════════════════════════════════════════════════════

def graph_diff_score(
    shared_concepts: int,
    nodes_a: int,
    nodes_b: int,
    deltas: int,
) -> float:
    """score = shared_ratio · (1 - delta_penalty · 0.5)."""
    shared_ratio = shared_concepts / max(nodes_a, nodes_b, 1)
    delta_penalty = deltas / max(deltas, 1)
    return shared_ratio * (1.0 - delta_penalty * 0.5)


# ═══════════════════════════════════════════════════════════════
# PRODUCT LOGIC (Category operations)
# ═══════════════════════════════════════════════════════════════

def product_logic_implication(src_cat: int, tgt_cat: int) -> bool:
    """Implication valid iff src_cat ≤ tgt_cat."""
    return src_cat <= tgt_cat


def product_logic_equivalence(a: int, b: int) -> bool:
    return a == b


# ═══════════════════════════════════════════════════════════════
# INVARIANT CHECKS
# ═══════════════════════════════════════════════════════════════

def check_opinion_invariant(
    b: float, d: float, u: float,
    epsilon: float = 0.01,
) -> bool:
    """|b + d + u - 1.0| < epsilon."""
    return abs(b + d + u - 1.0) < epsilon


def check_cycle_free(
    node_ids: set[UUID],
    edges: list,
) -> bool:
    """Check if the directed graph is acyclic."""
    in_degree: dict[UUID, int] = {nid: 0 for nid in node_ids}
    children: dict[UUID, list[UUID]] = {nid: [] for nid in node_ids}
    for edge in edges:
        src = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        tgt = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        if src in in_degree and tgt in in_degree:
            in_degree[tgt] += 1
            children[src].append(tgt)
    queue = [nid for nid, d in in_degree.items() if d == 0]
    visited = 0
    while queue:
        nid = queue.pop(0)
        visited += 1
        for child in children.get(nid, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return visited == len(node_ids)


def check_category_monotonicity(
    edges: list,
    get_src_cat_fn,
    get_tgt_cat_fn,
) -> list:
    """Return list of (edge, src_cat, tgt_cat) where monotonicity is violated."""
    violations = []
    for edge in edges:
        src = edge.source_id if hasattr(edge, 'source_id') else edge[0]
        tgt = edge.target_id if hasattr(edge, 'target_id') else edge[1]
        src_cat = get_src_cat_fn(src)
        tgt_cat = get_tgt_cat_fn(tgt)
        if not product_logic_implication(src_cat, tgt_cat):
            violations.append((edge, src_cat, tgt_cat))
    return violations


# ═══════════════════════════════════════════════════════════════
# CROSS-DOMAIN COMPLEXITY METRICS
# ═══════════════════════════════════════════════════════════════


def cross_domain_edge_density(
    nodes: dict[UUID, Any],
    edges: dict[UUID, Any],
    communities: dict[UUID, int],
) -> float:
    """Ratio of edges crossing community boundaries to total edges.

    Args:
        nodes: {node_id: node} mapping.
        edges: {edge_id: edge} mapping.
        communities: {node_id: community_id} mapping.

    Returns:
        Float in [0, 1]. 0 = all edges intra-community, 1 = all inter.
    """
    if not edges:
        return 0.0
    cross = 0
    for edge in edges.values():
        src_comm = communities.get(edge.source_id)
        tgt_comm = communities.get(edge.target_id)
        if src_comm is not None and tgt_comm is not None and src_comm != tgt_comm:
            cross += 1
    return cross / len(edges)


def domain_entanglement(
    nodes: dict[UUID, Any],
    edges: dict[UUID, Any],
    communities: dict[UUID, int],
) -> float:
    """Shannon entropy of node-type distribution across communities.

    High entropy = types are evenly spread across communities (entangled).
    Low entropy = types are concentrated in specific communities (segregated).

    Returns:
        Float >= 0. Max value = log2(num_distinct_types).
    """
    import math as _math
    if not nodes or not communities:
        return 0.0

    # Count (community, type) pairs
    counts: dict[tuple[int, str], int] = {}
    total = 0
    for nid, node in nodes.items():
        comm = communities.get(nid)
        if comm is None:
            continue
        key = (comm, node.type.name)
        counts[key] = counts.get(key, 0) + 1
        total += 1

    if total == 0:
        return 0.0

    # Per-community type distributions, then average entropy
    comm_types: dict[int, dict[str, int]] = {}
    for (comm, type_name), count in counts.items():
        comm_types.setdefault(comm, {})[type_name] = count

    entropy = 0.0
    for comm, types in comm_types.items():
        comm_total = sum(types.values())
        for count in types.values():
            p = count / comm_total
            if p > 0:
                entropy -= p * _math.log2(p)

    # Normalize by max possible entropy (uniform distribution across all types)
    num_types = len(set(node.type.name for node in nodes.values()))
    if num_types <= 1:
        return 0.0
    max_entropy = _math.log2(num_types)
    return entropy / (max_entropy * len(comm_types)) if comm_types else 0.0


def causal_chain_depth(
    nodes: dict[UUID, Any],
    edges: dict[UUID, Any],
) -> int:
    """Longest chain of CAUSES edges (topological depth).

    Returns the maximum number of sequential causal dependencies.
    Returns 0 if no CAUSES edges exist.
    """
    # Build adjacency for CAUSES edges only
    children: dict[UUID, list[UUID]] = {}
    in_degree: dict[UUID, int] = {nid: 0 for nid in nodes}
    for edge in edges.values():
        if edge.type.name == "CAUSES":
            src = edge.source_id
            tgt = edge.target_id
            children.setdefault(src, []).append(tgt)
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

    # Topological depth via BFS
    depths: dict[UUID, int] = {nid: 0 for nid in nodes}
    queue = [nid for nid, d in in_degree.items() if d == 0]
    while queue:
        nid = queue.pop(0)
        for child in children.get(nid, []):
            depths[child] = max(depths[child], depths[nid] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return max(depths.values()) if depths else 0


def feedback_loop_count(
    nodes: dict[UUID, Any],
    edges: dict[UUID, Any],
) -> int:
    """Count directed cycles in the CAUSES subgraph.

    Uses DFS-based cycle detection on CAUSES edges only.
    """
    # Build adjacency for CAUSES edges
    children: dict[UUID, list[UUID]] = {}
    for edge in edges.values():
        if edge.type.name == "CAUSES":
            children.setdefault(edge.source_id, []).append(edge.target_id)

    # DFS cycle detection (count strongly connected components with >1 node)
    visited: set[UUID] = set()
    cycles = 0

    def _dfs(nid: UUID, stack: set[UUID]) -> bool:
        nonlocal cycles
        visited.add(nid)
        stack.add(nid)
        for child in children.get(nid, []):
            if child not in visited:
                if _dfs(child, stack):
                    return True
            elif child in stack:
                cycles += 1
                return True
        stack.discard(nid)
        return False

    for nid in nodes:
        if nid not in visited:
            _dfs(nid, set())

    return cycles


# ═══════════════════════════════════════════════════════════════
# CONTEXT SIMILARITY FOR BELIEF TRANSFER
# ═══════════════════════════════════════════════════════════════


def context_similarity(
    node_a_id: UUID,
    node_b_id: UUID,
    nodes: dict[UUID, Any],
    edges: dict[UUID, Any],
    communities: Optional[dict[UUID, int]] = None,
) -> float:
    """Compute contextual similarity between two nodes for belief transfer.

    Factors:
      - Type match: same NodeType = 1.0, different = 0.0
      - Community match: same community = 1.0, different = 0.0
      - Edge neighborhood overlap: Jaccard similarity of neighbors

    Returns:
        Float in [0, 1]. Higher = more similar = safer to transfer beliefs.
    """
    node_a = nodes.get(node_a_id)
    node_b = nodes.get(node_b_id)
    if node_a is None or node_b is None:
        return 0.0

    scores = []

    # Type similarity
    type_sim = 1.0 if node_a.type == node_b.type else 0.0
    scores.append(type_sim)

    # Community similarity
    if communities:
        comm_a = communities.get(node_a_id)
        comm_b = communities.get(node_b_id)
        if comm_a is not None and comm_b is not None:
            scores.append(1.0 if comm_a == comm_b else 0.0)
        else:
            scores.append(0.5)  # unknown communities → neutral

    # Edge neighborhood overlap (Jaccard)
    neighbors_a = set()
    neighbors_b = set()
    for edge in edges.values():
        if edge.source_id == node_a_id:
            neighbors_a.add(edge.target_id)
        if edge.target_id == node_a_id:
            neighbors_a.add(edge.source_id)
        if edge.source_id == node_b_id:
            neighbors_b.add(edge.target_id)
        if edge.target_id == node_b_id:
            neighbors_b.add(edge.source_id)

    if neighbors_a or neighbors_b:
        intersection = len(neighbors_a & neighbors_b)
        union = len(neighbors_a | neighbors_b)
        scores.append(intersection / union if union > 0 else 0.0)
    else:
        scores.append(0.5)  # both isolated → neutral

    return sum(scores) / len(scores) if scores else 0.0


# ═══════════════════════════════════════════════════════════════
# RISK ATTITUDE MODELING
# ═══════════════════════════════════════════════════════════════


def risk_adjusted_belief(
    belief: float,
    alpha: float = 1.0,
    risk_measure: str = "power",
) -> float:
    """Apply risk attitude adjustment to a belief value.

    Models how a stakeholder's risk preference affects their perceived
    belief strength. Uses prospect-theory-style value functions.

    Args:
        belief: Raw belief value in [0, 1].
        alpha: Risk attitude parameter.
            alpha = 1.0: risk-neutral (no adjustment)
            alpha < 1.0: risk-averse (conservative, lowers high beliefs)
            alpha > 1.0: risk-seeking (amplifies high beliefs)
        risk_measure: Utility function type.
            "power": CE = b^α (standard power utility)
            "cara": CE = (1 - e^(-α*b)) / (1 - e^(-α)) (constant absolute risk aversion)

    Returns:
        Risk-adjusted belief in [0, 1].
    """
    if alpha == 1.0:
        return belief

    b = max(0.0, min(1.0, belief))

    if risk_measure == "power":
        if b <= 0.0:
            return 0.0
        return b ** alpha
    elif risk_measure == "cara":
        if alpha == 0.0:
            return b
        return (1.0 - math.exp(-alpha * b)) / (1.0 - math.exp(-alpha))
    else:
        return b


def stakeholder_utility(
    belief: float,
    alpha: float = 1.0,
    loss_aversion: float = 2.25,
) -> float:
    """Prospect-theory utility for a stakeholder.

    Combines risk attitude with loss aversion (losses hurt ~2.25x more
    than equivalent gains, per Kahneman & Tversky).

    Args:
        belief: Raw belief in [0, 1].
        alpha: Risk attitude parameter (same as risk_adjusted_belief).
        loss_aversion: Loss aversion coefficient (default 2.25).

    Returns:
        Utility value. Positive = gain frame, negative = loss frame.
    """
    b = max(0.0, min(1.0, belief))
    reference = 0.5  # reference point is midpoint

    if b >= reference:
        gain = b - reference
        return gain ** alpha if alpha > 0 else gain
    else:
        loss = reference - b
        return -(loss_aversion * (loss ** alpha)) if alpha > 0 else -loss_aversion * loss


def aggregate_stakeholder_beliefs(
    beliefs: list[tuple[float, float]],
) -> float:
    """Aggregate beliefs from multiple stakeholders with risk attitudes.

    Args:
        beliefs: List of (risk_adjusted_belief, weight) tuples.
            weight represents stakeholder influence/importance.

    Returns:
        Weighted average belief.
    """
    if not beliefs:
        return 0.0
    total_weight = sum(w for _, w in beliefs)
    if total_weight == 0:
        return 0.0
    return sum(b * w for b, w in beliefs) / total_weight


# ═══════════════════════════════════════════════════════════════
# INFORMATION THEORY UTILITIES (MDM Complexity Ruler)
# ═══════════════════════════════════════════════════════════════

def shannon_entropy(probs: list[float]) -> float:
    """Shannon Entropy: H(X) = -Σ p(x)·log₂(p(x))

    Measures how surprising or unpredictable a distribution is.

    Interpretation:
        0.0    = completely predictable (all weight on one outcome)
        Higher = more spread out, unpredictable, complex

    Use cases:
        - Measure system complexity (high entropy = complex system)
        - Measure balance (high entropy = balanced, low = concentrated)
        - Identify bottlenecks (low entropy = single point of failure)

    Example:
        Port A: 90%, Port B: 10% → entropy = 0.47 (low = fragile)
        Port A: 33%, Port B: 33%, Port C: 33% → entropy = 1.58 (high = resilient)

    Args:
        probs: Probability distribution (must sum to 1.0).

    Returns:
        Entropy in bits (0.0 to log₂(n)).

    Reference: Shannon (1948) "A Mathematical Theory of Communication"
    """
    if not probs:
        return 0.0
    return -sum(p * math.log2(p) for p in probs if p > 0)


def normalized_shannon_entropy(probs: list[float]) -> float:
    """Normalized Shannon Entropy: H(X) / H_max

    Normalizes entropy to [0, 1] range.

    Interpretation:
        0.0 = completely predictable
        1.0 = maximum unpredictability

    Args:
        probs: Probability distribution (must sum to 1.0).

    Returns:
        Normalized entropy in [0, 1].
    """
    if not probs:
        return 0.0
    h = shannon_entropy(probs)
    h_max = math.log2(len(probs)) if len(probs) > 1 else 1.0
    return h / h_max if h_max > 0 else 0.0


def pointwise_mutual_information(
    p_xy: float,
    p_x: float,
    p_y: float,
) -> float:
    """Pointwise Mutual Information (PMI): log₂(P(x,y) / (P(x)·P(y)))

    Measures how much two events co-occur relative to chance.

    Interpretation:
        > 0    = co-occur more than expected by chance
        = 0    = independent
        < 0    = co-occur less than expected by chance

    Use cases:
        - Discover which interactions happen together more than chance
        - Find functional modules in the MDM
        - Identify communities in the graph

    Example:
        "Port congestion" + "ship delays": PMI = 2.3 (strong association)
        "Port congestion" + "revenue loss": PMI = 0.1 (weak association)

    Args:
        p_xy: Joint probability P(x,y).
        p_x: Marginal probability P(x).
        p_y: Marginal probability P(y).

    Returns:
        PMI value in bits.

    Reference: Church & Hanks (1989) "Word Association Norms, Mutual
    Information, and Lexicography"
    """
    if p_xy == 0 or p_x == 0 or p_y == 0:
        return 0.0
    return math.log2(p_xy / (p_x * p_y))


def normalized_pmi(pmi: float, p_xy: float) -> float:
    """Normalized PMI (NPMI): PMI / (-log₂ P(x,y))

    Normalizes PMI to [-1, 1] range.

    Interpretation:
        -1.0 = never co-occur
         0.0 = independent
         1.0 = always co-occur

    Args:
        pmi: PMI value.
        p_xy: Joint probability P(x,y).

    Returns:
        Normalized PMI in [-1, 1].
    """
    if p_xy == 0:
        return 0.0
    return pmi / (-math.log2(p_xy))


def mutual_information_from_counts(
    count_xy: int,
    count_x: int,
    count_y: int,
    total: int,
) -> float:
    """Compute PMI from counts (convenience function).

    Args:
        count_xy: Number of times x and y co-occur.
        count_x: Number of times x occurs.
        count_y: Number of times y occurs.
        total: Total number of observations.

    Returns:
        PMI value in bits.
    """
    if total == 0 or count_xy == 0 or count_x == 0 or count_y == 0:
        return 0.0
    p_xy = count_xy / total
    p_x = count_x / total
    p_y = count_y / total
    return pointwise_mutual_information(p_xy, p_x, p_y)


def entropy_of_distribution(values: list[float]) -> float:
    """Compute Shannon entropy from a list of values (not probabilities).

    Normalizes values to probabilities first.

    Args:
        values: List of non-negative values.

    Returns:
        Shannon entropy in bits.
    """
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return shannon_entropy(probs)


def interaction_complexity(
    interactions: dict[str, int],
) -> float:
    """Measure complexity of interaction type distribution.

    High entropy = many different interaction types = complex system
    Low entropy = mostly one interaction type = simple system

    Args:
        interactions: Dict mapping interaction_type_name → count.

    Returns:
        Shannon entropy in bits.
    """
    if not interactions:
        return 0.0
    total = sum(interactions.values())
    if total == 0:
        return 0.0
    probs = [count / total for count in interactions.values() if count > 0]
    return shannon_entropy(probs)


def blob_type_entropy(
    blob_types: dict[str, int],
) -> float:
    """Measure how entangled different blob types are in a domain.

    High entropy = many different blob types mixed together = entangled
    Low entropy = mostly one blob type = separated

    Args:
        blob_types: Dict mapping blob_type_name → count.

    Returns:
        Shannon entropy in bits.
    """
    if not blob_types:
        return 0.0
    total = sum(blob_types.values())
    if total == 0:
        return 0.0
    probs = [count / total for count in blob_types.values() if count > 0]
    return shannon_entropy(probs)
