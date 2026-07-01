#!/usr/bin/env python3
"""Decision toy — ScenarioComparison + filter + grade_scenarios + rank end-to-end.

Demonstrates:
  1. Inline Turtle KB with supplier facts, budget, goals
  2. Inline .sysd model with KB_QUERY in aux expressions
  3. 4 ScenarioDefs: Do nothing, Switch to B, Increase stock, Delay orders
  4. Constraint filter via SPARQL ASK (eliminates budget-violating scenarios)
  5. Per-goal grading via grade_scenarios() + aggregated rank()
  6. Comparison table

Usage::

    python examples/decision_toy.py
"""

from dynafx import (
    KBSimBridge,
    ScenarioComparison,
    ScenarioDef,
    TripleStore,
    grade_queries,
)
from dynafx.dynamics import parse_sysd
from dynafx.knowledge.model import NamedNode, Literal, Triple
from dynafx.knowledge.turtle import parse_turtle
from dynafx.core.models import Opinion


# ═══════════════════════════════════════════════════════════════
# 1.  Knowledge Base (inline Turtle)
# ═══════════════════════════════════════════════════════════════

TURTLE = """\
@prefix sc: <http://sc.org/> .

# Supplier facts
sc:Supplier_A sc:status "delayed" .
sc:Supplier_A sc:cost 1.0 .

sc:Supplier_B sc:status "available" .
sc:Supplier_B sc:cost 1.03 .

# Budget
sc:Budget sc:limit 50000 .
sc:Budget sc:exceeded "false" .

# Contract
sc:Contract sc:minFillRate 0.70 .
sc:Contract sc:violated "false" .
"""

store = parse_turtle(TURTLE)
bridge = KBSimBridge(store)

NS = "http://sc.org/"
SCEN = NamedNode(f"{NS}Scenario")
FILL = NamedNode(f"{NS}fillRateScore")
COST = NamedNode(f"{NS}costScore")
PROFIT = NamedNode(f"{NS}profitScore")

# ── Common SPARQL query strings for KB_QUERY in the model ──
SUP_A_DELAYED_Q = f"ASK {{ <{NS}Supplier_A> <{NS}status> \"delayed\" }}"
SUP_B_AVAIL_Q = f"ASK {{ <{NS}Supplier_B> <{NS}status> \"available\" }}"

# ═══════════════════════════════════════════════════════════════
# 2.  Model (inline .sysd)
# ═══════════════════════════════════════════════════════════════

MODEL_SRC = """\
T
dt 0.5
from 0 to 30

// Parameters (overridden by scenario params)
aux supplier_reliability: 0.85
aux safety_stock: 300
aux demand: 100
aux unit_cost: 10.0

// KB_QUERY — string params passed at simulate time
aux sup_a_delayed: KB_QUERY(sup_a_q)
aux sup_b_available: KB_QUERY(sup_b_q)

// Effective reliability based on supplier choice
aux eff_reliability: IF(supplier == 1, IF(sup_b_available > 0, 0.95, 0.85), IF(sup_a_delayed > 0, 0.50, 0.85))

// Fulfillment
aux replenishment: eff_reliability * demand * (1 - delay_active * 0.3)
aux retail_sales: MIN(Inventory / dt, demand)

// Holding cost — penalizes large inventory buffers
aux inventory_holding_cost: (Inventory / 100) * 0.5

// Financial tracking
aux cost_rate: unit_cost * retail_sales + inventory_holding_cost
aux revenue_rate: 15.0 * retail_sales

stock Inventory: safety_stock
  + replenishment
  - retail_sales

stock Cumulative_Demand: 0
  + demand

stock Cumulative_Met: 0
  + retail_sales

stock Total_Cost: 0
  + cost_rate

stock Total_Revenue: 0
  + revenue_rate
"""

model = parse_sysd(MODEL_SRC)

# ═══════════════════════════════════════════════════════════════
# 3.  Scenario Definitions
# ═══════════════════════════════════════════════════════════════

COMMON_PARAMS = {
    "sup_a_q": SUP_A_DELAYED_Q,
    "sup_b_q": SUP_B_AVAIL_Q,
}

sdefs = [
    ScenarioDef("Do nothing", {
        **COMMON_PARAMS,
        "supplier": 0.0,
        "safety_stock": 300.0,
        "delay_active": 0.0,
        "unit_cost": 10.0,
    }),
    ScenarioDef("Switch to B", {
        **COMMON_PARAMS,
        "supplier": 1.0,
        "safety_stock": 300.0,
        "delay_active": 0.0,
        "unit_cost": 10.30,
    }),
    ScenarioDef("Increase stock", {
        **COMMON_PARAMS,
        "supplier": 0.0,
        "safety_stock": 500.0,
        "delay_active": 0.0,
        "unit_cost": 10.0,
    }),
    ScenarioDef("Delay orders", {
        **COMMON_PARAMS,
        "supplier": 0.0,
        "safety_stock": 300.0,
        "delay_active": 1.0,
        "unit_cost": 10.0,
    }),
]

# ═══════════════════════════════════════════════════════════════
# 4.  Evidence map for grading
# ═══════════════════════════════════════════════════════════════

MAX_DEMAND = 3000.0    # 60 steps × 100/day over 30 days
MAX_COST = 35000.0      # ceiling for total cost normalization
MAX_REVENUE = 45000.0   # ceiling for total revenue normalization


ev_map = [
    ("Cumulative_Met", SCEN, FILL,
     lambda init, final: (
         min(1.0, max(0.0, (final[-1] - init[0]) / MAX_DEMAND))
     )),
    ("Total_Cost", SCEN, COST,
     lambda init, final: (
         min(1.0, max(0.0, 1.0 - (final[-1] - init[0]) / MAX_COST))
     )),
    ("Total_Revenue", SCEN, PROFIT,
     lambda init, final: (
         min(1.0, max(0.0, (final[-1] - init[0]) / MAX_REVENUE))
     )),
]

# ═══════════════════════════════════════════════════════════════
# 5.  Grade specs (SPARQL SELECT → float)
# ═══════════════════════════════════════════════════════════════

P = f"PREFIX sc: <{NS}>"

grade_specs = [
    (f"{P} SELECT ?v WHERE {{ <{NS}Scenario> sc:fillRateScore ?v }}", "v", 0.0, 0.0),
    (f"{P} SELECT ?v WHERE {{ <{NS}Scenario> sc:costScore ?v }}", "v", 0.0, 0.0),
    (f"{P} SELECT ?v WHERE {{ <{NS}Scenario> sc:profitScore ?v }}", "v", 0.0, 0.0),
]

# ═══════════════════════════════════════════════════════════════
# 6.  Constraint queries (SPARQL ASK, per-scenario evidence)
# ═══════════════════════════════════════════════════════════════

constraint_queries = [
    # Minimum fill rate (eliminates scenarios that can't meet demand)
    f"{P} ASK {{ <{NS}Scenario> sc:fillRateScore ?f . FILTER(?f >= 0.4) }}",
    # Minimum cost efficiency (eliminates scenarios that exceed cost ceiling)
    f"{P} ASK {{ <{NS}Scenario> sc:costScore ?c . FILTER(?c >= 0.15) }}",
]

# ═══════════════════════════════════════════════════════════════
# 7.  Run pipeline
# ═══════════════════════════════════════════════════════════════

GRADE_LABELS = ["Fill", "Cost", "Profit"]

print("=" * 72)
print("  Decision Toy — Scenario Ranking")
print("=" * 72)

comp = ScenarioComparison(model, sdefs, method="rk4", kb=store)

n_before = len(comp.scenarios)
comp.filter(store, constraint_queries, evidence_map=ev_map, bridge=bridge)
n_after = len(comp.scenarios)
n_filtered = n_before - n_after

if n_filtered:
    print(f"\n  Filtered out {n_filtered} scenario(s) failing ASK constraints\n")
else:
    print("\n  All scenarios passed constraint checks\n")

grades = comp.grade_scenarios(grade_specs, store, evidence_map=ev_map, bridge=bridge)
ranked = comp.rank(grade_specs, store, evidence_map=ev_map, bridge=bridge,
                   agg="mean")

# ── Table ─────────────────────────────────────────────────────
header = f"  {'Rank':<6} {'Scenario':<18}"
for lbl in GRADE_LABELS:
    header += f" {lbl:<7}"
header += " Score"
print(header)
print("  " + "-" * (6 + 18 + 8 * len(GRADE_LABELS) + 2))

for rank_idx, (sname, score) in enumerate(ranked, 1):
    g = grades[sname]
    vals = [g.get(f"{sname}_0", 0.0), g.get(f"{sname}_1", 0.0), g.get(f"{sname}_2", 0.0)]
    row = f"  {rank_idx:<6} {sname:<18}"
    for v in vals:
        row += f" {v:<7.3f}"
    row += f" {score:.3f}"
    print(row)

print()
best = ranked[0][0] if ranked else "N/A"
print(f"  Recommendation: {best}")
print()

# ═══════════════════════════════════════════════════════════════
# 8.  Explanation for top-ranked scenario
# ═══════════════════════════════════════════════════════════════

if ranked:
    exp = comp.explain_scenario(
        ranked[0][0], store, evidence_map=ev_map, bridge=bridge,
        grade_specs=grade_specs, grades=grades, ranked=ranked,
    )
    print(f"  ── Explanation: {exp['name']} (rank {exp['rank']}/{exp['num_scenarios']}, "
          f"score {exp['total_score']}) ──")
    for g in exp["goals"]:
        print(f"\n    {g['label']:<16} {g['score']:.4f}")
        print(f"    {'':>16} Chain: {' → '.join(g['causal_chain'][:10])}")
    if exp["kb_facts"]:
        print(f"\n    KB facts ({len(exp['kb_facts'])}):")
        for fact in exp["kb_facts"]:
            print(f"      • {fact}")

print("=" * 72)
