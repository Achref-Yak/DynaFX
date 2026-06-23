"""
Supply Chain Multi-Echelon Demo
================================
Demonstrates framework capabilities:
  - DELAY3, DELAY_FIXED (higher-order delays)
  - SMOOTH (demand smoothing)
  - Feedback loop detection (bullwhip effect analysis)
  - Causal tracing (stockout root cause analysis)
  - Sensitivity analysis (Monte Carlo)
  - Scenario comparison

Run: python examples/supply_chain_demo.py
"""
import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TMPDIR", "/tmp")

from cognitive_engine.system.dsl import parse_sysd_file, parse_sysd
from cognitive_engine.system.causal import causes_tree, effects_tree, causal_trace
from cognitive_engine.system.feedback import detect_feedback_loops, loops_for_variable


# ═══════════════════════════════════════════════════════════════
# 1. PARSE THE MODEL
# ═══════════════════════════════════════════════════════════════

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "supply_chain_demo.sysd")
model = parse_sysd_file(model_path)

print("=" * 70)
print("MULTI-ECHELON SUPPLY CHAIN MODEL")
print("=" * 70)
print(f"Stocks:      {[s.name for s in model.stocks]}")
print(f"Auxes:       {len(model.aux_vars)} auxiliary variables")
print(f"Parameters:  t_span={model.t_span}, dt={model.dt}")
print()

all_exprs = " ".join(a.expr for a in model.aux_vars).upper()
print("Features used in model:")
print(f"  DELAY3:      {all_exprs.count('DELAY3')} call(s) — 3-stage shipping delays")
print(f"  DELAY_FIXED: {all_exprs.count('DELAY_FIXED')} call(s) — fixed processing lead time")
print(f"  SMOOTH:      {all_exprs.count('SMOOTH')} call(s) — demand smoothing")
print(f"  PULSE, NOISE, SIN, IF, MIN, MAX")
print()


# ═══════════════════════════════════════════════════════════════
# 2. BASE SIMULATION
# ═══════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    "factory_capacity": 500,
    "base_demand": 200,
    "reorder_point": 1500,
    "smoothing_time": 5,
    "shipping_delay": 10,
}

print("--- Base Simulation ---")
r = model.simulate(params=DEFAULT_PARAMS)

print(f"  Duration: {r.times[-1]:.0f} weeks")
print(f"  Steps:    {r.steps}")
print()
print("  Final inventory levels:")
for stock in r["stocks"]:
    val = r["values"][stock][-1]
    print(f"    {stock:25s}: {val:>10.1f}")
print()

cum_demand = r["values"]["Cumulative_Demand"][-1]
cum_met = r["values"]["Cumulative_Met"][-1]
fill_rate = cum_met / cum_demand if cum_demand > 0 else 1.0
print(f"  Overall fill rate: {fill_rate:.3f}")
print(f"  Total demand:      {cum_demand:.0f}")
print(f"  Demand met:        {cum_met:.0f}")
print()


# ═══════════════════════════════════════════════════════════════
# 3. FEEDBACK LOOP DETECTION
# ═══════════════════════════════════════════════════════════════

print("--- Feedback Loop Detection ---")
analysis = detect_feedback_loops(model)
d = analysis.to_dict()
print(f"  Total loops found: {len(analysis.loops)}")
print(f"  Reinforcing loops: {d['num_reinforcing']}")
print(f"  Balancing loops:   {d['num_balancing']}")
print()

for loop in analysis.loops[:8]:
    nodes_str = " → ".join(loop.nodes)
    print(f"  {loop.name}: {nodes_str} [{loop.polarity}]")
print()

for var in ["Retailer_Inventory", "Factory_Inventory", "Warehouse_Inventory"]:
    var_loops = loops_for_variable(analysis, var)
    if var_loops:
        print(f"  Loops involving {var}:")
        for vl in var_loops:
            print(f"    {vl.name}: {' → '.join(vl.nodes)} [{vl.polarity}]")
print()


# ═══════════════════════════════════════════════════════════════
# 4. CAUSAL TRACING
# ═══════════════════════════════════════════════════════════════

print("--- Causal Tracing: Retailer_Inventory ---")
state = {}
for name in r["stocks"]:
    state[name] = r["values"][name][-1]

trace = causal_trace(model, "Retailer_Inventory", state)

print("  Causes tree (upstream):")
if trace["causes"]:
    def print_tree(node, indent=2):
        prefix = " " * indent
        val = state.get(node["name"], 0)
        print(f"{prefix}{node['name']}: {val:.1f}")
        for child in node["children"][:3]:
            print_tree(child, indent + 4)
    print_tree(trace["causes"])

print()
print("  Effects tree (downstream):")
if trace["effects"]:
    def print_effects(node, indent=2):
        prefix = " " * indent
        print(f"{prefix}{node['name']}")
        for child in node["children"][:3]:
            print_effects(child, indent + 4)
    print_effects(trace["effects"])

print()
print("  Value decomposition:")
if trace["strip"]:
    for factor in trace["strip"]["factors"][:5]:
        print(f"    {factor['name']:30s} = {factor['value']:>10.1f}")
print()


# ═══════════════════════════════════════════════════════════════
# 5. SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

print("--- Sensitivity Analysis (Monte Carlo) ---")
rng = random.Random(42)

sensitivity_params = []
for _ in range(20):
    p = DEFAULT_PARAMS.copy()
    p["base_demand"] = rng.uniform(150, 350)
    p["smoothing_time"] = rng.uniform(2, 15)
    p["shipping_delay"] = rng.uniform(5, 20)
    sensitivity_params.append(p)

results = []
for p in sensitivity_params:
    res = model.simulate(params=p)
    cd = res["values"]["Cumulative_Demand"][-1]
    cm = res["values"]["Cumulative_Met"][-1]
    fill = cm / cd if cd > 0 else 1.0
    inv = sum(res["values"][s][-1] for s in ["Factory_Inventory", "Warehouse_Inventory", "Retailer_Inventory"])
    results.append({"fill_rate": fill, "total_inv": inv})

avg_fill = sum(r["fill_rate"] for r in results) / len(results)
min_fill = min(r["fill_rate"] for r in results)
max_fill = max(r["fill_rate"] for r in results)
avg_inv = sum(r["total_inv"] for r in results) / len(results)
print(f"  Runs: {len(results)}")
print(f"  Fill rate: avg={avg_fill:.3f}, min={min_fill:.3f}, max={max_fill:.3f}")
print(f"  Total inventory: avg={avg_inv:.0f}")
print()


# ═══════════════════════════════════════════════════════════════
# 6. SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════════

print("--- Scenario Comparison ---")
scenarios = {
    "Base": DEFAULT_PARAMS,
    "High Demand": {**DEFAULT_PARAMS, "base_demand": 350},
    "Fast Shipping": {**DEFAULT_PARAMS, "shipping_delay": 5},
    "Slow Smoothing": {**DEFAULT_PARAMS, "smoothing_time": 15},
}

print(f"  {'Scenario':20s} {'Fill Rate':>10s} {'Factory':>10s} {'Warehouse':>10s} {'Retailer':>10s}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for name, params in scenarios.items():
    res = model.simulate(params=params)
    cd = res["values"]["Cumulative_Demand"][-1]
    cm = res["values"]["Cumulative_Met"][-1]
    fill = cm / cd if cd > 0 else 1.0
    fi = res["values"]["Factory_Inventory"][-1]
    wi = res["values"]["Warehouse_Inventory"][-1]
    ri = res["values"]["Retailer_Inventory"][-1]
    print(f"  {name:20s} {fill:>10.3f} {fi:>10.1f} {wi:>10.1f} {ri:>10.1f}")
print()


# ═══════════════════════════════════════════════════════════════
# 7. SUMMARY
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("DEMO COMPLETE — All features demonstrated:")
print("  ✓ DELAY3, DELAY_FIXED (higher-order delays)")
print("  ✓ SMOOTH (demand smoothing)")
print("  ✓ Feedback loop detection (bullwhip analysis)")
print("  ✓ Causal tracing (stockout root cause)")
print("  ✓ Sensitivity analysis (Monte Carlo)")
print("  ✓ Scenario comparison")
print("  ✓ Seasonal dynamics (SIN, PULSE, NOISE)")
print("=" * 70)
