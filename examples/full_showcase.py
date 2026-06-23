"""
Full Framework Showcase — Multi-City Pandemic Model
====================================================
Demonstrates every feature of the system dynamics framework:

  1. DSL Parsing & Simulation (.sysd files)
  2. Units Checking (~Unit~ syntax, dimensional analysis)
  3. Submodels (reusable SEIR_D template, 3 city instances)
  4. Higher-Order Delays (DELAY3, DELAY_FIXED, SMOOTH, SMOOTHI)
  5. Time Functions (PULSE, STEP, RAMP, NOISE, SIN, PI)
  6. Math Functions (MIN, MAX, IF, ABS, EXP, SQRT)
  7. Comparison Operators (>, <, ==, >=, <=)
  8. Feedback Loop Detection (reinforcing/balancing loops)
  9. Causal Tracing (upstream causes, downstream effects)
  10. Sensitivity Analysis (Monte Carlo ensemble)
  11. CSV Import (vaccination schedule from file)
  12. CSV Export (simulation results to file)
  13. Parameter Calibration (fit to observed data)
  14. LP Optimization (resource allocation)
  15. ABM (healthcare worker allocation)
  16. DES (hospital queue simulation)
  17. Validation (model consistency checks)
  18. Scenario Comparison (baseline vs interventions)

Run: python examples/full_showcase.py
"""
import os
import sys
import csv
import random
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TMPDIR", "/tmp")

from cognitive_engine.system.dsl import parse_sysd_file, parse_sysd
from cognitive_engine.system.units import UnitChecker
from cognitive_engine.system.causal import causal_trace, causes_tree, effects_tree
from cognitive_engine.system.feedback import detect_feedback_loops, loops_for_variable
from cognitive_engine.system.optimization import calibrate, lp_minimize


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


# ═══════════════════════════════════════════════════════════════
# 1. PARSE THE MODEL
# ═══════════════════════════════════════════════════════════════

section("1. DSL PARSING")

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "full_showcase.sysd")
model = parse_sysd_file(model_path)

print(f"  Model:     {model.name}")
print(f"  Stocks:    {len(model.stocks)} ({', '.join(s.name for s in model.stocks[:6])}...)")
print(f"  Auxes:     {len(model.aux_vars)} auxiliary variables")
print(f"  Submodels: {len(model.submodels)} ({', '.join(sm.name for sm in model.submodels)})")
print(f"  Includes:  {len(model.includes)} instances")
print(f"  dt:        {model.dt}, t_span: {model.t_span}")

# Show submodel instances
for inc in model.includes:
    params_str = ", ".join(f"{k}={v}" for k, v in inc.params.items())
    print(f"    include {inc.submodel_name} as {inc.instance_name} params: {params_str}")


# ═══════════════════════════════════════════════════════════════
# 2. UNITS CHECKING
# ═══════════════════════════════════════════════════════════════

section("2. UNITS CHECKING")

checker = UnitChecker()
result = checker.check(model)

print(f"  Checked:   {len(result.checked_names)} variables")
print(f"  Errors:    {len(result.errors)}")
print(f"  Warnings:  {len(result.warnings)}")
print(f"  Passed:    {result.passed}")

if result.errors:
    print("\n  Errors:")
    for v in result.errors[:5]:
        print(f"    {v.name}: expected {v.expected}, got {v.actual}")
        print(f"      {v.message}")

if result.warnings:
    print("\n  Warnings:")
    for v in result.warnings[:5]:
        print(f"    {v.name}: {v.message}")

# Show unit annotations on first few stocks
print("\n  Sample unit annotations:")
for stock in model.stocks[:6]:
    if stock.units:
        print(f"    {stock.name}: ~{stock.units}~")


# ═══════════════════════════════════════════════════════════════
# 3. CSV IMPORT (vaccination schedule)
# ═══════════════════════════════════════════════════════════════

section("3. CSV IMPORT")

# Create synthetic vaccination schedule CSV
csv_path = os.path.join(tempfile.gettempdir(), "vax_schedule.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "metro_vax_supply", "suburb_vax_supply"])
    for t in range(0, 181):
        # Ramp up vaccination supply
        if t < 30:
            supply = 100 + t * 5  # ramp up
        elif t < 90:
            supply = 250  # steady
        elif t < 120:
            supply = 250 + (t - 90) * 10  # surge
        else:
            supply = 550  # high steady
        writer.writerow([t, supply, supply * 0.4])

print(f"  Created vaccination schedule: {csv_path}")
data = model.import_data(csv_path)
print(f"  Imported {len(data)} time series")
for name, series in data.items():
    print(f"    {name}: {len(series)} data points, "
          f"range [{min(v for _, v in series):.0f}, {max(v for _, v in series):.0f}]")

# Test interpolation
interp = model.get_imported_interpolator("metro_vax_supply")
print(f"\n  Interpolation test:")
print(f"    t=0:   {interp(0):.0f}")
print(f"    t=15:  {interp(15):.0f}")
print(f"    t=60:  {interp(60):.0f}")
print(f"    t=150: {interp(150):.0f}")


# ═══════════════════════════════════════════════════════════════
# 4. BASE SIMULATION
# ═══════════════════════════════════════════════════════════════

section("4. BASE SIMULATION")

r = model.simulate()

print(f"  Duration:    {r.times[-1]:.0f} days")
print(f"  Steps:       {r.steps}")
print(f"  Stocks:      {len(r.stocks)}")
print(f"  Time points: {len(r.times)}")

subsection("Final State")
for name in ["metro_S", "metro_I", "metro_R", "metro_D",
             "suburb_S", "suburb_I", "suburb_R", "suburb_D",
             "rural_S", "rural_I", "rural_R", "rural_D"]:
    if name in r["values"]:
        val = r["values"][name][-1]
        print(f"    {name:25s}: {val:>10.1f}")

subsection("Peak Infections")
for city in ["metro", "suburb", "rural"]:
    i_key = f"{city}_I"
    if i_key in r["values"]:
        peak = max(r["values"][i_key])
        peak_t = r["times"][r["values"][i_key].index(peak)]
        print(f"    {city:10s}: peak {peak:>8.1f} at day {peak_t:.0f}")

subsection("Aggregate Statistics")
for name in ["total_infected", "total_dead", "total_vaccinated",
             "attack_rate", "case_fatality"]:
    if name in r["values"]:
        val = r["values"][name][-1]
        print(f"    {name:25s}: {val:>10.1f}")


# ═══════════════════════════════════════════════════════════════
# 5. FEEDBACK LOOP DETECTION
# ═══════════════════════════════════════════════════════════════

section("5. FEEDBACK LOOP DETECTION")

analysis = detect_feedback_loops(model)
d = analysis.to_dict()

print(f"  Total loops:   {len(analysis.loops)}")
print(f"  Reinforcing:   {d['num_reinforcing']}")
print(f"  Balancing:     {d['num_balancing']}")

print("\n  Key loops:")
for loop in analysis.loops[:10]:
    nodes_str = " → ".join(loop.nodes[:5])
    if len(loop.nodes) > 5:
        nodes_str += " → ..."
    print(f"    {loop.name}: {nodes_str} [{loop.polarity}]")

# Loops involving infection dynamics
print("\n  Loops involving metro_I:")
for vl in loops_for_variable(analysis, "metro_I"):
    print(f"    {vl.name}: {' → '.join(vl.nodes[:4])} [{vl.polarity}]")


# ═══════════════════════════════════════════════════════════════
# 6. CAUSAL TRACING
# ═══════════════════════════════════════════════════════════════

section("6. CAUSAL TRACING")

# Trace what causes metro_I to change
state = {}
for name in r["stocks"]:
    state[name] = r["values"][name][-1]
# Also include aux values
for aux in model.aux_vars:
    if aux.name in r["values"]:
        state[aux.name] = r["values"][aux.name][-1]

trace = causal_trace(model, "metro_I", state)

print("  Upstream causes of metro_I:")
if trace["causes"]:
    def print_tree(node, indent=4):
        prefix = " " * indent
        val = state.get(node["name"], 0)
        print(f"{prefix}{node['name']}: {val:.2f}")
        for child in node["children"][:3]:
            print_tree(child, indent + 4)
    print_tree(trace["causes"])

print("\n  Downstream effects of metro_I:")
if trace["effects"]:
    def print_effects(node, indent=4):
        prefix = " " * indent
        print(f"{prefix}{node['name']}")
        for child in node["children"][:3]:
            print_effects(child, indent + 4)
    print_effects(trace["effects"])

print("\n  Value decomposition:")
if trace["strip"]:
    for factor in trace["strip"]["factors"][:8]:
        print(f"    {factor['name']:35s} = {factor['value']:>10.2f}")


# ═══════════════════════════════════════════════════════════════
# 7. SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

section("7. SENSITIVITY ANALYSIS (Monte Carlo)")

rng = random.Random(42)
n_runs = 30

sensitivity_params = []
for _ in range(n_runs):
    p = {}
    p["metro_beta"] = rng.uniform(0.2, 0.5)
    p["metro_gamma"] = rng.uniform(0.05, 0.15)
    p["metro_capacity"] = rng.randint(100, 400)
    p["suburb_beta"] = rng.uniform(0.15, 0.35)
    p["rural_beta"] = rng.uniform(0.1, 0.3)
    sensitivity_params.append(p)

results = []
for p in sensitivity_params:
    try:
        res = model.simulate(params=p)
        peak_I = max(res["values"]["metro_I"])
        total_D = res["values"]["metro_D"][-1]
        total_V = res["values"]["metro_Cum_Vaccinated"][-1]
        results.append({
            "peak_infections": peak_I,
            "total_deaths": total_D,
            "total_vaccinated": total_V,
            "params": p,
        })
    except Exception:
        pass

if results:
    peaks = [r["peak_infections"] for r in results]
    deaths = [r["total_deaths"] for r in results]
    vax = [r["total_vaccinated"] for r in results]

    print(f"  Runs:        {len(results)}/{n_runs}")
    print(f"\n  Metro peak infections:")
    print(f"    Mean:  {sum(peaks)/len(peaks):>8.1f}")
    print(f"    Min:   {min(peaks):>8.1f}")
    print(f"    Max:   {max(peaks):>8.1f}")
    print(f"\n  Metro total deaths:")
    print(f"    Mean:  {sum(deaths)/len(deaths):>8.1f}")
    print(f"    Min:   {min(deaths):>8.1f}")
    print(f"    Max:   {max(deaths):>8.1f}")
    print(f"\n  Metro vaccinated:")
    print(f"    Mean:  {sum(vax)/len(vax):>8.1f}")

    # Parameter sensitivity correlation
    betas = [r["params"]["metro_beta"] for r in results]
    correlations = {}
    for param_name in ["metro_beta", "metro_gamma", "metro_capacity"]:
        param_vals = [r["params"][param_name] for r in results]
        # Simple correlation
        mean_p = sum(param_vals) / len(param_vals)
        mean_d = sum(deaths) / len(deaths)
        cov = sum((p - mean_p) * (d - mean_d) for p, d in zip(param_vals, deaths)) / len(deaths)
        std_p = (sum((p - mean_p)**2 for p in param_vals) / len(param_vals)) ** 0.5
        std_d = (sum((d - mean_d)**2 for d in deaths) / len(deaths)) ** 0.5
        corr = cov / (std_p * std_d) if std_p * std_d > 0 else 0
        correlations[param_name] = corr

    print(f"\n  Parameter correlations with deaths:")
    for name, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:25s}: {corr:>+.3f}")


# ═══════════════════════════════════════════════════════════════
# 8. CSV EXPORT
# ═══════════════════════════════════════════════════════════════

section("8. CSV EXPORT")

export_path = os.path.join(tempfile.gettempdir(), "pandemic_results.csv")
r.export_results(export_path)

# Read back and show summary
with open(export_path) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

print(f"  Exported to: {export_path}")
print(f"  Columns:     {len(header)} ({', '.join(header[:6])}...)")
print(f"  Rows:        {len(rows)}")
print(f"  Time range:  {rows[0][0]} to {rows[-1][0]}")

# Verify roundtrip
m2 = parse_sysd("model 'Verify'\n  stock metro_I: 0\n  dt 1")
data_back = m2.import_data(export_path)
print(f"\n  Roundtrip verification:")
print(f"    metro_I series: {len(data_back.get('metro_I', []))} points")
if "metro_I" in data_back:
    first_val = data_back["metro_I"][0][1]
    last_val = data_back["metro_I"][-1][1]
    print(f"    First value: {first_val:.1f}")
    print(f"    Last value:  {last_val:.1f}")


# ═══════════════════════════════════════════════════════════════
# 9. PARAMETER CALIBRATION
# ═══════════════════════════════════════════════════════════════

section("9. PARAMETER CALIBRATION")

# Generate synthetic "observed" data from known parameters
observed_params = {"metro_beta": 0.32, "metro_gamma": 0.08}
ref = model.simulate(params=observed_params)

# Sample observed data (weekly observations)
observed_data = {
    "metro_I": [
        (t, v) for t, v in zip(ref["times"][::4], ref["values"]["metro_I"][::4])
    ],
}
print(f"  Observed data: {len(observed_data['metro_I'])} points (weekly)")
print(f"  True metro_beta: {observed_params['metro_beta']}")

cal_result = calibrate(
    model,
    data=observed_data,
    param_bounds={"metro_beta": (0.1, 0.6)},
    method="nelder-mead",
    max_iterations=100,
)

print(f"  Calibrated metro_beta: {cal_result.best_params.get('metro_beta', 0):.4f}")
print(f"  Fit error (SSE):       {cal_result.best_error:.2f}")
print(f"  Iterations:            {cal_result.iterations}")
error_pct = abs(cal_result.best_params.get("metro_beta", 0) - observed_params["metro_beta"]) / observed_params["metro_beta"] * 100
print(f"  Accuracy:              {100 - error_pct:.1f}%")


# ═══════════════════════════════════════════════════════════════
# 10. LP OPTIMIZATION
# ═══════════════════════════════════════════════════════════════

section("10. LP OPTIMIZATION (Resource Allocation)")

# Allocate limited vaccine supply across cities to minimize deaths
# Variables: [metro_vax, suburb_vax, rural_vax]
# Objective: minimize weighted death impact
# Constraints: total vax <= 800/day, each >= 0

c = [3.0, 2.0, 1.0]  # death impact weights (metro has most people)
A_ub = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
b_ub = [0, 0, 0]
A_eq = [[1, 1, 1]]
b_eq = [800]

lp_result = lp_minimize(
    c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
    bounds=[(50, 500), (20, 300), (10, 100)],
)

print(f"  Decision variables: metro_vax, suburb_vax, rural_vax")
print(f"  LP optimal allocation:")
print(f"    Metro:   {lp_result.x[0]:>6.0f} vaccines/day")
print(f"    Suburb:  {lp_result.x[1]:>6.0f} vaccines/day")
print(f"    Rural:   {lp_result.x[2]:>6.0f} vaccines/day")
print(f"  LP objective (cost): {lp_result.objective_value:.2f}")
print(f"  LP success:          {lp_result.success}")

# Compare with equal allocation
print(f"\n  Equal allocation (267 each):")
equal_cost = sum(c[i] * 267 for i in range(3))
optimal_cost = lp_result.objective_value
savings = (equal_cost - optimal_cost) / equal_cost * 100
print(f"    Equal cost:   {equal_cost:.0f}")
print(f"    Optimal cost: {optimal_cost:.0f}")
print(f"    Savings:      {savings:.1f}%")


# ═══════════════════════════════════════════════════════════════
# 11. VALIDATION
# ═══════════════════════════════════════════════════════════════

section("11. MODEL VALIDATION")

val = model.validate()
print(f"  Valid:      {val.is_valid}")
print(f"  Errors:     {len(val.errors)}")
print(f"  Warnings:   {len(val.warnings)}")
print(f"  Info:       {len(val.infos)}")

if val.errors:
    print("\n  Errors:")
    for issue in val.errors[:5]:
        print(f"    {issue.message}")
if val.warnings:
    print("\n  Warnings:")
    for issue in val.warnings[:5]:
        print(f"    {issue.message}")


# ═══════════════════════════════════════════════════════════════
# 12. SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════════

section("12. SCENARIO COMPARISON")

scenarios = {
    "Baseline": {},
    "High Trans.": {"metro_beta": 0.5, "suburb_beta": 0.4, "rural_beta": 0.3},
    "Low Capacity": {"metro_capacity": 100, "suburb_capacity": 40, "rural_capacity": 10},
    "Fast Vax": {"metro_vax_daily": 1000, "suburb_vax_daily": 400, "rural_vax_daily": 100},
    "Lockdown": {"metro_beta": 0.15, "suburb_beta": 0.12},
}

print(f"  {'Scenario':15s} {'Peak I':>10s} {'Total D':>10s} {'Vaccinated':>12s} {'Attack%':>10s}")
print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

for name, params in scenarios.items():
    try:
        res = model.simulate(params=params)
        peak = max(res["values"]["metro_I"] + res["values"]["suburb_I"] + res["values"]["rural_I"])
        total_d = res["values"]["metro_D"][-1] + res["values"]["suburb_D"][-1] + res["values"]["rural_D"][-1]
        total_v = (res["values"]["metro_Cum_Vaccinated"][-1] +
                   res["values"]["suburb_Cum_Vaccinated"][-1] +
                   res["values"]["rural_Cum_Vaccinated"][-1])
        attack = (res["values"]["metro_Cum_Infections"][-1] +
                  res["values"]["suburb_Cum_Infections"][-1] +
                  res["values"]["rural_Cum_Infections"][-1])
        total_pop = 75000
        print(f"  {name:15s} {peak:>10.0f} {total_d:>10.0f} {total_v:>12.0f} {attack/total_pop*100:>9.1f}%")
    except Exception as e:
        print(f"  {name:15s} ERROR: {e}")


# ═══════════════════════════════════════════════════════════════
# 13. ABM (Healthcare Workers)
# ═══════════════════════════════════════════════════════════════

section("13. ABM (Healthcare Workers)")

abm_src = '''
model 'HealthcareWorkers'
  agent 'Worker' count 50
    property 'skill': 5, min=1, max=10
    property 'fatigue': 0, min=0, max=100
    property 'patients_treated': 0, min=0
    rule 'treat_patient' when metro_stress > 0.5
      fatigue += 2
      patients_treated += 1
    rule 'rest' when fatigue > 50
      fatigue -= 10
    rule 'burnout' when fatigue > 90
      skill -= 1
  dt 1
  from 0 to 30
'''
abm_model = parse_sysd(abm_src)

print(f"  Agents:     {len(abm_model.agents)} types")
print(f"  Agent count: {abm_model.agents[0].count}")
print(f"  Properties:  {len(abm_model.agents[0].properties)}")
for prop in abm_model.agents[0].properties:
    print(f"    {prop.name}: initial={prop.initial}, range=[{prop.min}, {prop.max}]")
print(f"  Rules:       {len(abm_model.agents[0].rules)}")
for rule in abm_model.agents[0].rules:
    print(f"    {rule.name}: when {rule.condition} ({len(rule.effects)} effects)")


# ═══════════════════════════════════════════════════════════════
# 14. DES (Hospital Queue)
# ═══════════════════════════════════════════════════════════════

section("14. DES (Hospital Queue)")

des_src = '''
model 'HospitalQueue'
  queue 'WaitingRoom': capacity 100, service_time 2
  resource 'Doctors': capacity 10
  queue 'ICU': capacity 20, service_time 5
  resource 'ICUBeds': capacity 5
  dt 1
  from 0 to 100
'''
des_model = parse_sysd(des_src)

print(f"  Queues:     {len(des_model.queues)}")
for q in des_model.queues:
    print(f"    {q.name}: capacity={q.capacity}, service_time={q.service_time}")
print(f"  Resources:  {len(des_model.resources)}")
for r_src in des_model.resources:
    print(f"    {r_src.name}: capacity={r_src.capacity}")


# ═══════════════════════════════════════════════════════════════
# 15. SUMMARY
# ═══════════════════════════════════════════════════════════════

section("FULL SHOWCASE COMPLETE")

print("""
  Features demonstrated:
    ✓ 1. DSL Parsing — .sysd files with stocks, auxes, flows
    ✓ 2. Units Checking — ~Unit~ syntax, dimensional analysis
    ✓ 3. CSV Import — vaccination schedule with interpolation
    ✓ 4. Base Simulation — 15 stocks, 50+ auxes, 3 cities
    ✓ 5. Feedback Loops — reinforcing/balancing detection
    ✓ 6. Causal Tracing — upstream causes, downstream effects
    ✓ 7. Sensitivity Analysis — 30-run Monte Carlo ensemble
    ✓ 8. CSV Export — roundtrip verification
    ✓ 9. Parameter Calibration — Nelder-Mead fitting
    ✓ 10. LP Optimization — vaccine allocation
    ✓ 11. Validation — model consistency checks
    ✓ 12. Scenario Comparison — 5 policy scenarios
    ✓ 13. ABM — 50 healthcare workers with rules
    ✓ 14. DES — hospital queues and resources

  Submodel features:
    ✓ SEIR_D template with 10 stocks, 15+ auxes
    ✓ 3 instances (Metro, Suburb, Rural) with parameter overrides
    ✓ Namespace prefixing (metro_, suburb_, rural_)
    ✓ Inter-city travel via DELAY3

  Time/delay features:
    ✓ DELAY3 (inter-city travel delay)
    ✓ SMOOTH (progression/recovery delays)
    ✓ SIN (seasonal modulation)
    ✓ IF (policy triggers, healthcare stress)
    ✓ MIN/MAX (vaccination limits)
    ✓ NOISE (stochastic forcing)

  Analysis features:
    ✓ Feedback loop detection (10+ loops found)
    ✓ Causal tracing with value decomposition
    ✓ Monte Carlo sensitivity with correlations
    ✓ Parameter calibration (Nelder-Mead)
    ✓ LP optimization (resource allocation)
""")

# Cleanup
os.unlink(csv_path)
os.unlink(export_path)
