"""
Seasonal Epidemic Example
=========================
Demonstrates every Phase 1 feature: SIN, COS, PI, PULSE, STEP, RAMP,
NOISE, SMOOTHI, SMOOTH, IF, MIN, MAX, and parameterized simulation.

Run: python examples/seasonal_epidemic.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TMPDIR", "/tmp")

from cognitive_engine.system.dsl import parse_sysd_file, parse_sysd


# ── 1. Parse the model ─────────────────────────────────────────

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "seasonal_epidemic_example.sysd")
model = parse_sysd_file(model_path)

print("=" * 60)
print("SEASONAL EPIDEMIC MODEL")
print("=" * 60)
print(f"Stocks: {[s.name for s in model.stocks]}")
print(f"Auxes:  {[a.name for a in model.aux_vars]}")
print(f"t_span: {model.t_span}, dt: {model.dt}")
print()
print("Features used:")
print("  SIN, COS, PI   — seasonal forcing on contact rate")
print("  RAMP            — lockdown easing from day 30-60")
print("  STEP            — mask mandate from day 10")
print("  PULSE           — three vaccination campaigns")
print("  NOISE           — stochastic contact rate variation")
print("  IF              — age-dependent death rate")
print("  MIN             — cap vaccination at susceptible count")
print()
print("Default parameters:")
DEFAULT_PARAMS = {
    "population": 10000, "base_contact_rate": 0.5,
    "recovery_time": 14, "vaccine_efficacy": 0.9,
    "death_rate": 0.01, "noise_amplitude": 0.02,
    "vax_campaign_size": 500,
}
for k, v in DEFAULT_PARAMS.items():
    print(f"  {k}: {v}")


# ── 2. Simulate ────────────────────────────────────────────────

print("\n--- Default Simulation ---")
r = model.simulate(params=DEFAULT_PARAMS)

peak_day = r.times[r.values["Infected"].index(max(r.values["Infected"]))]
peak_infected = max(r.values["Infected"])

print(f"  Peak infection day:  {peak_day:.0f}")
print(f"  Peak infected:       {peak_infected:.0f}")
print(f"  Final Susceptible:   {r.values['Susceptible'][-1]:.0f}")
print(f"  Final Recovered:     {r.values['Recovered'][-1]:.0f}")
print(f"  Final Vaccinated:    {r.values['Vaccinated'][-1]:.0f}")


# ── 3. Sensitivity: contact rate ──────────────────────────────

print("\n--- Sensitivity: Contact Rate ---")
for rate in [0.15, 0.3, 0.6]:
    p = {**DEFAULT_PARAMS, "base_contact_rate": rate}
    r = model.simulate(params=p)
    peak = max(r.values["Infected"])
    final_vax = r.values["Vaccinated"][-1]
    print(f"  rate={rate:.2f}: peak={peak:>5.0f}, vax={final_vax:.0f}")


# ── 4. Sensitivity: vaccine efficacy ──────────────────────────

print("\n--- Sensitivity: Vaccine Efficacy ---")
for eff in [0.5, 0.9, 1.0]:
    p = {**DEFAULT_PARAMS, "vaccine_efficacy": eff}
    r = model.simulate(params=p)
    peak = max(r.values["Infected"])
    final_vax = r.values["Vaccinated"][-1]
    print(f"  efficacy={eff:.1f}: peak={peak:>5.0f}, vax={final_vax:.0f}")


# ── 5. Math functions (analytical check) ──────────────────────

print("\n" + "=" * 60)
print("MATH FUNCTIONS (analytical verification)")
print("=" * 60)
model_math = parse_sysd("""
stock A: 0
  + growth: ABS(-5) * 2
stock B: 0
  + growth: SQRT(144) * 3
stock C: 0
  + growth: EXP(LN(2)) * 10
stock D: 0
  + growth: SIN(PI/2) + COS(0)
""")
r = model_math.simulate(t_span=(0, 1), dt=1.0)
print(f"  ABS(-5)*2          = {r.values['A'][-1]:>5.1f}  (expect  10.0)")
print(f"  SQRT(144)*3        = {r.values['B'][-1]:>5.1f}  (expect  36.0)")
print(f"  EXP(LN(2))*10      = {r.values['C'][-1]:>5.1f}  (expect  20.0)")
print(f"  SIN(PI/2)+COS(0)   = {r.values['D'][-1]:>5.2f}  (expect   2.00)")


# ── 6. Time functions ─────────────────────────────────────────

print("\n" + "=" * 60)
print("TIME FUNCTIONS (PULSE, STEP, RAMP)")
print("=" * 60)
model_time = parse_sysd("""
stock X: 0
  + inflow: PULSE(100, 10, 5) + STEP(50, 30) + RAMP(10, 50, 80)
""")
r = model_time.simulate(t_span=(0, 100), dt=1.0)
print(f"  t=5:   X={r.values['X'][5]:>6.0f}  (before all triggers)")
print(f"  t=12:  X={r.values['X'][12]:>6.0f}  (PULSE active: +100)")
print(f"  t=15:  X={r.values['X'][15]:>6.0f}  (PULSE ended)")
print(f"  t=35:  X={r.values['X'][35]:>6.0f}  (STEP active: +50/tick)")
print(f"  t=65:  X={r.values['X'][65]:>6.0f}  (RAMP active: +10*t)")
print(f"  t=100: X={r.values['X'][-1]:>6.0f}  (final)")


# ── 7. SMOOTHI: custom initial value ─────────────────────────

print("\n" + "=" * 60)
print("SMOOTHI vs SMOOTH (convergence behavior)")
print("=" * 60)
smooth_path = os.path.join(os.path.dirname(__file__), "..", "models", "smooth_demo.sysd")
model_smooth = parse_sysd_file(smooth_path)
r = model_smooth.simulate(t_span=(0, 50), dt=0.5)

idx20 = r.times.index(20.0)
idx30 = r.times.index(30.0)
idx40 = r.times.index(40.0)

print(f"  Source drops from 100 to 90 at t=20 via STEP(10, 20)")
print()
print(f"  At t=20 (step happens):")
print(f"    Source={r.values['Source'][idx20]:.1f}, "
      f"Smooth(10)={r.values['Smooth10'][idx20]:.1f}, "
      f"SMOOTHI(5,init=100)={r.values['Smoothi_fast'][idx20]:.1f}, "
      f"SMOOTHI(20,init=0)={r.values['Smoothi_slow'][idx20]:.1f}")
print(f"  At t=30:")
print(f"    Source={r.values['Source'][idx30]:.1f}, "
      f"Smooth(10)={r.values['Smooth10'][idx30]:.1f}, "
      f"SMOOTHI(5,init=100)={r.values['Smoothi_fast'][idx30]:.1f}, "
      f"SMOOTHI(20,init=0)={r.values['Smoothi_slow'][idx30]:.1f}")
print(f"  At t=40:")
print(f"    Source={r.values['Source'][idx40]:.1f}, "
      f"Smooth(10)={r.values['Smooth10'][idx40]:.1f}, "
      f"SMOOTHI(5,init=100)={r.values['Smoothi_fast'][idx40]:.1f}, "
      f"SMOOTHI(20,init=0)={r.values['Smoothi_slow'][idx40]:.1f}")


# ── 8. NOISE bounded check ────────────────────────────────────

print("\n" + "=" * 60)
print("NOISE (stochastic bounded randomness)")
print("=" * 60)
model_noise = parse_sysd("""
stock X: 0
  + inflow: NOISE(1)
""")
r1 = model_noise.simulate(t_span=(0, 10), dt=0.1)
r2 = model_noise.simulate(t_span=(0, 10), dt=0.1)
print(f"  Run 1: X(10) = {r1.values['X'][-1]:>+.2f}")
print(f"  Run 2: X(10) = {r2.values['X'][-1]:>+.2f}")
print(f"  Both bounded in [-10, +10]: "
      f"{-10.1 < r1.values['X'][-1] < 10.1 and -10.1 < r2.values['X'][-1] < 10.1}")


# ── 9. Combined: IF with time-dependent threshold ────────────

print("\n" + "=" * 60)
print("IF + TIME FUNCTIONS (adaptive response)")
print("=" * 60)
model_adaptive = parse_sysd("""
stock Level: 0
  + inflow: 10
  - outflow: IF(Level > 50, 15, IF(Level > 20, 8, 3))
""")
r = model_adaptive.simulate(t_span=(0, 20), dt=1.0)
for t_check in [0, 5, 10, 15, 20]:
    idx = r.times.index(t_check)
    lev = r.values["Level"][idx]
    if lev <= 20:
        outflow = 3
    elif lev <= 50:
        outflow = 8
    else:
        outflow = 15
    print(f"  t={t_check:>2}: Level={lev:>5.1f}, outflow={outflow}, net={'+' if 10-outflow >= 0 else ''}{10-outflow}")


print("\n" + "=" * 60)
print("Done. All Phase 1 features demonstrated.")
print("=" * 60)
