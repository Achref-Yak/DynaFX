"""
SEIRV-H Pandemic Model — Full Pipeline Demo
============================================

Demonstrates every feature of the .sysd toolkit:
  1. Load a complex 7-stock model from .sysd
  2. Validate with parameter awareness
  3. Run baseline simulation
  4. Compare counterfactuals (no vaccination vs early vaccination)
  5. Plot results
  6. Run ensemble sensitivity analysis
  7. Print structured diagnostics

Usage:
    python examples/pandemic_pipeline.py
"""

import os
import sys

# ── Ensure the package is importable ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynafx.system.dsl import parse_sysd_file

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pandemic_seirvh.sysd")

BASELINE_PARAMS = {
    "N": 10_000_000,
    "R0": 3.0,
    "incubation_period": 5.2,
    "infectious_period": 7.0,
    "severe_fraction": 0.07,
    "base_mortality": 0.03,
    "overload_multiplier": 2.0,
    "initial_capacity": 5000,
    "capacity_expansion_rate": 25,
    "discharge_time": 14,
    "icu_mortality": 0.15,
    "vax_start": 90,
    "vax_rate_max": 8000,
    "attrition_rate": 0.002,
}

NO_VAX_PARAMS = {**BASELINE_PARAMS, "vax_rate_max": 0, "vax_start": 999}

EARLY_VAX_PARAMS = {**BASELINE_PARAMS, "vax_start": 40, "vax_rate_max": 15000}

LOW_R0_PARAMS = {**BASELINE_PARAMS, "R0": 1.8}

LOW_CAPACITY_PARAMS = {**BASELINE_PARAMS, "initial_capacity": 1000}


def step_1_load_and_validate():
    print("=" * 60)
    print("STEP 1: Load & Validate")
    print("=" * 60)

    model = parse_sysd_file(MODEL_PATH)
    print(f"  Model: {model.name}")
    print(f"  Stocks ({len(model.stocks)}): {[s.name for s in model.stocks]}")
    print(f"  Aux vars ({len(model.aux_vars)}): {[a.name for a in model.aux_vars]}")
    print(f"  Tables ({len(model.tables)}): {[t.name for t in model.tables]}")
    print(f"  dt={model.dt}, t_span={model.t_span}")

    vr = model.validate(params=set(BASELINE_PARAMS.keys()))
    print(f"  Valid: {'YES' if vr.is_valid else 'NO'}")
    if vr.errors:
        for e in vr.errors:
            print(f"    ERROR: {e.message}")
    for w in vr.warnings[:3]:
        print(f"    WARNING: {w.message}")
    if len(vr.warnings) > 3:
        print(f"    ... and {len(vr.warnings) - 3} more warnings")

    return model


def step_2_baseline_simulation(model):
    print()
    print("=" * 60)
    print("STEP 2: Baseline Simulation (100 days, dt=0.25, RK4)")
    print("=" * 60)

    result = model.simulate(method="rk4", params=BASELINE_PARAMS, t_span=(0.0, 100.0))

    header = f"{'Day':>6}"
    for name in result.stocks:
        header += f" {name:>20}"
    print(header)
    print("-" * len(header))

    for i in range(0, len(result.times), 20):
        line = f"{result.times[i]:6.0f}"
        for name in result.stocks:
            line += f" {result.values[name][i]:>20.0f}"
        print(line)

    # Key metrics
    peak_infected = max(result.values["Infected"])
    peak_day = result.times[result.values["Infected"].index(peak_infected)]
    peak_hospitalized = max(result.values["Hospitalized"])
    total_fatalities = result.final_state[result.stocks.index("Fatalities")]

    print()
    print(f"  Peak Infected:       {peak_infected:>10,.0f} on day {peak_day:.0f}")
    print(f"  Peak Hospitalized:   {peak_hospitalized:>10,.0f}")
    print(f"  Total Fatalities:    {total_fatalities:>10,.0f}")
    return result


def step_3_counterfactuals(model):
    print()
    print("=" * 60)
    print("STEP 3: Counterfactual Comparison")
    print("=" * 60)

    no_vax = model.simulate(method="rk4", params=NO_VAX_PARAMS)
    baseline = model.simulate(method="rk4", params=BASELINE_PARAMS)
    low_r0 = model.simulate(method="rk4", params=LOW_R0_PARAMS)
    low_cap = model.simulate(method="rk4", params=LOW_CAPACITY_PARAMS)

    f_idx = baseline.stocks.index("Fatalities")

    def _peak_stress(r):
        return max(
            r.values["Hospitalized"][i] / max(1, r.values["Healthcare_Capacity"][i])
            for i in range(len(r.times))
        )

    scenarios = [
        ("No vaccination", no_vax),
        ("Baseline (vax day 90)", baseline),
        ("Low R0 = 1.8", low_r0),
        ("Low initial capacity (1000)", low_cap),
    ]

    print(f"  {'Scenario':<30} {'Peak Infected':>15} {'Fatalities':>15} {'Max Stress':>12}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*12}")
    for label, r in scenarios:
        peak = max(r.values["Infected"])
        fatal = r.final_state[f_idx]
        stress = _peak_stress(r)
        print(f"  {label:<30} {peak:>15,.0f} {fatal:>15,.0f} {stress:>11.1f}x")

    return {"no_vax": no_vax, "baseline": baseline, "low_r0": low_r0, "low_capacity": low_cap}


def step_4_sensitivity(model):
    print()
    print("=" * 60)
    print("STEP 4: Sensitivity — R0 uncertainty (uniform 2.5–3.5)")
    print("=" * 60)

    ens = model.simulate_ensemble(
        params={
            "R0": (2.5, 3.5, "uniform"),
            "capacity_expansion_rate": (10, 50, "uniform"),
        },
        fixed_params={
            "N": 10_000_000,
            "incubation_period": 5.2,
            "infectious_period": 7.0,
            "severe_fraction": 0.07,
            "base_mortality": 0.03,
            "overload_multiplier": 2.0,
            "initial_capacity": 5000,
            "discharge_time": 14,
            "icu_mortality": 0.15,
            "vax_start": 90,
            "vax_rate_max": 8000,
            "attrition_rate": 0.002,
        },
        n=20,
        method="rk4",
        seed=42,
    )

    f_idx = ens["stocks"].index("Fatalities")
    final_fatalities = [
        t.final_state[f_idx] for t in ens["trajectories"]
    ]
    mean_fatal = ens["mean"]["Fatalities"][-1]
    p5_fatal = ens["p5"]["Fatalities"][-1]
    p95_fatal = ens["p95"]["Fatalities"][-1]
    spread = (p95_fatal - p5_fatal) / mean_fatal

    print(f"  Ensemble size:        {len(ens['trajectories'])}")
    print(f"  Mean fatalities:      {mean_fatal:>10,.0f}")
    print(f"  5th percentile:       {p5_fatal:>10,.0f}")
    print(f"  95th percentile:      {p95_fatal:>10,.0f}")
    print(f"  Relative spread:      {spread:.1%}")

    peak_infected_vals = sorted(max(t.values["Infected"]) for t in ens["trajectories"])
    mean_peak = sum(peak_infected_vals) / len(peak_infected_vals)

    print(f"\n  Infection peak sensitivity:")
    print(f"    Mean peak:  {mean_peak:>10,.0f}")
    print(f"    Range:      {peak_infected_vals[0]:>10,.0f} – {peak_infected_vals[-1]:>10,.0f}")

    return ens


def step_5_plot(model, results, ensemble):
    print()
    print("=" * 60)
    print("STEP 5: Plotting")
    print("=" * 60)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    # Baseline trajectory
    path1 = os.path.join(out_dir, "pandemic_trajectory.png")
    results["baseline"].plot(
        path1,
        stocks=["Infected", "Hospitalized", "Healthcare_Capacity", "Fatalities"],
        title="SEIRV-H Pandemic — Baseline",
    )
    print(f"  Trajectory plot → {path1}")

    # Subplots
    path2 = os.path.join(out_dir, "pandemic_subplots.png")
    results["baseline"].plot(path2, subplots=True, title="SEIRV-H — Per-Stock View")
    print(f"  Subplot view   → {path2}")

    # Sensitivity bands
    path3 = os.path.join(out_dir, "pandemic_sensitivity.png")
    results["baseline"].plot_with_bands(
        path3,
        mean=ensemble["mean"],
        std=ensemble["std"],
        p5=ensemble["p5"],
        p95=ensemble["p95"],
    )
    print(f"  Sensitivity    → {path3}")

    print(f"\n  Plots saved to {out_dir}/")


def step_6_full_run_cli(model):
    print()
    print("=" * 60)
    print("STEP 6: CLI Equivalent")
    print("=" * 60)

    cli_sim = (
        "python -m dynafx.system simulate models/pandemic_seirvh.sysd "
        "--param N=10000000 --param R0=3.0 --param incubation_period=5.2 "
        "--param infectious_period=7.0 --param severe_fraction=0.07 "
        "--param base_mortality=0.03 --param vax_start=90 "
        "--param vax_rate_max=8000 --t-end 100 --output output/cli_plot.png --csv output/cli_data.csv"
    )
    print(f"  {cli_sim}")
    print()
    cli_val = "python -m dynafx.system validate models/pandemic_seirvh.sysd"
    print(f"  {cli_val}")
    print()
    print("  (CLI commands require manual execution with params)")

    # Run validate via Python API
    print()
    vr = model.validate(params=set(BASELINE_PARAMS.keys()))
    print(f"  Validation via API: {'PASS' if vr.is_valid else 'FAIL'}")


# ── Main ────────────────────────────────────────────────────────

def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  SEIRV-H Pandemic Model — Full Pipeline Demo          ║")
    print("╚" + "═" * 58 + "╝")
    print()

    model = step_1_load_and_validate()
    baseline = step_2_baseline_simulation(model)
    counterfactuals = step_3_counterfactuals(model)
    ensemble = step_4_sensitivity(model)
    step_5_plot(model, counterfactuals, ensemble)
    step_6_full_run_cli(model)

    print()
    print("─" * 60)
    print("All pipeline steps completed successfully.")
    print("─" * 60)


if __name__ == "__main__":
    main()
