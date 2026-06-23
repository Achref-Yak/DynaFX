"""Example: Pandemic Response Model — SD + ABM + DES

Demonstrates how to use the cognitive_engine.system DSL to simulate
a multi-paradigm model with system dynamics, agent-based modeling,
and discrete event simulation.

Usage:
    python examples/pandemic_response.py
"""

from cognitive_engine.system.dsl import parse_sysd_file


def main() -> None:
    # ── 1. Parse the model ───────────────────────────────────
    model = parse_sysd_file("models/pandemic_response.sysd")

    print(f"Model: {model.name}")
    print(f"  Stocks:  {len(model.stocks)}")
    print(f"  Agents:  {sum(a.count for a in model.agents)} across {len(model.agents)} types")
    print(f"  Queues:  {len(model.queues)}")
    print(f"  Resources: {len(model.resources)}")
    print()

    # ── 2. Define parameters ─────────────────────────────────
    params = {
        # Epidemiological
        "beta": 0.3,               # transmission rate
        "gamma": 0.1,              # recovery rate (1/10 days)
        "mu": 0.02,                # baseline hospital fatality
        "vax_rate": 5000,          # vaccinations per day
        "hospital_rate": 0.05,     # fraction needing hospital
        "avg_hospital_stay": 7,    # days
        "population": 1_000_000,   # city size
    }

    # ── 3. Run simulation ────────────────────────────────────
    result = model.simulate(params=params)

    print(f"Simulation: {result.method}, {result.steps} steps")
    print()

    # ── 4. Examine SD results ────────────────────────────────
    print("=== System Dynamics — Final Population ===")
    for name in result.stocks:
        final = result.values[name][-1]
        print(f"  {name:>15}: {final:>12,.0f}")
    print()

    # Peak infection timing
    infected = result.values["Infected"]
    peak_idx = max(range(len(infected)), key=lambda i: infected[i])
    peak_time = result.times[peak_idx]
    peak_val = infected[peak_idx]
    print(f"  Peak infection: {peak_val:,.0f} at day {peak_time:.1f}")
    print()

    # ── 5. Examine ABM results ───────────────────────────────
    print("=== Agent-Based Model — Behavioral Cohorts ===")
    if result.abm_engine:
        agents_by_type: dict[str, list] = {}
        for inst in result.abm_engine.instances:
            agents_by_type.setdefault(inst.agent_def.name, []).append(inst)

        for type_name, instances in agents_by_type.items():
            n = len(instances)
            vaccinated = sum(1 for i in instances if i.state.get("vaccinated", 0) > 0)
            hospitalized = sum(1 for i in instances if i.state.get("hospitalized", 0) > 0)
            avg_symptoms = sum(i.state.get("symptom_score", 0) for i in instances) / n
            print(f"  {type_name}:")
            print(f"    Agents: {n}")
            print(f"    Vaccinated: {vaccinated}/{n} ({100*vaccinated/n:.0f}%)")
            print(f"    Hospitalized: {hospitalized}/{n}")
            print(f"    Avg symptom score: {avg_symptoms:.1f}")
    print()

    # ── 6. Examine DES results ───────────────────────────────
    print("=== Discrete Event Simulation — Hospital Resources ===")
    if result.des_engine:
        for name, stats in result.des_engine.get_all_stats().items():
            if isinstance(stats, dict):
                cap = stats.get("capacity", "N/A")
                util = stats.get("utilization", 0)
                arrivals = stats.get("total_arrivals", 0)
                print(f"  {name}:")
                print(f"    Capacity: {cap}")
                print(f"    Utilization: {util:.1%}")
                if arrivals:
                    print(f"    Total arrivals: {arrivals}")
    print()

    # ── 7. Sensitivity analysis ──────────────────────────────
    print("=== Sensitivity: Varying beta (transmission rate) ===")
    ensemble = model.simulate_ensemble(
        params={
            "beta": (0.15, 0.45),  # vary beta from 0.15 to 0.45
        },
        fixed_params=params,
        n=2,
        seed=42,
    )
    print(f"  Ran {len(ensemble['trajectories'])} ensemble members")
    print(f"  Mean final infected: {ensemble['mean']['Infected'][-1]:,.0f}")
    print(f"  Std final infected:  {ensemble['std']['Infected'][-1]:,.0f}")
    print(f"  5th percentile dead: {ensemble['p5']['Dead'][-1]:,.0f}")
    print(f"  95th percentile dead: {ensemble['p95']['Dead'][-1]:,.0f}")
    print()

    # ── 8. Plot ──────────────────────────────────────────────
    output_path = "pandemic_response.png"
    result.plot(output_path, stocks=["Infected", "Recovered", "Dead", "Vaccinated"])
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
