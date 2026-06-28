"""SEIR Epidemic — .sysd DSL with seasonal forcing, capacity, and policy."""

from dynafx.system.dsl import parse_sysd

DSL = """
model 'SEIR Epidemic'
  dt 0.25
  from 0 to 365

  // Seasonal transmission: higher in winter
  table 'seasonality'
    x: [0, 90, 180, 270, 365]
    y: [0.8, 1.2, 1.5, 0.9, 0.8]

  stock 'Susceptible': 990000
    - 'Infection': Susceptible * Infected * 3e-7 * seasonality(t)

  stock 'Exposed': 0
    + 'Infection': Susceptible * Infected * 3e-7 * seasonality(t)
    - 'Incubation': Exposed * 0.2

  stock 'Infected': 10000
    + 'Incubation': Exposed * 0.2
    - 'Recovery': Infected * MIN(0.07, ICU_Beds / Infected)

  stock 'Recovered': 0
    + 'Recovery': Infected * MIN(0.07, ICU_Beds / Infected)

  // Endogenous capacity policy
  stock 'ICU_Beds': 20000
    + 'Expansion': IF(SMOOTH(Infected, 7) > 40000, 500, 50)
    - 'Phaseout': ICU_Beds * 0.008
"""


def main():
    model = parse_sysd(DSL)

    print("=" * 72)
    print(f"  {model.name}")
    print(f"  dt={model.dt}  |  t: {model.t_span[0]} → {model.t_span[1]}")
    print(f"  Stocks: {len(model.stocks)}  |  Tables: {len(model.tables)}")
    print()

    # Decompose into a graph and print structure
    decomposer = model.to_decomposer()
    g = decomposer.graph

    print("  Decomposition")
    for nid, node in g.nodes.items():
        meta = node.metadata
        p = meta.get("parameter")
        init = f" = {p.value}" if p else ""
        print(f"    [{node.type.name:>4s}]  {node.text}{init}")
    print()

    # Print causal edges
    for eid, edge in g.edges.items():
        src = g.nodes[edge.source_id].text
        tgt = g.nodes[edge.target_id].text
        pol = "+" if edge.polarity and edge.polarity > 0 else "−"
        print(f"      {src:>20s}  ──{pol}──→  {tgt}")
    print()

    # Simulate
    print("  Simulation (RK4)")
    print(f"  {'t':>6s}  {'Susceptible':>12s}  {'Exposed':>10s}  {'Infected':>10s}", end="")
    print(f"  {'Recovered':>10s}  {'ICU_Beds':>10s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    result = model.simulate(method="rk4")
    times = result["times"]
    vals = result["values"]
    stride = max(1, len(times) // 15)

    for i, t in enumerate(times):
        if i % stride != 0 and i != len(times) - 1:
            continue
        s = vals["Susceptible"][i]
        e = vals["Exposed"][i]
        inf = vals["Infected"][i]
        r = vals["Recovered"][i]
        icu = vals["ICU_Beds"][i]
        print(f"  {t:6.1f}  {s:12.1f}  {e:10.1f}  {inf:10.1f}  {r:10.1f}  {icu:10.0f}")

    print()
    print(f"  Final: S={result['final_state'][0]:.0f}", end="")
    print(f"  E={result['final_state'][1]:.0f}", end="")
    print(f"  I={result['final_state'][2]:.1f}", end="")
    print(f"  R={result['final_state'][3]:.0f}", end="")
    print(f"  ICU={result['final_state'][4]:.0f}")
    print(f"  Steps: {result['steps']}")
    print("=" * 72)

    # Plot if available
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(times, vals["Susceptible"], label="Susceptible")
        ax1.plot(times, vals["Exposed"], label="Exposed")
        ax1.plot(times, vals["Infected"], label="Infected")
        ax1.plot(times, vals["Recovered"], label="Recovered")
        ax1.set_ylabel("Population")
        ax1.set_title(model.name)
        ax1.legend()
        ax1.grid(True)

        ax2.plot(times, vals["ICU_Beds"], label="ICU Beds", color="red")
        ax2.set_xlabel("Time (days)")
        ax2.set_ylabel("Beds")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("examples/epidemic.png", dpi=150)
        print("  Plot → examples/epidemic.png")
    except ImportError:
        print("  (install matplotlib for plots)")


if __name__ == "__main__":
    main()
