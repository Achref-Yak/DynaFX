"""Port-Warehouse supply chain — .sysd DSL with capacity, backlog, and policy."""

from dynafx.dynamics.dsl import parse_sysd, LookupTable

DSL = """
model 'Port-Warehouse Supply Chain'
  dt 0.25
  from 0 to 200

  // Seasonal customer demand (units/day)
  table 'demand'
    x: [0, 50, 100, 120, 140, 160, 200]
    y: [500, 800, 1100, 1300, 1200, 900, 600]

  // ── Shipping pipeline ────────────────────────────────
  // Goods in transit from overseas suppliers
  stock 'Transit': 6000
    // Order rate: expected demand + inventory gap (non-negative), capped
    + 'Orders': MIN(2000, MAX(0, SMOOTH(demand(t), 5) + (5000 - Warehouse_Stock) / 5))
    // Port receiving capacity: 1500 units/day (can't drain more than available)
    - 'Arrivals': MIN(1500, MAX(0, Transit) / dt)

  // ── Port buffer ──────────────────────────────────────
  stock 'Port_Stock': 1500
    + 'Arrivals': MIN(1500, MAX(0, Transit) / dt)
    // Trucking bottleneck: 1100/day (below peak demand — deliberate)
    - 'Trucked': MIN(1100, MAX(0, Port_Stock) / dt)

  // ── Warehouse ────────────────────────────────────────
  stock 'Warehouse_Stock': 5000
    + 'Trucked': MIN(1100, MAX(0, Port_Stock) / dt)
    // Fulfill demand + clear backlog at 1/5 per day
    - 'Shipped': MIN(MAX(0, Warehouse_Stock) / dt, MAX(0, demand(t) + Backlog / 5))

  // ── Backlog ──────────────────────────────────────────
  stock 'Backlog': 0
    + 'Orders_Placed': demand(t)
    // Same quantity as warehouse outflow (conservation)
    - 'Shipped': MIN(MAX(0, Warehouse_Stock) / dt, MAX(0, demand(t) + Backlog / 5))
"""


def main():
    model = parse_sysd(DSL)

    print("=" * 72)
    print(f"  {model.name}")
    print(f"  dt={model.dt}  |  t: {model.t_span[0]} → {model.t_span[1]}")
    print(f"  Stocks: {len(model.stocks)}  |  Tables: {len(model.tables)}")
    print()

    # ── Decomposition ──
    decomposer = model.to_decomposer()
    g = decomposer.graph

    print("  Stocks / Flows")
    for nid, node in g.nodes.items():
        meta = node.metadata
        p = meta.get("parameter")
        t = "  " if node.type.name == "STOCK" else "    "
        init = f"  = {p.value}" if p else ""
        print(f"    [{node.type.name:>4s}]{t}{node.text}{init}")
    print()

    print("  Causal edges")
    for eid, edge in g.edges.items():
        src = g.nodes[edge.source_id].text
        tgt = g.nodes[edge.target_id].text
        pol = "+" if edge.polarity and edge.polarity > 0 else "−"
        print(f"    {src:>20s}  ──{pol}──→  {tgt}")
    print()

    # ── Simulate ──
    print("  Simulation (RK4)")
    hdr = f"  {'t':>5s}  {'Transit':>9s}  {'Port':>8s}  {'WH':>8s}  {'Backlog':>8s}  {'Demand':>8s}"
    print(hdr)
    print(f"  {'─'*5}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    result = model.simulate(method="rk4")
    times = result["times"]
    vals = result["values"]
    stride = max(1, len(times) // 18)

    dtbl = LookupTable([0, 50, 100, 120, 140, 160, 200],
                       [500, 800, 1100, 1300, 1200, 900, 600])

    for i, t in enumerate(times):
        if i % stride != 0 and i != len(times) - 1:
            continue
        print(f"  {t:5.1f}  {vals['Transit'][i]:9.0f}  {vals['Port_Stock'][i]:8.0f}"
              f"  {vals['Warehouse_Stock'][i]:8.0f}  {vals['Backlog'][i]:8.0f}"
              f"  {dtbl(t):8.0f}")

    print()
    print(f"  Steps: {result['steps']}")
    print("=" * 72)

    # ── Plot ──
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        ax1.plot(times, vals["Transit"], label="Transit (ships)", lw=1.5)
        ax1.plot(times, vals["Port_Stock"], label="Port Stock", lw=1.5)
        ax1.plot(times, vals["Warehouse_Stock"], label="Warehouse", lw=1.5)
        ax1.plot(times, vals["Backlog"], label="Backlog", lw=2, ls="--")
        ax1.set_ylabel("Units")
        ax1.set_title("Port-Warehouse Supply Chain (bullwhip effect)")
        ax1.legend()
        ax1.grid(True)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        demand_series = [dtbl(t) for t in times]
        ax2.plot(times, demand_series, label="Customer Demand", color="green", lw=2)
        ax2.axhline(y=1100, color="gray", ls=":", label="Truck capacity (1100/day)")
        ax2.set_xlabel("Time (days)")
        ax2.set_ylabel("Units / day")
        ax2.legend()
        ax2.grid(True)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        plt.tight_layout()
        plt.savefig("examples/logistics.png", dpi=150)
        print("  Plot → examples/logistics.png")
    except ImportError:
        print("  (install matplotlib for plots)")


if __name__ == "__main__":
    main()
