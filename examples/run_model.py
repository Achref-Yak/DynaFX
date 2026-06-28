"""Decompose a .sysd model, inspect the graph, then simulate."""

import sys
from pathlib import Path

from dynafx.system.dsl import parse_sysd


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "predator_prey.sysd"
    model = parse_sysd(path.read_text())

    print("=" * 60)
    print(f"Model: {model.name}")
    print(f"  dt:       {model.dt}")
    print(f"  time:     {model.t_span[0]} → {model.t_span[1]}")
    print(f"  stocks:   {len(model.stocks)}")
    print(f"  tables:   {len(model.tables)}")
    print()

    # Decompose the system into a structured graph
    decomposer = model.to_decomposer()
    g = decomposer.graph

    print("─" * 60)
    print("Decomposition (graph nodes):")
    for nid, node in g.nodes.items():
        meta = node.metadata
        p = meta.get("parameter")
        init = f" = {p.value}" if p else ""
        print(f"  [{node.type.name:>4s}] {node.text}{init}")
    print()

    print("─" * 60)
    print("Causal edges:")
    for eid, edge in g.edges.items():
        source = g.nodes[edge.source_id].text
        target = g.nodes[edge.target_id].text
        pol = "+" if edge.polarity and edge.polarity > 0 else "−"
        print(f"  {source} ──[{pol}]──→ {target}")
    print()

    print("─" * 60)
    print("Simulating with RK4...")
    result = model.simulate(method="rk4")

    # Compact table: print every Nth step
    times = result["times"]
    stocks = result["stocks"]
    n = max(1, len(times) // 15)  # ~15 rows

    header = f"{'t':>5s}  " + "  ".join(f"{s:>12s}" for s in stocks)
    print(header)
    print("-" * len(header))
    for i, t in enumerate(times):
        if i % n == 0 or i == len(times) - 1:
            vals = "  ".join(f"{result['values'][s][i]:12.1f}" for s in stocks)
            print(f"{t:5.1f}  {vals}")
    print()

    print(f"Final state:  Prey={result['final_state'][0]:.1f}  Predators={result['final_state'][1]:.1f}")
    print(f"Steps taken:  {result['steps']}")

    # Plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        for s in stocks:
            plt.plot(times, result["values"][s], label=s)
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.title(f"{model.name}  (dt={model.dt})")
        plt.legend()
        plt.grid(True)
        out = path.with_suffix(".png")
        plt.savefig(out, dpi=150)
        print(f"Plot saved  → {out}")
    except ImportError:
        print("(install matplotlib to generate plots)")


if __name__ == "__main__":
    main()
