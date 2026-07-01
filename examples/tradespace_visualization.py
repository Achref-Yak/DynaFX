#!/usr/bin/env python3
"""Tradespace visualization — Pareto frontier analysis for logistics fleet.

Sweeps two parameters across a grid, collects objective metrics
(end_cash, max_delivery_time_ratio), computes the Pareto frontier
of non-dominated solutions, and produces a publication-ready figure
with scatter + trajectory panels.

Sweeps:
  productivity  — driver_productivity (1.5-5.0) × salary_per_driver (80-200)
  market       — revenue_per_delivery (180-320) × customer_sensitivity (0.03-0.25)

Usage:
    python examples/tradespace_visualization.py
    python examples/tradespace_visualization.py --sweep market
    python examples/tradespace_visualization.py --fast
    python examples/tradespace_visualization.py --n 10 --output my_fig.png
"""

import sys
import os
import io
import itertools
import argparse
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

from dynafx.dynamics.dsl import parse_sysd_file

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "logistics_fleet.sysd")

FIXED_PARAMS = dict(
    initial_customers=3000, order_base=180, market_size=15000,
    revenue_per_delivery=250, fuel_cost_per_km=0.40, avg_daily_km=150,
    truck_cost=120000, truck_lifetime=3650, acquisition_time=120,
    fleet_productivity=3.0, driver_productivity=3.0,
    salary_per_driver=120, warehouse_operating_cost=5000,
    target_delivery_time=3, customer_sensitivity=0.10,
    initial_cash=3000000,
    maintenance_time=2, mechanic_count=5,
)

SWEEPS = {
    "productivity": {
        "title": "Driver Productivity vs Salary \u2014 Pareto Frontier",
        "param_x": "driver_productivity",
        "range_x": (1.5, 5.0),
        "param_y": "salary_per_driver",
        "range_y": (80, 200),
        "x_label": "Max delivery time ratio (lower is better)",
        "y_label": "End cash ($) (higher is better)",
        "z_metric": "end_fleet",
        "z_label": "End fleet",
        "trajectory_stock": "Cash_Reserves",
    },
    "market": {
        "title": "Revenue vs Churn Sensitivity \u2014 Pareto Frontier",
        "param_x": "revenue_per_delivery",
        "range_x": (180, 320),
        "param_y": "customer_sensitivity",
        "range_y": (0.03, 0.25),
        "x_label": "Max delivery time ratio (lower is better)",
        "y_label": "End cash ($) (higher is better)",
        "z_metric": "end_customers",
        "z_label": "End customers",
        "trajectory_stock": "Cash_Reserves",
    },
}


@dataclass
class RunResult:
    params: dict
    end_cash: float
    max_dtr: float
    end_fleet: float
    max_backlog: float
    end_customers: float
    min_cash: float
    succeeded: bool = True
    error: str = ""


def run_single(model, params):
    try:
        r = model.simulate(params=params, method="euler")
        return RunResult(
            params=dict(params),
            end_cash=r.values["Cash_Reserves"][-1],
            max_dtr=max(r.aux_values["delivery_time_ratio"]),
            end_fleet=r.values["Fleet"][-1],
            max_backlog=max(r.values["Orders_Backlog"]),
            end_customers=r.values["Customers"][-1],
            min_cash=min(r.values["Cash_Reserves"]),
        )
    except Exception as e:
        return RunResult(
            params=dict(params),
            end_cash=float("-inf"),
            max_dtr=float("inf"),
            end_fleet=0,
            max_backlog=0,
            end_customers=0,
            min_cash=0,
            succeeded=False,
            error=str(e),
        )


def run_grid(model, sweep, n=7):
    x_vals = np.linspace(sweep["range_x"][0], sweep["range_x"][1], n)
    y_vals = np.linspace(sweep["range_y"][0], sweep["range_y"][1], n)
    total = n * n
    results = []
    for idx, (xv, yv) in enumerate(itertools.product(x_vals, y_vals), 1):
        p = dict(FIXED_PARAMS)
        p[sweep["param_x"]] = round(float(xv), 4)
        p[sweep["param_y"]] = round(float(yv), 4)
        sys.stdout.write(
            f"\r  {idx:3d}/{total}  {sweep['param_x']}={xv:.2f}  {sweep['param_y']}={yv:.3f}"
        )
        sys.stdout.flush()
        results.append(run_single(model, p))
    print()
    return results


def pareto_front(results, higher_better=("end_cash",), lower_better=("max_dtr",)):
    n = len(results)
    front = []
    for i in range(n):
        if not results[i].succeeded:
            continue
        dominated = False
        for j in range(n):
            if i == j or not results[j].succeeded:
                continue
            strictly_better = False
            at_least_as_good = True
            for key in higher_better:
                if getattr(results[j], key) > getattr(results[i], key):
                    strictly_better = True
                elif getattr(results[j], key) < getattr(results[i], key):
                    at_least_as_good = False
            for key in lower_better:
                if getattr(results[j], key) < getattr(results[i], key):
                    strictly_better = True
                elif getattr(results[j], key) > getattr(results[i], key):
                    at_least_as_good = False
            if strictly_better and at_least_as_good:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def plot_tradespace(results, front_indices, sweep, model, n=7):
    cash = np.array([r.end_cash for r in results])
    dtr = np.array([r.max_dtr for r in results])
    ok = np.array([r.succeeded for r in results])
    z = np.array([getattr(r, sweep["z_metric"]) for r in results])

    dom = [i for i in range(len(results)) if i not in front_indices and ok[i]]
    failed = [i for i in range(len(results)) if not ok[i]]

    p_results = [results[i] for i in front_indices]
    p_cash = np.array([r.end_cash for r in p_results])
    p_dtr = np.array([r.max_dtr for r in p_results])

    srt = np.argsort(p_dtr)
    p_dtr_s = p_dtr[srt]
    p_cash_s = p_cash[srt]

    traj_indices = []
    if len(front_indices) >= 3:
        sf = sorted(front_indices, key=lambda i: results[i].max_dtr)
        traj_indices = [sf[0], sf[len(sf) // 2], sf[-1]]
    elif len(front_indices) > 0:
        traj_indices = front_indices[: min(3, len(front_indices))]

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.28)

    # ── Main scatter ────────────────────────────────────────
    ax = fig.add_subplot(gs[0])

    if failed:
        ax.scatter(
            dtr[failed], cash[failed],
            c="red", marker="x", s=35, alpha=0.5, label="Failed",
        )
    if dom:
        ax.scatter(
            dtr[dom], cash[dom],
            c="lightblue", marker="o", s=35, alpha=0.5, edgecolors="gray", linewidth=0.4,
            label="Dominated",
        )

    if len(p_dtr) > 0:
        sc = ax.scatter(
            p_dtr, p_cash,
            c=[getattr(r, sweep["z_metric"]) for r in p_results],
            cmap="viridis", marker="D", s=70, zorder=5,
            edgecolors="darkgreen", linewidth=0.8,
            label="Pareto front",
        )
        ax.plot(
            p_dtr_s, p_cash_s,
            color="green", linestyle="--", linewidth=1.3, alpha=0.6,
        )

        if len(front_indices) >= 3:
            labels = ["Best service", "Balanced", "Best cash"]
            for idx, label in zip([traj_indices[0], traj_indices[1], traj_indices[-1]], labels):
                ax.annotate(
                    label,
                    (results[idx].max_dtr, results[idx].end_cash),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=7, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8),
                )

        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(sweep["z_label"], fontsize=9)

        cash_min = min(p_cash)
        cash_max = max(p_cash)
        dtr_min = min(p_dtr)
        dtr_max = max(p_dtr)
        front_text = (
            f"Pareto front: {len(front_indices)} of {len(results)} non-dominated points   |   "
            f"Grid: {n}\u00d7{n}   |   "
            f"Cash range on front: ${cash_min:,.0f} \u2013 ${cash_max:,.0f}   |   "
            f"DTR range: {dtr_min:.2f}x \u2013 {dtr_max:.2f}x"
        )
    else:
        sc = None
        front_text = f"No non-dominated points found ({len(results)} simulated)"

    ax.set_xlabel(sweep["x_label"])
    ax.set_ylabel(sweep["y_label"])
    ax.set_title(sweep["title"], fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.85)
    ax.grid(True, alpha=0.25)

    ax.text(
        0.5, -0.18, front_text, transform=ax.transAxes,
        ha="center", va="top", fontsize=7.5, color="gray",
    )

    # ── Trajectory panel ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    colors_traj = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    traj_stock = sweep.get("trajectory_stock", "Cash_Reserves")

    for idx, color in zip(traj_indices, colors_traj):
        r = results[idx]
        p = dict(FIXED_PARAMS)
        p[sweep["param_x"]] = r.params[sweep["param_x"]]
        p[sweep["param_y"]] = r.params[sweep["param_y"]]
        rr = model.simulate(params=p, method="euler")
        label_x = r.params[sweep["param_x"]]
        label_y = r.params[sweep["param_y"]]
        ax2.plot(
            rr.times, rr.values[traj_stock],
            color=color, linewidth=1.2,
            label=f"{sweep['param_x']}={label_x:.1f}, {sweep['param_y']}={label_y:.1f}",
        )

    ax2.set_xlabel("Days")
    ax2.set_ylabel(traj_stock.replace("_", " "))
    ax2.set_title(f"{traj_stock.replace('_', ' ')} trajectories for selected Pareto points", fontsize=10)
    ax2.legend(fontsize=7, framealpha=0.85)
    ax2.grid(True, alpha=0.25)

    return fig


def _fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


class Report(FPDF):
    def _s(self, t):
        return t.encode("latin-1", errors="replace").decode("latin-1")

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, self._s("Logistics Fleet \u2014 Tradespace Analysis"), align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, self._s(f"Page {self.page_no()}/{{nb}}"), align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 60, 120)
        self.cell(0, 12, self._s(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, self._s(text))
        self.ln(3)

    def add_chart_page(self, title, fig, conclusion=""):
        self.add_page()
        self.section(title)
        img = _fig_bytes(fig)
        self.image(img, x=self.l_margin, w=170)
        if conclusion:
            self.ln(3)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(50, 50, 50)
            self.multi_cell(0, 4.5, self._s(conclusion))


def _make_pdf(results, front, sweep, model, n, out, total):
    fig = plot_tradespace(results, front, sweep, model, n=n)

    p_results = [results[i] for i in front]
    p_cash = [r.end_cash for r in p_results]
    p_dtr = [r.max_dtr for r in p_results]

    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Page 1: Title + summary ──────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 15, pdf._s(sweep["title"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, pdf._s(f"Sweep: {sweep['param_x']} \u00d7 {sweep['param_y']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, pdf._s(f"Grid: {n}\u00d7{n} = {total} simulations"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.section("Summary")
    pdf.body(
        f"Tested {total} parameter combinations: {sweep['param_x']} "
        f"({sweep['range_x'][0]:.1f} \u2013 {sweep['range_x'][1]:.1f}) "
        f"\u00d7 {sweep['param_y']} "
        f"({sweep['range_y'][0]:.1f} \u2013 {sweep['range_y'][1]:.1f}). "
        f"Pareto front: {len(front)} of {total} non-dominated points. "
        + (
            f"Cash ranges from ${min(p_cash):,.0f} to ${max(p_cash):,.0f}. "
            f"Max DTR from {min(p_dtr):.2f}x to {max(p_dtr):.2f}x."
            if p_cash
            else ""
        )
    )

    pdf.ln(2)
    pdf.section("Key Takeaways")
    takeaways = []
    succeeded = sum(1 for r in results if r.succeeded)
    failed = total - succeeded
    takeaways.append(f"{succeeded} of {total} simulations completed successfully.")
    if failed:
        takeaways.append(f"{failed} runs failed (negative cash / bankruptcy).")
    if p_cash:
        takeaways.append(
            f"Best cash: ${max(p_cash):,.0f} at DTR {p_dtr[p_cash.index(max(p_cash))]:.2f}x. "
        )
        takeaways.append(
            f"Best service: DTR {min(p_dtr):.2f}x at cash ${p_cash[p_dtr.index(min(p_dtr))]:,.0f}."
        )
    for t in takeaways:
        pdf.body(f"\u2022 {t}")

    # ── Pages 2 & 3: Charts ─────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pdf.add_page()
    pdf.section("Pareto Frontier")
    buf.seek(0)
    pdf.image(buf, x=pdf.l_margin, w=170)

    if front:
        pdf.add_page()
        pdf.section("Selected Trajectories")
        pdf.body(
            pdf._s("Showing Cash Reserves over time for Pareto-optimal points on the frontier. "
                   "The curve illustrates how different parameter combinations "
                   "produce different financial trajectories.")
        )
        pdf.ln(2)
        buf.seek(0)
        pdf.image(buf, x=pdf.l_margin, w=170)

    pdf.output(out)
    print(f"Saved: {out} ({pdf.pages_count} pages)")


def main():
    parser = argparse.ArgumentParser(
        description="Tradespace visualization for logistics fleet model"
    )
    parser.add_argument(
        "--sweep", choices=list(SWEEPS.keys()), default="productivity",
        help="Which parameter sweep to run (default: productivity)",
    )
    parser.add_argument(
        "--n", type=int, default=7,
        help="Grid points per dimension (total runs = n\u00b2, default: 7)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick mode: n=5 (25 runs)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output filename (default: tradespace_<sweep>.png)",
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help="Path to .sysd model file",
    )
    args = parser.parse_args()

    if args.fast:
        args.n = 5

    print(f"Loading model: {args.model}")
    model = parse_sysd_file(args.model)
    print(f"  Model: {model.name}")
    print(f"  {len(model.stocks)} stocks, {len(model.agents)} agents, "
          f"{len(model.queues)} queues")

    sweep = SWEEPS[args.sweep]
    total = args.n * args.n
    print(f"\nSweep: {sweep['title']}")
    print(f"  X: {sweep['param_x']}  {sweep['range_x']}")
    print(f"  Y: {sweep['param_y']}  {sweep['range_y']}")
    print(f"  Grid: {args.n}\u00d7{args.n} = {total} simulations\n")

    results = run_grid(model, sweep, n=args.n)

    succeeded = sum(1 for r in results if r.succeeded)
    failed = total - succeeded
    print(f"\n  Succeeded: {succeeded}, Failed: {failed}")

    front = pareto_front(results)
    print(f"  Pareto front: {len(front)} non-dominated points")

    out = args.output or f"tradespace_{args.sweep}.png"

    if out.lower().endswith(".pdf"):
        print("\nGenerating PDF report...")
        _make_pdf(results, front, sweep, model, n=args.n, out=out, total=total)
    else:
        print("\nPlotting...")
        fig = plot_tradespace(results, front, sweep, model, n=args.n)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out} ({total} runs, {len(front)} Pareto points)")


if __name__ == "__main__":
    main()
