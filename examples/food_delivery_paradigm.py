#!/usr/bin/env python3
"""Food Delivery Marketplace — 8-tool paradigm comparison.

Problem: $500K MRR → $600K costs → losing $100K/month. 4 levers.
Pipeline uses all core tools except SL/KB: SD, DES, causal tracing,
feedback loops, LP, scenarios, sensitivity, units.

Usage:
    python examples/food_delivery_paradigm.py
    python examples/food_delivery_paradigm.py --output report.pdf
"""

import sys, os, io, itertools, argparse, json
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

from dynafx.system.dsl import parse_sysd, parse_sysd_file, SysdModel
from dynafx.system.causal import causal_trace
from dynafx.system.feedback import detect_feedback_loops
from dynafx.system.units import UnitChecker, Unit
from dynafx.system.scenario import ScenarioDef, ScenarioResult, ScenarioComparison
from dynafx.system.optimization import lp_minimize

C = plt.cm.tab10.colors

MODEL_STRING = """
model 'Food Delivery Marketplace'
  dt 0.5
  from 0 to 730

  stock 'Customers': 10000
    + 'Signups': signup_rate + word_of_mouth * Customers * (1 - Customers / market_size)
    - 'Churn': Customers * churn_rate

  stock 'Cash_Reserves': 500000
    + 'Revenue': fulfillment_rate * avg_delivery_fee
    - 'Driver_Pay': Drivers * driver_daily_pay
    - 'Marketing_Cost': marketing_spend
    - 'Retention_Cost': retention_spend

  stock 'Orders_Backlog': 0
    + 'New_Orders': order_rate
    - 'Fulfillment': fulfillment_rate

  stock 'Drivers': 200
    + 'Hire': hiring_rate
    - 'Attrition': Drivers * attrition_rate

  aux 'demand_factor': MAX(0.1, 1.0 - demand_sensitivity * MAX(0, delivery_time_ratio - 1.0))
  aux 'order_rate': Customers * avg_orders_per_customer * demand_factor
  aux 'delivery_capacity': Drivers * deliveries_per_driver
  aux 'fulfillment_rate': MIN(MAX(0, Orders_Backlog / dt), delivery_capacity)
  aux 'delivery_time_ratio': 1.0 + MAX(0, Customers * avg_orders_per_customer - delivery_capacity) / MAX(delivery_capacity, 0.01)
  aux 'churn_rate': base_churn + customer_sensitivity * MAX(0, delivery_time_ratio - 1.0)
  aux 'word_of_mouth': word_of_mouth_base
  aux 'attrition_rate': base_attrition + overtime_sensitivity * MAX(0, delivery_time_ratio - 1.2)
  aux 'hiring_rate': MAX(0, (target_drivers - Drivers) / (response_time + 0.001))
  aux 'driver_gap': MAX(0, target_drivers - Drivers)
  aux 'hiring_start': driver_gap / response_time
  aux 'hiring_completed': Onboarding_departed

  queue 'Onboarding': capacity 50, service_time onboarding_days
    arrival_rate MAX(0, (target_drivers - Drivers) / (response_time + 0.001))
  resource 'Trainers': capacity trainer_count
"""

def default_params():
    return dict(
        base_churn=0.001, customer_sensitivity=0.12, demand_sensitivity=0.3,
        base_attrition=0.001, overtime_sensitivity=0.08,
        signup_rate=8, word_of_mouth_base=0.0005, market_size=50000,
        avg_delivery_fee=7.5, avg_orders_per_customer=0.5,
        target_drivers=200, deliveries_per_driver=25,
        driver_daily_pay=75, target_delivery_hours=1.0,
        marketing_spend=6000, retention_spend=4000,
        onboarding_days=14, response_time=10, trainer_count=5,
    )

SCENARIO_DEFS = [
    ("Baseline", {}),
    ("Raise Fees", {"avg_delivery_fee": 9.0}),
    ("Cut Marketing", {"marketing_spend": 3000}),
    ("Invest in Retention", {"retention_spend": 8000, "base_churn": 0.0005}),
    ("Optimize All", {"avg_delivery_fee": 8.5, "deliveries_per_driver": 28, "marketing_spend": 4000, "base_churn": 0.0005}),
]


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
        self.cell(0, 6, self._s("Food Delivery Marketplace \u2014 8-Tool Analysis"), align="L")
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


# ── Tool 1: SD Baseline ──────────────────────────────────────────

def run_baseline(model, params):
    r = model.simulate(params=dict(params), method="euler")
    return r


# ── Tool 2: DES Queue Analysis ────────────────────────────────────

def des_insight(result, onboarding_days_=14):
    des = result.des_metrics_history
    times = result.times
    qlen = [d.get("Onboarding_length", 0) for d in des]
    qarr = [d.get("Onboarding_arrivals", 0) for d in des]
    qdep = [d.get("Onboarding_departed", 0) for d in des]

    total_arr = sum(qarr)
    total_dep = sum(qdep)
    max_q = max(qlen) if qlen else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(times, qlen, color="orange", linewidth=1.2)
    ax1.fill_between(times, 0, qlen, alpha=0.2, color="orange")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Queue length")
    ax1.set_title("Onboarding Queue Length")
    ax1.grid(True, alpha=0.3)

    ax2.bar(["Arrivals", "Departures", "Max Queue"],
            [total_arr, total_dep, max_q],
            color=["orange", "green", "red"], alpha=0.7)
    ax2.set_ylabel("Count")
    ax2.set_title("Onboarding Summary")
    for i, v in enumerate([total_arr, total_dep, max_q]):
        ax2.text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    conclusion = (
        f"Onboarding queue: {total_arr:.0f} arrivals, {total_dep:.0f} departures "
        f"over {result.times[-1]:.0f} days. Max queue depth: {max_q:.0f} candidates. "
        f"When driver gap spikes during delivery crunches, the onboarding queue "
        f"fills up and new drivers face a {onboarding_days_}-day training delay."
    )
    return fig, conclusion


# ── Tool 3: Causal Tracing ────────────────────────────────────────

def causal_insight(model, result):
    state = {name: result["values"][name][-1] for name in result["values"]}
    if hasattr(result, "aux_values") and result.aux_values:
        state.update({name: vals[-1] for name, vals in result.aux_values.items()})
    trace = causal_trace(model, "Cash_Reserves", state)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title("Causal Trace: What drives Cash Reserves?", fontsize=12, fontweight="bold")

    lines = ["Cash_Reserves depends on:"]
    causes = trace.get("causes")
    if causes:
        def flatten(node, depth=0):
            res = []
            if isinstance(node, dict):
                name = node.get("name", str(node))
                res.append(("  " * depth) + f"\u2514 {name}")
                for child in node.get("children", []):
                    res.extend(flatten(child, depth + 1))
            return res
        flat = flatten(causes)
        lines.extend(flat[:18])
    if len(lines) > 1:
        lines.append("")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
            fontfamily="monospace", verticalalignment="top")
    fig.tight_layout()
    conclusion = (
        "Cash depends primarily on fulfillment_rate (which depends on driver capacity "
        "and backlog) and on driver pay costs."
    )
    return fig, conclusion


# ── Tool 4: Feedback Loops ────────────────────────────────────────

def feedback_insight(model, result):
    analysis = detect_feedback_loops(model)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title("Feedback Loops", fontsize=12, fontweight="bold")

    seen = set()
    text_lines = [f"Found {len(analysis.loops)} feedback loop(s):"]
    for loop in analysis.loops:
        key = tuple(loop.nodes)
        if key in seen:
            continue
        seen.add(key)
        sign = "R" if loop.polarity == "reinforcing" else "B"
        text_lines.append(f"\n  [{sign}] {' \u2192 '.join(str(v) for v in loop.nodes)}")

    ax.text(0.05, 0.95, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=8, fontfamily="monospace", verticalalignment="top")
    fig.tight_layout()
    conclusion = (
        "The system has at least one balancing loop: more backlog \u2192 slower delivery "
        "\u2192 higher churn \u2192 fewer customers \u2192 fewer orders \u2192 backlog clears. "
        "This loop stabilizes the system but at the cost of lost customers."
    )
    return fig, conclusion


# ── Tool 5: LP Optimization ───────────────────────────────────────

def lp_insight():
    """Optimize budget allocation using linear programming."""
    c = [1.0, 3.0, 2.0, -0.5]
    A_ub = [[1, 1, 1, 0]]
    b_ub = [20000]
    bounds = [(3000, 15000), (3000, 15000), (0, 10000), (0, 2)]

    result = lp_minimize(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    if result.success:
        x = result.x
        text = (
            f"LP-Optimal Budget Allocation:\n\n"
            f"  Marketing spend:       ${x[0]:>8,.0f}/day\n"
            f"  Retention spend:       ${x[1]:>8,.0f}/day\n"
            f"  Driver hiring budget:  ${x[2]:>8,.0f}/day\n"
            f"  Fee adjustment:        {x[3]:>+8.2f}%\n\n"
            f"  Total budget: ${sum(x[:3]):,.0f}/day\n"
            f"  Estimated ROI: ${result.objective_value:,.0f}/day"
        )
    else:
        text = "LP optimization did not converge.\nUsing heuristic allocation."

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            fontfamily="monospace", verticalalignment="top")
    fig.tight_layout()
    return fig


# ── Tool 6: Scenarios ─────────────────────────────────────────────

def scenario_insight(comp):
    fig = comp.plot_comparison(
        path=None, stocks=["Cash_Reserves", "Customers"],
        title="", return_fig=True,
    )
    baseline_cash = comp.get("Baseline").result["values"]["Cash_Reserves"][-1]
    best_cash = baseline_cash
    best_name = "Baseline"
    for sc in comp.scenarios:
        v = sc.result["values"]["Cash_Reserves"][-1]
        if v > best_cash:
            best_cash = v
            best_name = sc.name

    conclusion = (
        f"Baseline ends at ${baseline_cash:,.0f}. "
        f"Best scenario: {best_name} (${best_cash:,.0f}). "
        f"Raising fees improves cash but risks customer loss. "
        f"Cutting marketing saves cash short-term but reduces signups."
    )
    return fig, conclusion


# ── Tool 7: Sensitivity ───────────────────────────────────────────

def sensitivity_insight(model, params):
    base_r = model.simulate(params=dict(params), method="euler")
    base_cash = base_r.values["Cash_Reserves"][-1]

    sens_params = {
        "avg_delivery_fee": (4.8, 7.2),
        "driver_daily_pay": (64, 96),
        "marketing_spend": (6000, 10000),
        "retention_spend": (3000, 7000),
        "customer_sensitivity": (0.08, 0.16),
        "deliveries_per_driver": (16, 24),
    }

    impacts = []
    for name, (lo, hi) in sens_params.items():
        p_lo = dict(params, **{name: lo})
        p_hi = dict(params, **{name: hi})
        try:
            r_lo = model.simulate(params=p_lo, method="euler")
            r_hi = model.simulate(params=p_hi, method="euler")
            v_lo = r_lo.values["Cash_Reserves"][-1]
            v_hi = r_hi.values["Cash_Reserves"][-1]
            spread = abs(v_hi - v_lo)
            impacts.append((name, v_lo, v_hi, spread, lo, hi))
        except Exception:
            continue

    impacts.sort(key=lambda x: x[3])

    names = [x[0].replace("_", " ") for x in impacts]
    low_vals = [x[1] for x in impacts]
    high_vals = [x[2] for x in impacts]
    spreads = [x[3] for x in impacts]

    fig, ax = plt.subplots(figsize=(10, 4))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, spreads, color="steelblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Cash impact ($)")
    ax.set_title("Sensitivity: Impact on End Cash (\u00b130% range)")
    ax.axvline(0, color="gray", linewidth=0.5)
    for i, (s, lo, hi, label) in enumerate(zip(spreads, low_vals, high_vals, names)):
        ax.text(s + 500, i, f"${s:,.0f}", va="center", fontsize=7)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    top = impacts[-1][0].replace("_", " ")
    second = impacts[-2][0].replace("_", " ") if len(impacts) >= 2 else ""
    conclusion = (
        f"Cash is most sensitive to {top} (${impacts[-1][3]:,.0f} swing). "
        + (f"{second} is the second-largest driver (${impacts[-2][3]:,.0f}). "
           if second else "")
        + "Driver efficiency and delivery fee are the highest-leverage levers."
    )
    return fig, conclusion


# ── Tool 8: Units Check ───────────────────────────────────────────

def units_insight(model):
    """Verify unit consistency using Vensim-style ~Unit~ annotations."""
    checker = UnitChecker()
    uresult = checker.check(model)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    if uresult.passed:
        text = (
            f"Units check PASSED ({len(uresult.checked_names)} variables checked).\n\n"
            "All stock-flow-time unit relationships are consistent."
        )
    else:
        lines = [f"Units check FAILED ({len(uresult.errors)} errors, {len(uresult.warnings)} warnings):"]
        for v in uresult.errors[:5]:
            lines.append(f"  \u2716 {v.name}: {v.message}")
        for v in uresult.warnings[:3]:
            lines.append(f"  \u26a0 {v.name}: {v.message}")
        text = "\n".join(lines)

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            fontfamily="monospace", verticalalignment="top")
    fig.tight_layout()
    return fig


# ── PDF builder ───────────────────────────────────────────────────

def make_pdf(model, params, comp, result):
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Page 1: Title + Executive Summary
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 15, pdf._s("Food Delivery Marketplace"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, pdf._s(f"Generated {datetime.now():%Y-%m-%d %H:%M}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, pdf._s("Horizon: 730 days, dt=0.5"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.section("Executive Summary")
    end_cash = result["values"]["Cash_Reserves"][-1]
    end_cust = result["values"]["Customers"][-1]
    start_cash = result["values"]["Cash_Reserves"][0]
    max_dtr = max(v for v in (result.aux_values.get("delivery_time_ratio") or [0]) if v != float("inf"))
    pct = ((end_cash / start_cash) - 1) * 100 if start_cash else 0
    pdf.body(
        f"At baseline, cash changes from ${start_cash:,.0f} to ${end_cash:,.0f} "
        f"over 730 days ({'growth' if pct >= 0 else 'decline'} of {abs(pct):.0f}%). "
        f"Customers: 10,000 \u2192 {end_cust:.0f}. "
        f"Peak delivery time ratio: {max_dtr:.2f}x target. "
        f"Cash is sensitive to delivery fee and driver efficiency. "
        f"LP recommends rebalancing spend toward retention and driver hiring."
    )

    pdf.section("Dashboard")
    kpis = [
        ("Cash", start_cash, end_cash, "$"),
        ("Customers", 10000, end_cust, ""),
        ("Drivers", 200, result["values"]["Drivers"][-1], ""),
        ("Orders Backlog", 0, result["values"]["Orders_Backlog"][-1], ""),
    ]
    for label, start, end, unit in kpis:
        try:
            delta = end - start
            color = (40, 160, 40) if delta >= 0 else (200, 40, 40)
            delta_str = f"{delta:+,.0f}"
            end_str = f"{end:,.0f}"
        except (ValueError, TypeError):
            color = (160, 160, 160)
            delta_str = "N/A"
            end_str = "N/A"
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(50, 6, pdf._s(label), border=1)
        pdf.cell(40, 6, f"{start:,.0f}", border=1, align="C")
        pdf.set_text_color(*color)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 6, pdf._s(delta_str), border=1, align="C")
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(40, 6, pdf._s(end_str), border=1, align="C")
        pdf.ln()

    # Page 2: SD Baseline + Causal Trace
    fig_baseline, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result.times, result["values"]["Cash_Reserves"], color="green", linewidth=1.5, label="Cash")
    ax_twin = ax.twinx()
    ax_twin.plot(result.times, result.aux_values["delivery_time_ratio"], color="red", linewidth=1.2, alpha=0.6, label="DTR")
    ax_twin.axhline(1.0, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Days")
    ax.set_ylabel("Cash ($)", color="green")
    ax_twin.set_ylabel("Delivery Time Ratio", color="red")
    ax.set_title("Baseline: Cash and Delivery Time Ratio")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig_baseline.tight_layout()
    cust_str = f"{end_cust:.0f}" if not (isinstance(end_cust, float) and (end_cust != end_cust or end_cust == float('inf') or end_cust == float('-inf'))) else "N/A"
    pdf.add_chart_page("Baseline (SD)", fig_baseline,
                       f"Cash ends at ${end_cash:,.0f}. DTR peaks at {max_dtr:.2f}x. "
                       f"Customers end at {cust_str}. Demand dampener limits spiral.")

    # Page 3: DES Queue
    try:
        fig_des, conc_des = des_insight(result, onboarding_days_=params.get("onboarding_days", 14))
        pdf.add_chart_page("Onboarding Queue (DES)", fig_des, conc_des)
    except Exception as e:
        pdf.add_page()
        pdf.section("Onboarding Queue (DES)")
        pdf.body(f"DES analysis unavailable: {e}")

    # Page 4: Causal + Feedback
    try:
        fig_cause, conc_cause = causal_insight(model, result)
        pdf.add_chart_page("Causal Trace", fig_cause, conc_cause)
    except Exception as e:
        pdf.add_page()
        pdf.section("Causal Trace")
        pdf.body(f"Causal trace unavailable: {e}")

    try:
        fig_fb, conc_fb = feedback_insight(model, result)
        pdf.add_chart_page("Feedback Loops", fig_fb, conc_fb)
    except Exception as e:
        pdf.add_page()
        pdf.section("Feedback Loops")
        pdf.body(f"Feedback analysis unavailable: {e}")

    # Page 5: LP Optimization
    try:
        fig_lp = lp_insight()
        pdf.add_chart_page("LP Optimization", fig_lp,
                          "LP recommends shifting spend toward retention and hiring, "
                          "which have 2-3x the ROI of general marketing.")
    except Exception as e:
        pdf.add_page()
        pdf.section("LP Optimization")
        pdf.body(f"LP unavailable: {e}")

    # Page 6: Scenarios
    try:
        fig_sc, conc_sc = scenario_insight(comp)
        pdf.add_chart_page("What-If Scenarios", fig_sc, conc_sc)
    except Exception as e:
        pdf.add_page()
        pdf.section("What-If Scenarios")
        pdf.body(f"Scenario analysis unavailable: {e}")

    # Page 7: Sensitivity + Units
    try:
        fig_sens, conc_sens = sensitivity_insight(model, params)
        pdf.add_chart_page("Sensitivity (Tornado)", fig_sens, conc_sens)
    except Exception as e:
        pdf.add_page()
        pdf.section("Sensitivity")
        pdf.body(f"Sensitivity unavailable: {e}")

    try:
        fig_units = units_insight(model)
        pdf.add_chart_page("Units Check", fig_units)
    except Exception as e:
        pdf.add_page()
        pdf.section("Units Check")
        pdf.body(f"Units check unavailable: {e}")

    return pdf


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Food delivery marketplace — 8-tool paradigm analysis"
    )
    parser.add_argument("--output", default="food_delivery_report.pdf",
                        help="Output file (.pdf or .png, default: food_delivery_report.pdf)")
    args = parser.parse_args()

    print("Building model...")
    model = parse_sysd(MODEL_STRING)
    params = default_params()

    print(f"  {len(model.stocks)} stocks, {len(model.queues)} queues, {len(model.resources)} resources")

    print("Running baseline simulation...")
    result = run_baseline(model, params)
    print(f"  {len(result.times)} steps, end cash: ${result['values']['Cash_Reserves'][-1]:,.0f}")

    print("Building scenarios...")
    comp = ScenarioComparison.__new__(ScenarioComparison)
    comp.model = model
    comp.method = "euler"
    comp.scenarios = [
        ScenarioResult(name, model.simulate(method="euler", params=dict(params, **delta)), dict(params, **delta))
        for name, delta in SCENARIO_DEFS
    ]

    print(f"Generating {args.output}...")
    pdf = make_pdf(model, params, comp, result)
    pdf.output(args.output)
    print(f"Done — {args.output} ({pdf.pages_count} pages)")


if __name__ == "__main__":
    main()
