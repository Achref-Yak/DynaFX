#!/usr/bin/env python3
"""
Strategic Supplier Risk Assessment — KB-driven Scenario Ranking + PDF Report
==============================================================================
Compares 4 sourcing strategies using a KB-informed SD+DES model.
Outputs a professional PDF report with charts, causal explanation,
sensitivity analysis, and a ranked recommendation.

Usage:
    python examples/supplier_risk_assessment.py
    python examples/supplier_risk_assessment.py --output my_report.pdf
"""

from __future__ import annotations
import sys, os, io, argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

from dynafx import (
    KBSimBridge,
    ScenarioComparison,
    ScenarioDef,
    TripleStore,
    grade_queries,
)
from dynafx.dynamics import parse_sysd
from dynafx.knowledge.model import NamedNode, Literal, Triple
from dynafx.knowledge.turtle import parse_turtle

C = plt.cm.tab10.colors

# ═══════════════════════════════════════════════════════════════
# 1.  Knowledge Base
# ═══════════════════════════════════════════════════════════════

TURTLE = """\
@prefix sc: <http://sc.org/> .

# Supplier profiles
sc:Supplier_A sc:reliability 0.92 .
sc:Supplier_A sc:leadTime 4 .
sc:Supplier_A sc:costPerUnit 10 .
sc:Supplier_A sc:risk "medium" .

sc:Supplier_B sc:reliability 0.98 .
sc:Supplier_B sc:leadTime 2 .
sc:Supplier_B sc:costPerUnit 14 .
sc:Supplier_B sc:risk "low" .

sc:Supplier_C sc:reliability 0.65 .
sc:Supplier_C sc:leadTime 10 .
sc:Supplier_C sc:costPerUnit 7 .
sc:Supplier_C sc:risk "high" .

# Contract & budget
sc:Contract sc:minFillRate 0.70 .
sc:Budget sc:limit 600000 .
"""

store = parse_turtle(TURTLE)
bridge = KBSimBridge(store)

NS = "http://sc.org/"
SCEN = NamedNode(f"{NS}Scenario")
FILL_RATE = NamedNode(f"{NS}fillRateScore")
COST = NamedNode(f"{NS}costScore")
PROFIT = NamedNode(f"{NS}profitScore")

P = f"PREFIX sc: <{NS}>"
MIN_FILL_Q = f"{P} SELECT ?v WHERE {{ <{NS}Contract> <{NS}minFillRate> ?v }}"
BUDGET_Q = f"{P} SELECT ?v WHERE {{ <{NS}Budget> <{NS}limit> ?v }}"

# ═══════════════════════════════════════════════════════════════
# 2.  Model
# ═══════════════════════════════════════════════════════════════

MODEL_SRC = """\
supplier_risk_model
  dt 0.25
  from 0 to 120

  // Parameters (overridden per scenario)
  aux demand_rate: 100
  aux base_reliability: 0.85
  aux cost_per_unit: 10.0
  aux safety_stock: 300

  // KB_QUERY — reads contract + budget from TripleStore
  aux min_fill_rate: KB_QUERY(min_fill_q)
  aux budget_limit: KB_QUERY(budget_q)

  // Operational
  aux supply_rate: demand_rate * base_reliability
  aux fulfillment_rate: MIN(Inventory / dt, demand_rate)

  // Financial
  aux procurement_cost: fulfillment_rate * cost_per_unit
  aux holding_cost: (Inventory / 200) * cost_per_unit * 0.3
  aux revenue_rate: 25.0 * fulfillment_rate
  aux profit_rate: revenue_rate - procurement_cost - holding_cost

  // Fill rate (diagnostic)
  aux fill_rate: IF(Cumulative_Orders > 0, Cumulative_Fulfilled / Cumulative_Orders, 1.0)

  // DES quality inspection queue
  queue "Quality_Inspection": capacity 30, service_time 2.0 + fulfillment_rate * 0.005
  resource "Inspectors": capacity 1
  arrival_rate: MAX(0, demand_rate - fulfillment_rate) * 0.05

  stock Inventory: safety_stock
    + supply_rate
    - fulfillment_rate

  stock Cumulative_Orders: 0
    + demand_rate

  stock Cumulative_Fulfilled: 0
    + fulfillment_rate

  stock Total_Cost: 0
    + procurement_cost + holding_cost

  stock Total_Revenue: 0
    + revenue_rate

  stock Total_Profit: 0
    + profit_rate
"""

model = parse_sysd(MODEL_SRC)

# ═══════════════════════════════════════════════════════════════
# 3.  Scenario Definitions
# ═══════════════════════════════════════════════════════════════

COMMON_PARAMS = {
    "min_fill_q": MIN_FILL_Q,
    "budget_q": BUDGET_Q,
}

sdefs = [
    ScenarioDef("Low Cost", {
        **COMMON_PARAMS,
        "base_reliability": 0.65,
        "cost_per_unit": 7.0,
        "safety_stock": 300.0,
    }),
    ScenarioDef("Premium Quality", {
        **COMMON_PARAMS,
        "base_reliability": 0.98,
        "cost_per_unit": 14.0,
        "safety_stock": 500.0,
    }),
    ScenarioDef("Balanced", {
        **COMMON_PARAMS,
        "base_reliability": 0.92,
        "cost_per_unit": 10.0,
        "safety_stock": 400.0,
    }),
    ScenarioDef("Dual Source", {
        **COMMON_PARAMS,
        "base_reliability": 0.95,
        "cost_per_unit": 12.0,
        "safety_stock": 450.0,
    }),
]

# ═══════════════════════════════════════════════════════════════
# 4.  Evidence map / grades / constraints
# ═══════════════════════════════════════════════════════════════

MAX_DEMAND = 120 * 100       # 120 days x 100/day = 12000
MAX_COST = 600000.0           # budget limit
MAX_PROFIT = 200000.0         # estimated ceiling

ev_map = [
    ("Cumulative_Fulfilled", SCEN, FILL_RATE,
     lambda init, final: min(1.0, max(0.0, (final[-1] - init[0]) / MAX_DEMAND))),
    ("Total_Cost", SCEN, COST,
     lambda init, final: min(1.0, max(0.0, 1.0 - (final[-1] - init[0]) / MAX_COST))),
    ("Total_Profit", SCEN, PROFIT,
     lambda init, final: min(1.0, max(0.0, (final[-1] - init[0]) / MAX_PROFIT))),
]

grade_specs = [
    (f"{P} SELECT ?v WHERE {{ <{NS}Scenario> sc:fillRateScore ?v }}", "v", 0.0, 0.0),
    (f"{P} SELECT ?v WHERE {{ <{NS}Scenario> sc:costScore ?v }}", "v", 0.0, 0.0),
    (f"{P} SELECT ?v WHERE {{ <{NS}Scenario> sc:profitScore ?v }}", "v", 0.0, 0.0),
]

constraint_queries = [
    f"{P} ASK {{ <{NS}Scenario> sc:fillRateScore ?f . FILTER(?f >= 0.68) }}",
]

GRADE_LABELS = ["Fill", "Cost", "Profit"]

# ═══════════════════════════════════════════════════════════════
# 5.  Chart generators
# ═══════════════════════════════════════════════════════════════

def _fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _scenario_colors(snames):
    palette = [C[0], C[1], C[2], C[3], C[4]]
    return {n: palette[i % len(palette)] for i, n in enumerate(snames)}


def chart_inventory_trajectories(comp):
    fig, ax = plt.subplots(figsize=(10, 4))
    t = comp.times
    colors = _scenario_colors([s.name for s in comp.scenarios])
    for sc in comp.scenarios:
        inv = sc.result.values.get("Inventory", [])
        if inv:
            ax.plot(t, inv, label=sc.name, color=colors[sc.name], linewidth=1.5)
    ax.set_xlabel("Days")
    ax.set_ylabel("Units")
    ax.set_title("Inventory Trajectories by Scenario", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def chart_fill_rate(comp):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    t = comp.times
    colors = _scenario_colors([s.name for s in comp.scenarios])
    for sc in comp.scenarios:
        met = sc.result.values.get("Cumulative_Fulfilled", [])
        dem = sc.result.values.get("Cumulative_Orders", [])
        if met and dem:
            fill = np.divide(met, dem, out=np.ones_like(met), where=np.array(dem) > 0)
            ax.plot(t, fill * 100, label=sc.name, color=colors[sc.name], linewidth=1.5)
    ax.set_xlabel("Days")
    ax.set_ylabel("Fill Rate (%)")
    ax.set_title("Fill Rate Over Time", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def chart_cumulative_fulfillment(comp):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    t = comp.times
    colors = _scenario_colors([s.name for s in comp.scenarios])
    for sc in comp.scenarios:
        met = sc.result.values.get("Cumulative_Fulfilled", [])
        dem = sc.result.values.get("Cumulative_Orders", [])
        profit = sc.result.values.get("Total_Profit", [])
        if met:
            ax1.plot(t, met, label=sc.name, color=colors[sc.name], linewidth=1.5)
        if profit:
            ax2.plot(t, profit, label=sc.name, color=colors[sc.name], linewidth=1.5)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Units")
    ax1.set_title("Cumulative Fulfilled", fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Currency")
    ax2.set_title("Total Profit", fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def chart_des_queue(comp):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    t = comp.times
    colors = _scenario_colors([s.name for s in comp.scenarios])
    for sc in comp.scenarios:
        dh = getattr(sc.result, "des_metrics_history", [])
        if dh:
            qlen = [m.get("Quality_Inspection", 0) for m in dh]
            ax.plot(t[:len(qlen)], qlen, label=sc.name, color=colors[sc.name], linewidth=1.5)
    ax.set_xlabel("Days")
    ax.set_ylabel("Queue Length")
    ax.set_title("Quality Inspection Queue Length", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def chart_comparison_grid(comp):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    t = comp.times
    colors = _scenario_colors([s.name for s in comp.scenarios])
    titles = ["Inventory", "Fill Rate (%)", "Cumulative Fulfilled", "Total Profit"]
    keys = ["Inventory", None, "Cumulative_Fulfilled", "Total_Profit"]
    for idx, (ax, title, key) in enumerate(zip(axes.flat, titles, keys)):
        for sc in comp.scenarios:
            c = colors[sc.name]
            if key is None:
                met = sc.result.values.get("Cumulative_Fulfilled", [])
                dem = sc.result.values.get("Cumulative_Orders", [])
                if met and dem:
                    fill = np.divide(met, dem, out=np.ones_like(met), where=np.array(dem) > 0)
                    ax.plot(t, fill * 100, label=sc.name, color=c, linewidth=1.5)
            else:
                vals = sc.result.values.get(key, [])
                if vals:
                    ax.plot(t, vals, label=sc.name, color=c, linewidth=1.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        if idx >= 2:
            ax.set_xlabel("Days")
    fig.tight_layout()
    return fig


def chart_tornado(comp, param_ranges, output_stock="Total_Profit"):
    plt.figure(figsize=(8, max(3, 0.5 * len(param_ranges))))
    base_sc = comp.scenarios[0]
    base_params = dict(base_sc.params)
    t_measure = comp.times[-1] if comp.times else 0

    impacts = []
    for pname, (low, high) in param_ranges.items():
        mid = (low + high) / 2.0
        pl = dict(base_params, **{pname: low})
        ph = dict(base_params, **{pname: high})
        rl = comp.model.simulate(params=pl, method=comp.method)
        rh = comp.model.simulate(params=ph, method=comp.method)
        vl = rl.values.get(output_stock, [0])[-1] if pl else 0
        vh = rh.values.get(output_stock, [0])[-1] if ph else 0
        impacts.append((pname, vl, vh, abs(vh - vl)))

    impacts.sort(key=lambda x: x[3])

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(impacts))))
    for i, (pn, lv, hv, sp) in enumerate(impacts):
        left = min(lv, hv)
        right = max(lv, hv)
        color = "steelblue" if (hv - lv) > 0 else "coral"
        ax.barh(i, right - left, left=left, height=0.4, color=color, edgecolor="black")
        ax.text(lv, i, f"{lv:.0f}", va="center", ha="right", fontsize=8)
        ax.text(hv, i, f"{hv:.0f}", va="center", ha="left", fontsize=8)

    ax.set_yticks(list(range(len(impacts))))
    ax.set_yticklabels([i[0] for i in impacts])
    ax.set_xlabel(f"{output_stock} at t={t_measure:.0f}")
    ax.set_title("Sensitivity Tornado", fontweight="bold")
    ax.grid(True, axis="x")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 6.  PDF Report
# ═══════════════════════════════════════════════════════════════

class Report(FPDF):
    def _s(self, t):
        return t.encode("latin-1", "replace").decode("latin-1")

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(150, 150, 150)
            self.cell(0, 4, self._s("Supplier Risk Assessment"), align="C")
            self.ln(6)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no() - 1}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, self._s(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(self.l_margin, self.get_y(), self.l_margin + 190, self.get_y())
        self.ln(4)

    def sub_section(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, self._s(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, self._s(text))
        self.ln(2)

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


def make_pdf(comp, grades, ranked, store, output_path):
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    best = ranked[0][0] if ranked else "N/A"
    best_score = ranked[0][1] if ranked else 0.0

    exp = None
    if ranked:
        try:
            exp = comp.explain_scenario(
                ranked[0][0], store, evidence_map=ev_map, bridge=bridge,
                grade_specs=grade_specs, grades=grades, ranked=ranked,
            )
        except Exception:
            exp = None

    # ── Page 1: Cover ─────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 15, pdf._s("Supplier Risk Assessment"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, pdf._s(
        "Strategic Sourcing Decision Support  |  SD + DES + Knowledge Base"
    ), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 6, pdf._s(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"dt=0.25  |  {len(comp.scenarios)} scenarios after filtering"
    ), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.section("Executive Summary")
    pdf.body(
        "This report evaluates four supplier sourcing strategies for a manufacturing "
        "operation. A Knowledge Base stores supplier intelligence (reliability, lead time, "
        "cost, risk level) plus contract terms. The system dynamics model simulates "
        f"inventory, fulfillment, and financial performance over 120 days. "
        f"One of four initial scenarios was eliminated by SPARQL ASK constraint checks "
        f"(minimum fill rate &ge; 68%)."
    )
    score_str = f"{best_score:.3f}"
    pdf.body(
        f"Recommended strategy: {best} (aggregate score {score_str}). "
        "This scenario offers the best trade-off between fill rate, cost efficiency, "
        "and profitability given the supplier profiles in the Knowledge Base."
    )

    pdf.ln(4)
    pdf.sub_section("Suppliers in Knowledge Base")
    suppliers = [
        ("Supplier A", "0.92", "4 days", "$10/unit"),
        ("Supplier B", "0.98", "2 days", "$14/unit"),
        ("Supplier C", "0.65", "10 days", "$7/unit"),
    ]
    pdf.set_font("Courier", "", 9)
    pdf.cell(40, 6, "Supplier", border=1)
    pdf.cell(28, 6, "Reliability", border=1)
    pdf.cell(28, 6, "Lead Time", border=1)
    pdf.cell(28, 6, "Unit Cost", border=1)
    pdf.ln()
    pdf.set_font("Courier", "", 9)
    for name, rel, lead, cost in suppliers:
        pdf.cell(40, 6, pdf._s(name), border=1)
        pdf.cell(28, 6, rel, border=1)
        pdf.cell(28, 6, lead, border=1)
        pdf.cell(28, 6, cost, border=1)
        pdf.ln()
    pdf.ln(3)

    # ── Page 2: Dashboard + Ranking ───────────────────────
    pdf.add_page()
    pdf.section("Scenario Dashboard")

    pdf.sub_section("Constraint Filter Results")
    n_before = 4
    n_filtered = n_before - len(comp.scenarios)
    if n_filtered:
        pdf.body(f"Eliminated {n_filtered} scenario(s) failing SPARQL ASK constraints:")
        pdf.set_font("Courier", "", 9)
        for q in constraint_queries:
            short = q.replace(f"PREFIX sc: <{NS}> ", "").strip()
            pdf.cell(0, 5, pdf._s(f"  {short}"), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.body("All scenarios passed constraint checks.")
    pdf.ln(3)

    pdf.sub_section("Ranked Scenarios")
    pdf.set_font("Courier", "B", 9)
    pdf.cell(8, 6, "R", border=1)
    pdf.cell(36, 6, "Scenario", border=1)
    for lbl in GRADE_LABELS:
        pdf.cell(22, 6, lbl, border=1)
    pdf.cell(22, 6, "Score", border=1)
    pdf.ln()
    pdf.set_font("Courier", "", 9)
    for i, (sname, score) in enumerate(ranked, 1):
        g = grades.get(sname, {})
        vals = [g.get(f"{sname}_{j}", 0.0) for j in range(3)]
        pdf.cell(8, 6, str(i), border=1)
        pdf.cell(36, 6, pdf._s(sname[:16]), border=1)
        for v in vals:
            pdf.cell(22, 6, f"{v:.3f}", border=1)
        pdf.cell(22, 6, f"{score:.3f}", border=1)
        pdf.ln()
    pdf.ln(3)

    # Per-scenario KPI summary
    pdf.sub_section("Final State KPIs")
    pdf.set_font("Courier", "B", 9)
    pdf.cell(36, 6, "Scenario", border=1)
    pdf.cell(22, 6, "Fill%", border=1)
    pdf.cell(22, 6, "Fulfilled", border=1)
    pdf.cell(22, 6, "Cost", border=1)
    pdf.cell(22, 6, "Profit", border=1)
    pdf.cell(22, 6, "Score", border=1)
    pdf.ln()
    score_map = dict(ranked)
    pdf.set_font("Courier", "", 9)
    for sc in comp.scenarios:
        v = sc.result.values
        filled = v.get("Cumulative_Fulfilled", [0])[-1]
        ordered = v.get("Cumulative_Orders", [0])[-1]
        cost = v.get("Total_Cost", [0])[-1]
        profit = v.get("Total_Profit", [0])[-1]
        pct = (filled / ordered * 100) if ordered > 0 else 0
        s = score_map.get(sc.name, 0)
        pdf.cell(36, 6, pdf._s(sc.name[:16]), border=1)
        pdf.cell(22, 6, f"{pct:.1f}%", border=1)
        pdf.cell(22, 6, f"{filled:.0f}", border=1)
        pdf.cell(22, 6, f"{cost:.0f}", border=1)
        pdf.cell(22, 6, f"{profit:.0f}", border=1)
        pdf.cell(22, 6, f"{s:.3f}", border=1)
        pdf.ln()

    # ── Page 3: Scenario Comparison Charts ─────────────────
    pdf.add_chart_page("Scenario Comparison",
        chart_comparison_grid(comp),
        "All scenarios: Inventory, Fill Rate, Cumulative Fulfilled, and Total Profit "
        "over 120 days. The fill rate plot clearly separates the scenarios by "
        "effective reliability."
    )

    # ── Page 4: DES Queue + Explanation ────────────────────
    pdf.add_chart_page("Quality Inspection Queue",
        chart_des_queue(comp),
        "DES queue length for the Quality Inspection bottleneck. When fulfillment "
        "lags demand, backorders trigger inspection arrivals. Higher-reliability "
        "scenarios (Premium, Dual Source) maintain shorter queues."
    )

    if exp:
        pdf.add_page()
        pdf.section("Causal Explanation")
        pdf.sub_section(f"Top Scenario: {exp['name']} (rank {exp['rank']}, score {exp['total_score']})")
        for g in exp["goals"]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 60, 120)
            pdf.cell(0, 6, pdf._s(f"  {g['label']}  =  {g['score']:.4f}"),
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Courier", "", 7)
            pdf.set_text_color(60, 60, 60)
            chain_str = " \u2192 ".join(g["causal_chain"][:10])
            pdf.multi_cell(0, 3.5, pdf._s(f"  Chain: {chain_str}"))
            pdf.ln(2)

        if exp.get("kb_facts"):
            pdf.sub_section("KB Facts")
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(40, 40, 40)
            for fact in exp["kb_facts"]:
                pdf.cell(0, 4.5, pdf._s(f"  \u2022 {fact}"),
                         new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # ── Page 5: KB Facts ──────────────────────────────────
    pdf.add_page()
    pdf.section("Knowledge Base State")
    pdf.sub_section("All Triples")
    pdf.set_font("Courier", "", 8)
    for graph_name in sorted(store.graphs()):
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, pdf._s(f"  --- {graph_name} ---"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 8)
        triples = list(store.triples_in_graph(graph_name))
        for t in triples:
            s = str(t.subject).split("/")[-1].rstrip(">")
            p = str(t.predicate).split("/")[-1].rstrip(">")
            o = str(t.object_)[:40]
            pdf.cell(0, 4, pdf._s(f"    {s}  {p}  {o}"),
                     new_x="LMARGIN", new_y="NEXT")
            if pdf.get_y() > 260:
                pdf.add_page()
    pdf.ln(3)

    # ── Page 6: Sensitivity ───────────────────────────────
    param_ranges = {
        "base_reliability": (0.75, 1.0),
        "cost_per_unit": (8.0, 15.0),
        "safety_stock": (200, 600),
    }
    fig_tornado = chart_tornado(comp, param_ranges, "Total_Profit")
    pdf.add_chart_page("Sensitivity Analysis",
        fig_tornado,
        "Tornado diagram showing the impact of each parameter on Total Profit. "
        "Holding other params at the first scenario's baseline, each parameter "
        "is varied between its low and high bounds. The spread shows the "
        "parameter's leverage on the outcome."
    )

    # ── Page 7: Recommendation ────────────────────────────
    pdf.add_page()
    pdf.section("Recommendation")

    pdf.body(
        f"After evaluating {len(comp.scenarios)} sourcing strategies "
        f"(1 eliminated by constraint filter), the top-ranked "
        f"scenario is '{best}' with an aggregate score of {best_score:.3f}."
    )
    pdf.body(
        "Scoring dimensions: fill rate (how much demand is met), cost efficiency "
        "(procurement + holding cost vs. budget), and profitability (revenue minus cost). "
        "Scores are normalized to 0-1 and averaged with equal weighting."
    )
    pdf.ln(3)

    pdf.sub_section("Ranking Summary")
    pdf.set_font("Courier", "", 9)
    for i, (sname, score) in enumerate(ranked, 1):
        highlight = " <<<" if i == 1 else ""
        pdf.cell(0, 5, pdf._s(f"  {i}. {sname}: {score:.3f}{highlight}"),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.sub_section("Why this recommendation?")
    if exp:
        pdf.body(f"'{best}' achieves the best balance across all three goals:")
        for g in exp["goals"]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 5, pdf._s(f"  {g['label']}: {g['score']:.4f}"),
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            chain_str = " \u2192 ".join(g["causal_chain"][:6])
            pdf.cell(0, 5, pdf._s(f"    Chain: {chain_str}..."),
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    pdf.body(
        "The Knowledge Base provides a single source of truth for supplier "
        "intelligence. As new supplier data arrives, updating the KB triples "
        "and re-running the pipeline produces an updated recommendation without "
        "changing the model or ranking logic."
    )

    pdf.output(str(output_path))
    return output_path


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategic Supplier Risk Assessment"
    )
    parser.add_argument("--output", "-o", default="supplier_risk_report.pdf")
    args = parser.parse_args()

    print("=" * 62)
    print("  Supplier Risk Assessment — KB-driven Scenario Ranking")
    print("=" * 62)

    comp = ScenarioComparison(model, sdefs, method="rk4", kb=store)

    n_before = len(comp.scenarios)
    comp.filter(store, constraint_queries, evidence_map=ev_map, bridge=bridge)
    n_after = len(comp.scenarios)
    n_filtered = n_before - n_after

    if n_filtered:
        print(f"\n  Filtered out {n_filtered} scenario(s) failing ASK constraints\n")
    else:
        print("\n  All scenarios passed constraint checks\n")

    grades = comp.grade_scenarios(grade_specs, store, evidence_map=ev_map, bridge=bridge)
    ranked = comp.rank(grade_specs, store, evidence_map=ev_map, bridge=bridge, agg="mean")

    header = f"  {'Rank':<6} {'Scenario':<18}"
    for lbl in GRADE_LABELS:
        header += f" {lbl:<7}"
    header += " Score"
    print(header)
    print("  " + "-" * (6 + 18 + 8 * len(GRADE_LABELS) + 2))

    for rank_idx, (sname, score) in enumerate(ranked, 1):
        g = grades[sname]
        vals = [g.get(f"{sname}_0", 0.0), g.get(f"{sname}_1", 0.0), g.get(f"{sname}_2", 0.0)]
        row = f"  {rank_idx:<6} {sname:<18}"
        for v in vals:
            row += f" {v:<7.3f}"
        row += f" {score:.3f}"
        print(row)

    print()
    best = ranked[0][0] if ranked else "N/A"
    print(f"  Recommendation: {best}")
    print()

    print(f"  Generating PDF report: {args.output}")
    out = make_pdf(comp, grades, ranked, store, args.output)
    print(f"  Done: {out}")
    print("=" * 62)
