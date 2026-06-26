#!/usr/bin/env python3
"""Insights demo — generates a multi-page PDF report from a SysdModel simulation.

Usage:
    cd <project-root>
    python examples/insights_demo.py

Output: pandemic_insights.pdf in the current directory.
"""

import sys, os, io, json, textwrap
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from fpdf import FPDF

def _sanitize(text: str) -> str:
    return text.replace("\u2014", "-").replace("\u2013", "-").replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2026", "...")

from cognitive_engine.system.dsl import parse_sysd_file
from cognitive_engine.system.causal import causal_trace
from cognitive_engine.system.feedback import detect_feedback_loops
from cognitive_engine.system.emergent import run_consistency_checks
from cognitive_engine.system.units import UnitChecker

# ── Configuration ───────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pandemic_seirvh.sysd")
OUTPUT_PDF = "pandemic_insights.pdf"
SIM_T_SPAN = (0.0, 200.0)
SIM_DT = 0.5
BASELINE_PARAMS = dict(
    N=10_000_000, R0=3.0, incubation_period=5.2, infectious_period=7.0,
    severe_fraction=0.07, base_mortality=0.03, overload_multiplier=2.0,
    initial_capacity=5000, capacity_expansion_rate=25, discharge_time=14,
    icu_mortality=0.15, vax_start=90, vax_rate_max=8000, vax_duration=180,
    attrition_rate=0.002,
)
SCENARIO_PARAMS = [
    ("No vaccination", {**BASELINE_PARAMS, "vax_start": 9999}),
    ("Baseline", BASELINE_PARAMS),
    ("High R0", {**BASELINE_PARAMS, "R0": 4.5}),
]
KEY_STOCKS = ["Susceptible", "Exposed", "Infected", "Recovered",
              "Hospitalized", "Fatalities", "Healthcare_Capacity"]

# ── Matplotlib helpers ──────────────────────────────────────────

COLORS = plt.cm.tab10.colors


def fig_to_bytes(fig: plt.Figure) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_timeseries_plot(result, stock_names: list[str], title: str = "Simulation") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, s in enumerate(stock_names):
        vals = result["values"][s]
        ax.plot(result.times, vals, color=COLORS[i % len(COLORS)], label=s, linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_causal_tree_plot(model, state: dict[str, float], stock_names: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, title_txt in enumerate(["Causes (upstream drivers)", "Effects (downstream impacts)"]):
        ax = axes[ax_idx]
        target = stock_names[2] if len(stock_names) > 2 else stock_names[0]
        trace = causal_trace(model, target, state=state, max_depth=4)
        node = trace.get("causes" if ax_idx == 0 else "effects", None)
        if node is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title_txt)
            continue

        pos = {}
        labels = {}
        edges = []
        _flatten_tree(node, pos=pos, labels=labels, edges=edges, x=0, y=0, depth=0)

        rows = sorted(set(p[1] for p in pos.values()))
        if rows:
            for name, (x, y) in pos.items():
                new_y = (y - min(rows)) / (max(rows) - min(rows) + 1) * 2 - 1
                pos[name] = (x, new_y)

        G = nx.DiGraph()
        G.add_nodes_from(labels.keys())
        G.add_edges_from(edges)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600, node_color="lightblue",
                               edgecolors="gray", linewidths=1)
        polarities = {}
        _extract_polarities(node, polarities)
        edge_colors = []
        for u, v in G.edges():
            key = (u, v)
            pol = polarities.get(key, 1)
            edge_colors.append("green" if pol > 0 else "red")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, arrows=True,
                               arrowsize=12, width=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

        legend_elements = [
            mpatches.Patch(color="green", label="Positive (+1)"),
            mpatches.Patch(color="red", label="Negative (-1)"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
        ax.set_title(title_txt)
        ax.axis("off")
    fig.tight_layout()
    return fig


def _flatten_tree(node, pos, labels, edges, x, y, depth):
    name = node["name"]
    label = name.replace("_", "\n") if len(name) > 10 else name
    if name not in pos:
        pos[name] = (x, y)
        labels[name] = label
    children = node.get("children", [])
    for i, child in enumerate(children):
        child_name = child["name"]
        child_x = x + 1.5
        child_y = y - len(children) / 2 + i + 0.5
        pos[child_name] = (child_x, child_y)
        labels[child_name] = child_name.replace("_", "\n") if len(child_name) > 10 else child_name
        edges.append((name, child_name))
        _flatten_tree(child, pos, labels, edges, child_x, child_y, depth + 1)


def _extract_polarities(node, polarities, parent_name=None):
    name = node["name"]
    pol = node.get("polarity", 1)
    if parent_name is not None:
        polarities[(parent_name, name)] = pol
    for child in node.get("children", []):
        _extract_polarities(child, polarities, name)


def make_feedback_plot(analysis, max_loops: int = 8) -> plt.Figure:
    loops = analysis.loops[:max_loops]
    n = len(loops)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, loop in enumerate(loops):
        ax = axes[idx]
        G = nx.DiGraph()
        for i in range(len(loop.nodes)):
            src = loop.nodes[i]
            dst = loop.nodes[(i + 1) % len(loop.nodes)]
            edge_key = (src, dst)
            pol = loop.edge_polarities.get(edge_key, 1)
            G.add_edge(src, dst, polarity=pol)

        pos = nx.circular_layout(G)
        edge_colors = []
        for u, v in G.edges():
            pol = G.edges[u, v]["polarity"]
            edge_colors.append("green" if pol > 0 else "red")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=400, node_color="lightyellow",
                               edgecolors="gray")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, arrows=True,
                               arrowsize=10, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
        pol_label = "R" if loop.polarity == "reinforcing" else "B"
        ax.set_title(f"{loop.name}  [{pol_label}]  ({len(loop.nodes)} nodes)", fontsize=9)
        ax.axis("off")

    for idx in range(len(loops), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"Feedback Loops (showing {len(loops)} of {len(analysis.loops)})", fontsize=12)
    fig.tight_layout()
    return fig


def make_scenario_comparison_plot(scenarios, results, stock_names: list[str]) -> plt.Figure:
    n_stocks = min(4, len(stock_names))
    fig, axes = plt.subplots(1, n_stocks, figsize=(5 * n_stocks, 4))
    if n_stocks == 1:
        axes = [axes]
    for i, s in enumerate(stock_names[:n_stocks]):
        ax = axes[i]
        for j, (name, res) in enumerate(zip(scenarios, results)):
            ax.plot(res.times, res["values"][s], color=COLORS[j % len(COLORS)],
                    label=name, linewidth=1.2)
        ax.set_title(s, fontsize=9)
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Scenario Comparison", fontsize=12)
    fig.tight_layout()
    return fig


# ── fpdf2 PDF builder ───────────────────────────────────────────


class InsightPDF(FPDF):
    """Custom PDF with header/footer for insight reports."""

    def _s(self, text: str) -> str:
        """Sanitize text for latin-1 encoding."""
        return text.replace("\u2014", "-").replace("\u2013", "-") \
                    .replace("\u2018", "'").replace("\u2019", "'") \
                    .replace("\u201c", '"').replace("\u201d", '"') \
                    .replace("\u2026", "...").replace("\u2022", "*") \
                    .replace("\u00b0", " deg") \
                    .encode("latin-1", errors="replace").decode("latin-1")

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, self._s("Cognitive Engine - Insights Report"), align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, self._s(f"Page {self.page_no()}/{{nb}}"), align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 60, 120)
        self.cell(0, 12, self._s(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, self._s(text))
        self.ln(2)

    def key_value_row(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(60, 6, self._s(key))
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, self._s(value), new_x="LMARGIN", new_y="NEXT")

    def add_image_page(self, title: str, fig: plt.Figure):
        self.add_page()
        self.section_title(title)
        buf = fig_to_bytes(fig)
        img_w = 170
        self.image(buf, x=self.l_margin, w=img_w)

    def safe_cell(self, w, h, text, **kwargs):
        self.cell(w, h, self._s(text), **kwargs)

    def safe_multi_cell(self, w, h, text, **kwargs):
        self.multi_cell(w, h, self._s(text), **kwargs)


def build_report():
    """Run all analyses and produce the PDF."""

    # ── Load model ─────────────────────────────────────────────
    print("Loading model...")
    model = parse_sysd_file(MODEL_PATH)
    stock_names = [s.name for s in model.stocks]
    aux_names = [a.name for a in model.aux_vars]
    print(f"  {model.name}: {len(stock_names)} stocks, {len(aux_names)} auxes")

    # ── Validation ─────────────────────────────────────────────
    print("Running validation...")
    val = model.validate(params=set(BASELINE_PARAMS.keys()))
    cc = run_consistency_checks(model)
    print(f"  Errors: {len(val.errors)}, Warnings: {len(val.warnings)}")
    print(f"  Consistency: {cc.is_valid} ({cc.checks_passed}/{cc.checks_run})")

    # ── Baseline simulation ─────────────────────────────────────
    print("Running baseline simulation...")
    baseline_result = model.simulate(method="euler", t_span=SIM_T_SPAN, dt=SIM_DT,
                                     params=BASELINE_PARAMS)
    print(f"  {len(baseline_result.times)} steps")

    # ── Causal trace ────────────────────────────────────────────
    print("Computing causal traces...")
    final_state = {s: baseline_result["values"][s][-1] for s in stock_names}
    causal_data = {}
    for s in stock_names:
        trace = causal_trace(model, s, state=final_state, max_depth=4)
        if trace:
            causal_data[s] = trace
    print(f"  {len(causal_data)} stocks traced")

    # ── Feedback loops ─────────────────────────────────────────
    print("Detecting feedback loops...")
    loop_analysis = detect_feedback_loops(model)
    print(f"  {len(loop_analysis.loops)} loops found")

    # ── Scenario comparison ─────────────────────────────────────
    print("Running scenario comparison...")
    scenario_results = []
    scenario_names = [s[0] for s in SCENARIO_PARAMS]
    scenario_param_sets = [s[1] for s in SCENARIO_PARAMS]
    for name, params in SCENARIO_PARAMS:
        print(f"  Scenario: {name}")
        res = model.simulate(method="euler", t_span=SIM_T_SPAN, dt=SIM_DT, params=params)
        scenario_results.append(res)

    # ── Units check ─────────────────────────────────────────────
    print("Checking units...")
    checker = UnitChecker()
    unit_result = checker.check(model)
    print(f"  {len(unit_result.errors)} unit errors, {len(unit_result.warnings)} warnings")

    # ── Build PDF ─────────────────────────────────────────────
    print(f"\nGenerating PDF: {OUTPUT_PDF}")
    pdf = InsightPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Page 1: Title + Model Summary ──────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 60, 120)
    pdf.safe_cell(0, 15, model.name, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.safe_cell(0, 6, "Generated by Cognitive Engine", new_x="LMARGIN", new_y="NEXT")
    pdf.safe_cell(0, 6, f"{datetime.now():%Y-%m-%d %H:%M}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.section_title("Model Summary")
    pdf.key_value_row("Stocks", f"{len(stock_names)}")
    pdf.key_value_row("Auxiliaries", f"{len(aux_names)}")
    pdf.key_value_row("Time span", f"{model.t_span[0]} to {model.t_span[1]}")
    pdf.key_value_row("Time step (dt)", f"{model.dt}")
    pdf.key_value_row("Integration", f"Euler ({SIM_DT} dt)")
    pdf.key_value_row("Feedback loops", f"{len(loop_analysis.loops)}")

    val_badge = "PASS" if val.is_valid else "FAIL"
    val_color = (40, 160, 40) if val.is_valid else (200, 40, 40)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*val_color)
    pdf.safe_cell(60, 6, "Validation")
    pdf.set_font("Helvetica", "", 10)
    pdf.safe_cell(0, 6, f"{val_badge} ({len(val.errors)} errors, {len(val.warnings)} warnings)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(40, 40, 40)

    cc_badge = "PASS" if cc.is_valid else "FAIL"
    cc_color = (40, 160, 40) if cc.is_valid else (200, 40, 40)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*cc_color)
    pdf.safe_cell(60, 6, "Consistency")
    pdf.set_font("Helvetica", "", 10)
    pdf.safe_cell(0, 6, f"{cc_badge} ({cc.checks_passed}/{cc.checks_run})",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(40, 40, 40)

    pdf.ln(4)
    pdf.section_title("Parameters")
    pdf.set_font("Helvetica", "", 9)
    for k, v in sorted(BASELINE_PARAMS.items()):
        pdf.safe_cell(60, 5, k)
        pdf.safe_cell(0, 5, str(v), new_x="LMARGIN", new_y="NEXT")

    # ── Page 2: Simulation Chart ───────────────────────────────
    print("  Rendering simulation chart...")
    sim_fig = make_timeseries_plot(baseline_result, KEY_STOCKS,
                                   f"{model.name} — Baseline Scenario")
    pdf.add_image_page("Simulation Timeseries", sim_fig)

    # ── Page 3: Causal Tree ────────────────────────────────────
    print("  Rendering causal tree...")
    causal_fig = make_causal_tree_plot(model, final_state, stock_names)
    pdf.add_image_page("Causal Analysis", causal_fig)

    # ── Page 4: Feedback Loops ─────────────────────────────────
    print("  Rendering feedback loops...")
    loop_fig = make_feedback_plot(loop_analysis)
    pdf.add_image_page("Feedback Loops", loop_fig)

    # ── Page 5: Scenario Comparison ────────────────────────────
    print("  Rendering scenario comparison...")
    scenario_fig = make_scenario_comparison_plot(scenario_names, scenario_results, KEY_STOCKS)
    pdf.add_image_page("Scenario Comparison", scenario_fig)

    # ── Page 6: Validation & Consistency ───────────────────────
    print("  Rendering validation page...")
    pdf.add_page()
    pdf.section_title("Validation Report")

    if val.errors:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(200, 40, 40)
        pdf.safe_cell(0, 6, f"Errors ({len(val.errors)})", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 9)
        for v in val.errors:
            pdf.safe_multi_cell(0, 5, f"  [{v.location}] {v.message}")
        pdf.ln(3)

    if val.warnings:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(200, 160, 40)
        pdf.safe_cell(0, 6, f"Warnings ({len(val.warnings)})", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 9)
        for v in val.warnings:
            pdf.set_x(pdf.l_margin)
            pdf.safe_multi_cell(0, 5, f"  [{v.location}] {v.message}")
        pdf.ln(3)

    pdf.section_title("Consistency Check")
    for v in cc.violations:
        lvl_color = (200, 40, 40) if v.level == "error" else (200, 160, 40)
        pdf.set_text_color(*lvl_color)
        pdf.set_font("Helvetica", "B", 9)
        pdf.safe_cell(30, 5, f"[{v.level.upper()}]")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.safe_cell(0, 5, v.message, new_x="LMARGIN", new_y="NEXT")
    if not cc.violations:
        pdf.set_text_color(40, 160, 40)
        pdf.set_font("Helvetica", "", 10)
        pdf.safe_cell(0, 6, "No consistency violations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(40, 40, 40)

    pdf.section_title("Unit Check")
    if unit_result.violations:
        for v in unit_result.violations:
            lvl_color = (200, 40, 40) if v.severity == "error" else (200, 160, 40)
            pdf.set_text_color(*lvl_color)
            pdf.set_font("Helvetica", "B", 9)
            pdf.safe_cell(30, 5, f"[{v.severity.upper()}]")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            pdf.safe_multi_cell(0, 5, v.message)
    else:
        pdf.set_text_color(40, 160, 40)
        pdf.set_font("Helvetica", "", 10)
        pdf.safe_cell(0, 6, "No unit violations — all ~Unit~ annotations consistent",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(40, 40, 40)

    # ── Page 7: Scenario Details ───────────────────────────────
    print("  Rendering scenario details...")
    pdf.add_page()
    pdf.section_title("Scenario Comparison — Final Values")

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    col_w = 170 // (len(scenario_names) + 1)
    col_w = max(col_w, 25)
    pdf.safe_cell(col_w, 7, "Stock", border=1, fill=True)
    for name in scenario_names:
        pdf.safe_cell(col_w, 7, name[:12], border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(40, 40, 40)

    # Table rows
    pdf.set_font("Helvetica", "", 8)
    for s in stock_names:
        pdf.safe_cell(col_w, 6, s[:col_w // 2], border=1)
        for res in scenario_results:
            fval = res["values"][s][-1]
            pdf.safe_cell(col_w, 6, f"{fval:,.0f}", border=1, align="C")
        pdf.ln()
        if pdf.get_y() > 260:
            pdf.add_page()

    pdf.ln(6)
    pdf.section_title("Scenario Comparison — Summary Stats")
    for si, name in enumerate(scenario_names):
        res = scenario_results[si]
        pdf.set_font("Helvetica", "B", 9)
        pdf.safe_cell(0, 6, name, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        for s in stock_names:
            vals = res["values"][s]
            pdf.safe_cell(80, 5, f"  {s}:", align="R")
            pdf.safe_cell(40, 5, f"min={min(vals):,.0f}", align="C")
            pdf.safe_cell(40, 5, f"max={max(vals):,.0f}", align="C")
            pdf.safe_cell(0, 5, f"end={vals[-1]:,.0f}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # ── Save ───────────────────────────────────────────────────
    pdf.output(OUTPUT_PDF)
    print(f"\nDone — {OUTPUT_PDF} ({pdf.pages_count} pages)")
    print(f"  Model: {model.name}")
    print(f"  Stocks: {len(stock_names)}, Loops: {len(loop_analysis.loops)}")
    print(f"  Scenarios: {len(scenario_names)}")
    print(f"  Validation: {'PASS' if val.is_valid else 'FAIL'}")


if __name__ == "__main__":
    build_report()
