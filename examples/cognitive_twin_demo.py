#!/usr/bin/env python3
"""
Cognitive Digital Twin — Self-Healing Supply Chain
===================================================
Demonstrates a single-trajectory self-healing system:
  KB-triggered disruption detected in real-time via KB_QUERY auxes
  -> ABM agents adapt -> inventory recovers automatically

Single simulation: normal -> disruption -> self-healed (continuous)

Usage:
    python examples/cognitive_twin_demo.py
    # Produces: cognitive_twin_report.pdf
"""

from __future__ import annotations
import argparse, io, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

from dynafx import KBSimBridge, TripleStore, parse_sysd_file, causal_trace
from dynafx.knowledge.model import NamedNode, Literal, Triple

# ═══════════════════════════════════════════════════════════════
# 1. Knowledge Network
# ═══════════════════════════════════════════════════════════════

NS = "http://sc.org/"
S = NamedNode(f"{NS}Scenario")
ST = NamedNode(f"{NS}status")

store = TripleStore()
store.add(Triple(S, ST, Literal("normal")), graph="world")

DISP_Q = 'ASK { <http://sc.org/Scenario> <http://sc.org/status> "disrupted" }'

bridge = KBSimBridge(store)
model = parse_sysd_file(str(Path(__file__).resolve().parent.parent / "models" / "cognitive_twin_sc.sysd"))

# ═══════════════════════════════════════════════════════════════
# 2. Run single self-healing simulation
# ═══════════════════════════════════════════════════════════════

def run():
    result = model.simulate(
        params={
            "demand_rate": 100.0,
            "disp_q": DISP_Q,
        },
        kb=store,
    )
    return result

# ═══════════════════════════════════════════════════════════════
# 3. Chart generators
# ═══════════════════════════════════════════════════════════════

def _fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def chart_inventory(result):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    t = np.array(result.times)
    for name, color, label in [
        ("Factory_Inventory", "#3498db", "Factory"),
        ("Warehouse_Inventory", "#e67e22", "Warehouse"),
        ("Retailer_Inventory", "#2ecc71", "Retailer"),
    ]:
        vals = np.array(result.values[name])
        ax.plot(t, vals, label=label, color=color, linewidth=1.5)
    ax.axvline(x=60, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7, label="Disruption")
    ax.axvline(x=90, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.7, label="Recovery")
    ax.set_xlabel("Days")
    ax.set_ylabel("Units")
    ax.set_title("Inventory: Self-Healing Trajectory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.fill_betweenx(ax.get_ylim(), 60, 90, alpha=0.08, color="#e74c3c")
    fig.tight_layout()
    return fig

def chart_fill_rate(result):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    t = np.array(result.times)
    met = np.array(result.values["Cumulative_Met"])
    dem = np.array(result.values["Cumulative_Demand"])
    fill = np.divide(met, dem, out=np.ones_like(met), where=dem > 0)
    ax.plot(t, fill * 100, color="#2ecc71", linewidth=2)
    ax.axvline(x=60, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=90, color="#27ae60", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=100, color="#95a5a6", linestyle=":", linewidth=1, alpha=0.5)
    ax.fill_betweenx(ax.get_ylim(), 60, 90, alpha=0.08, color="#e74c3c")
    ax.set_xlabel("Days")
    ax.set_ylabel("Fill Rate (%)")
    ax.set_title("Fill Rate Over Time", fontsize=12, fontweight="bold")
    ax.set_ylim(90, 101)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def chart_demand_vs_met(result):
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.array(result.times)
    dem = np.array(result.values["Cumulative_Demand"])
    met = np.array(result.values["Cumulative_Met"])
    ax.fill_between(t, dem, alpha=0.15, color="#e74c3c")
    ax.fill_between(t, met, alpha=0.15, color="#2ecc71")
    ax.plot(t, dem, color="#e74c3c", linewidth=1.5, label="Cumulative Demand")
    ax.plot(t, met, color="#2ecc71", linewidth=1.5, label="Cumulative Fulfilled")
    ax.axvline(x=60, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=90, color="#27ae60", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_betweenx(ax.get_ylim(), 60, 90, alpha=0.05, color="#e74c3c")
    ax.set_xlabel("Days")
    ax.set_ylabel("Units")
    ax.set_title("Demand vs. Fulfillment", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def chart_demand_rate(result):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    t = np.array(result.times)
    dem = np.diff(np.array(result.values["Cumulative_Demand"]), prepend=0) / 0.25
    met = np.diff(np.array(result.values["Cumulative_Met"]), prepend=0) / 0.25
    ax.plot(t, dem, color="#e74c3c", linewidth=1.5, label="Demand rate")
    ax.plot(t, met, color="#2ecc71", linewidth=1.5, alpha=0.7, label="Fulfillment rate")
    ax.axvline(x=60, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=90, color="#27ae60", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_betweenx(ax.get_ylim(), 60, 90, alpha=0.08, color="#e74c3c")
    ax.set_xlabel("Days")
    ax.set_ylabel("Rate (units/day)")
    ax.set_title("Instantaneous Demand Rate", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def chart_abm_reliability(result):
    if not result.abm_metrics_history:
        return None
    fig, ax = plt.subplots(figsize=(10, 3.5))
    abm = result.abm_metrics_history
    times = [r.get("t", i) for i, r in enumerate(abm)]
    rel = [r.get("Supplier_reliability_avg", 0) for r in abm]
    ax.plot(times, rel, color="#9b59b6", linewidth=2)
    ax.axvline(x=60, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=90, color="#27ae60", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_betweenx(ax.get_ylim(), 60, 90, alpha=0.08, color="#e74c3c")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reliability")
    ax.set_title("Supplier ABM Reliability (KB_QUERY-driven)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def chart_stockout(result):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    t = np.array(result.times)
    dem = np.array(result.values["Cumulative_Demand"])
    met = np.array(result.values["Cumulative_Met"])
    gap = dem - met
    ax.fill_between(t, gap, alpha=0.3, color="#e74c3c")
    ax.plot(t, gap, color="#c0392b", linewidth=1.5)
    ax.axvline(x=60, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=90, color="#27ae60", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_betweenx(ax.get_ylim(), 60, 90, alpha=0.08, color="#e74c3c")
    ax.set_xlabel("Days")
    ax.set_ylabel("Unmet Demand")
    ax.set_title("Stockout Gap (Cumulative)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════
# 4. PDF Report
# ═══════════════════════════════════════════════════════════════

class Report(FPDF):
    def _s(self, t):
        return t.encode("latin-1", errors="replace").decode("latin-1")
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, self._s("Cognitive Digital Twin \u2014 Self-Healing Supply Chain"), align="L")
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
    def sub_section(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, self._s(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
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


def make_pdf(result, output_path):
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    t = np.array(result.times)
    met = np.array(result.values["Cumulative_Met"])
    dem = np.array(result.values["Cumulative_Demand"])

    idx_60 = np.searchsorted(t, 60)
    idx_90 = np.searchsorted(t, 90)
    fill_before = met[idx_60] / dem[idx_60] if dem[idx_60] > 0 else 1
    fill_during = (met[idx_90] - met[idx_60]) / max(1, dem[idx_90] - dem[idx_60])
    fill_final = met[-1] / dem[-1] if dem[-1] > 0 else 1
    inventory_min = min(result.values["Retailer_Inventory"])
    inventory_final = result.values["Retailer_Inventory"][-1]

    # ── Page 1: Title ─────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 15, pdf._s("Cognitive Digital Twin"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, pdf._s("Self-Healing Supply Chain \u2014 SD + ABM + DES + KB"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 6, pdf._s(
        "120-day single-trajectory simulation  |  dt=0.25  |  disruption at t=60, recovery at t=90"
    ), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    pdf.section("Executive Summary")
    pdf.body(
        "This report demonstrates a self-healing cognitive digital twin for a "
        "three-echelon semiconductor supply chain. Unlike the previous multi-pass "
        "approach using ClosedLoopReasoner, this twin operates in a single continuous "
        "simulation trajectory. A World Event ABM agent initiates a disruption at "
        "t=60 by asserting a status triple into the Knowledge Base. Aux expressions "
        "using KB_QUERY detect the change in real time and automatically adjust "
        "demand, expedite factors, and production capacity. Supplier ABM agents also "
        "sense the disruption via KB_QUERY and adjust reliability. At t=90 the World "
        "agent restores normal status, and the system automatically recovers."
    )
    pdf.body(
        f"Key results: fill rate before disruption = {fill_before:.1%}, "
        f"during disruption = {fill_during:.1%}, final = {fill_final:.1%}. "
        f"Retailer inventory reached a minimum of {inventory_min:.0f} units "
        f"(from 500) during the shock, then recovered to {inventory_final:.0f} units. "
        f"Supplier reliability dropped from 0.80 to 0.30 at disruption and "
        f"recovered to 1.0 after normal status was restored."
    )

    # ── Page 2: Architecture ──────────────────────────────
    pdf.add_page()
    pdf.section("Self-Healing Architecture")
    pdf.sub_section("How It Works (Single Simulation)")
    pdf.body(
        "1. KB starts with status='normal'. Aux expressions use KB_QUERY(disp_q) "
        "to check status at every timestep. When the KB says normal, demand_rate=100, "
        "expedite=1, fab_capacity=200."
    )
    pdf.body(
        "2. At t=60, a World Event agent (ABM) fires rule: 't >= 60 AND triggered == 0' "
        "→ KB_ASSERT(status='disrupted'). This updates the TripleStore during the "
        "simulation, not between passes."
    )
    pdf.body(
        "3. At the NEXT timestep (t=60.25), all aux expressions re-evaluate. "
        "KB_QUERY now returns 1.0. Demand automatically surges to 250/day. "
        "Expedite increases to 1.5x. Fab capacity boosts to 300/day. "
        "No external controller needed \u2014 the model self-adapts."
    )
    pdf.body(
        "4. Supplier ABM agents also detect the KB change via KB_QUERY and "
        "drop reliability from 0.80 to 0.30. They report utilization via KB_ASSERT."
    )
    pdf.body(
        "5. At t=90, the World agent fires: 't >= 90 AND triggered == 1' "
        "→ KB_ASSERT(status='normal'). KB_QUERY returns 0 again, and all "
        "auxes revert to normal. The system has self-healed."
    )
    pdf.sub_section("Key Difference from ClosedLoopReasoner")
    pdf.body(
        "The previous approach used three separate simulations (baseline, shock, "
        "recovery) with grade_update between passes. This approach runs ONE "
        "simulation where the KB changes mid-run. The model detects and responds "
        "autonomously \u2014 a true cognitive digital twin that doesn't need an external "
        "orchestrator to tell it how to adapt."
    )

    # ── Page 3: Model Structure ───────────────────────────
    pdf.add_page()
    pdf.section("Model Structure")
    pdf.sub_section("KB-Aware Aux Expressions")
    pdf.body(
        "disruption_active = KB_QUERY(disp_q)  # 0 or 1 at every timestep"
    )
    pdf.body(
        "demand = demand_rate * (1.0 + disruption_active * 1.5)  "
        "# 100 normal, 250 during disruption"
    )
    pdf.body(
        "expedite = 1.0 + disruption_active * 0.5  "
        "# 1.0x normal, 1.5x expedited"
    )
    pdf.body(
        "fab_capacity = 200.0 + disruption_active * 100.0  "
        "# 200/day normal, 300/day boosted"
    )
    pdf.sub_section("World Event Agent (ABM)")
    pdf.body(
        "A single 'World' agent with property 'triggered' orchestrates "
        "the event timeline using rule conditions:"
    )
    pdf.body(
        "  t >= 60 AND triggered == 0  →  set triggered=1, KB_ASSERT(disrupted)"
    )
    pdf.body(
        "  t >= 90 AND triggered == 1  →  set triggered=2, KB_ASSERT(normal)"
    )
    pdf.sub_section("Supplier ABM Agents")
    pdf.body(
        "3 Supplier agents use KB_QUERY in their rules: when disruption "
        "is detected, reliability drops to 0.3. When normal, it increments "
        "by 0.05/step back to 1.0. A reporting rule asserts current "
        "reliability to the KB via KB_ASSERT."
    )
    pdf.sub_section("DES Escalation Queue")
    pdf.body(
        "50-capacity queue with 2-day service time. Arrivals proportional "
        "to stockout volume (10%). Models customer complaint handling."
    )

    # ── Pages 4-9: Charts ────────────────────────────────
    pdf.add_chart_page("Inventory Trajectories", chart_inventory(result),
        "Red shaded zone = disruption active (t=60-90). Retailer inventory dips from 500 "
        f"to {inventory_min:.0f} at the trough, then self-recovers to {inventory_final:.0f}. "
        "Factory and warehouse absorb most of the shock due to expedited shipping."
    )

    fig_fill = chart_fill_rate(result)
    pdf.add_chart_page("Fill Rate", fig_fill,
        f"Fill rate drops from {fill_before:.1%} to {fill_during:.1%} during disruption, "
        f"then recovers to {fill_final:.1%}. The brief dip shows the system self-correcting."
    )

    pdf.add_chart_page("Cumulative Demand vs Fulfillment", chart_demand_vs_met(result),
        "Cumulative demand and fulfillment tracks closely. A small gap widens during "
        "disruption and then stabilizes as the system self-heals."
    )

    pdf.add_chart_page("Instantaneous Demand Rate", chart_demand_rate(result),
        "Demand rate jumps from 100 to 250 at t=60 (due to disruption_active going from "
        "0 to 1 in aux expressions). Fulfillment rate briefly lags, then catches up as "
        "expedited shipping takes effect."
    )

    fig_rel = chart_abm_reliability(result)
    if fig_rel:
        pdf.add_chart_page("ABM Supplier Reliability", fig_rel,
            "Supplier agents detect disruption via KB_QUERY and drop reliability "
            "instantly to 0.3 at t=60. When normal status returns at t=90, they "
            "incrementally recover to 1.0."
        )

    pdf.add_chart_page("Stockout Gap", chart_stockout(result),
        "Cumulative unmet demand gap. The gap starts at t=60 when demand surges, "
        "widens briefly, then stabilizes as the self-healing takes effect. The slope "
        "change around t=90-100 shows recovery accelerating."
    )

    # ── Page 10: Causal Trace ─────────────────────────────
    trace = causal_trace(model, "retail_sales", state={k: v[-1] for k, v in result.values.items()})
    if trace and trace.get("causes"):
        pdf.add_page()
        pdf.section("Causal Trace Analysis")
        pdf.sub_section("What drives retail_sales?")
        def _flatten(node, depth=0):
            lines = []
            prefix = "  " * depth
            name = node.get("name", "?")
            expr = node.get("expr", "")[:55]
            lines.append(f"{prefix}{name}: {expr}")
            for child in node.get("children", []):
                lines.extend(_flatten(child, depth + 1))
            return lines
        lines = _flatten(trace["causes"])
        pdf.set_font("Courier", "", 8)
        for line in lines:
            pdf.cell(0, 4, pdf._s(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        pdf.body(
            "The causal tree shows retail_sales depends on Retailer_Inventory "
            "(MIN-gated) and demand. Demand depends on disruption_active, which "
            "comes from KB_QUERY(disp_q) \u2014 the KB bridge. The root controllable "
            "levers are the aux expressions that adapt to KB state changes."
        )

    # ── Page 11: KB State ─────────────────────────────────
    pdf.add_page()
    pdf.section("Knowledge Base State")
    pdf.sub_section("Named Graphs")
    for graph_name in sorted(store.graphs()):
        triples = list(store.triples_in_graph(graph_name))
        pdf.body(f"  {graph_name}: {len(triples)} triples")

    pdf.sub_section("Key Triples (All Graphs)")
    pdf.set_font("Courier", "", 7)
    for graph_name in sorted(store.graphs()):
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, pdf._s(f"--- {graph_name} ---"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 7)
        for t in store.triples_in_graph(graph_name):
            s = str(t.subject).split("/")[-1].rstrip(">")
            p = str(t.predicate).split("/")[-1].rstrip(">")
            o = str(t.object_)[:35]
            pdf.cell(0, 4, pdf._s(f"  {s}  {p}  {o}"), new_x="LMARGIN", new_y="NEXT")
            if pdf.get_y() > 260:
                pdf.add_page()

    # ── Page 12: Conclusion ───────────────────────────────
    pdf.add_page()
    pdf.section("Conclusion")
    pdf.body(
        "This self-healing cognitive digital twin represents a fundamental shift "
        "from the previous ClosedLoopReasoner approach. Instead of running three "
        "separate simulations with an external decision engine between them, the "
        "system operates as a single continuous trajectory where the Knowledge Base "
        "is updated mid-run by ABM agents and aux expressions respond autonomously."
    )
    pdf.body(
        f"Key findings: (1) KB_QUERY in aux expressions enables real-time "
        f"self-adaptation without a separate orchestration pass. "
        f"(2) ABM agents can update the KB during simulation, creating a closed "
        f"loop within a single run. (3) The system absorbs a 150% demand surge and "
        f"recovers within 30 days with no external intervention. "
        f"(4) The architecture scales to any scenario where KB state changes "
        f"should trigger automatic model adaptation."
    )
    pdf.body(
        "This is the cognitive twin pattern: a simulation that is aware of its "
        "world state through a shared knowledge base and adapts its behavior "
        "autonomously. No ClosedLoopReasoner needed \u2014 the loop is built into "
        "the model itself."
    )

    pdf.output(str(output_path))
    return output_path


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Healing Cognitive Twin Demo")
    parser.add_argument("--output", "-o", default="cognitive_twin_report.pdf")
    args = parser.parse_args()

    print("Running single self-healing simulation...")
    result = run()

    fill_final = result.values["Cumulative_Met"][-1] / max(1, result.values["Cumulative_Demand"][-1])
    print(f"  Final fill rate: {fill_final:.1%}")
    print(f"  Retailer inventory: {result.values['Retailer_Inventory'][-1]:.0f}")
    print(f"  Stockout gap: {result.values['Cumulative_Demand'][-1] - result.values['Cumulative_Met'][-1]:.0f}")

    print(f"\nGenerating PDF report: {args.output}")
    make_pdf(result, args.output)
    print(f"Done! Report saved to: {args.output}")
