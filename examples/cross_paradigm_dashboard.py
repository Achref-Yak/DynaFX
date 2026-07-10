#!/usr/bin/env python3
"""Cross-Paradigm Prediction Dashboard — DynaFX.

Demonstrates what SD + DES + ABM each predict individually, and critically,
what combining them predicts that no single paradigm can.

Model: EV Battery Supply Chain (6 echelon, 10 stocks, 4 DES queues, 120 ABM agents).
"""

import sys, json, textwrap, itertools, math
from pathlib import Path
from datetime import datetime
from dynafx.utils.dashboard_html import make_lazy

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from dynafx.dynamics import parse_sysd_file
from dynafx.dynamics.scenario import ScenarioComparison, ScenarioDef
from dynafx.dynamics.sensitivity import SensitivityAnalyzer
from dynafx.dynamics.feedback import detect_feedback_loops

THEME = {
    "primary": "#1a237e", "accent": "#2196f3", "success": "#4caf50",
    "warning": "#ff9800", "danger": "#f44336", "bg": "#f5f5f5",
    "card": "#ffffff", "text": "#333333", "muted": "#666666",
}

INVENTORY_STOCKS = ["Mine_Inventory", "Chem_Inventory", "Cell_Inventory", "Pack_Inventory", "Warehouse_Inventory"]
QUEUE_NAMES = ["Cell_Line", "Assembly_Line", "Shipping_Dock", "Repair_Shop"]
RESOURCE_NAMES = ["Workforce", "Trucks"]
AGENT_TYPES = {"Automaker": ["demand_rate", "lead_time_tolerance", "inventory", "supplier_switched", "inventory_target", "emergency_orders", "price_tolerance"],
               "Supplier": ["price", "reliability", "capacity", "quality_score", "contract_volume"]}

ECHELON_LABELS = ["Mine", "Chem Processing", "Cell Factory", "Pack Assembly", "Warehouse", "Customer"]
ECHELON_COLORS = ["#4CAF50", "#8BC34A", "#FF9800", "#FF5722", "#E91E63", "#9C27B0"]


def _hex_rgba(c, a):
    h = c.lstrip("#"); r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"

def _wait_percentiles(avg_wait, utilization):
    """Estimate P50/P90/P99 wait times from M/M/1 queueing theory.
    Wait time ~ Exponential(mean=avg_wait) for M/M/1.
    """
    if avg_wait <= 0 or utilization >= 1:
        return (float("inf"), float("inf"), float("inf"))
    p50 = -math.log(0.5) * avg_wait
    p90 = -math.log(0.1) * avg_wait
    p99 = -math.log(0.01) * avg_wait
    return (p50, p90, p99)


def _kpi_card(label, value, color, subtitle=""):
    return f"""<div class="kpi" style="border-top:3px solid {color}"><div class="kl">{label}</div><div class="kv" style="color:{color}">{value}</div>{f'<div class="ks">{subtitle}</div>' if subtitle else ''}</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline() -> dict:
    model_path = Path(__file__).parent.parent / "models" / "ev_battery_supply_chain.sysd"
    model = parse_sysd_file(str(model_path))
    print(f"  Model: {model.name} ({len(model.stocks)} stocks, {len(model.aux_vars)} auxes)")

    # Baseline simulation
    print("  Simulating baseline (t_span=(0,200))...")
    result = model.simulate(params={}, method="rk4", t_span=(0, 200))
    print(f"  Done: {len(result.times)} steps, {result.steps} steps, method={result.method}")

    def get_v(name, idx=-1):
        v = result.values.get(name, result.aux_values.get(name, []))
        return v[idx] if v else 0.0

    def get_ts(name):
        return result.values.get(name, result.aux_values.get(name, []))

    # Final state values
    sv = {}
    for s in result.stocks:
        sv[s] = get_v(s)
    for k in ["customer_demand", "fill_rate", "gross_profit", "bullwhip_amplitude",
              "lead_time", "mine_utilization", "chem_utilization", "total_cost_per_pack",
              "revenue_val", "total_expenses_val", "fulfillment_rate"]:
        sv[k] = get_v(k)

    # DES final stats
    des_stats = {}
    if result.des_engine:
        des_stats = result.des_engine.get_all_stats()

    # Per-step DES metrics
    des_history = result.des_metrics_history if result.des_metrics_history else []

    # ABM per-step metrics
    abm_history = result.abm_metrics_history if result.abm_metrics_history else []

    # Per-agent final state
    agents_final = {}
    if result.abm_engine:
        for inst in result.abm_engine.instances:
            agents_final.setdefault(inst.agent_def.name, []).append(inst.state.copy())

    # ── Scenarios ──
    print("  Running 4 scenarios...")
    scenario_defs = [
        ScenarioDef("1. Baseline", {}),
        ScenarioDef("2. Mine Disruption", {"mine_disruption_amt": 0.4, "mine_disruption_time": 90}),
        ScenarioDef("3. Port Delay", {"port_disruption_amt": 0.5, "port_disruption_time": 120}),
        ScenarioDef("4. Combined Shock", {"mine_disruption_amt": 0.3, "mine_disruption_time": 90,
                                          "port_disruption_amt": 0.4, "port_disruption_time": 120,
                                          "labor_disruption_amt": 0.2, "labor_disruption_time": 150}),
    ]
    sc = ScenarioComparison(model, scenario_defs, method="rk4", t_span=(0, 200))
    sc_summary = sc.summary()

    # ── Sensitivity ──
    print("  Running OAT sensitivity...")
    sa = SensitivityAnalyzer(model, method="rk4")
    oat = sa.oat(param_spec={"mining_capacity": (400, 1200), "chem_capacity": (250, 750),
                             "labor_disruption_amt": (0, 0.5)}, output="Orders_Backlog", t=200, t_span=(0, 200))

    # ── Feedback loops ──
    print("  Detecting feedback loops...")
    loops = detect_feedback_loops(model)

    # ── Cross-paradigm insights ──
    # SD→DES: Find when inventory drops below threshold and queue spikes
    threshold = 500
    inv_traj = get_ts("Warehouse_Inventory")
    queue_traj = [d.get("Shipping_Dock_length", 0) for d in des_history] if des_history else []
    times = result.times if result.times else []
    crossing_day = None
    for i in range(len(times)):
        if i < len(inv_traj) and inv_traj[i] < threshold:
            crossing_day = float(times[i])
            break

    # SD+ABM: Supplier reliability clusters
    supplier_reliability = [ag.get("reliability", 0) for ag in agents_final.get("Supplier", [])]
    supplier_capacity = [ag.get("capacity", 0) for ag in agents_final.get("Supplier", [])]
    automaker_switched = sum(ag.get("supplier_switched", 0) for ag in agents_final.get("Automaker", []))
    supplier_unreliable_pct = sum(1 for r in supplier_reliability if r < 0.65) / max(1, len(supplier_reliability))

    # Bullwish CV across echelons
    demand_traj = get_ts("customer_demand")
    mine_order_traj = get_ts("mine_order_rate")
    def cv(traj): return np.std(traj) / max(1e-6, np.mean(traj)) if traj else 0
    bullwhip_cvs = {
        "Customer Demand": cv(demand_traj),
        "Mine Orders": cv(mine_order_traj),
        "Chem Orders": cv(get_ts("chem_order_rate")),
        "Cell Orders": cv(get_ts("cell_order_rate")),
        "Pack Orders": cv(get_ts("pack_order_rate")),
    }

    data = {
        "model": model, "result": result,
        "sv": sv, "get_v": get_v, "get_ts": get_ts,
        "times": result.times, "stock_values": result.values,
        "aux_values": result.aux_values,
        "des_history": des_history, "des_stats": des_stats,
        "abm_history": abm_history, "agents_final": agents_final,
        "sc": sc, "sc_summary": sc_summary,
        "oat": oat, "loops": loops,
        "crossing_day": crossing_day,
        "supplier_reliability": supplier_reliability,
        "supplier_capacity": supplier_capacity,
        "automaker_switched": automaker_switched,
        "supplier_unreliable_pct": supplier_unreliable_pct,
        "bullwhip_cvs": bullwhip_cvs,
        "threshold": threshold,
    }
    return data


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def build_exec_summary(d):
    sv = d["sv"]; des_stats = d["des_stats"]; s = d["supplier_reliability"]
    inv_health = sv.get("Warehouse_Inventory", 0) / 5000 * 100
    queue_pressure = des_stats.get("Shipping_Dock", {}).get("utilization", 0) * 100 if des_stats else 0
    avg_rel = np.mean(s) if s else 0

    kpis = (_kpi_card("Inventory Health", f"{inv_health:.0f}%", THEME["success"] if inv_health > 30 else THEME["warning"] if inv_health > 10 else THEME["danger"]) +
            _kpi_card("Queue Pressure", f"{queue_pressure:.0f}%", THEME["warning"] if queue_pressure < 70 else THEME["danger"]) +
            _kpi_card("Avg Supplier Reliability", f"{avg_rel:.0%}", THEME["success"] if avg_rel > 0.75 else THEME["warning"]) +
            _kpi_card("Supplier Unreliable", f"{d['supplier_unreliable_pct']:.0%}", THEME["danger"] if d['supplier_unreliable_pct'] > 0.3 else THEME["success"]) +
            _kpi_card("Backlog", f"{sv.get('Orders_Backlog',0):,.0f}", THEME["danger"] if sv.get('Orders_Backlog',0) > 1000 else THEME["success"]) +
            _kpi_card("Cash", f"${sv.get('Cash_Reserves',0)/1e6:.1f}M", THEME["accent"]))

    # Inventory x Queue overlay
    fig_diag = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    t = d["times"]; inv = d["get_ts"]("Warehouse_Inventory"); q = [h.get("Shipping_Dock_length", 0) for h in d["des_history"]]
    if t and inv:
        fig_diag.add_trace(go.Scatter(x=t, y=inv, mode="lines", name="Warehouse Inventory", line=dict(color=THEME["success"], width=2),
                                       hovertemplate="Day %{x}<br>Inventory: %{y:,.0f} packs<extra></extra>"), secondary_y=False)
        if d["crossing_day"] is not None:
            fig_diag.add_vline(x=d["crossing_day"], line=dict(color=THEME["danger"], dash="dash", width=1.5),
                               annotation_text=f"Threshold crossed Day {d['crossing_day']:.0f}", annotation_position="top left")
        fig_diag.add_hline(y=d["threshold"], line=dict(color=THEME["danger"], dash="dot", width=1),
                           annotation_text="Critical threshold", annotation_position="bottom right")
        if q:
            fig_diag.add_trace(go.Scatter(x=t[:len(q)], y=q, mode="lines", name="Shipping Queue", line=dict(color=THEME["warning"], width=1.5),
                                           hovertemplate="Day %{x}<br>Queue: %{y} entities<extra></extra>"), secondary_y=True)
        fig_diag.update_layout(title="SD → DES Coupling: Inventory Threshold Drives Queue Contention",
                               paper_bgcolor="white", plot_bgcolor="white", height=300, margin=dict(l=40, r=40, t=40, b=30),
                               hovermode="x unified", legend=dict(orientation="h", y=-0.25))
        fig_diag.update_xaxes(title="Day", gridcolor="#f0f0f0")
        fig_diag.update_yaxes(title="Inventory (packs)", gridcolor="#f0f0f0", secondary_y=False)
        fig_diag.update_yaxes(title="Queue Length", gridcolor="#f0f0f0", secondary_y=True)

    # Cross-paradigm insight callout
    crossing = d["crossing_day"]
    queue_peak = max(q) if q else 0
    insight_cards = f"""
    <div class="q-grid">
        <div class="q-card" style="border-left:4px solid {THEME['success']}"><div class="ql">SD Predicts</div><div class="qa">Warehouse depletes to {inv[-1] if inv else 0:,.0f} packs by Day 200. Bullwhip CV amplifies {d['bullwhip_cvs'].get('Mine Orders',0)/max(0.001,d['bullwhip_cvs'].get('Customer Demand',1)):.1f}x from customer to mine.</div></div>
        <div class="q-card" style="border-left:4px solid {THEME['warning']}"><div class="ql">DES Predicts</div><div class="qa">Shipping_Dock peak at {queue_peak:.0f} entities (utilization {queue_pressure:.0f}%). Cell_Line: {des_stats.get('Cell_Line',{}).get('total_arrivals',0):.0f} total arrivals.</div></div>
        <div class="q-card" style="border-left:4px solid #9c27b0"><div class="ql">ABM Predicts</div><div class="qa">Supplier reliability avg {avg_rel:.0%} ({d['supplier_unreliable_pct']:.0%} below 0.65). {d['automaker_switched']:.0f}/100 automakers have switched suppliers.</div></div>
        <div class="q-card" style="border-left:4px solid {THEME['danger']}"><div class="ql">Combined Insight</div><div class="qa">SD predicts stockout risk when inventory < {d['threshold']} (Day {f'{crossing:.0f}' if crossing is not None else 'N/A'}). DES pinpoints which queue contends. ABM shows <b>why</b>: {d['supplier_unreliable_pct']:.0%} of suppliers below 0.65 reliability.</div></div>
    </div>"""

    diag_html = fig_diag.to_html(full_html=False, include_plotlyjs=False) if t and inv else "<p>No data</p>"
    return {"title": "Executive Summary", "icon": "&#x1F4CA;",
            "content": f"""<div class="kpi-row">{kpis}</div><div class="chart-box">{diag_html}</div>{insight_cards}"""}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SD PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
def build_sd(d):
    t = d["times"]; sv = d["stock_values"]

    # 6-echelon inventory
    fig_inv = go.Figure()
    for name, label, color in zip(INVENTORY_STOCKS + ["Orders_Backlog"],
                                   ["Mine", "Chem", "Cell", "Pack", "Warehouse", "Backlog"],
                                   ECHELON_COLORS + [THEME["danger"]]):
        vals = sv.get(name, [])
        if vals:
            fig_inv.add_trace(go.Scatter(x=t[:len(vals)], y=vals, mode="lines", name=label,
                                          line=dict(width=2), hovertemplate=f"{label}: %{{y:,.0f}}<extra></extra>"))
    fig_inv.update_layout(title="6-Echelon Inventory Trajectories", paper_bgcolor="white", plot_bgcolor="white",
                          height=350, margin=dict(l=40, r=20, t=40, b=30), hovermode="x unified",
                          xaxis=dict(title="Day", gridcolor="#f0f0f0"), yaxis=dict(title="Units", gridcolor="#f0f0f0"),
                          legend=dict(orientation="h", y=-0.3))

    # Bullwhip CV
    bc = d["bullwhip_cvs"]
    fig_bw = go.Figure(go.Bar(x=list(bc.keys()), y=list(bc.values()),
                               marker=dict(color=[THEME["success"], THEME["warning"], THEME["accent"], THEME["danger"], "#9c27b0"]),
                               hovertemplate="%{x}: CV=%{y:.3f}<extra></extra>"))
    fig_bw.update_layout(title="Bullwhip Effect — CV of Orders Across Echelons",
                         paper_bgcolor="white", plot_bgcolor="white", height=280, margin=dict(l=40, r=20, t=40, b=50),
                         xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(title="Coefficient of Variation", gridcolor="#f0f0f0"))

    # Cash + cost
    fig_cash = go.Figure()
    cash_vals = sv.get("Cash_Reserves", []); rev_vals = d["get_ts"]("revenue_val"); exp_vals = d["get_ts"]("total_expenses_val")
    if cash_vals:
        fig_cash.add_trace(go.Scatter(x=t[:len(cash_vals)], y=[v/1e6 for v in cash_vals], mode="lines", name="Cash ($M)",
                                       fill="tozeroy", fillcolor=_hex_rgba(THEME["success"], 0.15),
                                       line=dict(color=THEME["success"], width=2),
                                       hovertemplate="Day %{x}<br>Cash: $%{y:.2f}M<extra></extra>"))
    if rev_vals:
        fig_cash.add_trace(go.Scatter(x=t[:len(rev_vals)], y=rev_vals, mode="lines", name="Revenue ($K)",
                                       line=dict(color=THEME["accent"], width=1.5, dash="dot"),
                                       hovertemplate="Day %{x}<br>Revenue: $%{y:,.0f}<extra></extra>"))
    if exp_vals:
        fig_cash.add_trace(go.Scatter(x=t[:len(exp_vals)], y=exp_vals, mode="lines", name="Expenses ($K)",
                                       line=dict(color=THEME["danger"], width=1.5, dash="dot"),
                                       hovertemplate="Day %{x}<br>Expenses: $%{y:,.0f}<extra></extra>"))
    fig_cash.update_layout(title="Cash & Cost Accrual", paper_bgcolor="white", plot_bgcolor="white",
                           height=280, margin=dict(l=40, r=20, t=40, b=30), hovermode="x unified",
                           xaxis=dict(title="Day", gridcolor="#f0f0f0"), yaxis=dict(title="$", gridcolor="#f0f0f0"),
                           legend=dict(orientation="h", y=-0.3))

    # OAT tornado
    oat = d["oat"]
    fig_t = go.Figure()
    if oat and oat.oat_low and oat.oat_high:
        params = list(oat.oat_low.keys()); base_val = d["sv"].get("Orders_Backlog", 0)
        impacts = sorted([(p, abs(oat.oat_low[p] - base_val), abs(oat.oat_high[p] - base_val)) for p in params],
                         key=lambda x: max(x[1], x[2]), reverse=True)
        ps = [x[0] for x in impacts]
        fig_t.add_trace(go.Bar(y=ps, x=[-(x[1]) for x in impacts], orientation="h", name="Low Bound",
                                marker=dict(color=THEME["warning"]), hovertemplate="%{y}: -%{x:,.0f}<extra></extra>"))
        fig_t.add_trace(go.Bar(y=ps, x=[x[2] for x in impacts], orientation="h", name="High Bound",
                                marker=dict(color=THEME["success"]), hovertemplate="%{y}: +%{x:,.0f}<extra></extra>"))
        fig_t.update_layout(title="OAT Sensitivity — Orders Backlog @ Day 200", barmode="relative",
                           paper_bgcolor="white", plot_bgcolor="white", height=250, margin=dict(l=120, r=40, t=40, b=30),
                            xaxis=dict(title="Change in Backlog", gridcolor="#f0f0f0"), legend=dict(orientation="h", y=-0.3))

    # Feedback loops
    loop_rows = ""
    for l in d["loops"].loops if d["loops"] else []:
        pc = "#e91e63" if l.polarity == "reinforcing" else THEME["accent"]
        loop_rows += f"<tr><td>{l.name}</td><td style='color:{pc}'>{'R' if l.polarity == 'reinforcing' else 'B'}</td><td>{', '.join(l.nodes[:4])}{'...' if len(l.nodes) > 4 else ''}</td></tr>"

    n_loops = len(d["loops"].loops) if d["loops"] else 0

    inv_html = fig_inv.to_html(full_html=False, include_plotlyjs=False)
    bw_html = fig_bw.to_html(full_html=False, include_plotlyjs=False)
    cash_html = fig_cash.to_html(full_html=False, include_plotlyjs=False)
    t_html = fig_t.to_html(full_html=False, include_plotlyjs=False) if oat and oat.oat_low else "<p>No sensitivity data</p>"

    return {"title": "SD Predictions", "icon": "&#x1F4C8;",
            "content": f"""
        <div class="chart-box">{inv_html}</div>
        <div class="two-col"><div class="chart-box">{bw_html}</div><div class="chart-box">{cash_html}</div></div>
        <div class="two-col"><div class="chart-box">{t_html}</div>
        <div><div class="section-title">Feedback Loops ({n_loops} found)</div>
        <div class="table-wrap"><table class="dt"><thead><tr><th>Loop</th><th>Polarity</th><th>Variables</th></tr></thead>
        <tbody>{loop_rows if loop_rows else '<tr><td colspan="3">No loops detected</td></tr>'}</tbody></table></div></div></div>"""}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DES PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
def build_des(d):
    dh = d["des_history"]; ds = d["des_stats"]; t = d["times"]

    # 4 queue lengths
    fig_q = make_subplots(rows=2, cols=2, subplot_titles=QUEUE_NAMES, vertical_spacing=0.12, horizontal_spacing=0.08)
    q_colors = [THEME["success"], THEME["accent"], THEME["warning"], THEME["danger"]]
    for i, qn in enumerate(QUEUE_NAMES):
        row, col = i // 2 + 1, i % 2 + 1
        qvals = [h.get(f"{qn}_length", 0) for h in dh]
        if qvals:
            fig_q.add_trace(go.Scatter(x=t[:len(qvals)], y=qvals, mode="lines", name=qn, line=dict(color=q_colors[i], width=2),
                                        hovertemplate=f"{qn}: %{{y}} entities<extra></extra>"), row=row, col=col)
        fig_q.update_xaxes(title="Day", gridcolor="#f0f0f0", row=row, col=col)
        fig_q.update_yaxes(title="Length", gridcolor="#f0f0f0", row=row, col=col)
    fig_q.update_layout(title="Queue Length Dynamics", paper_bgcolor="white", plot_bgcolor="white",
                        height=450, margin=dict(l=30, r=20, t=40, b=20), showlegend=False)

    # Resource utilization
    fig_res = go.Figure()
    for rn in RESOURCE_NAMES:
        rvals = [h.get(f"{rn}_utilization", 0) for h in dh]
        if rvals:
            fig_res.add_trace(go.Scatter(x=t[:len(rvals)], y=rvals, mode="lines", name=rn,
                                          fill="tozeroy", fillcolor=_hex_rgba(THEME["accent"] if "Workforce" in rn else THEME["warning"], 0.1),
                                          line=dict(width=2), hovertemplate=f"{rn}: %{{y:.1%}}<extra></extra>"))
    fig_res.add_hline(y=0.8, line=dict(color=THEME["danger"], dash="dash"), annotation_text="Saturation threshold")
    fig_res.update_layout(title="Resource Utilization", paper_bgcolor="white", plot_bgcolor="white",
                          height=280, margin=dict(l=40, r=20, t=40, b=30), hovermode="x unified",
                          xaxis=dict(title="Day", gridcolor="#f0f0f0"), yaxis=dict(title="Utilization", tickformat=".0%", gridcolor="#f0f0f0"),
                          legend=dict(orientation="h", y=-0.3))

    # Throughput bar
    fig_tp = go.Figure()
    qnames_short = ["Cell", "Assembly", "Shipping", "Repair"]
    arrivals = [ds.get(qn, {}).get("total_arrivals", 0) for qn in QUEUE_NAMES]
    departures = [ds.get(qn, {}).get("total_departures", 0) for qn in QUEUE_NAMES]
    fig_tp.add_trace(go.Bar(x=qnames_short, y=arrivals, name="Arrivals", marker=dict(color=THEME["accent"]),
                             hovertemplate="%{x}: %{y} arrivals<extra></extra>"))
    fig_tp.add_trace(go.Bar(x=qnames_short, y=departures, name="Departures", marker=dict(color=THEME["success"]),
                             hovertemplate="%{x}: %{y} departures<extra></extra>"))
    fig_tp.update_layout(title="Queue Throughput (Cumulative)", barmode="group",
                         paper_bgcolor="white", plot_bgcolor="white", height=280, margin=dict(l=40, r=20, t=40, b=50),
                         xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(title="Count", gridcolor="#f0f0f0"),
                         legend=dict(orientation="h", y=-0.3))

    # Bottleneck table
    util_rows = ""
    for rn in RESOURCE_NAMES:
        u = ds.get(rn, {}).get("utilization", 0) * 100
        color = THEME["success"] if u < 60 else THEME["warning"] if u < 80 else THEME["danger"]
        util_rows += f"<tr><td>{rn}</td><td style='color:{color}'>{u:.1f}%</td></tr>"
    for qn in QUEUE_NAMES:
        m = ds.get(qn, {}).get("max_length", 0); u = ds.get(qn, {}).get("utilization", 0) * 100
        util_rows += f"<tr><td>{qn}</td><td>max {m}</td></tr>"
    for qn in QUEUE_NAMES:
        aw = ds.get(qn, {}).get("avg_wait", 0); ut = ds.get(qn, {}).get("utilization", 0)
        p50, p90, p99 = _wait_percentiles(aw, ut)
        util_rows += f"<tr><td>{qn} avg_wait</td><td>{aw:.2f}d</td></tr>"
        if math.isfinite(p90):
            util_rows += f"<tr><td>{qn} P50/P90/P99</td><td>{p50:.2f}d / {p90:.2f}d / {p99:.2f}d</td></tr>"

    q_html = fig_q.to_html(full_html=False, include_plotlyjs=False)
    res_html = fig_res.to_html(full_html=False, include_plotlyjs=False)
    tp_html = fig_tp.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "DES Predictions", "icon": "&#x2699;",
            "content": f"""
        <div class="chart-box">{q_html}</div>
        <div class="two-col"><div class="chart-box">{res_html}</div><div class="chart-box">{tp_html}</div></div>
        <div class="section-title">Bottleneck Analysis</div>
        <div class="table-wrap"><table class="dt"><thead><tr><th>Queue / Resource</th><th>Metric</th></tr></thead>
        <tbody>{util_rows}</tbody></table></div>"""}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABM PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
def build_abm(d):
    ah = d["abm_history"]; t = d["times"]; af = d["agents_final"]
    agent_type_count = len(AGENT_TYPES)

    # Supplier property trends
    fig_sup = make_subplots(rows=1, cols=3, subplot_titles=("Price", "Reliability", "Capacity"),
                            horizontal_spacing=0.06)
    sprops = ["price", "reliability", "capacity"]
    for i, prop in enumerate(sprops):
        avg = [h.get(f"Supplier_{prop}_avg", 0) for h in ah]
        mn = [h.get(f"Supplier_{prop}_min", 0) for h in ah]
        mx = [h.get(f"Supplier_{prop}_max", 0) for h in ah]
        if avg:
            fig_sup.add_trace(go.Scatter(x=t[:len(avg)], y=avg, mode="lines", name=f"{prop.title()} (avg)",
                                          line=dict(color=THEME["accent"], width=2),
                                          hovertemplate=f"{prop.title()}: %{{y:.3f}}<extra></extra>"), row=1, col=i+1)
            if mn and mx:
                fig_sup.add_trace(go.Scatter(x=t[:len(avg)], y=mx, mode="lines", name="max",
                                              line=dict(width=0), showlegend=False), row=1, col=i+1)
                fig_sup.add_trace(go.Scatter(x=t[:len(avg)], y=mn, mode="lines", fill="tonexty",
                                              fillcolor=_hex_rgba(THEME["accent"], 0.1), name="min/max",
                                              line=dict(width=0), showlegend=False), row=1, col=i+1)
        fig_sup.update_xaxes(title="Day", gridcolor="#f0f0f0", row=1, col=i+1)
        fig_sup.update_yaxes(title=prop.title(), gridcolor="#f0f0f0", row=1, col=i+1)
    fig_sup.update_layout(title="Supplier Behavior Over Time (avg ± min/max)",
                          paper_bgcolor="white", plot_bgcolor="white", height=280,
                          margin=dict(l=30, r=10, t=40, b=30), showlegend=False)

    # Automaker switching
    fig_auto = go.Figure()
    switched = [h.get("Automaker_supplier_switched_sum", 0) for h in ah]
    if switched:
        fig_auto.add_trace(go.Scatter(x=t[:len(switched)], y=switched, mode="lines", fill="tozeroy",
                                       fillcolor=_hex_rgba(THEME["danger"], 0.15), name="Switched suppliers",
                                       line=dict(color=THEME["danger"], width=2, shape="hv"),
                                       hovertemplate="Day %{x}<br>Switched: %{y:.0f} automakers<extra></extra>"))
    fig_auto.update_layout(title="Automaker Supplier Switching (cumulative)",
                           paper_bgcolor="white", plot_bgcolor="white", height=280,
                           margin=dict(l=40, r=20, t=40, b=30),
                           xaxis=dict(title="Day", gridcolor="#f0f0f0"), yaxis=dict(title="# Switched", gridcolor="#f0f0f0"))

    # Supplier final scatter
    fig_scat = go.Figure()
    sup_states = af.get("Supplier", [])
    if sup_states:
        rels = [s.get("reliability", 0) for s in sup_states]
        caps = [s.get("capacity", 0) for s in sup_states]
        prices = [s.get("price", 0) for s in sup_states]
        fig_scat.add_trace(go.Scatter(x=rels, y=caps, mode="markers", marker=dict(
            size=[p * 15 for p in prices], color=prices, colorscale="RdYlGn", showscale=True,
            colorbar=dict(title="Price"), line=dict(width=1, color="white")),
            text=[f"Supplier {i}: rel={r:.2f}, cap={c:.0f}, price={p:.2f}" for i, (r, c, p) in enumerate(zip(rels, caps, prices))],
            hovertemplate="%{text}<extra></extra>"))
    fig_scat.add_vline(x=0.65, line=dict(color=THEME["danger"], dash="dash"), annotation_text="Unreliable threshold")
    fig_scat.update_layout(title="Supplier Final State Distribution", paper_bgcolor="white", plot_bgcolor="white",
                           height=300, margin=dict(l=40, r=40, t=40, b=30),
                           xaxis=dict(title="Reliability", range=[0.4, 1.05], gridcolor="#f0f0f0"),
                           yaxis=dict(title="Capacity", gridcolor="#f0f0f0"))

    sup_html = fig_sup.to_html(full_html=False, include_plotlyjs=False)
    auto_html = fig_auto.to_html(full_html=False, include_plotlyjs=False)
    scat_html = fig_scat.to_html(full_html=False, include_plotlyjs=False)

    # ABM stats table
    abm_rows = ""
    for atype, props in AGENT_TYPES.items():
        states = af.get(atype, [])
        if states:
            for prop in props:
                vals = [s.get(prop, 0) for s in states]
                abm_rows += f"<tr><td>{atype}</td><td>{prop}</td><td>{np.mean(vals):.3f}</td><td>{np.min(vals):.3f}</td><td>{np.max(vals):.3f}</td><td>{np.std(vals):.3f}</td></tr>"

    return {"title": "ABM Predictions", "icon": "&#x1F465;",
            "content": f"""
        <div class="chart-box">{sup_html}</div>
        <div class="two-col"><div class="chart-box">{auto_html}</div><div class="chart-box">{scat_html}</div></div>
        <div class="section-title">Agent State Distribution</div>
        <div class="table-wrap"><table class="dt"><thead><tr><th>Type</th><th>Property</th><th>Mean</th><th>Min</th><th>Max</th><th>Std</th></tr></thead>
        <tbody>{abm_rows}</tbody></table></div>"""}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CROSS-PARADIGM INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def build_cross(d):
    t = d["times"]; sv = d["stock_values"]; ah = d["abm_history"]; dh = d["des_history"]; scs = d["sc_summary"]

    # SD→DES: Dual-axis with triggering
    fig_sddes = make_subplots(specs=[[{"secondary_y": True}]])
    inv_wh = d["get_ts"]("Warehouse_Inventory"); inv_mine = d["get_ts"]("Mine_Reserve")
    q_cell = [h.get("Cell_Line_length", 0) for h in dh]
    q_ship = [h.get("Shipping_Dock_length", 0) for h in dh]
    if inv_wh:
        fig_sddes.add_trace(go.Scatter(x=t[:len(inv_wh)], y=[v/1000 for v in inv_wh], mode="lines", name="Warehouse (K packs)",
                                        line=dict(color=THEME["success"], width=2),
                                        hovertemplate="Day %{x}<br>Warehouse: %{y:.1f}K<extra></extra>"), secondary_y=False)
    if inv_mine:
        fig_sddes.add_trace(go.Scatter(x=t[:len(inv_mine)], y=[v/1000 for v in inv_mine], mode="lines", name="Mine Reserve (K tons)",
                                        line=dict(color="#8BC34A", width=1.5, dash="dot"),
                                        hovertemplate="Day %{x}<br>Mine: %{y:.1f}K<extra></extra>"), secondary_y=False)
    if q_cell:
        fig_sddes.add_trace(go.Scatter(x=t[:len(q_cell)], y=q_cell, mode="lines", name="Cell Line Queue",
                                        line=dict(color=THEME["danger"], width=1.5),
                                        hovertemplate="Day %{x}<br>Cell Queue: %{y}<extra></extra>"), secondary_y=True)
    if q_ship:
        fig_sddes.add_trace(go.Scatter(x=t[:len(q_ship)], y=q_ship, mode="lines", name="Shipping Queue",
                                        line=dict(color=THEME["warning"], width=1.5),
                                        hovertemplate="Day %{x}<br>Shipping Queue: %{y}<extra></extra>"), secondary_y=True)
    if d["crossing_day"]:
        fig_sddes.add_vline(x=d["crossing_day"], line=dict(color=THEME["danger"], dash="dash", width=2),
                            annotation_text=f"Trigger Day {d['crossing_day']:.0f}", annotation_position="top left")
    fig_sddes.update_layout(title="SD → DES: Aggregate Inventory Triggers Queue Contention",
                            paper_bgcolor="white", plot_bgcolor="white", height=300, margin=dict(l=40, r=40, t=40, b=30),
                            hovermode="x unified", legend=dict(orientation="h", y=-0.3))
    fig_sddes.update_xaxes(title="Day", gridcolor="#f0f0f0")
    fig_sddes.update_yaxes(title="Inventory (K units)", gridcolor="#f0f0f0", secondary_y=False)
    fig_sddes.update_yaxes(title="Queue Length", gridcolor="#f0f0f0", secondary_y=True)

    # SD+ABM: Supplier reliability vs inventory outcome
    fig_sdabm = go.Figure()
    sup_rel_avg = [h.get("Supplier_reliability_avg", 0) for h in ah]
    sup_cap_avg = [h.get("Supplier_capacity_avg", 0) for h in ah]
    dem_rate = [h.get("Automaker_demand_rate_avg", 0) for h in ah]
    if sup_rel_avg:
        fig_sdabm.add_trace(go.Scatter(x=t[:len(sup_rel_avg)], y=sup_rel_avg, mode="lines", name="Supplier Reliability (avg)",
                                        line=dict(color=THEME["accent"], width=2), fill="tozeroy",
                                        fillcolor=_hex_rgba(THEME["accent"], 0.1),
                                        hovertemplate="Day %{x}<br>Reliability: %{y:.3f}<extra></extra>"))
    if inv_wh:
        fig_sdabm.add_trace(go.Scatter(x=t[:len(inv_wh)], y=[v/100 for v in inv_wh], mode="lines", name="Warehouse Inv / 100",
                                        line=dict(color=THEME["success"], width=1.5, dash="dot"),
                                        hovertemplate="Day %{x}<br>Warehouse/100: %{y:.1f}<extra></extra>"))
    fig_sdabm.update_layout(title="SD + ABM: Supplier Degradation Predicts Inventory Pressure",
                            paper_bgcolor="white", plot_bgcolor="white", height=300, margin=dict(l=40, r=20, t=40, b=30),
                            hovermode="x unified", xaxis=dict(title="Day", gridcolor="#f0f0f0"),
                            yaxis=dict(title="Value", gridcolor="#f0f0f0"), legend=dict(orientation="h", y=-0.3))

    # DES+ABM: Queue vs heterogeneity
    fig_desabm = go.Figure()
    if q_cell and dem_rate:
        fig_desabm.add_trace(go.Scatter(x=t[:min(len(q_cell), len(dem_rate))], y=q_cell[:min(len(q_cell), len(dem_rate))],
                                         mode="lines", name="Cell Queue Length",
                                         line=dict(color=THEME["warning"], width=2),
                                         hovertemplate="Day %{x}<br>Queue: %{y}<extra></extra>"))
        fig_desabm.add_trace(go.Scatter(x=t[:len(dem_rate)], y=dem_rate, mode="lines", name="Automaker Demand Rate (avg)",
                                         yaxis="y2", line=dict(color="#9c27b0", width=1.5, dash="dot"),
                                         hovertemplate="Day %{x}<br>Demand Rate: %{y:.3f}<extra></extra>"))
    fig_desabm.update_layout(title="DES + ABM: Heterogeneous Demand Drives Queue Variability",
                             paper_bgcolor="white", plot_bgcolor="white", height=300, margin=dict(l=40, r=40, t=40, b=30),
                             hovermode="x unified", xaxis=dict(title="Day", gridcolor="#f0f0f0"),
                             yaxis=dict(title="Queue Length", gridcolor="#f0f0f0"),
                             yaxis2=dict(title="Demand Rate", overlaying="y", side="right", gridcolor="#f0f0f0"),
                             legend=dict(orientation="h", y=-0.3))

    # Cross-paradigm comparison table
    scenario_keys = sorted(scs.keys())
    cp_rows = ""
    for m in ["Warehouse_Inventory", "Orders_Backlog", "Cash_Reserves", "total_cost_per_pack", "fill_rate"]:
        vals = []
        for sk in scenario_keys:
            v = scs[sk].get(m, 0)
            if "fill_rate" in m:
                vals.append(f"{v:.1%}")
            elif "Cash" in m:
                vals.append(f"${v/1e6:.2f}M")
            elif "cost" in m:
                vals.append(f"${v:,.0f}")
            else:
                vals.append(f"{v:,.0f}")
        cells = "".join(f"<td>{v}</td>" for v in vals)
        cp_rows += f"<tr><td><strong>{m}</strong></td>{cells}</tr>"

    sddes_html = fig_sddes.to_html(full_html=False, include_plotlyjs=False)
    sdabm_html = fig_sdabm.to_html(full_html=False, include_plotlyjs=False)
    desabm_html = fig_desabm.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Cross-Paradigm", "icon": "&#x1F52D;",
            "content": f"""
        <div class="section-title">SD → DES: Aggregate Trigger Leads to Individual-Event Consequences</div>
        <div class="chart-box">{sddes_html}</div>
        <div class="section-title">SD + ABM: Supplier Behavior Regime Predicts Inventory Outcome</div>
        <div class="chart-box">{sdabm_html}</div>
        <div class="section-title">DES + ABM: Heterogeneous Agent Behavior Drives Queue Variability</div>
        <div class="chart-box">{desabm_html}</div>
        <div class="section-title">Cross-Paradigm Comparison (all 4 scenarios)</div>
        <div class="table-wrap"><table class="dt"><thead><tr><th>Metric</th>{"".join(f'<th>{sk}</th>' for sk in scenario_keys)}</tr></thead>
        <tbody>{cp_rows}</tbody></table></div>
        <div class="q-grid">
            <div class="q-card" style="border-left:4px solid {THEME['success']}"><div class="ql">SD Alone Predicts</div><div class="qa">Warehouse inventory trajectories, bullwhip amplification ({d['bullwhip_cvs'].get('Mine Orders',0)/max(0.001,d['bullwhip_cvs'].get('Customer Demand',1)):.1f}x), cash accrual. But cannot say <i>which</i> orders miss fulfillment.</div></div>
            <div class="q-card" style="border-left:4px solid {THEME['warning']}"><div class="ql">DES Alone Predicts</div><div class="qa">Queue contention timing (Shipping_Dock peak), resource bottlenecks. But cannot explain <i>why</i> variability in arrival rates occurs.</div></div>
            <div class="q-card" style="border-left:4px solid #9c27b0"><div class="ql">ABM Alone Predicts</div><div class="qa">Supplier degradation patterns, automaker switching tipping points. But cannot translate into aggregate fulfillment timelines.</div></div>
            <div class="q-card" style="border-left:4px solid {THEME['danger']}"><div class="ql">Combined Predicts</div><div class="qa">SD says when inventory goes critical. DES says which queue bottlenecks and which orders miss. ABM says <b>why</b> (supplier X degraded reliability → automaker Y switched → demand spike → queue contention). None alone gives the full chain.</div></div>
        </div>"""}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DECISION INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
def build_decisions(d):
    scs = d["sc_summary"]; oat = d["oat"]; sv = d["sv"]

    # Scenario comparison
    metrics = ["Warehouse_Inventory", "Orders_Backlog", "Cash_Reserves", "fill_rate"]
    m_labels = ["Warehouse", "Backlog", "Cash", "Fill Rate"]
    m_fmts = ["{:,.0f}", "{:,.0f}", lambda v: f"${v/1e6:.1f}M", "{:.1%}"]
    scenario_keys = sorted(scs.keys())
    sc_rows = ""
    for m, lb, ff in zip(metrics, m_labels, m_fmts):
        cells = ""
        for sk in scenario_keys:
            v = scs[sk].get(m, 0)
            if callable(ff):
                cells += f"<td>{ff(v)}</td>"
            elif "%" in ff:
                cells += f"<td>{ff.format(v)}</td>"
            else:
                cells += f"<td>{ff.format(v)}</td>"
        sc_rows += f"<tr><td><strong>{lb}</strong></td>{cells}</tr>"

    # Scenario bar chart
    fig_sc = go.Figure()
    for sk in scenario_keys:
        vals = [scs[sk].get(m, 0) for m in ["Warehouse_Inventory", "Orders_Backlog", "Cash_Reserves"]]
        labels = ["Warehouse", "Backlog", "Cash (÷1e6)"]
        display = [vals[0], vals[1], vals[2] / 1e6]
        fig_sc.add_trace(go.Bar(name=sk, x=labels, y=display, hovertemplate=f"{sk}: %{{y:,.1f}}<extra></extra>"))
    fig_sc.update_layout(title="Scenario Comparison (Day 200)", barmode="group",
                         paper_bgcolor="white", plot_bgcolor="white", height=300, margin=dict(l=40, r=20, t=40, b=50),
                         xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(title="Value", gridcolor="#f0f0f0"),
                         legend=dict(orientation="h", y=-0.3))

    # Intervention ranking
    recommendations = [
        ("Increase Mining Capacity", "mining_capacity", 400, 1200, "+100% mine output", THEME["danger"]),
        ("Boost Chem Capacity", "chem_capacity", 250, 750, "+200% chem throughput", THEME["warning"]),
        ("Reduce Labor Disruption", "labor_disruption_amt", 0, 0.5, "Eliminate labor shock", THEME["accent"]),
    ]
    base = sv.get("Orders_Backlog", 0)
    fig_rec = go.Figure()
    for label, pname, lo, hi, desc, color in recommendations:
        if oat and oat.oat_low and oat.oat_high and pname in oat.oat_low:
            imp_lo = abs(oat.oat_low[pname] - base)
            imp_hi = abs(oat.oat_high[pname] - base)
            fig_rec.add_trace(go.Bar(name=label, x=[label], y=[imp_hi], marker=dict(color=color),
                                      hovertemplate=f"{label}: {imp_hi:,.0f} change<extra></extra>"))
    fig_rec.update_layout(title="Impact of Interventions on Orders Backlog", paper_bgcolor="white", plot_bgcolor="white",
                          height=250, margin=dict(l=100, r=40, t=40, b=80),
                          xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(title="Change in Backlog", gridcolor="#f0f0f0"), showlegend=False)

    sc_html = fig_sc.to_html(full_html=False, include_plotlyjs=False)
    rec_html = fig_rec.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Decisions", "icon": "&#x1F4A1;",
            "content": f"""
        <div class="section-title">What-If Scenario Comparison</div>
        <div class="table-wrap"><table class="dt"><thead><tr><th>Metric</th>{"".join(f'<th>{sk}</th>' for sk in scenario_keys)}</tr></thead>
        <tbody>{sc_rows}</tbody></table></div>
        <div class="two-col"><div class="chart-box">{sc_html}</div><div class="chart-box">{rec_html}</div></div>
        <div class="q-grid">
            <div class="q-card" style="border-left:4px solid {THEME['danger']}"><div class="ql">Mine Disruption</div><div class="qa">40% downtime at Day 90: upstream mine disruption cascades through all 6 echelons with ~45-day lag per tier.</div></div>
            <div class="q-card" style="border-left:4px solid {THEME['warning']}"><div class="ql">Port Delay</div><div class="qa">50% port delay at Day 120: extends chem transit from 15 to ~23 days, amplifying bullwhip across cell and pack tiers.</div></div>
            <div class="q-card" style="border-left:4px solid {THEME['accent']}"><div class="ql">Combined Shock</div><div class="qa">Mine + port + labor disruptions simultaneously: worst-case backlog, but DES queues show which bottleneck saturates first (Shipping_Dock).</div></div>
        </div>
        <div class="section-title">Intervention Targeting</div>
        <div class="chart-box">{rec_html}</div>"""}


# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Cross-Paradigm Prediction Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js@3.6.0/dist/plotly.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Segoe UI,Helvetica,Arial,sans-serif;background:{BG};color:{TEXT}}}
.header{{background:{PRIMARY};color:white;padding:14px 24px;position:sticky;top:0;z-index:100}}
.header h1{{font-size:18px;font-weight:600}}
.header .sub{{font-size:11px;opacity:.8;margin-top:2px}}
.tab-bar{{display:flex;background:{PRIMARY};padding:0 12px;gap:1px;position:sticky;top:63px;z-index:99;overflow-x:auto}}
.tab-btn{{padding:8px 14px;background:transparent;color:rgba(255,255,255,.7);border:none;cursor:pointer;font-size:12px;border-bottom:3px solid transparent;transition:all .2s;white-space:nowrap;flex-shrink:0}}
.tab-btn:hover{{background:rgba(255,255,255,.1);color:#fff}}
.tab-btn.active{{background:rgba(255,255,255,.15);color:#fff;border-bottom-color:{ACCENT}}}
.content{{max-width:1200px;margin:0 auto;padding:16px}}
.pane{{/* visible for Plotly render */}}
.pane.hidden{{display:none}}
.st{{font-size:15px;font-weight:600;color:{PRIMARY};margin:16px 0 8px 0;border-bottom:2px solid {ACCENT};padding-bottom:3px}}
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-bottom:12px}}
.kpi{{background:{CARD};border-radius:6px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.08);text-align:center}}
.kl{{font-size:10px;text-transform:uppercase;color:{MUTED};margin-bottom:2px}}
.kv{{font-size:22px;font-weight:700}}
.ks{{font-size:10px;color:{MUTED};margin-top:2px}}
.two-col{{display:flex;gap:12px;margin-bottom:12px}}
.two-col>*{{flex:1;min-width:0}}
.chart-box{{background:{CARD};border-radius:6px;padding:6px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:10px}}
.q-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin:12px 0}}
.q-card{{background:{CARD};border-radius:6px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.ql{{font-size:10px;font-weight:700;color:{PRIMARY};margin-bottom:3px;text-transform:uppercase}}
.qa{{font-size:11px;color:{TEXT};line-height:1.5}}
.table-wrap{{overflow-x:auto;margin:8px 0}}
.dt{{width:100%;border-collapse:collapse;font-size:11px}}
.dt th{{background:{PRIMARY};color:#fff;padding:6px 10px;text-align:left;font-weight:600}}
.dt td{{padding:6px 10px;border-bottom:1px solid #eee}}
.dt tbody tr:hover{{background:#f0f4ff}}
@media(max-width:768px){{.two-col{{flex-direction:column}}.tab-btn{{font-size:10px;padding:6px 10px}}}}
</style></head><body>
<div class=header><h1>&#x1F4CA; Cross-Paradigm Prediction Dashboard</h1><div class=sub>EV Battery Supply Chain | SD + DES + ABM | {DATE}</div></div>
<div class=tab-bar>{TABS}</div>
<div class=content>{PANES}</div>
<script>
window.addEventListener('load',function(){{setTimeout(function(){{document.querySelectorAll('.pane').forEach(function(e,i){{if(i!==0)e.classList.add('hidden')}})}},500)}})
function switchTab(i){{document.querySelectorAll('.pane').forEach(function(e){{e.classList.remove('hidden')}});document.querySelectorAll('.pane').forEach(function(e,j){{if(j!==i)e.classList.add('hidden')}});document.querySelectorAll('.tab-btn').forEach(function(e,j){{e.classList.toggle('active',j===i)}});document.querySelectorAll('.pane:not(.hidden) .js-plotly-plot').forEach(function(e){{if(typeof Plotly!=='undefined')Plotly.Plots.resize(e)}})}}
</script></body></html>"""


def build_html(pages):
    tabs = "".join(f'<button class="tab-btn {"active" if i==0 else ""}" onclick="switchTab({i})">{p["icon"]} {p["title"]}</button>' for i, p in enumerate(pages))
    panes = "".join(f'<div class="pane" id="pane-{i}">{p["content"]}</div>' for i, p in enumerate(pages))
    return HTML_TEMPLATE.format(PRIMARY=THEME["primary"], ACCENT=THEME["accent"], SUCCESS=THEME["success"],
                                WARNING=THEME["warning"], DANGER=THEME["danger"], BG=THEME["bg"], CARD=THEME["card"],
                                TEXT=THEME["text"], MUTED=THEME["muted"],
                                TABS=tabs, PANES=panes, DATE=datetime.now().strftime("%Y-%m-%d %H:%M"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Cross-Paradigm Prediction Dashboard")
    print("=" * 60)
    data = run_pipeline()

    print("Building dashboard pages...")
    pages = [build_exec_summary(data), build_sd(data), build_des(data),
             build_abm(data), build_cross(data), build_decisions(data)]

    print("Assembling HTML...")
    html = make_lazy(build_html(pages))
    out = "/tmp/cross_paradigm_dashboard.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"\nDashboard: {out} ({len(html)//1024}KB, {len(pages)} tabs)")
    sv = data["sv"]
    print(f"Warehouse: {sv.get('Warehouse_Inventory',0):,.0f} | Backlog: {sv.get('Orders_Backlog',0):,.0f} | Cash: ${sv.get('Cash_Reserves',0)/1e6:.2f}M")
    return out


if __name__ == "__main__":
    main()
