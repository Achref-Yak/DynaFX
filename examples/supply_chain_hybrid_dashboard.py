"""Supply Chain Crisis Response — Hybrid SD+ABM Dashboard.

Comprehensive 11-tab HTML dashboard integrating system dynamics, agent-based
modeling, message passing, strategy switching, and supply chain analytics.

Run: python -m examples.supply_chain_hybrid_dashboard
Output: /tmp/supply_chain_hybrid_dashboard.html
"""

import math, random, statistics, sys
from pathlib import Path
from datetime import datetime
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from dynafx.dynamics.dsl import (
    SysdModel, AgentDef, AgentPropDef, AgentRuleDef, AgentStrategy,
)
from dynafx.dynamics.agent import ABMEngine

random.seed(42)

NUM_BUYERS = 10
T_END = 40
THEME = {
    "primary": "#2B4570", "accent": "#4A6FA5", "success": "#3D8361",
    "warning": "#C77D2E", "danger": "#B23A48", "bg": "#FAFAF8",
    "card": "#FFFFFF", "text": "#333333", "muted": "#6B7280",
}
COLORS = ["#2B4570", "#3D8361", "#C77D2E", "#B23A48", "#4A6FA5",
          "#8B9DC3", "#D4A76A", "#6B7280", "#5A8F7B", "#A06B7B"]

ALERT_T = 10


def _hex_rgba(c, a):
    h = c.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"

def _kpi_card(label, value, color, subtitle=""):
    return f"""<div class="kpi" style="border-top:3px solid {color}">
      <div class="kl">{label}</div>
      <div class="kv" style="color:{color}">{value}</div>
      {f'<div class="ks">{subtitle}</div>' if subtitle else ''}
    </div>"""


# ── Simulation ───────────────────────────────────────────────────

def run_simulation() -> dict[str, Any]:
    buyer_params = [
        {"consumption": round(random.uniform(5, 15), 1),
         "emergency_threshold": round(random.uniform(5, 20), 1)}
        for _ in range(NUM_BUYERS)
    ]

    model = SysdModel("supply_chain")
    model.dt = 1.0
    model.t_start = 0.0
    model.t_end = float(T_END)

    with model.stock("Inventory", 1000.0) as s:
        s.inflow("Production", "60")
        s.outflow("Shipments",
                  "MIN(Inventory / dt, MAX(1, Buyer_order_size_sum))")
    model.aux("fill_rate", "Shipments / MAX(1, Buyer_order_size_sum)")

    model.agents.append(AgentDef(
        "AlertSystem", 1,
        rules=[AgentRuleDef("send_warning", "t >= 9.5 and t < 10.5",
                            ["SEND(Buyer, 'disruption_warning', severity=0.8)"])],
    ))

    for p in buyer_params:
        model.agents.append(AgentDef(
            "Buyer", 1,
            properties=[
                AgentPropDef("inventory", 100.0, min=0, max=500),
                AgentPropDef("consumption", p["consumption"], min=0),
                AgentPropDef("order_size", 0.0, min=0),
                AgentPropDef("emergency_threshold", p["emergency_threshold"],
                             min=0, max=50),
                AgentPropDef("in_emergency", 0.0),
            ],
            strategies=[
                AgentStrategy("standard", [
                    AgentRuleDef("consume", "always", ["inventory -= consumption"]),
                    AgentRuleDef("receive", "always", ["inventory += order_size * MIN(1, fill_rate)"]),
                    AgentRuleDef("calc_order", "always",
                                 ["order_size = MAX(0, consumption + (consumption * 3 - inventory) * 0.3)"]),
                ]),
                AgentStrategy("emergency", [
                    AgentRuleDef("consume", "always", ["inventory -= consumption"]),
                    AgentRuleDef("receive", "always", ["inventory += order_size * MIN(1, fill_rate)"]),
                    AgentRuleDef("panic_order", "always",
                                 ["order_size = consumption * 1.5 + MAX(0, consumption * 5 - inventory) * 0.5"]),
                ]),
            ],
            meta_rules=[
                AgentRuleDef("low_stock_check", "inventory < emergency_threshold",
                             ["SWITCH_STRATEGY('emergency', cooldown=5)"]),
                AgentRuleDef("alert_check", "inbox_disruption_warning > 0",
                             ["SWITCH_STRATEGY('emergency', cooldown=10)"]),
                AgentRuleDef("mark_emergency", "strategy == 'emergency'",
                             ["in_emergency = 1"]),
                AgentRuleDef("mark_standard", "strategy != 'emergency'",
                             ["in_emergency = 0"]),
            ],
        ))

    abm = ABMEngine(model.agents, seed=42)
    abm.initialize()

    buyer_idx = 0
    for inst in abm.instances:
        if inst.agent_def.name == "Buyer":
            p = buyer_params[buyer_idx]
            inst.state["consumption"] = p["consumption"]
            inst.state["emergency_threshold"] = p["emergency_threshold"]
            inst.state["inventory"] = p["consumption"] * 3
            buyer_idx += 1

    warehouse = 1000.0
    dt = 1.0

    history: dict[str, Any] = {
        "t": [], "total_orders": [], "incoming_orders": [],
        "avg_inventory": [], "warehouse_stock": [], "shipments": [],
        "fill_rate": [], "emergency_count": [], "buyer_orders": [],
        "buyer_inventories": [], "buyer_modes": [], "alerts_received": [],
    }

    for t in range(0, T_END):
        ft = float(t)

        incoming_orders = sum(
            inst.state["order_size"] for inst in abm.instances
            if inst.agent_def.name == "Buyer"
        )
        shipments = min(warehouse / dt, max(1.0, incoming_orders))
        fill_rate = shipments / max(1.0, incoming_orders)

        env = {"t": ft, "Inventory": warehouse, "fill_rate": fill_rate,
               "Buyer_order_size_sum": incoming_orders, "shipments": shipments}

        abm.step(ft, dt, env)

        orders_list: list[float] = []
        inv_list: list[float] = []
        mode_list: list[str] = []
        alert_list: list[int] = []
        emergency_count = 0
        for inst in abm.instances:
            if inst.agent_def.name == "Buyer":
                orders_list.append(inst.state["order_size"])
                inv_list.append(inst.state["inventory"])
                mode_list.append(inst.strategy or "standard")
                alert_list.append(len([m for m in inst.mailbox if not m.expired]))
                if inst.strategy == "emergency":
                    emergency_count += 1

        placed_orders = sum(orders_list)

        history["t"].append(t)
        history["total_orders"].append(placed_orders)
        history["incoming_orders"].append(incoming_orders)
        history["avg_inventory"].append(statistics.mean(inv_list))
        history["warehouse_stock"].append(warehouse)
        history["shipments"].append(shipments)
        history["fill_rate"].append(fill_rate)
        history["emergency_count"].append(emergency_count)
        history["buyer_orders"].append(orders_list[:])
        history["buyer_inventories"].append(inv_list[:])
        history["buyer_modes"].append(mode_list[:])
        history["alerts_received"].append(alert_list[:])

        production = 60.0
        warehouse += (production - shipments) * dt

    # Derived metrics (both use same post-step data = consistent)
    pre_mask = [5 <= t <= 9 for t in history["t"]]
    post_mask = [11 <= t <= 20 for t in history["t"]]
    pre_orders = [history["total_orders"][i] for i, m in enumerate(pre_mask) if m]
    post_orders = [history["total_orders"][i] for i, m in enumerate(post_mask) if m]
    pre_mean = statistics.mean(pre_orders) if pre_orders else 0
    post_mean = statistics.mean(post_orders) if post_orders else 0
    demand_surge = post_mean / max(1, pre_mean)

    per_buyer: list[dict] = []
    for a in range(NUM_BUYERS):
        orders = [history["buyer_orders"][s][a] for s in range(T_END)]
        inv = [history["buyer_inventories"][s][a] for s in range(T_END)]
        pre_o = [orders[i] for i in range(T_END) if 5 <= i <= 9]
        post_o = [orders[i] for i in range(T_END) if 11 <= i <= 20]
        switch_t = next((history["t"][s] for s in range(T_END)
                         if history["buyer_modes"][s][a] == "emergency"), None)
        per_buyer.append({
            "id": a,
            "consumption": buyer_params[a]["consumption"],
            "emergency_threshold": buyer_params[a]["emergency_threshold"],
            "avg_order": statistics.mean(orders),
            "max_order": max(orders),
            "pre_mean": statistics.mean(pre_o) if pre_o else 0,
            "post_mean": statistics.mean(post_o) if post_o else 0,
            "surge": statistics.mean(post_o) / max(1, statistics.mean(pre_o)) if pre_o else 0,
            "switch_t": switch_t,
            "final_inv": inv[-1],
            "orders": orders,
            "inventories": inv,
        })

    return {
        "history": history,
        "per_buyer": per_buyer,
        "buyer_params": buyer_params,
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "demand_surge": demand_surge,
        "peak_emergency": max(history["emergency_count"]),
        "first_emergency": next(
            (t for t, c in zip(history["t"], history["emergency_count"]) if c > 0), None),
        "all_emergency": next(
            (t for t, c in zip(history["t"], history["emergency_count"])
             if c == NUM_BUYERS), None),
        "final_warehouse": warehouse,
        "min_fill_rate": min(history["fill_rate"]),
    }


# ── Tab Builders ─────────────────────────────────────────────────

def build_executive_summary(d):
    h = d["history"]
    kpi = _kpi_card("Avg Order Rate (Before Alert)",
                    f"{d['pre_mean']:.0f}", THEME["accent"])
    kpi += _kpi_card("Avg Order Rate (After Alert)",
                     f"{d['post_mean']:.0f}", THEME["danger"])
    kpi += _kpi_card("Demand Surge", f"{d['demand_surge']:.2f}x",
                     THEME["danger"], "post-alert / pre-alert")
    kpi += _kpi_card("Buyers in Emergency Mode",
                     f"{d['peak_emergency']}/{NUM_BUYERS}",
                     THEME["warning"], f"started at t={d['first_emergency']}")
    kpi += _kpi_card("Lowest Fulfillment Rate",
                     f"{d['min_fill_rate']*100:.0f}%", THEME["danger"])
    kpi += _kpi_card("Ending Warehouse Stock",
                     f"{d['final_warehouse']:.0f}", THEME["muted"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["t"], y=h["incoming_orders"], mode="lines",
                              name="Orders Received", line=dict(color=COLORS[0])))
    fig.add_trace(go.Scatter(x=h["t"], y=h["shipments"], mode="lines",
                              name="Shipments", line=dict(color=COLORS[1])))
    fig.add_trace(go.Scatter(x=h["t"], y=h["fill_rate"], mode="lines",
                              name="Fulfillment Rate", yaxis="y2",
                              line=dict(color=COLORS[2], dash="dot")))
    fig.add_vline(x=ALERT_T, line_dash="dash", line_color=THEME["danger"],
                  annotation_text="Alert Sent")
    fig.add_vline(x=ALERT_T + 1, line_dash="dot", line_color=THEME["warning"],
                  annotation_text="Emergency")
    fig.update_layout(
        title="Supply Chain Overview",
        yaxis=dict(title="Units"),
        yaxis2=dict(title="Fulfillment Rate", overlaying="y",
                     side="right", range=[0, 1.05]),
        height=380, margin=dict(l=40, r=50, t=40, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
        hovermode="x unified", legend=dict(orientation="h", y=-0.15),
    )

    chart = fig.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Executive Summary", "icon": "&#x1F4CA;",
            "content": f"""
        <div class="kpi-row">{kpi}</div>
        <div class="st">Timeline Overview</div>
        <div class="chart-box">{chart}</div>
        <div class="q-grid">
          <div class="q-card">
            <div class="ql">What Happened</div>
            <div class="qa">An automated alert system sent a disruption warning to all 10 buyers at period 10. Buyers received the alert one period later and immediately switched to emergency ordering protocols.</div>
          </div>
          <div class="q-card">
            <div class="ql">Demand Surge</div>
            <div class="qa">Orders jumped from {d['pre_mean']:.0f} units/period (pre-alert) to {d['post_mean']:.0f} units/period (post-alert) — a {d['demand_surge']:.2f}x increase driven by buyers panic-ordering to rebuild safety buffers.</div>
          </div>
          <div class="q-card">
            <div class="ql">Warehouse Strain</div>
            <div class="qa">Fulfillment dropped from 100% to {d['min_fill_rate']*100:.0f}% as warehouse stock depleted. Ending warehouse inventory: {d['final_warehouse']:.0f} (started at 1,000). Production capacity was overwhelmed by the demand spike.</div>
          </div>
          <div class="q-card">
            <div class="ql">Coordinated Response</div>
            <div class="qa">All {NUM_BUYERS} buyers received the disruption alert simultaneously and entered emergency mode at the same time, creating a synchronized demand surge — a classic bullwhip scenario.</div>
          </div>
        </div>"""}


def build_supply_chain_flow(d):
    h = d["history"]
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Warehouse Stock &amp; Flow",
                                        "Orders &amp; Fulfillment Rate"),
                        vertical_spacing=0.15, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=h["t"], y=h["warehouse_stock"], mode="lines",
                              name="Warehouse Stock",
                              fill="tozeroy", line=dict(color=COLORS[0])),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=h["t"], y=h["shipments"], mode="lines",
                              name="Outbound Shipments",
                              line=dict(color=COLORS[1])), row=1, col=1)
    fig.add_trace(go.Scatter(x=h["t"], y=[60]*len(h["t"]), mode="lines",
                              name="Production Rate",
                              line=dict(color=COLORS[4], dash="dot")),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=h["t"], y=h["incoming_orders"], mode="lines",
                              name="Orders Received by Warehouse",
                              line=dict(color=COLORS[2])), row=2, col=1)
    fig.add_trace(go.Scatter(x=h["t"], y=h["fill_rate"], mode="lines",
                              name="Fulfillment Rate",
                              line=dict(color=COLORS[3]), fill="tozeroy"),
                  row=2, col=1)

    for row in (1, 2):
        fig.add_vline(x=ALERT_T, line_dash="dash", line_color=THEME["danger"],
                      row=row, col=1)

    fig.update_layout(
        height=550, margin=dict(l=40, r=20, t=50, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
        hovermode="x unified", legend=dict(orientation="h", y=1.02),
    )
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="Units", row=2, col=1)

    chart = fig.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Supply Chain Flow", "icon": "&#x1F504;",
            "content": f"""
        <div class="chart-box">{chart}</div>
        <div class="st">How the Supply Chain Works</div>
        <div class="q-grid">
          <div class="q-card">
            <div class="ql">Warehouse</div>
            <div class="qa">The central warehouse holds inventory (starting at 1,000 units). Stock increases through production at a steady rate of 60 units/period.</div>
          </div>
          <div class="q-card">
            <div class="ql">Outbound Shipments</div>
            <div class="qa">Shipments are driven by buyer orders received from the previous period, capped by available warehouse stock. When demand exceeds supply, shipments are rationed.</div>
          </div>
          <div class="q-card">
            <div class="ql">Fulfillment Rate</div>
            <div class="qa">The percentage of orders that could be fulfilled from current stock. Always 100% before the alert when demand was within production capacity.</div>
          </div>
          <div class="q-card">
            <div class="ql">Demand Signal</div>
            <div class="qa">Buyers' collective order volume feeds back into the system: high orders deplete warehouse stock, reducing future fulfillment, which can trigger even more aggressive ordering.</div>
          </div>
        </div>"""}


def build_buyer_ordering(d):
    h = d["history"]
    per_buyer = d["per_buyer"]

    fig = go.Figure()
    for a in range(NUM_BUYERS):
        orders = [h["buyer_orders"][s][a] for s in range(T_END)]
        fig.add_trace(go.Scatter(
            x=h["t"], y=orders, mode="lines",
            name=f"Buyer {a} (usage {per_buyer[a]['consumption']:.0f}/d)",
            line=dict(color=COLORS[a % len(COLORS)], width=1.5), opacity=0.7,
        ))
    fig.add_trace(go.Scatter(
        x=h["t"], y=[o / NUM_BUYERS for o in h["total_orders"]],
        mode="lines", name="Average Order",
        line=dict(color="#000", width=3, dash="dash")))
    fig.add_vline(x=ALERT_T, line_dash="dash", line_color=THEME["danger"])
    fig.update_layout(
        title="Orders Placed by Each Buyer Over Time",
        xaxis_title="Time Period", yaxis_title="Order Quantity",
        height=450, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
        hovermode="x unified",
        legend=dict(font=dict(size=9), orientation="h", y=-0.25),
    )
    chart = fig.to_html(full_html=False, include_plotlyjs=False)

    labels = [f"B{a}" for a in range(NUM_BUYERS)]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=labels, y=[per_buyer[a]["pre_mean"] for a in range(NUM_BUYERS)],
                           name="Before Alert (periods 5-9)",
                           marker_color=COLORS[4]))
    fig2.add_trace(go.Bar(x=labels, y=[per_buyer[a]["post_mean"] for a in range(NUM_BUYERS)],
                           name="After Alert (periods 11-20)",
                           marker_color=COLORS[3]))
    fig2.update_layout(
        title="Before vs After Alert: Average Orders by Buyer",
        xaxis_title="Buyer", yaxis_title="Avg Order Quantity",
        barmode="group", height=350, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
        legend=dict(orientation="h", y=-0.15),
    )
    chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Buyer Ordering", "icon": "&#x1F4E6;",
            "content": f"""
        <div class="chart-box">{chart}</div>
        <div class="chart-box">{chart2}</div>"""}


def build_crisis_response(d):
    h = d["history"]
    per_buyer = d["per_buyer"]

    mode_matrix: list[list[int]] = []
    for a in range(NUM_BUYERS):
        row = [1 if h["buyer_modes"][s][a] == "emergency" else 0
               for s in range(T_END)]
        mode_matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=mode_matrix,
        x=list(range(T_END)),
        y=[f"Buyer {a}" for a in range(NUM_BUYERS)],
        colorscale=[[0, COLORS[4]], [1, COLORS[3]]],
        showscale=False,
        hovertemplate="t=%{x}<br>Buyer %{y}<br>%{z}<extra></extra>",
    ))
    fig.add_vline(x=ALERT_T, line_dash="dash", line_color="#fff",
                  annotation_text="Alert")
    fig.update_layout(
        title="Operating Mode Timeline (Standard / Emergency)",
        xaxis_title="Time Period", yaxis_title="Buyer",
        height=300 + NUM_BUYERS * 20,
        margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=h["t"], y=h["emergency_count"],
                               mode="lines+markers",
                               name="Buyers in Emergency",
                               fill="tozeroy", line=dict(color=COLORS[3]),
                               marker=dict(size=8)))
    fig2.add_hline(y=NUM_BUYERS, line_dash="dot", line_color=COLORS[4],
                    annotation_text=f"All {NUM_BUYERS}")
    fig2.add_vline(x=ALERT_T, line_dash="dash", line_color=THEME["danger"])
    fig2.update_layout(
        title="Buyers in Emergency Mode Over Time",
        xaxis_title="Time Period", yaxis_title="Count",
        yaxis=dict(dtick=1), height=300,
        margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
    )

    rows = ""
    for a in range(NUM_BUYERS):
        switch_t = per_buyer[a]["switch_t"]
        reason = "Received disruption alert" if (switch_t and switch_t == 11) else \
                 ("Low inventory threshold" if switch_t else "Never")
        rows += f"<tr><td>Buyer {a}</td><td>{switch_t or '—'}</td>" \
                f"<td>{reason}</td><td>{per_buyer[a]['consumption']:.1f}</td>" \
                f"<td>{per_buyer[a]['emergency_threshold']:.1f}</td></tr>"

    chart = fig.to_html(full_html=False, include_plotlyjs=False)
    chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Crisis Response", "icon": "&#x1F3AF;",
            "content": f"""
        <div class="two-col">
          <div class="chart-box">{chart}</div>
          <div class="chart-box">{chart2}</div>
        </div>
        <div class="st">Mode Change Log</div>
        <div class="table-wrap">
          <table class="dt">
            <thead><tr><th>Buyer</th><th>Switch Time</th><th>Trigger</th><th>Daily Usage</th><th>Emergency Threshold</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""}


def build_buyer_inventories(d):
    h = d["history"]
    per_buyer = d["per_buyer"]

    fig = go.Figure()
    for a in range(NUM_BUYERS):
        fig.add_trace(go.Scatter(
            x=h["t"], y=per_buyer[a]["inventories"], mode="lines",
            name=f"Buyer {a}",
            line=dict(color=COLORS[a % len(COLORS)]), opacity=0.7,
        ))
    fig.add_trace(go.Scatter(x=h["t"], y=h["avg_inventory"], mode="lines",
                              name="Average Stock",
                              line=dict(color="#000", width=3, dash="dash")))
    fig.add_vline(x=ALERT_T, line_dash="dash", line_color=THEME["danger"])
    fig.update_layout(
        title="Buyer Stock Levels Over Time",
        xaxis_title="Time Period", yaxis_title="Stock on Hand",
        height=450, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
        hovermode="x unified",
        legend=dict(font=dict(size=9), orientation="h", y=-0.25),
    )

    final_invs = [per_buyer[a]["final_inv"] for a in range(NUM_BUYERS)]
    labels = [f"B{a}" for a in range(NUM_BUYERS)]
    fig2 = go.Figure(data=[go.Bar(x=labels, y=final_invs,
                                   marker_color=COLORS[:NUM_BUYERS])])
    fig2.add_hline(y=statistics.mean(final_invs), line_dash="dot",
                    line_color=THEME["muted"],
                    annotation_text=f"Mean: {statistics.mean(final_invs):.0f}")
    fig2.update_layout(
        title="Ending Stock by Buyer (Period 39)",
        xaxis_title="Buyer", yaxis_title="Stock on Hand",
        height=350, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
    )

    chart = fig.to_html(full_html=False, include_plotlyjs=False)
    chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Stock Levels", "icon": "&#x1F4E6;",
            "content": f"""
        <div class="chart-box">{chart}</div>
        <div class="chart-box">{chart2}</div>"""}


def build_alert_system(d):
    h = d["history"]

    fig = go.Figure()
    for a in range(NUM_BUYERS):
        alerts = [h["alerts_received"][s][a] for s in range(T_END)]
        fig.add_trace(go.Scatter(
            x=h["t"], y=alerts, mode="lines",
            name=f"Buyer {a}",
            line=dict(color=COLORS[a % len(COLORS)], width=1.5), opacity=0.7,
        ))
    fig.add_vline(x=ALERT_T, line_dash="dash", line_color=THEME["danger"],
                  annotation_text="Alert Sent")
    fig.update_layout(
        title="Notifications Received by Each Buyer",
        xaxis_title="Time Period", yaxis_title="Messages in Inbox",
        height=350, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
        hovermode="x unified",
        legend=dict(font=dict(size=9), orientation="h", y=-0.3),
    )

    chart = fig.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Alert System", "icon": "&#x1F4E8;",
            "content": f"""
        <div class="st">Alert Notification Flow</div>
        <div class="kpi-row">
          {_kpi_card("Alerts Sent", "1 broadcast", THEME["accent"], "to all 10 buyers")}
          {_kpi_card("Recipients", f"{NUM_BUYERS} buyers", THEME["accent"])}
          {_kpi_card("Delivery Time", "1 period", THEME["success"], "next-period delivery")}
          {_kpi_card("Auto-Response", "Immediate", THEME["warning"], "buyers switch modes")}
        </div>
        <div class="chart-box">{chart}</div>
        <div class="q-grid">
          <div class="q-card">
            <div class="ql">Alert Broadcast</div>
            <div class="qa">At period 10, the automated alert system sends a disruption warning to all buyers simultaneously. The alert includes a severity rating of 0.8 (high).</div>
          </div>
          <div class="q-card">
            <div class="ql">Delivery &amp; Action</div>
            <div class="qa">Alerts arrive in buyer inboxes one period later (period 11). Each buyer's automated policy detects the alert and immediately switches to emergency ordering mode — no manual intervention needed.</div>
          </div>
          <div class="q-card">
            <div class="ql">One-Time Processing</div>
            <div class="qa">Each notification is processed once and then expires. Buyers see each alert exactly once, preventing repeated reactions to the same event.</div>
          </div>
        </div>"""}


def build_buyer_profiles(d):
    per_buyer = d["per_buyer"]
    buyer_params = d["buyer_params"]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Daily Usage per Buyer",
                                        "Emergency Threshold per Buyer",
                                        "Usage vs Average Order",
                                        "Threshold vs Response Time"),
                        vertical_spacing=0.15, horizontal_spacing=0.12)

    labels = [f"B{a}" for a in range(NUM_BUYERS)]
    cons = [buyer_params[a]["consumption"] for a in range(NUM_BUYERS)]
    thresh = [buyer_params[a]["emergency_threshold"] for a in range(NUM_BUYERS)]
    avg_orders = [per_buyer[a]["avg_order"] for a in range(NUM_BUYERS)]
    switch_ts = [per_buyer[a]["switch_t"] or 0 for a in range(NUM_BUYERS)]

    fig.add_trace(go.Bar(x=labels, y=cons, marker_color=COLORS[:NUM_BUYERS],
                          name="Usage", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=thresh, marker_color=COLORS[:NUM_BUYERS],
                          name="Threshold", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=cons, y=avg_orders, mode="markers+text",
                              text=labels, textposition="top center",
                              marker=dict(size=10, color=COLORS[0], opacity=0.8),
                              name="", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=thresh, y=switch_ts, mode="markers+text",
                              text=labels, textposition="top center",
                              marker=dict(size=10, color=COLORS[3], opacity=0.8),
                              name="", showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Buyer", row=1, col=1)
    fig.update_xaxes(title_text="Buyer", row=1, col=2)
    fig.update_xaxes(title_text="Daily Usage", row=2, col=1)
    fig.update_xaxes(title_text="Emergency Threshold", row=2, col=2)
    fig.update_yaxes(title_text="Units", row=1, col=1)
    fig.update_yaxes(title_text="Units", row=1, col=2)
    fig.update_yaxes(title_text="Avg Order Size", row=2, col=1)
    fig.update_yaxes(title_text="Response Time (period)", row=2, col=2, dtick=1)

    fig.update_layout(
        height=550, margin=dict(l=40, r=20, t=60, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
    )

    chart = fig.to_html(full_html=False, include_plotlyjs=False)

    rows = ""
    for a in range(NUM_BUYERS):
        p = buyer_params[a]
        pa = per_buyer[a]
        rows += f"<tr><td>Buyer {a}</td><td>{p['consumption']:.1f}</td>" \
                f"<td>{p['emergency_threshold']:.1f}</td>" \
                f"<td>{pa['avg_order']:.1f}</td>" \
                f"<td>{pa['max_order']:.1f}</td>" \
                f"<td>{pa['surge']:.2f}x</td>" \
                f"<td>{pa['switch_t'] or '—'}</td></tr>"

    return {"title": "Buyer Profiles", "icon": "&#x1F9EE;",
            "content": f"""
        <div class="chart-box">{chart}</div>
        <div class="st">Buyer Parameters &amp; Metrics</div>
        <div class="table-wrap">
          <table class="dt">
            <thead><tr><th>Buyer</th><th>Daily Usage</th><th>Emergency Threshold</th><th>Avg Order</th><th>Max Order</th><th>Demand Surge</th><th>Response Time</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""}


def build_supply_chain_health(d):
    h = d["history"]
    per_buyer = d["per_buyer"]

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Fulfillment Rate &amp; Warehouse Stock",
                                        "Order Variability"),
                        vertical_spacing=0.15, shared_xaxes=True)

    fill_color = _hex_rgba(THEME["danger"], 0.3)
    fig.add_trace(go.Scatter(x=h["t"], y=h["fill_rate"], mode="lines",
                              name="Fulfillment Rate", fill="tozeroy",
                              line=dict(color=COLORS[3], width=2),
                              fillcolor=fill_color), row=1, col=1)
    fig.add_trace(go.Scatter(x=h["t"], y=[s / 1000 for s in h["warehouse_stock"]],
                              mode="lines",
                              name="Warehouse Stock (scaled /1000)",
                              line=dict(color=COLORS[0], dash="dot")),
                  row=1, col=1)

    window = 5
    order_var = []
    for i in range(T_END):
        if i < window:
            order_var.append(0)
        else:
            seg = h["total_orders"][i - window:i]
            order_var.append(statistics.variance(seg) if len(seg) > 1 else 0)

    fig.add_trace(go.Scatter(x=h["t"], y=order_var, mode="lines",
                              name="Order Variability (5-period rolling)",
                              fill="tozeroy",
                              line=dict(color=COLORS[2], width=2),
                              fillcolor=_hex_rgba(COLORS[2], 0.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=h["t"], y=h["total_orders"], mode="lines",
                              name="Total Orders (reference)",
                              line=dict(color=COLORS[0], dash="dot")), row=2, col=1)

    for row in (1, 2):
        fig.add_vline(x=ALERT_T, line_dash="dash", line_color=THEME["danger"],
                      row=row, col=1)

    fig.update_layout(
        height=500, margin=dict(l=40, r=20, t=50, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
        hovermode="x unified", legend=dict(orientation="h", y=1.02),
    )
    fig.update_yaxes(title_text="Fulfillment Rate", row=1, col=1)
    fig.update_yaxes(title_text="Variability", row=2, col=1)

    labels = [f"B{a}" for a in range(NUM_BUYERS)]
    fig2 = go.Figure(data=[go.Bar(x=labels,
                                   y=[per_buyer[a]["surge"] for a in range(NUM_BUYERS)],
                                   marker_color=COLORS[:NUM_BUYERS])])
    fig2.add_hline(y=d["demand_surge"], line_dash="dot",
                    line_color=THEME["danger"],
                    annotation_text=f"System average: {d['demand_surge']:.2f}x")
    fig2.update_layout(
        title="Demand Surge by Buyer",
        xaxis_title="Buyer", yaxis_title="Surge (post-alert / pre-alert ratio)",
        height=300, margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
    )

    chart = fig.to_html(full_html=False, include_plotlyjs=False)
    chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Supply Chain Health", "icon": "&#x1F9F1;",
            "content": f"""
        <div class="st">Supply Chain Health Metrics</div>
        <div class="kpi-row">
          {_kpi_card("System Demand Surge", f"{d['demand_surge']:.2f}x", THEME["danger"], "post-alert / pre-alert")}
          {_kpi_card("Pre-Alert Avg Orders", f"{d['pre_mean']:.0f}", THEME["accent"], "periods 5-9")}
          {_kpi_card("Post-Alert Avg Orders", f"{d['post_mean']:.0f}", THEME["danger"], "periods 11-20")}
          {_kpi_card("Lowest Fulfillment", f"{d['min_fill_rate']*100:.0f}%", THEME["warning"], f"at period {h['fill_rate'].index(min(h['fill_rate']))}")}
        </div>
        <div class="chart-box">{chart}</div>
        <div class="chart-box">{chart2}</div>"""}


def build_system_behavior(d):
    h = d["history"]
    per_buyer = d["per_buyer"]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Order Patterns vs Stock Levels",
                                        "Demand Surge Distribution",
                                        "Operating Mode Breakdown",
                                        "Fulfillment Shortfall"),
                        vertical_spacing=0.15, horizontal_spacing=0.12)

    all_orders: list[float] = []
    all_inv: list[float] = []
    colors_scatter: list[str] = []
    for s in range(0, T_END, 2):
        for a in range(NUM_BUYERS):
            all_orders.append(h["buyer_orders"][s][a])
            all_inv.append(h["buyer_inventories"][s][a])
            mode = h["buyer_modes"][s][a]
            colors_scatter.append(COLORS[3] if mode == "emergency" else COLORS[4])

    fig.add_trace(go.Scatter(x=all_inv, y=all_orders, mode="markers",
                              marker=dict(size=4, color=colors_scatter, opacity=0.5),
                              name="", showlegend=False), row=1, col=1)

    surge_vals = [per_buyer[a]["surge"] for a in range(NUM_BUYERS)]
    fig.add_trace(go.Histogram(x=surge_vals, nbinsx=8,
                                marker_color=COLORS[3], name="", showlegend=False),
                  row=1, col=2)

    emergency_ts = [h["emergency_count"][s] for s in range(T_END)]
    standard_ts = [NUM_BUYERS - c for c in emergency_ts]
    fig.add_trace(go.Scatter(x=h["t"], y=standard_ts, mode="lines",
                              name="Standard Mode", stackgroup="one",
                              line=dict(color=COLORS[4]),
                              fillcolor=_hex_rgba(COLORS[4], 0.7)), row=2, col=1)
    fig.add_trace(go.Scatter(x=h["t"], y=emergency_ts, mode="lines",
                              name="Emergency Mode", stackgroup="one",
                              line=dict(color=COLORS[3]),
                              fillcolor=_hex_rgba(COLORS[3], 0.7)), row=2, col=1)

    gap = [h["shipments"][s] - h["incoming_orders"][s] for s in range(T_END)]
    fig.add_trace(go.Scatter(x=h["t"], y=gap, mode="lines",
                              name="Shortfall (shipments - orders received)",
                              fill="tozeroy",
                              line=dict(color=COLORS[2], width=2)), row=2, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color=THEME["muted"], row=2, col=2)

    for row in (1, 2):
        for col in (1, 2):
            try:
                fig.add_vline(x=ALERT_T, line_dash="dash",
                              line_color=THEME["danger"], opacity=0.3,
                              row=row, col=col)
            except Exception:
                pass

    fig.update_xaxes(title_text="Stock on Hand", row=1, col=1)
    fig.update_xaxes(title_text="Demand Surge Ratio", row=1, col=2)
    fig.update_xaxes(title_text="Time Period", row=2, col=1)
    fig.update_xaxes(title_text="Time Period", row=2, col=2)
    fig.update_yaxes(title_text="Order Size", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Buyer Count", row=2, col=1)
    fig.update_yaxes(title_text="Units", row=2, col=2)

    fig.update_layout(
        height=600, margin=dict(l=40, r=20, t=60, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
        legend=dict(orientation="h", y=1.02),
    )

    chart = fig.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "System Behavior", "icon": "&#x1F916;",
            "content": f"""
        <div class="st">How the System Responded</div>
        <div class="chart-box">{chart}</div>
        <div class="q-grid">
          <div class="q-card">
            <div class="ql">Standard Ordering</div>
            <div class="qa">Under normal conditions, each buyer orders enough to maintain 3 days of safety stock: order = daily usage + (safety target - current stock) × 0.3. This keeps the system stable.</div>
          </div>
          <div class="q-card">
            <div class="ql">Emergency Ordering</div>
            <div class="qa">In emergency mode, buyers order aggressively to build 5 days of safety buffer: order = usage × 1.5 + (emergency target - current stock) × 0.5. This 67% increase in target stock drives the demand surge.</div>
          </div>
          <div class="q-card">
            <div class="ql">Automated Policies</div>
            <div class="qa">Two policies trigger mode changes: (1) receiving a disruption alert immediately switches to emergency mode, and (2) if stock drops below a buyer-specific threshold, it also triggers emergency mode. A cooldown prevents rapid switching.</div>
          </div>
          <div class="q-card">
            <div class="ql">Diversity Matters</div>
            <div class="qa">Buyers have different daily usage rates (5-15 units) and different emergency thresholds (5-20 units). In this scenario, all 10 buyers received the alert simultaneously, so all switched at the same time regardless of individual thresholds.</div>
          </div>
        </div>"""}


def build_root_cause_analysis(d):
    h = d["history"]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Order Volume vs Fulfillment Rate",
                                        "Warehouse Stock vs Orders (feedback)",
                                        "Order Pattern Persistence",
                                        "Emergency Mode vs Order Volume"),
                        vertical_spacing=0.15, horizontal_spacing=0.12)

    pre_mask_val = [5 <= t <= 9 for t in h["t"]]
    post_mask_val = [11 <= t <= 20 for t in h["t"]]
    pre_orders_s = [h["incoming_orders"][i] for i, m in enumerate(pre_mask_val) if m]
    pre_fill = [h["fill_rate"][i] for i, m in enumerate(pre_mask_val) if m]
    post_orders_s = [h["incoming_orders"][i] for i, m in enumerate(post_mask_val) if m]
    post_fill = [h["fill_rate"][i] for i, m in enumerate(post_mask_val) if m]

    fig.add_trace(go.Scatter(x=pre_orders_s, y=pre_fill, mode="markers",
                              name="Before Alert",
                              marker=dict(color=COLORS[4], size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=post_orders_s, y=post_fill, mode="markers",
                              name="After Alert",
                              marker=dict(color=COLORS[3], size=8)), row=1, col=1)

    fig.add_trace(go.Scatter(x=h["warehouse_stock"], y=h["total_orders"],
                              mode="lines+markers",
                              marker=dict(color=h["t"], colorscale="Viridis",
                                          size=6, showscale=True,
                                          colorbar=dict(title="t")),
                              line=dict(color="#ccc", width=1),
                              name="Time trajectory"), row=1, col=2)

    vals = h["total_orders"]
    autocorrs = []
    for lag in range(1, 8):
        if len(vals) > lag:
            corr = np.corrcoef(vals[:-lag], vals[lag:])[0, 1] \
                if statistics.stdev(vals[:-lag]) > 0 \
                   and statistics.stdev(vals[lag:]) > 0 else 0
            autocorrs.append(corr)
        else:
            autocorrs.append(0)
    fig.add_trace(go.Bar(x=list(range(1, len(autocorrs) + 1)), y=autocorrs,
                          marker_color=COLORS[0], name="", showlegend=False),
                  row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=THEME["muted"], row=2, col=1)

    fig.add_trace(go.Scatter(x=h["emergency_count"], y=h["total_orders"],
                              mode="markers",
                              marker=dict(size=10,
                                          color=[COLORS[3] if c > 0 else COLORS[4]
                                                 for c in h["emergency_count"]],
                                          opacity=0.7),
                              text=[f"t={t}" for t in h["t"]],
                              name="", showlegend=False), row=2, col=2)

    for row in (1, 2):
        for col in (1, 2):
            try:
                fig.update_xaxes(title_text="Order Volume", row=1, col=1)
                fig.update_xaxes(title_text="Warehouse Stock", row=1, col=2)
                fig.update_xaxes(title_text="Lag (periods)", row=2, col=1, dtick=1)
                fig.update_xaxes(title_text="Buyers in Emergency", row=2, col=2, dtick=1)
                fig.update_yaxes(title_text="Fulfillment Rate", row=1, col=1)
                fig.update_yaxes(title_text="Order Volume", row=1, col=2)
                fig.update_yaxes(title_text="Auto-Correlation", row=2, col=1)
                fig.update_yaxes(title_text="Order Volume", row=2, col=2)
            except Exception:
                pass

    fig.update_layout(
        height=600, margin=dict(l=40, r=50, t=60, b=30),
        paper_bgcolor=THEME["card"], plot_bgcolor=THEME["bg"],
        legend=dict(orientation="h", y=1.02),
    )

    chart = fig.to_html(full_html=False, include_plotlyjs=False)

    return {"title": "Root Cause Analysis", "icon": "&#x1F9E0;",
            "content": f"""
        <div class="st">Why Did This Happen?</div>
        <div class="chart-box">{chart}</div>
        <div class="q-grid">
          <div class="q-card">
            <div class="ql">Order Volume → Fulfillment Rate</div>
            <div class="qa">Before the alert: supply comfortably exceeds demand, so every order is filled at 100%. After the alert: order volume spikes past production capacity, fulfillment collapses to {d['min_fill_rate']*100:.0f}%.</div>
          </div>
          <div class="q-card">
            <div class="ql">The Vicious Cycle</div>
            <div class="qa">Warehouse stock drops → shipments are constrained → fulfillment rate falls → buyers order even more (emergency mode) → demand surges further → warehouse stock drops further. This self-reinforcing loop is the classic bullwhip effect.</div>
          </div>
          <div class="q-card">
            <div class="ql">Order Momentum</div>
            <div class="qa">Orders show strong trend persistence (high auto-correlation at short lags). Once the surge begins, it sustains itself for many periods because each buyer's emergency ordering rule keeps re-ordering based on depleted stock.</div>
          </div>
          <div class="q-card">
            <div class="ql">The Trigger Point</div>
            <div class="qa">The transition is sharp: 0 buyers in emergency mode before the alert is delivered, all 10 immediately after. The synchronized response creates a wall of demand that the supply system cannot absorb.</div>
          </div>
        </div>"""}


def build_configuration(d):
    h = d["history"]
    per_buyer = d["per_buyer"]

    sample_rows = ""
    for s in range(0, T_END, 5):
        t = h["t"][s]
        em = h["emergency_count"][s]
        modes = h["buyer_modes"][s]
        em_str = f"{em}/{NUM_BUYERS}"
        sample_rows += f"<tr><td>{t}</td>" \
                       f"<td>{h['total_orders'][s]:.1f}</td>" \
                       f"<td>{h['shipments'][s]:.1f}</td>" \
                       f"<td>{h['fill_rate'][s]*100:.0f}%</td>" \
                       f"<td>{h['warehouse_stock'][s]:.0f}</td>" \
                       f"<td>{h['avg_inventory'][s]:.1f}</td>" \
                       f"<td>{em_str}</td>" \
                       f"<td>{','.join(m[:3] for m in modes[:3])}...</td></tr>"

    stats_rows = ""
    for a in range(NUM_BUYERS):
        pa = per_buyer[a]
        p = d["buyer_params"][a]
        stats_rows += f"<tr><td>Buyer {a}</td>" \
                      f"<td>{p['consumption']:.1f}</td>" \
                      f"<td>{p['emergency_threshold']:.1f}</td>" \
                      f"<td>{pa['avg_order']:.1f}</td>" \
                      f"<td>{pa['max_order']:.1f}</td>" \
                      f"<td>{pa['surge']:.2f}x</td>" \
                      f"<td>{pa['switch_t'] or '—'}</td>" \
                      f"<td>{pa['final_inv']:.1f}</td></tr>"

    return {"title": "Configuration", "icon": "&#x2699;",
            "content": f"""
        <div class="st">System Parameters</div>
        <div class="q-grid">
          <div class="q-card">
            <div class="ql">Supply Chain</div>
            <div class="qa">Centralized warehouse with capacity-constrained production (60 units/period). Single-echelon distribution: warehouse ships directly to buyers. Simulation runs 40 periods at daily resolution.</div>
          </div>
          <div class="q-card">
            <div class="ql">Buyer Network</div>
            <div class="qa">{NUM_BUYERS} independent buyers with heterogeneous usage rates (5-15 units/day) and emergency stock thresholds (5-20 units). Each buyer carries 3 days of safety stock under normal conditions.</div>
          </div>
          <div class="q-card">
            <div class="ql">Alert System</div>
            <div class="qa">Centralized alert broadcasts disruption warnings to all buyers simultaneously. One-period delivery delay. Buyers are configured with automated policies that respond to alerts without human intervention.</div>
          </div>
          <div class="q-card">
            <div class="ql">Feedback Loop</div>
            <div class="qa">Buyers perceive current fulfillment rate and warehouse status each period. Their ordering decisions feed back into the system: orders → shipments → stock changes → fulfillment → next period's orders.</div>
          </div>
        </div>

        <div class="st">Simulation Data (every 5 periods)</div>
        <div class="table-wrap">
          <table class="dt">
            <thead><tr><th>Period</th><th>Orders</th><th>Shipments</th><th>Fulfill%</th><th>Warehouse</th><th>Avg Buyer Stock</th><th>Emergency</th><th>Modes</th></tr></thead>
            <tbody>{sample_rows}</tbody>
          </table>
        </div>

        <div class="st">Per-Buyer Summary</div>
        <div class="table-wrap">
          <table class="dt">
            <thead><tr><th>Buyer</th><th>Daily Usage</th><th>Emergency Threshold</th><th>Avg Order</th><th>Max Order</th><th>Demand Surge</th><th>Response Time</th><th>Final Stock</th></tr></thead>
            <tbody>{stats_rows}</tbody>
          </table>
        </div>"""}


# ── HTML Template ────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Supply Chain Crisis Response — Demand Surge Analysis Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Segoe UI,Helvetica,Arial,sans-serif;background:{BG};color:{TEXT};font-size:13px}}
.header{{background:{PRIMARY};color:white;padding:12px 20px;position:sticky;top:0;z-index:100}}
.header h1{{font-size:16px;font-weight:600}}
.header .sub{{font-size:11px;opacity:0.8;margin-top:2px}}
.tab-bar{{display:flex;background:{PRIMARY};padding:0 10px;gap:1px;position:sticky;top:57px;z-index:99;overflow-x:auto}}
.tab-btn{{padding:6px 10px;background:transparent;color:rgba(255,255,255,.7);border:none;cursor:pointer;font-size:10px;border-bottom:3px solid transparent;transition:all .2s;white-space:nowrap;flex-shrink:0}}
.tab-btn:hover{{background:rgba(255,255,255,.1);color:#fff}}
.tab-btn.active{{background:rgba(255,255,255,.15);color:#fff;border-bottom-color:{ACCENT}}}
.content{{max-width:1400px;margin:0 auto;padding:12px}}
.pane{{}}
.pane.hidden{{display:none}}
.st{{font-size:13px;font-weight:600;color:{PRIMARY};margin:12px 0 6px 0;border-bottom:2px solid {ACCENT};padding-bottom:2px}}
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:6px;margin-bottom:10px}}
.kpi{{background:{CARD};border-radius:5px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,.07);text-align:center}}
.kl{{font-size:9px;text-transform:uppercase;color:{MUTED};margin-bottom:1px}}
.kv{{font-size:18px;font-weight:700}}
.ks{{font-size:9px;color:{MUTED};margin-top:1px}}
.two-col{{display:flex;gap:10px;margin-bottom:10px}}
.two-col>*{{flex:1;min-width:0}}
.chart-box{{background:{CARD};border-radius:5px;padding:5px;box-shadow:0 1px 3px rgba(0,0,0,.07);margin-bottom:8px}}
.q-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px;margin:10px 0}}
.q-card{{background:{CARD};border-radius:5px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,.07)}}
.ql{{font-size:9px;font-weight:700;color:{PRIMARY};margin-bottom:2px;text-transform:uppercase}}
.qa{{font-size:10px;color:{TEXT};line-height:1.5}}
.qa code{{background:#f0f0f0;padding:1px 4px;border-radius:2px;font-size:9px}}
.table-wrap{{overflow-x:auto;margin:6px 0}}
.dt{{width:100%;border-collapse:collapse;font-size:10px}}
.dt th{{background:{PRIMARY};color:#fff;padding:4px 8px;text-align:left;font-weight:600}}
.dt td{{padding:4px 8px;border-bottom:1px solid #eee}}
.dt tbody tr:hover{{background:#f0f4ff}}
@media(max-width:768px){{.two-col{{flex-direction:column}}.tab-btn{{font-size:9px;padding:5px 8px}}}}
</style></head><body>
<div class=header><h1>&#x1F310; Supply Chain Crisis Response Dashboard</h1><div class=sub>Demand surge analysis: automated alerts, buyer behavior, and supply chain impact | {DATE}</div></div>
<div class=tab-bar>{TABS}</div>
<div class=content>{PANES}</div>
<script>
window.addEventListener('load',function(){{setTimeout(function(){{document.querySelectorAll('.pane').forEach(function(e,i){{if(i!==0)e.classList.add('hidden')}})}},500)}})
function switchTab(i){{document.querySelectorAll('.pane').forEach(function(e){{e.classList.remove('hidden')}});document.querySelectorAll('.pane').forEach(function(e,j){{if(j!==i)e.classList.add('hidden')}});document.querySelectorAll('.tab-btn').forEach(function(e,j){{e.classList.toggle('active',j===i)}});document.querySelectorAll('.pane:not(.hidden) .js-plotly-plot').forEach(function(e){{if(typeof Plotly!=='undefined')Plotly.Plots.resize(e)}})}}
</script></body></html>"""


def build_html(pages):
    tabs = "".join(
        f'<button class="tab-btn {"active" if i==0 else ""}" '
        f'onclick="switchTab({i})">{p["icon"]} {p["title"]}</button>'
        for i, p in enumerate(pages)
    )
    panes = "".join(
        f'<div class="pane" id="pane-{i}">{p["content"]}</div>'
        for i, p in enumerate(pages)
    )
    return HTML_TEMPLATE.format(
        PRIMARY=THEME["primary"], ACCENT=THEME["accent"],
        SUCCESS=THEME["success"], WARNING=THEME["warning"],
        DANGER=THEME["danger"], BG=THEME["bg"], CARD=THEME["card"],
        TEXT=THEME["text"], MUTED=THEME["muted"],
        TABS=tabs, PANES=panes,
        DATE=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


def main():
    print("Supply Chain Crisis Response Dashboard")
    print("=" * 60)
    data = run_simulation()
    print(f"\n  Demand surge: {data['demand_surge']:.2f}x")
    print(f"  Pre-alert avg: {data['pre_mean']:.0f}, Post-alert avg: {data['post_mean']:.0f}")
    print(f"  First emergency: t={data['first_emergency']}, Peak: {data['peak_emergency']}/{NUM_BUYERS}")
    print(f"  Lowest fulfillment: {data['min_fill_rate']*100:.0f}%")

    print("\nBuilding 11 dashboard tabs...")
    builders = [
        build_executive_summary,
        build_supply_chain_flow,
        build_buyer_ordering,
        build_crisis_response,
        build_buyer_inventories,
        build_alert_system,
        build_buyer_profiles,
        build_supply_chain_health,
        build_system_behavior,
        build_root_cause_analysis,
        build_configuration,
    ]
    pages = [b(data) for b in builders]
    print(f"  Built {len(pages)} pages")

    print("Assembling HTML...")
    html = build_html(pages)
    out = "/tmp/supply_chain_hybrid_dashboard.html"
    with open(out, "w") as f:
        f.write(html)
    size_kb = len(html) // 1024
    print(f"\nDashboard: {out} ({size_kb}KB, {len(pages)} tabs)")
    return out


if __name__ == "__main__":
    main()
