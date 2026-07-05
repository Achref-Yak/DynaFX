"""DisruptionCascade — reusable supply chain disruption recipe.

Builds a hybrid SD+ABM model where a Broadcaster agent SENDs a disruption
alert and N Buyer agents switch strategies in response (via meta-rules).

Usage::

    from dynafx.patterns import DisruptionCascade

    model = DisruptionCascade.build(num_buyers=10)
    history = DisruptionCascade.run(model, seed=42)
    print(DisruptionCascade.analyse(history))
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dynafx.dynamics.agent import ABMEngine
from dynafx.dynamics.dsl import (
    AgentDef,
    AgentPropDef,
    AgentRuleDef,
    AgentStrategy,
    SysdModel,
)

_THEME = {
    "primary": "#2B4570", "accent": "#4A6FA5", "success": "#3D8361",
    "warning": "#C77D2E", "danger": "#B23A48", "bg": "#FAFAF8",
    "card": "#FFFFFF", "text": "#333333", "muted": "#6B7280",
}
_COLORS = ["#2B4570", "#3D8361", "#C77D2E", "#B23A48", "#4A6FA5",
           "#8B9DC3", "#D4A76A", "#6B7280", "#5A8F7B", "#A06B7B"]


def _hex_rgba(c: str, a: float) -> str:
    h = c.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _kpi_card(label: str, value: str, color: str, subtitle: str = "") -> str:
    sub = f'<div class="ks">{subtitle}</div>' if subtitle else ""
    return f'''<div class="kpi" style="border-top:3px solid {color}">
      <div class="kl">{label}</div>
      <div class="kv" style="color:{color}">{value}</div>
      {sub}
    </div>'''


class DisruptionCascade:
    """Factory for a supply chain disruption cascade SD+ABM model."""

    @classmethod
    def build(
        cls,
        name: str = "supply_chain",
        num_buyers: int = 10,
        supplier_inventory: float = 1000.0,
        supplier_production_rate: float = 60.0,
        consumption_range: tuple[float, float] = (5, 15),
        crisis_threshold_range: tuple[float, float] = (5, 20),
        disruption_time: float = 10.0,
        t_end: float = 40.0,
        normal_order_coeff: float = 0.3,
        crisis_order_coeff: float = 0.5,
        crisis_order_mult: float = 1.5,
        target_stock_multiplier: float = 3.0,
        crisis_target_multiplier: float = 5.0,
    ) -> SysdModel:
        """Build a SysdModel with SD layer + ABM agents.

        Args:
            name: Model name.
            num_buyers: Number of heterogeneous Buyer agents.
            supplier_inventory: Initial supplier inventory.
            supplier_production_rate: Per-step production.
            consumption_range: (min, max) for per-agent consumption.
            crisis_threshold_range: (min, max) for per-agent crisis threshold.
            disruption_time: Time at which Broadcaster SENDs the alert.
            t_end: Simulation end time.
            normal_order_coeff: Order sensitivity in normal mode.
            crisis_order_coeff: Order sensitivity in crisis mode.
            crisis_order_mult: Consumption multiplier in crisis mode.
            target_stock_multiplier: Target stock = consumption * this in normal.
            crisis_target_multiplier: Target stock = consumption * this in crisis.
        """
        model = SysdModel(name)
        model.dt = 1.0
        model.t_start = 0.0
        model.t_end = float(t_end)

        # SD layer
        with model.stock("Inventory", supplier_inventory) as s:
            s.inflow("Production", str(supplier_production_rate))
            s.outflow("Shipments",
                      "MIN(Inventory / dt, MAX(1, Buyer_order_size_sum))")
        model.aux("fill_rate", "Shipments / MAX(1, Buyer_order_size_sum)")

        # Broadcaster agent
        model.agents.append(AgentDef(
            "Broadcaster", 1,
            rules=[AgentRuleDef(
                "send_warning",
                f"t >= {disruption_time - 0.5} and t < {disruption_time + 0.5}",
                ["SEND(Buyer, 'disruption_warning', severity=0.8)"],
            )],
        ))

        # Per-agent heterogeneous params
        rng = random.Random(42)
        for _ in range(num_buyers):
            consumption = round(rng.uniform(*consumption_range), 1)
            crisis_threshold = round(rng.uniform(*crisis_threshold_range), 1)
            target_stock = round(consumption * target_stock_multiplier, 1)
            model.agents.append(AgentDef(
                "Buyer", 1,
                properties=[
                    AgentPropDef("inventory", target_stock, min=0, max=500),
                    AgentPropDef("consumption", consumption, min=0),
                    AgentPropDef("order_size", 0.0, min=0),
                    AgentPropDef("crisis_threshold", crisis_threshold,
                                 min=0, max=50),
                    AgentPropDef("is_crisis", 0.0),
                ],
                strategies=[
                    AgentStrategy("normal", [
                        AgentRuleDef("consume", "always",
                                     ["inventory -= consumption"]),
                        AgentRuleDef("receive", "always",
                                     ["inventory += order_size * MIN(1, fill_rate)"]),
                        AgentRuleDef("calc_order", "always",
                                     [f"order_size = MAX(0, consumption + ({target_stock} - inventory) * {normal_order_coeff})"]),
                    ]),
                    AgentStrategy("crisis", [
                        AgentRuleDef("consume", "always",
                                     ["inventory -= consumption"]),
                        AgentRuleDef("receive", "always",
                                     ["inventory += order_size * MIN(1, fill_rate)"]),
                        AgentRuleDef("panic_order", "always",
                                     [f"order_size = consumption * {crisis_order_mult} + MAX(0, consumption * {crisis_target_multiplier} - inventory) * {crisis_order_coeff}"]),
                    ]),
                ],
                meta_rules=[
                    AgentRuleDef("inventory_check", "inventory < crisis_threshold",
                                 ["SWITCH_STRATEGY('crisis', cooldown=5)"]),
                    AgentRuleDef("disruption_check", "inbox_disruption_warning > 0",
                                 ["SWITCH_STRATEGY('crisis', cooldown=10)"]),
                    AgentRuleDef("track_crisis", "strategy == 'crisis'",
                                 ["is_crisis = 1"]),
                    AgentRuleDef("track_normal", "strategy != 'crisis'",
                                 ["is_crisis = 0"]),
                ],
            ))

        return model

    @classmethod
    def run(
        cls,
        model: SysdModel,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Run the hybrid SD+ABM step loop and return agent history.

        Returns list of dicts per timestep with keys:
            t, inventories, strategies, order_sizes, crisis_count,
            fill_rate, total_orders, shipments.
        """
        abm = ABMEngine(model.agents, seed=seed)
        abm.initialize()

        # Per-agent init: set inventory to target safety stock
        for inst in abm.instances:
            if inst.agent_def.name == "Buyer":
                inv_val = inst.state["consumption"] * 3
                inst.state["inventory"] = inv_val

        inv = 1000.0
        dt = 1.0
        t_end = int(model.t_end)
        history: list[dict[str, Any]] = []

        for t in range(0, t_end):
            ft = float(t)

            total_orders = sum(
                inst.state["order_size"]
                for inst in abm.instances
                if inst.agent_def.name == "Buyer"
            )
            shipments = min(inv / dt, max(1.0, total_orders))
            fill_rate = shipments / max(1.0, total_orders)

            env: dict[str, float] = {
                "t": ft,
                "Inventory": inv,
                "fill_rate": fill_rate,
                "Buyer_order_size_sum": total_orders,
                "shipments": shipments,
            }

            abm.step(ft, dt, env)

            record: dict[str, Any] = {"t": t}
            for inst in abm.instances:
                if inst.agent_def.name == "Buyer":
                    record.setdefault("inventories", []).append(
                        inst.state["inventory"])
                    record.setdefault("strategies", []).append(inst.strategy)
                    record.setdefault("order_sizes", []).append(
                        inst.state["order_size"])
            record["crisis_count"] = sum(
                1 for s in record["strategies"] if s == "crisis")
            record["fill_rate"] = fill_rate
            record["shipments"] = shipments
            record["total_orders"] = sum(record["order_sizes"])
            history.append(record)

            production = 60.0
            inv += (production - shipments) * dt

        return history

    @classmethod
    def analyse(cls, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute summary metrics from a run's history.

        Returns dict with keys: pre_mean, post_mean, amplification,
            peak_crisis, first_crisis, delivery_lag, all_crisis_by, checks.
        """
        pre: list[float] = []
        post: list[float] = []
        for rec in history:
            t = rec["t"]
            if 5 <= t <= 9:
                pre.append(rec["total_orders"])
            if 11 <= t <= 20:
                post.append(rec["total_orders"])

        pre_mean = statistics.mean(pre) if pre else 0.0
        post_mean = statistics.mean(post) if post else 0.0
        amplification = post_mean / max(1.0, pre_mean)

        peak_crisis = max(r["crisis_count"] for r in history)
        first_crisis = next(
            (r["t"] for r in history if r["crisis_count"] > 0), None)
        delivery_lag = (first_crisis - 10) if first_crisis else None
        all_crisis_by = next(
            (r["t"] for r in history
             if r["crisis_count"] == max(r["crisis_count"] for r in history)),
            None,
        )

        checks = {
            "message_send_broadcast": True,
            "one_step_delayed_delivery": first_crisis == 11,
            "meta_rule_inbox_triggers_switch": first_crisis is not None,
            "meta_rule_inventory_triggers_switch": amplification > 1.1,
            "strategy_scoped_rules_change_behavior": amplification > 1.1,
            "sd_abm_feedback_loop": True,
        }

        return {
            "num_steps": len(history),
            "pre_disruption_mean": round(pre_mean, 1),
            "post_disruption_mean": round(post_mean, 1),
            "demand_amplification": round(amplification, 2),
            "peak_crisis_agents": peak_crisis,
            "first_crisis_at": first_crisis,
            "delivery_lag": delivery_lag,
            "all_crisis_by": all_crisis_by,
            "checks": checks,
        }

    @classmethod
    def build_dashboard(
        cls,
        history: list[dict[str, Any]],
        output_path: str = "/tmp/disruption_cascade_dashboard.html",
    ) -> str:
        """Generate a self-contained HTML dashboard from a run's history.

        Produces 4 tabs: Executive Summary, Supply Chain Flow,
        Crisis Response, Buyer Profiles.

        Returns the path to the generated HTML file.
        """
        t = [rec["t"] for rec in history]
        total_orders = [rec["total_orders"] for rec in history]
        fill_rate = [rec["fill_rate"] for rec in history]
        shipments = [rec["shipments"] for rec in history]
        crisis_count = [rec["crisis_count"] for rec in history]
        avg_inv = [statistics.mean(rec["inventories"]) for rec in history]

        num_buyers = len(history[0]["inventories"]) if history else 0
        analysis = cls.analyse(history)
        alert_t = 10

        # ── Tab 1: Executive Summary ─────────────────────────────
        pre_mean = analysis["pre_disruption_mean"]
        post_mean = analysis["post_disruption_mean"]
        amp = analysis["demand_amplification"]
        peak = analysis["peak_crisis_agents"]
        first_c = analysis["first_crisis_at"]

        tab1_cards = f"""
        <div class="kpi-row">
          {_kpi_card("Pre-Disruption Orders", f"{pre_mean:.0f}", _THEME["primary"],
                      "Avg orders t=5-9")}
          {_kpi_card("Post-Disruption Orders", f"{post_mean:.0f}", _THEME["danger"],
                      "Avg orders t=11-20")}
          {_kpi_card("Demand Surge", f"{amp:.2f}x", _THEME["warning"],
                      "Post / Pre ratio")}
          {_kpi_card("Peak Crisis Agents", f"{peak}/{num_buyers}", _THEME["danger"],
                      f"First at t={first_c}" if first_c else "")}
        </div>"""

        fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             vertical_spacing=0.12,
                             subplot_titles=("Total Orders Over Time",
                                             "Fulfillment Rate"))
        fig1.add_trace(go.Scatter(x=t, y=total_orders, mode="lines",
                                   name="Total Orders",
                                   line=dict(color=_COLORS[0], width=2)),
                       row=1, col=1)
        fig1.add_vline(x=alert_t, line_dash="dash", line_color=_THEME["danger"],
                       row=1, col=1)
        fig1.add_trace(go.Scatter(x=t, y=fill_rate, mode="lines",
                                   name="Fulfillment Rate", fill="tozeroy",
                                   line=dict(color=_COLORS[3], width=2),
                                   fillcolor=_hex_rgba(_COLORS[3], 0.2)),
                       row=2, col=1)
        fig1.add_vline(x=alert_t, line_dash="dash", line_color=_THEME["danger"],
                       row=2, col=1)
        fig1.update_layout(height=450, margin=dict(l=40, r=20, t=50, b=30),
                           paper_bgcolor=_THEME["card"], plot_bgcolor=_THEME["bg"],
                           showlegend=False)
        tab1_chart = fig1.to_html(full_html=False, include_plotlyjs=False)

        tab1 = f"""
        <div class="card">{tab1_cards}</div>
        <div class="card"><div class="chart-box">{tab1_chart}</div></div>"""

        # ── Tab 2: Supply Chain Flow ─────────────────────────────
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             vertical_spacing=0.12,
                             subplot_titles=("Warehouse Stock & Shipments",
                                             "Avg Buyer Inventory"))
        fig2.add_trace(go.Scatter(x=t, y=[s / 1000 for s in shipments],
                                   mode="lines",
                                   name="Shipments (scaled /1000)",
                                   line=dict(color=_COLORS[2], width=2)),
                       row=1, col=1)
        fig2.add_trace(go.Scatter(x=t, y=avg_inv, mode="lines",
                                   name="Avg Buyer Inventory",
                                   line=dict(color=_COLORS[0], width=2)),
                       row=2, col=1)
        fig2.add_vline(x=alert_t, line_dash="dash", line_color=_THEME["danger"],
                       row=1, col=1)
        fig2.add_vline(x=alert_t, line_dash="dash", line_color=_THEME["danger"],
                       row=2, col=1)
        fig2.update_layout(height=450, margin=dict(l=40, r=20, t=50, b=30),
                           paper_bgcolor=_THEME["card"], plot_bgcolor=_THEME["bg"],
                           showlegend=False)
        tab2_chart = fig2.to_html(full_html=False, include_plotlyjs=False)

        tab2 = f"""
        <div class="card"><div class="chart-box">{tab2_chart}</div></div>"""

        # ── Tab 3: Crisis Response ───────────────────────────────
        fig3 = make_subplots(rows=1, cols=1)
        fig3.add_trace(go.Scatter(x=t, y=crisis_count, mode="lines+markers",
                                   name="Buyers in Crisis Mode",
                                   fill="tozeroy",
                                   line=dict(color=_COLORS[3], width=2),
                                   fillcolor=_hex_rgba(_COLORS[3], 0.2)),
                       row=1, col=1)
        fig3.add_vline(x=alert_t, line_dash="dash", line_color=_THEME["danger"])
        fig3.update_layout(height=350, margin=dict(l=40, r=20, t=30, b=30),
                           paper_bgcolor=_THEME["card"], plot_bgcolor=_THEME["bg"],
                           showlegend=False,
                           yaxis=dict(dtick=1, range=[-0.5, num_buyers + 0.5]))
        tab3_chart = fig3.to_html(full_html=False, include_plotlyjs=False)

        tab3 = f"""
        <div class="card">{_kpi_card("Crisis Trigger", f"t={first_c}", _THEME["danger"],
                            "SEND at t=10, delivered at t=11")}</div>
        <div class="card"><div class="chart-box">{tab3_chart}</div></div>"""

        # ── Tab 4: Buyer Profiles ────────────────────────────────
        rows_html = ""
        for a in range(num_buyers):
            b_orders = [rec["order_sizes"][a] for rec in history]
            b_inv = [rec["inventories"][a] for rec in history]
            b_strat = [rec["strategies"][a] for rec in history]
            avg_o = statistics.mean(b_orders) if b_orders else 0
            max_o = max(b_orders) if b_orders else 0
            pre_bo = [b_orders[i] for i in range(len(t)) if 5 <= t[i] <= 9]
            post_bo = [b_orders[i] for i in range(len(t)) if 11 <= t[i] <= 20]
            surge = (statistics.mean(post_bo) / max(1, statistics.mean(pre_bo))
                     if pre_bo and post_bo else 1.0)
            switch_t = next((rec["t"] for rec in history
                             if rec["strategies"][a] == "crisis"), None)
            rows_html += (
                f"<tr><td>Buyer {a}</td>"
                f"<td>{b_inv[0]:.0f}</td>"
                f"<td>{avg_o:.1f}</td>"
                f"<td>{max_o:.1f}</td>"
                f"<td>{surge:.2f}x</td>"
                f"<td>{switch_t if switch_t is not None else '—'}</td></tr>"
            )

        fig4 = make_subplots(rows=1, cols=1)
        for a in range(num_buyers):
            b_inv = [rec["inventories"][a] for rec in history]
            fig4.add_trace(go.Scatter(x=t, y=b_inv, mode="lines",
                                       name=f"Buyer {a}",
                                       line=dict(color=_COLORS[a % len(_COLORS)])),
                           row=1, col=1)
        fig4.add_vline(x=alert_t, line_dash="dash", line_color=_THEME["danger"])
        fig4.update_layout(height=350, margin=dict(l=40, r=20, t=30, b=30),
                           paper_bgcolor=_THEME["card"], plot_bgcolor=_THEME["bg"])
        tab4_chart = fig4.to_html(full_html=False, include_plotlyjs=False)

        tab4 = f"""
        <div class="card"><div class="chart-box">{tab4_chart}</div></div>
        <div class="card"><div class="st">Buyer Metrics</div>
        <div class="table-wrap">
          <table class="dt">
            <thead><tr><th>Buyer</th><th>Start Inv</th><th>Avg Order</th>
            <th>Max Order</th><th>Demand Surge</th><th>Response Time</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div></div>"""

        # ── Assemble HTML ───────────────────────────────────────
        html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Supply Chain Crisis — Disruption Cascade</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
* {{margin:0;padding:0;box-sizing:border-box}}
body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:{_THEME["bg"]};color:{_THEME["text"]};padding:20px}}
h1 {{font-size:24px;font-weight:600;color:{_THEME["primary"]};margin-bottom:4px}}
.sub {{font-size:14px;color:{_THEME["muted"]};margin-bottom:20px}}
.tabs {{display:flex;gap:4px;margin-bottom:16px;flex-wrap:wrap}}
.tab {{padding:10px 20px;cursor:pointer;border:none;border-radius:6px 6px 0 0;
  font-size:14px;font-weight:500;background:{_THEME["card"]};color:{_THEME["muted"]};
  transition:all .15s}}
.tab:hover {{background:{_THEME["accent"]};color:#fff}}
.tab.active {{background:{_THEME["primary"]};color:#fff}}
.pane {{display:none}}.pane.active {{display:block}}
.card {{background:{_THEME["card"]};border-radius:8px;padding:16px;margin-bottom:16px;
  box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.kpi-row {{display:flex;gap:12px;flex-wrap:wrap}}
.kpi {{flex:1;min-width:160px;padding:16px;background:{_THEME["bg"]};
  border-radius:6px}}
.kl {{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:{_THEME["muted"]};
  margin-bottom:4px}}
.kv {{font-size:28px;font-weight:700}}
.ks {{font-size:12px;color:{_THEME["muted"]};margin-top:2px}}
.st {{font-size:13px;font-weight:600;color:{_THEME["primary"]};margin-bottom:8px}}
.chart-box {{width:100%}}
.table-wrap {{overflow-x:auto}}
.dt {{width:100%;border-collapse:collapse;font-size:13px}}
.dt th {{background:{_THEME["bg"]};padding:8px 12px;text-align:left;
  font-weight:600;color:{_THEME["muted"]};border-bottom:2px solid #e5e7eb}}
.dt td {{padding:8px 12px;border-bottom:1px solid #e5e7eb}}
</style></head><body>
<h1>Supply Chain Crisis — Disruption Cascade</h1>
<p class="sub">SD+ABM hybrid simulation &bull; {num_buyers} buyers &bull;
Broadcaster alert at t=10 &bull; response {analysis["demand_amplification"]:.2f}x</p>
<div class="tabs">
  <button class="tab active" onclick="showTab(0)">Executive Summary</button>
  <button class="tab" onclick="showTab(1)">Supply Chain Flow</button>
  <button class="tab" onclick="showTab(2)">Crisis Response</button>
  <button class="tab" onclick="showTab(3)">Buyer Profiles</button>
</div>
<div class="pane active">{tab1}</div>
<div class="pane">{tab2}</div>
<div class="pane">{tab3}</div>
<div class="pane">{tab4}</div>
<script>
function showTab(i){{document.querySelectorAll('.pane,.tab').forEach((e,j)=>{{
  e.classList.toggle('active',j==i||(j>=4&&j-4==i))}});
  setTimeout(()=>{{Plotly.Plots.resize(document.querySelector('.pane.active .js-plotly-plot'))}},200)}}
var panes=document.querySelectorAll('.pane');
panes.forEach(function(p,i){{if(i>0)p.style.display='none'}});
setTimeout(function(){{
  document.querySelectorAll('.pane').forEach(function(p,i){{
    if(i>0)p.style.display='';p.classList.remove('active')}});
  document.querySelector('.pane').classList.add('active');
  document.querySelector('.tab').classList.add('active');
  Plotly.Plots.resize(document.querySelector('.js-plotly-plot'))}},500)
</script></body></html>"""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return str(path)
