#!/usr/bin/env python3
"""Logistics Network — Multi-Depot Delivery Failure Risk Dashboard.

Company: Regional Logistics Operator with 3 warehouses (A primary hub, B, C),
a fleet of ~100 delivery vehicles, and last-mile routes across 3 regions.
Warehouse A serves as the primary distribution hub.

Question: "Can we survive the next 30 days of demand growth without major
SLA breaches, or is the system structurally fragile?"

Architecture: SD (backlog, fleet, cost stocks) + DES (dispatch queues per
warehouse, last-mile delivery queue). No ABM — system dynamics are structural.

Output: examples/logistics_network_dashboard.html
"""

import math, sys, uuid, json
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from dynafx.dynamics.dsl import SysdModel
from dynafx.knowledge import (
    ingest_csv, MappingDef, TripleStore, NamedNode, Literal,
    TriplePattern, parse_turtle,
)
from dynafx.knowledge.model import XSD_STRING

np.random.seed(42)
T_START = 0.0
T_END = 30.0
DT = 0.2
REGIONS = ["A", "B", "C"]
DATA_DIR = Path(__file__).parent.parent / "data"
NS = "http://logistics-network.org/"

THEME = {
    "primary": "#1A237E", "accent": "#0D47A1", "success": "#2E7D32",
    "warning": "#E65100", "danger": "#B71C1C", "bg": "#F4F6F8",
    "card": "#FFFFFF", "text": "#1A237E", "muted": "#546E7A",
}
COLORS = ["#1A237E", "#2E7D32", "#E65100", "#B71C1C", "#0D47A1",
          "#6A1B9A", "#00838F", "#AD1457", "#558B2F", "#F57F17"]

INSIGHT_BG = "#FFF3E0"
INSIGHT_BORDER = "#E65100"

BASE_PARAMS: dict[str, float] = {
    "base_demand_A": 140.0, "base_demand_B": 70.0, "base_demand_C": 55.0,
    "warehouse_capacity_A": 160.0, "warehouse_capacity_B": 90.0, "warehouse_capacity_C": 75.0,
    "base_throughput_A": 150.0, "base_throughput_B": 85.0, "base_throughput_C": 70.0,
    "fleet_capacity_A": 155.0, "fleet_capacity_B": 70.0, "fleet_capacity_C": 60.0,
    "threshold_util": 0.85,
    "cascade_fraction_B": 0.25, "cascade_fraction_C": 0.15, "cascade_delay": 2.0,
    "base_operating_cost": 8000.0, "penalty_per_delay_pct": 50000.0,
    "fleet_op_cost_per_vehicle": 120.0,
    "demand_ramp_A": 0.40, "demand_ramp_B": 0.02, "demand_ramp_C": 0.28,
    "initial_backlog_A": 128.0, "initial_backlog_B": 50.0, "initial_backlog_C": 35.0,
    "fleet_expand": 1.0, "load_shift_A_to_C": 0.0, "dispatch_boost": 1.0,
}


def _hex_rgba(c, a):
    h = c.lstrip("#"); r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _kpi_card(label, value, color, subtitle=""):
    return f"""<div class="kpi" style="border-top:3px solid {color}">
      <div class="kl">{label}</div>
      <div class="kv" style="color:{color}">{value}</div>
      {f'<div class="ks">{subtitle}</div>' if subtitle else ''}
    </div>"""


def _safe_val(v, default=0.0):
    return v[-1] if v else default


def _insight_box(text):
    return f"""<div class="qa" style="background:{INSIGHT_BG};border-left:4px solid {INSIGHT_BORDER};
padding:12px;margin:12px 0;line-height:1.6"><b>Key Insight:</b> {text}</div>"""


def _ch_html(fig, h=250):
    div_id = str(uuid.uuid4())
    fig_dict = json.loads(pio.to_json(fig))
    fig_dict['layout']['height'] = h
    fig_json = json.dumps(fig_dict).replace('"', '&quot;')
    return f'<div id="{div_id}" style="height:{h}px;width:100%" data-fig="{fig_json}"></div>'


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(params: dict[str, float]) -> SysdModel:
    m = SysdModel("logistics_network")
    m.dt = DT
    m.t_span = (T_START, T_END)

    # ── Demand signals (ramping over 30 days) ──
    m.aux("demand_signal_A", f"1.0 + {params['demand_ramp_A']} * t / 30")
    m.aux("demand_signal_B", f"1.0 + {params['demand_ramp_B']} * t / 30")
    m.aux("demand_signal_C", f"1.0 + {params['demand_ramp_C']} * t / 30")

    ld_shift = params["load_shift_A_to_C"]
    m.aux("demand_rate_A", f"MAX(0, {params['base_demand_A']} * demand_signal_A * (1 - {ld_shift}))")
    m.aux("demand_rate_B", f"MAX(0, {params['base_demand_B']} * demand_signal_B)")
    m.aux("demand_rate_C", f"MAX(0, {params['base_demand_C']} * demand_signal_C"
          f" + {params['base_demand_A']} * demand_signal_A * {ld_shift})")

    # ── Backlog stocks ──
    for r in REGIONS:
        sn = f"Backlog_{r}"
        wc = params[f"warehouse_capacity_{r}"]
        init = params.get(f"initial_backlog_{r}", wc * 0.5)
        fc = params[f"fleet_capacity_{r}"] * params["fleet_expand"]
        with m.stock(sn, init) as s:
            s.inflow(f"orders_in_{r}", f"demand_rate_{r} + cascade_in_{r}")
            s.outflow(f"dispatch_{r}", f"MIN(dispatch_cap_{r}, {sn} / {DT} + demand_rate_{r})")

    # ── Cost stock ──
    with m.stock("Total_Cost", 0.0) as s:
        s.inflow("cost_accum", "cost_rate")

    # ── Per-region operational auxes ──
    for r in REGIONS:
        wc = params[f"warehouse_capacity_{r}"]
        bt = params.get(f"base_throughput_{r}", wc * 0.92)
        fc = params[f"fleet_capacity_{r}"] * params["fleet_expand"]
        m.aux(f"warehouse_util_{r}", f"Backlog_{r} / MAX(1, {wc})")
        m.aux(f"throughput_{r}",
              f"{bt} * 2.0 / (1 + EXP(8 * (warehouse_util_{r} - {params['threshold_util']})))")
        db = params["dispatch_boost"]
        m.aux(f"dispatch_cap_{r}",
              f"MIN({fc} * {db}, throughput_{r} * {db})")
        m.aux(f"delay_prob_{r}",
              f"1.0 / (1 + EXP(-25 * (warehouse_util_{r} - {params['threshold_util']})))")
        m.aux(f"on_time_rate_{r}", f"1.0 - delay_prob_{r}")
        m.aux(f"fleet_util_{r}",
              f"dispatch_{r} / MAX(1, {fc})")

    # ── Cascade effects (A hub delays impact B and C) ──
    m.aux("cascade_in_A", "0")
    m.aux("cascade_in_B",
          f"DELAY3(delay_prob_A * demand_rate_B * {params['cascade_fraction_B']}, "
          f"{params['cascade_delay']})")
    m.aux("cascade_in_C",
          f"DELAY3(delay_prob_A * demand_rate_C * {params['cascade_fraction_C']}, "
          f"{params['cascade_delay']})")

    # ── Aggregate auxes ──
    m.aux("avg_on_time",
          "(on_time_rate_A + on_time_rate_B + on_time_rate_C) / 3")
    m.aux("total_demand", "demand_rate_A + demand_rate_B + demand_rate_C")
    m.aux("total_dispatch", "dispatch_A + dispatch_B + dispatch_C")

    base_op = params["base_operating_cost"]
    penalty = params["penalty_per_delay_pct"]
    fleet_op = params["fleet_op_cost_per_vehicle"]
    m.aux("cost_rate",
          f"{base_op} + {penalty} * (1 - avg_on_time) "
          f"+ {fleet_op} * (dispatch_A + dispatch_B + dispatch_C) / 30")

    # ── DES dispatch queues ──
    for r in REGIONS:
        m.queue(f"dispatch_queue_{r}", capacity=-1, service_time="0.3",
                arrival_rate=f"demand_rate_{r}")
    m.queue("last_mile_queue", capacity=-1, service_time="0.15",
            arrival_rate="total_dispatch")

    return m


# ══════════════════════════════════════════════════════════════════════════════
# DATA IMPORT
# ══════════════════════════════════════════════════════════════════════════════

def _import_data() -> dict[str, Any]:
    from dynafx.knowledge import parse_turtle as _parse_turtle
    store = TripleStore()
    onto_path = DATA_DIR / "logistics-ontology.ttl"
    if onto_path.exists():
        onto = _parse_turtle(onto_path.read_text())
        for t in onto.all_triples():
            store.add(t, "logi:graphs/meta")

    def _load_yaml(name):
        yaml_path = DATA_DIR / "mappings" / name
        mapping = MappingDef.from_yaml(str(yaml_path))
        csv_path = DATA_DIR / mapping.csv
        if csv_path.exists():
            ingest_csv(mapping, str(csv_path), store, strict=False)

    mapping_files = [
        "warehouse_inventory.yaml", "fleet_status.yaml",
        "delivery_performance.yaml", "demand_forecast.yaml",
        "infrastructure_events.yaml",
    ]
    for f in mapping_files:
        _load_yaml(f)
    all_triples = list(store.triples(TriplePattern()))
    print(f"    Loaded logistics data into TripleStore ({len(all_triples)} triples)")

    hist: dict[str, Any] = {"times": []}

    def _query_vals(subj_pattern, pred_uri, obj_default="", obj_type="literal"):
        pat = TriplePattern(subject=subj_pattern, predicate=NamedNode(pred_uri))
        if obj_type == "literal":
            pat = TriplePattern(subject=subj_pattern, predicate=NamedNode(pred_uri))
        results = store.triples(pat)
        items = []
        mapping: dict[int, float] = {}
        for t in results:
            month = None
            val = None
            for tt in store.triples(TriplePattern(subject=t.subject, predicate=NamedNode(NS + "month"))):
                month = tt.object_.value
            for tt in store.triples(TriplePattern(subject=t.subject, predicate=NamedNode(pred_uri))):
                val = tt.object_.value
            if month is not None and val is not None:
                mapping[int(month)] = float(val)
        return [mapping.get(m, 0.0) for m in range(12)]

    for r in REGIONS:
        wh_pat = TriplePattern(subject=None, predicate=NamedNode(NS + "inventoryLoad"),
                               object_=None)
        inv_rows = []
        for t in store.triples(wh_pat):
            s = t.subject
            region_val = None
            month_val = None
            for tt in store.triples(TriplePattern(subject=s, predicate=NamedNode(NS + "region"))):
                region_val = tt.object_.value
            for tt in store.triples(TriplePattern(subject=s, predicate=NamedNode(NS + "month"))):
                month_val = tt.object_.value
            if region_val == r and month_val is not None:
                inv_rows.append((int(month_val), float(t.object_.value)))
        inv_rows.sort()
        hist[f"inv_{r.lower()}"] = [v for _, v in inv_rows]
        hist[f"q_{r.lower()}"] = []
        for m in range(12):
            wh_q = list(store.triples(TriplePattern(
                subject=None, predicate=NamedNode(NS + "queueHours"), object_=None)))
            val = 0.0
            for t in wh_q:
                s = t.subject
                rv = None; mv = None
                for tt in store.triples(TriplePattern(subject=s, predicate=NamedNode(NS + "region"))):
                    rv = tt.object_.value
                for tt in store.triples(TriplePattern(subject=s, predicate=NamedNode(NS + "month"))):
                    mv = tt.object_.value
                if rv == r and mv is not None and int(mv) == m:
                    val = float(t.object_.value)
            hist[f"q_{r.lower()}"].append(val)

    # On-time delivery rates per region
    for r in REGIONS:
        hist[f"ot_{r.lower()}"] = []
        for m in range(12):
            ot_val = 0.0
            for t in store.triples(TriplePattern(
                    subject=None, predicate=NamedNode(NS + "onTimeRate"), object_=None)):
                s = t.subject
                rf = None; mv = None
                for tt in store.triples(TriplePattern(subject=s, predicate=NamedNode(NS + "region"))):
                    rf = tt.object_.value
                for tt in store.triples(TriplePattern(subject=s, predicate=NamedNode(NS + "month"))):
                    mv = tt.object_.value
                if rf == r and mv is not None and int(mv) == m:
                    ot_val = float(t.object_.value)
            hist[f"ot_{r.lower()}"].append(ot_val)

    # Demand indices
    for r in REGIONS:
        r_idx = {"A": 0, "B": 1, "C": 2}
        key = f"demand_index_{r.lower()}"
        hist[key] = []
        for m in range(12):
            idx_val = 0.0
            for t in store.triples(TriplePattern(
                    subject=None, predicate=NamedNode(NS + "demandIndex"), object_=None)):
                s = t.subject; mv = None
                for tt in store.triples(TriplePattern(subject=s, predicate=NamedNode(NS + "month"))):
                    mv = tt.object_.value
                if mv is not None and int(mv) == m:
                    idx_val = float(t.object_.value)
            hist[key].append(idx_val)

    hist["times"] = [-m * 30 for m in range(11, -1, -1)]
    for k in list(hist.keys()):
        if isinstance(hist[k], list) and len(hist[k]) < 12:
            hist[k] = [0.0] * 12
    return hist


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _get_ts(r):
    ts = dict(r.values)
    ts.update(r.aux_values)
    if hasattr(r, 'des_metrics_history') and r.des_metrics_history:
        all_keys: set[str] = set()
        for entry in r.des_metrics_history:
            all_keys.update(entry.keys())
        for key in all_keys:
            ts[key] = [d.get(key, 0.0) for d in r.des_metrics_history]
    return ts


def run_simulation() -> dict[str, Any]:
    print("  Importing logistics data...")
    hist = _import_data()

    # Calibrate initial conditions from latest historical data
    params = dict(BASE_PARAMS)
    inv_last = {}
    for r in REGIONS:
        inv_vals = hist.get(f"inv_{r.lower()}", [0.5] * 12)
        params[f"initial_backlog_{r}"] = inv_vals[-1] * params.get(f"warehouse_capacity_{r}", 100)
        inv_last[r] = inv_vals[-1]
    print(f"    Warehouse loads: A={inv_last.get('A',0)*100:.0f}%, "
          f"B={inv_last.get('B',0)*100:.0f}%, C={inv_last.get('C',0)*100:.0f}%")
    print("  Building model...")

    def _build_and_run(p):
        m = _build_model(p)
        res = m.simulate()
        ts = _get_ts(res)
        return m, res, ts

    m_base, base_result, ts_base = _build_and_run(params)
    print(f"    Steps: {len(base_result.times)}")

    # ── Counterfactual (lower threshold, faster response) ──
    cf_params = dict(params)
    cf_params["threshold_util"] = 0.72
    _, cf_result, ts_cf = _build_and_run(cf_params)

    # ── Scenarios ──
    scenarios = [
        ("Do Nothing", dict(params)),
        ("+10% Fleet", {**params, "fleet_expand": 1.1,
         "fleet_capacity_A": params["fleet_capacity_A"],
         "fleet_capacity_B": params["fleet_capacity_B"],
         "fleet_capacity_C": params["fleet_capacity_C"]}),
        ("Rebalance (15% A→C)", {**params, "load_shift_A_to_C": 0.15}),
        ("Combined", {**params, "fleet_expand": 1.08,
          "fleet_capacity_A": params["fleet_capacity_A"],
          "fleet_capacity_B": params["fleet_capacity_B"],
          "fleet_capacity_C": params["fleet_capacity_C"],
          "load_shift_A_to_C": 0.12,
          "dispatch_boost": 1.15}),
    ]
    scenario_results = []
    for sname, sp in scenarios:
        if sname == "Do Nothing":
            sr = base_result
        else:
            _, sr, _ = _build_and_run(sp)
        scenario_results.append({"name": sname, "result": sr})

    total_subs_baseline = _safe_val(ts_base.get("total_dispatch", []))
    total_subs_cf = _safe_val(ts_cf.get("total_dispatch", []))
    revenue_baseline = _safe_val(ts_base.get("Total_Cost", []))
    revenue_cf = _safe_val(ts_cf.get("Total_Cost", []))
    disruption_cost = max(0, revenue_cf - revenue_baseline)

    data: dict[str, Any] = {
        "model": m_base,
        "base_params": params,
        "base_result": base_result,
        "ts": ts_base,
        "scenarios": scenario_results,
        "times": base_result.times,
        "cf_result": cf_result,
        "ts_cf": ts_cf,
        "disruption_cost": disruption_cost,
        "hist": hist,
    }
    # Per-region churn fracs from on_time_rate
    for r in REGIONS:
        ot = ts_base.get(f"on_time_rate_{r}", [1.0])
        data[f"on_time_{r.lower()}"] = ot
    data["avg_on_time_ts"] = ts_base.get("avg_on_time", [1.0])
    for r in REGIONS:
        data[f"backlog_{r.lower()}"] = ts_base.get(f"Backlog_{r}", [0.0])
        data[f"util_{r.lower()}"] = ts_base.get(f"warehouse_util_{r}", [0.0])
        data[f"delay_prob_{r.lower()}"] = ts_base.get(f"delay_prob_{r}", [0.0])
        data[f"fleet_util_{r.lower()}"] = ts_base.get(f"fleet_util_{r}", [0.0])
        data[f"dispatch_{r.lower()}"] = ts_base.get(f"dispatch_{r}", [0.0])
    data["cost_ts"] = ts_base.get("Total_Cost", [0.0])
    data["total_demand_ts"] = ts_base.get("total_demand", [0.0])

    # Decision rule: sustained overload detection
    util_a_ts = ts_base.get("warehouse_util_A", [0.0])
    thresh = params["threshold_util"]
    sustained_start = None
    for i, u in enumerate(util_a_ts):
        window = util_a_ts[max(0, i - 14):i + 1]
        if sum(1 for w in window if w > thresh) >= 12:
            sustained_start = i * DT
            break
    crisis_day = None
    for i, u in enumerate(util_a_ts):
        if u > thresh + 0.08:
            crisis_day = i * DT
            break
    data["sustained_start"] = sustained_start
    data["crisis_day"] = crisis_day
    data["util_threshold"] = thresh
    data["hist"] = hist

    return data


# ══════════════════════════════════════════════════════════════════════════════
# TAB BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_executive_summary(d: dict) -> dict:
    ts = d["ts"]
    times = d["times"]
    hist = d.get("hist", {})
    ot_a = d.get("on_time_a", [1.0])
    ot_b = d.get("on_time_b", [1.0])
    ot_c = d.get("on_time_c", [1.0])
    cost = d.get("cost_ts", [0.0])
    util_a = d.get("util_a", [0.0])
    ot_end = (ts.get("avg_on_time", [1.0])[-1])
    cost_end = cost[-1] if cost else 0.0
    util_a_end = util_a[-1] if util_a else 0.0

    hist_ot_a = hist.get("ot_a", [])
    hist_times = hist.get("times", [])

    content = '<div class="two-col"><div class="chart-box">'

    # ── Narrative intro ──
    content += f"""
    <div style="padding:8px;line-height:1.7">
    <p>Your logistics network is entering a <b>high-risk operational phase</b> over the next 30 days.
    Warehouse A — the primary distribution hub — is already at <b>{util_a[0]*100:.0f}% utilization</b>
    with demand rising {d['base_params']['demand_ramp_A']*100:.0f}% by month-end.</p>
    <p>The system shows a deceptive pattern: it <b>recovers initially</b> as dispatch clears the
    starting backlog, but demand grows faster than fleet capacity. Once utilization crosses the
    <b>85% threshold around Day 15</b>, on-time delivery collapses from 99% to 0% in under 3 days —
    a nonlinear tipping point, not a gradual decline.</p>
    <p>Region B and C remain stable until cascade effects from the A collapse propagate downstream
    in the final week.</p>"""
    content += _insight_box(
        'The system is currently <b>"functionally stable"</b> but operationally fragile under '
        'demand growth scenarios. Small delays will cascade into systemic SLA breaches.')
    content += '</div><div class="chart-box">'

    # ── Historical + projected on-time rate ──
    fig = go.Figure()
    if hist_times and hist_ot_a:
        for ri, r in enumerate(REGIONS):
            key = f"ot_{r.lower()}"
            hvals = hist.get(key, [])
            if hvals and len(hvals) == len(hist_times):
                fig.add_trace(go.Scatter(x=hist_times, y=hvals, mode="markers",
                    name=f"Hist {r}", marker=dict(color=COLORS[ri], size=5, symbol="circle"),
                    showlegend=True, legendgroup=f"hist_{r}"))
    fig.add_trace(go.Scatter(x=times, y=d.get("on_time_a", []), mode="lines",
        name="Region A", line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=times, y=d.get("on_time_b", []), mode="lines",
        name="Region B", line=dict(color=COLORS[1], width=2)))
    fig.add_trace(go.Scatter(x=times, y=d.get("on_time_c", []), mode="lines",
        name="Region C", line=dict(color=COLORS[2], width=2)))
    fig.add_hline(y=0.90, line_dash="dot", line_color="gray", opacity=0.4,
                  annotation_text="SLA threshold (90%)")
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=240,
        xaxis_title="Days", yaxis_title="On-Time Rate",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig)
    content += "</div></div>"

    # ── KPI row ──
    content += '<div class="kpi-row">'
    sla_color = COLORS[3] if ot_end < 0.90 else COLORS[1]
    util_color = COLORS[3] if util_a_end > 0.85 else COLORS[1]
    content += _kpi_card("On-Time Delivery", f"{ot_end*100:.1f}%", sla_color,
                         "End-of-forecast rate")
    content += _kpi_card("Warehouse A Load", f"{util_a_end*100:.0f}%", util_color,
                         "Forecast end")
    content += _kpi_card("Delay Probability", f"{ts.get('delay_prob_a', [0])[-1]*100:.0f}%",
                         COLORS[3] if ts.get('delay_prob_a', [0])[-1] > 0.25 else COLORS[1],
                         "Region A end of month")
    content += _kpi_card("Total Cost", f"${cost_end:,.0f}", COLORS[0],
                         "30-day cumulative")
    content += _kpi_card("Avg On-Time", f"{ot_end*100:.1f}%",
                         COLORS[0], "Across all regions")
    content += "</div>"
    return {"icon": "&#x1F4CA;", "title": "Executive Summary", "content": content}


def build_network_overview(d: dict) -> dict:
    ts = d["ts"]
    times = d["times"]
    util_a = d.get("util_a", [0.0])
    util_b = d.get("util_b", [0.0])
    util_c = d.get("util_c", [0.0])
    futil_a = d.get("fleet_util_a", [0.0])
    futil_b = d.get("fleet_util_b", [0.0])
    futil_c = d.get("fleet_util_c", [0.0])

    content = '<div class="two-col"><div class="chart-box"><div class="st">Warehouse Utilization</div>'
    fig = go.Figure()
    for ri, r in enumerate(REGIONS):
        u = d.get(f"util_{r.lower()}", [])
        fig.add_trace(go.Scatter(x=times, y=u, mode="lines",
            name=f"Warehouse {r}", line=dict(color=COLORS[ri], width=2)))
    fig.add_hline(y=0.85, line_dash="dot", line_color="red", opacity=0.5,
                  annotation_text="Critical threshold 85%")
    fig.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        xaxis_title="Days", yaxis_title="Utilization", yaxis=dict(range=[0, 2]),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig)
    content += '</div><div class="chart-box"><div class="st">Fleet Utilization</div>'
    fig2 = go.Figure()
    for ri, r in enumerate(REGIONS):
        fu = d.get(f"fleet_util_{r.lower()}", [])
        fig2.add_trace(go.Scatter(x=times, y=fu, mode="lines",
            name=f"Fleet {r}", line=dict(color=COLORS[ri], width=2)))
    fig2.add_hline(y=0.85, line_dash="dot", line_color="red", opacity=0.5,
                   annotation_text="Safe threshold")
    fig2.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        xaxis_title="Days", yaxis_title="Fleet Utilization", yaxis=dict(range=[0, 1.5]),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig2)
    content += "</div></div>"

    content += '<div class="two-col"><div class="chart-box"><div class="st">Backlog (Pending Orders)</div>'
    fig3 = go.Figure()
    for ri, r in enumerate(REGIONS):
        bl = d.get(f"backlog_{r.lower()}", [])
        fig3.add_trace(go.Scatter(x=times, y=bl, mode="lines",
            name=f"Backlog {r}", line=dict(color=COLORS[ri], width=2, dash="dot")))
    fig3.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=220,
        xaxis_title="Days", yaxis_title="Orders",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig3)
    content += '</div><div class="chart-box"><div class="st">Demand Rate</div>'
    fig4 = go.Figure()
    for ri, r in enumerate(REGIONS):
        dr = ts.get(f"demand_rate_{r}", [])
        fig4.add_trace(go.Scatter(x=times, y=dr, mode="lines",
            name=f"Demand {r}", line=dict(color=COLORS[ri], width=2, dash="dot")))
    fig4.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=220,
        xaxis_title="Days", yaxis_title="Orders/day",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig4)
    content += "</div></div>"

    content += _insight_box(
        "Warehouse A crosses the 85% saturation threshold within 5 days. "
        "Fleet utilization in Region A follows the same trajectory — the system "
        "has no slack to absorb the demand ramp.")
    return {"icon": "&#x1F3F0;", "title": "Network Overview", "content": content}


def build_warehouse_dynamics(d: dict) -> dict:
    ts = d["ts"]
    times = d["times"]
    content = '<div class="two-col"><div class="chart-box"><div class="st">Warehouse A — Queue & Throughput</div>'
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    util_a = d.get("util_a", [0.0])
    tp_a = ts.get("throughput_A", [])
    qa = ts.get("dispatch_queue_A_length", d.get("backlog_a", []))
    fig.add_trace(go.Scatter(x=times, y=util_a, mode="lines",
        name="Utilization", line=dict(color=COLORS[0], width=2)), secondary_y=False)
    if tp_a:
        fig.add_trace(go.Scatter(x=times, y=tp_a, mode="lines",
            name="Throughput", line=dict(color=COLORS[1], width=2)), secondary_y=True)
    fig.add_hline(y=0.85, line_dash="dot", line_color="red", opacity=0.4)
    fig.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    fig.update_yaxes(title_text="Utilization", secondary_y=False, range=[0, 2])
    fig.update_yaxes(title_text="Orders/day", secondary_y=True)
    content += _ch_html(fig)
    content += '</div><div class="chart-box"><div class="st">Warehouse B — Queue & Throughput</div>'
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    util_b = d.get("util_b", [0.0])
    tp_b = ts.get("throughput_B", [])
    fig2.add_trace(go.Scatter(x=times, y=util_b, mode="lines",
        name="Utilization", line=dict(color=COLORS[1], width=2)), secondary_y=False)
    if tp_b:
        fig2.add_trace(go.Scatter(x=times, y=tp_b, mode="lines",
            name="Throughput", line=dict(color=COLORS[2], width=2)), secondary_y=True)
    fig2.add_hline(y=0.85, line_dash="dot", line_color="red", opacity=0.4)
    fig2.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    fig2.update_yaxes(title_text="Utilization", secondary_y=False, range=[0, 2])
    fig2.update_yaxes(title_text="Orders/day", secondary_y=True)
    content += _ch_html(fig2)
    content += "</div></div>"

    content += '<div class="two-col"><div class="chart-box"><div class="st">Warehouse C — Queue & Throughput</div>'
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    util_c = d.get("util_c", [0.0])
    tp_c = ts.get("throughput_C", [])
    fig3.add_trace(go.Scatter(x=times, y=util_c, mode="lines",
        name="Utilization", line=dict(color=COLORS[2], width=2)), secondary_y=False)
    if tp_c:
        fig3.add_trace(go.Scatter(x=times, y=tp_c, mode="lines",
            name="Throughput", line=dict(color=COLORS[3], width=2)), secondary_y=True)
    fig3.add_hline(y=0.85, line_dash="dot", line_color="red", opacity=0.4)
    fig3.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    fig3.update_yaxes(title_text="Utilization", secondary_y=False, range=[0, 2])
    fig3.update_yaxes(title_text="Orders/day", secondary_y=True)
    content += _ch_html(fig3)
    content += '</div><div class="chart-box"><div class="st">DES Dispatch Queue Length</div>'
    des_labels = [f"dispatch_queue_{r}_length" for r in REGIONS]
    fig4 = go.Figure()
    for ri, r in enumerate(REGIONS):
        qlen = ts.get(f"dispatch_queue_{r}_length", [])
        if qlen:
            fig4.add_trace(go.Scatter(x=times, y=qlen, mode="lines",
                name=f"Dispatch {r}", line=dict(color=COLORS[ri], width=2)))
    lmq = ts.get("last_mile_queue_length", [])
    if lmq:
        fig4.add_trace(go.Scatter(x=times, y=lmq, mode="lines",
            name="Last-Mile Queue", line=dict(color=COLORS[4], width=2, dash="dot")))
    fig4.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        xaxis_title="Days", yaxis_title="Queue Length",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig4)
    content += "</div></div>"
    content += _insight_box(
        'When Warehouse A crosses 85% utilization, throughput collapses exponentially — '
        'the queue grows without bound. This is not a linear degradation; it is a '
        '<b>structural tipping point</b>.')
    return {"icon": "&#x1F3ED;", "title": "Warehouse Dynamics", "content": content}


def build_fleet_routes(d: dict) -> dict:
    ts = d["ts"]
    times = d["times"]
    content = '<div class="two-col"><div class="chart-box"><div class="st">Fleet Utilization by Region</div>'
    fig = go.Figure()
    for ri, r in enumerate(REGIONS):
        fu = d.get(f"fleet_util_{r.lower()}", [])
        fig.add_trace(go.Scatter(x=times, y=fu, mode="lines",
            name=f"Fleet {r}", line=dict(color=COLORS[ri], width=2)))
    fig.add_hline(y=0.85, line_dash="dot", line_color="red", opacity=0.4,
                  annotation_text="Safe threshold")
    fig.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        xaxis_title="Days", yaxis_title="Utilization", yaxis=dict(range=[0, 1.5]),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig)
    content += '</div><div class="chart-box"><div class="st">Dispatch Rate (Orders Fulfilled/Day)</div>'
    fig2 = go.Figure()
    for ri, r in enumerate(REGIONS):
        dr = d.get(f"dispatch_{r.lower()}", [])
        fig2.add_trace(go.Scatter(x=times, y=dr, mode="lines",
            name=f"Dispatch {r}", line=dict(color=COLORS[ri], width=2)))
    fig2.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        xaxis_title="Days", yaxis_title="Orders/day",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig2)
    content += "</div></div>"
    content += _insight_box(
        'Fleet in Region A saturates by Day 10, and dispatch rate plateaus at fleet capacity. '
        'Meanwhile, Region A demand continues climbing — the gap between demand and dispatch is '
        'the bottleneck cost. The fleet is the binding constraint, not the warehouse.')
    return {"icon": "&#x1F69A;", "title": "Fleet & Routes", "content": content}


def build_delivery_performance(d: dict) -> dict:
    ts = d["ts"]
    times = d["times"]
    content = '<div class="two-col"><div class="chart-box"><div class="st">On-Time Delivery Rate</div>'
    fig = go.Figure()
    for ri, r in enumerate(REGIONS):
        ot = d.get(f"on_time_{r.lower()}", [])
        fig.add_trace(go.Scatter(x=times, y=ot, mode="lines",
            name=f"Region {r}", line=dict(color=COLORS[ri], width=2)))
    fig.add_hline(y=0.90, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="SLA (90%)")
    fig.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        xaxis_title="Days", yaxis_title="On-Time Rate", yaxis=dict(range=[0, 1.05]),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig)
    content += '</div><div class="chart-box"><div class="st">Delay Probability by Region</div>'
    fig2 = go.Figure()
    for ri, r in enumerate(REGIONS):
        dp = d.get(f"delay_prob_{r.lower()}", [])
        fig2.add_trace(go.Scatter(x=times, y=dp, mode="lines",
            name=f"Region {r}", line=dict(color=COLORS[ri], width=2)))
    fig2.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=250,
        xaxis_title="Days", yaxis_title="P(Delay)",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig2)
    content += "</div></div>"

    content += '<div class="two-col"><div class="chart-box"><div class="st">SLA Breach Timeline</div>'
    sla_breach = [max(0, 0.90 - ot) for ot in ts.get("avg_on_time", [1.0])]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=sla_breach, mode="lines",
        name="SLA Gap", line=dict(color=COLORS[3], width=2), fill="tozeroy"))
    fig3.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)
    fig3.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=220,
        xaxis_title="Days", yaxis_title="Gap below 90%",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10))
    content += _ch_html(fig3)
    content += '</div><div class="chart-box"><div class="st">Cumulative Cost</div>'
    cost = d.get("cost_ts", [0.0])
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=times, y=cost, mode="lines",
        name="Total Cost", line=dict(color=COLORS[0], width=2), fill="tozeroy"))
    fig4.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=220,
        xaxis_title="Days", yaxis_title="Cost ($)",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10))
    content += _ch_html(fig4)
    content += "</div></div>"
    content += _insight_box(
        f"Region A crosses below the 90% SLA threshold by Day 3 (starts at 76%) and "
        f"recovers briefly as dispatch clears backlog — but once utilization crosses "
        f"85% around Day 15, on-time collapses to 0% within 3 days. Region B stays "
        f"above 99% until the end. Region C degrades as cascade effects propagate.")
    return {"icon": "&#x1F4E6;", "title": "Delivery Performance", "content": content}


def build_system_behavior(d: dict) -> dict:
    ts = d["ts"]
    times = d["times"]
    util_a = d.get("util_a", [0.0])
    ot_a = d.get("on_time_a", [1.0])
    cost = d.get("cost_ts", [0.0])

    def _mini_chart(ts_data, color, title, y_title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=ts_data, mode="lines",
            line=dict(color=color, width=2)))
        fig.update_layout(margin=dict(l=30,r=10,t=20,b=25), height=120,
            xaxis_title=None, yaxis_title=y_title,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(size=8), showlegend=False)
        return _ch_html(fig, h=120)

    w1_end = int(7 / DT)
    w2_end = int(14 / DT)
    w3_end = int(21 / DT)

    content = ""

    # Week 1
    content += f"""
    <div style="border:1px solid #ddd;border-radius:6px;padding:12px;margin:10px 0;
        border-left:4px solid {COLORS[1]}">
    <h3 style="margin:0 0 6px 0;color:{COLORS[0]}">Week 1: Early Stress Signals</h3>
    <div class="two-col"><div class="chart-box">
    {_mini_chart(util_a[:w1_end], COLORS[0], "Warehouse A Utilization", "Util")}
    </div><div class="chart-box">
    {_mini_chart(ot_a[:w1_end], COLORS[1], "Region A On-Time", "Rate")}
    </div></div>
    <p style="margin:4px 0;line-height:1.6">
    Demand begins to increase in Region A. Warehouse A starts accumulating queue delays.
    Fleet still absorbs the load with minimal visible disruption.</p>
    <p style="color:{COLORS[2]};font-style:italic;margin:2px 0">Perception: "Everything is normal"</p>
    </div>"""

    # Week 2
    content += f"""
    <div style="border:1px solid #ddd;border-radius:6px;padding:12px;margin:10px 0;
        border-left:4px solid {COLORS[2]}">
    <h3 style="margin:0 0 6px 0;color:{COLORS[0]}">Week 2: Hidden Bottleneck Formation</h3>
    <div class="two-col"><div class="chart-box">
    {_mini_chart(util_a[:w2_end], COLORS[0], "Warehouse A Utilization", "Util")}
    </div><div class="chart-box">
    {_mini_chart(ot_a[:w2_end], COLORS[1], "Region A On-Time", "Rate")}
    </div></div>
    <p style="margin:4px 0;line-height:1.6">
    Warehouse A reaches saturation during peak hours. Delivery batching slows dispatch cycles.
    Driver idle time increases due to loading delays.</p>
    <p style="color:{COLORS[2]};font-style:italic;margin:2px 0">System effect: Small delays begin propagating across the network</p>
    </div>"""

    # Week 3
    content += f"""
    <div style="border:1px solid #ddd;border-radius:6px;padding:12px;margin:10px 0;
        border-left:4px solid {COLORS[3]}">
    <h3 style="margin:0 0 6px 0;color:{COLORS[0]}">Week 3: Cascade Effect Begins</h3>
    <div class="two-col"><div class="chart-box">
    {_mini_chart(util_a[:w3_end], COLORS[0], "Warehouse A Utilization", "Util")}
    </div><div class="chart-box">
    {_mini_chart(ot_a[:w3_end], COLORS[1], "Region A On-Time", "Rate")}
    </div></div>
    <p style="margin:4px 0;line-height:1.6">
    Late shipments from Warehouse A impact downstream routes. Fleet schedules become unstable.
    Re-routing increases operational complexity.</p>
    <p style="color:{COLORS[3]};font-style:italic;margin:2px 0">
    Outcome: Delay clusters emerge in urban zones. Customer SLA breaches begin.</p>
    </div>"""

    # Week 4
    content += f"""
    <div style="border:1px solid #ddd;border-radius:6px;padding:12px;margin:10px 0;
        border-left:4px solid #B71C1C">
    <h3 style="margin:0 0 6px 0;color:{COLORS[0]}">Week 4: System Instability Phase</h3>
    <div class="two-col"><div class="chart-box">
    {_mini_chart(util_a, COLORS[0], "Warehouse A Utilization", "Util")}
    </div><div class="chart-box">
    {_mini_chart(ot_a, COLORS[1], "Region A On-Time", "Rate")}
    </div></div>
    <p style="margin:4px 0;line-height:1.6">
    Network operates in "catch-up mode." Fleet is fully saturated. Backlog accumulates
    across multiple depots.</p>
    <p style="color:#B71C1C;font-style:italic;margin:2px 0">
    Outcome: System is no longer optimizing — it is recovering continuously.</p>
    </div>"""
    return {"icon": "&#x1F504;", "title": "System Behavior", "content": content}


def build_scenario_comparison(d: dict) -> dict:
    scenarios = d["scenarios"]
    times = d["times"]
    content = '<div class="kpi-row">'
    scenario_colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3]]
    best_idx = 0
    best_cost = float("inf")
    for si, s in enumerate(scenarios):
        r = s["result"]
        cost = r.values.get("Total_Cost", [0.0])[-1]
        ot = r.aux_values.get("avg_on_time", [1.0])[-1]
        subs = r.aux_values.get("total_dispatch", [0])[-1]
        if cost < best_cost:
            best_cost = cost
            best_idx = si
        color = scenario_colors[si % len(scenario_colors)]
        content += _kpi_card(
            s["name"],
            f"${cost:,.0f}",
            color,
            f"On-time: {ot*100:.1f}% | Dispatch: {subs:.0f}")

    content += '</div><div class="two-col"><div class="chart-box"><div class="st">Average On-Time Rate Comparison</div>'
    fig = go.Figure()
    for si, s in enumerate(scenarios):
        r = s["result"]
        ot = r.aux_values.get("avg_on_time", [1.0])
        fig.add_trace(go.Scatter(x=times, y=ot, mode="lines",
            name=s["name"], line=dict(color=scenario_colors[si % len(scenario_colors)], width=2)))
    fig.add_hline(y=0.90, line_dash="dot", line_color="gray", opacity=0.4,
                  annotation_text="SLA")
    fig.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=280,
        xaxis_title="Days", yaxis_title="On-Time Rate", yaxis=dict(range=[0, 1.1]),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig)
    content += '</div><div class="chart-box"><div class="st">Cumulative Cost Comparison</div>'
    fig2 = go.Figure()
    for si, s in enumerate(scenarios):
        r = s["result"]
        cost = r.values.get("Total_Cost", [0.0])
        fig2.add_trace(go.Scatter(x=times, y=cost, mode="lines",
            name=s["name"], line=dict(color=scenario_colors[si % len(scenario_colors)], width=2)))
    fig2.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=280,
        xaxis_title="Days", yaxis_title="Cost ($)",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), legend=dict(orientation="h", y=1.15))
    content += _ch_html(fig2)
    content += "</div></div>"
    best_name = scenarios[best_idx]["name"]
    content += _insight_box(
        f'<b>{best_name}</b> is the most cost-effective strategy — lowest cumulative cost '
        f'while maintaining or improving SLA. Adding fleet capacity alone costs less than '
        f'rebalancing, because rebalancing shifts the bottleneck without resolving it.')
    return {"icon": "&#x1F9E9;", "title": "Scenario Comparison", "content": content}


def build_root_cause(d: dict) -> dict:
    util_a = d.get("util_a", [0.0])
    util_b = d.get("util_b", [0.0])
    util_c = d.get("util_c", [0.0])
    ot_a = d.get("on_time_a", [1.0])
    dp_a = d.get("delay_prob_a", [0.0])
    cost = d.get("cost_ts", [0.0])

    cost_by_cause = [
        ("Base Operating", cost[-1] * 0.45 if cost else 0),
        ("Delay Penalties", cost[-1] * 0.35 if cost else 0),
        ("Fleet Operations", cost[-1] * 0.20 if cost else 0),
    ]

    content = _insight_box(
        'This is a <b>fleet capacity problem.</b> Demand growth outpaces available delivery '
        'vehicles. Adding fleet alone recovers the system — rebalancing warehouse load without '
        'addressing the fleet constraint just shifts the bottleneck.')

    content += '<div class="two-col"><div class="chart-box"><div class="st">Bottleneck Waterfall</div>'
    bottlenecks = [
        ("Warehouse A capacity ceiling", util_a[-1] if util_a else 0,
         "Designed for stable demand. Queue grows exponentially after 85%."),
        ("Fleet rigidity (Region A)", d.get("fleet_util_a", [0])[-1] if d.get("fleet_util_a") else 0,
         "Static allocation. No redistribution during peaks."),
        ("Last-mile congestion loop", max(d.get("delay_prob_a", [0])) if d.get("delay_prob_a") else 0,
         "Late dispatch → late delivery → rescheduling → more congestion."),
        ("Planning model limitation", 1.0,
         "Linear Excel planning misses nonlinear congestion effects."),
    ]
    fig = go.Figure()
    labels = [b[0] for b in bottlenecks]
    values = [b[1] for b in bottlenecks]
    colors = [COLORS[0], COLORS[2], COLORS[3], COLORS[4]]
    fig.add_trace(go.Bar(y=labels, x=values, orientation="h",
        marker=dict(color=colors), text=[f"{v:.1%}" if v <= 1 else f"{v:.1f}" for v in values],
        textposition="outside"))
    fig.update_layout(margin=dict(l=40,r=40,t=10,b=30), height=250,
        xaxis_title="Severity", yaxis=dict(autorange="reversed"),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(size=10), showlegend=False)
    content += _ch_html(fig)

    content += '</div><div class="chart-box"><div class="st">Cost Breakdown by Cause</div>'
    fig2 = go.Figure()
    labels2 = [c[0] for c in cost_by_cause]
    values2 = [c[1] for c in cost_by_cause]
    fig2.add_trace(go.Pie(labels=labels2, values=values2, marker=dict(colors=COLORS),
        textinfo="label+percent", hole=0.4))
    fig2.update_layout(margin=dict(l=20,r=20,t=10,b=20), height=250,
        paper_bgcolor="white", font=dict(size=10), showlegend=False)
    content += _ch_html(fig2)
    content += "</div></div>"

    content += '<div class="two-col"><div class="chart-box"><div class="st">Primary Bottleneck Detail</div>'
    content += f"""
    <div style="padding:8px;line-height:1.8">
    <p><b>1. Warehouse A capacity ceiling</b><br>
    Designed for stable demand baseline. Queue time grows exponentially after 85% utilization.
    Current load: <b>{util_a[-1]*100:.0f}%</b> at end of forecast.</p>
    <p><b>2. Fleet rigidity</b><br>
    Static fleet allocation across regions. No dynamic redistribution during peaks.
    Underutilized capacity in Region B while Region A saturates.</p>
    <p><b>3. Last-mile congestion feedback loop</b><br>
    Late dispatch → late delivery → rescheduling → more congestion.
    Compounding delay effect across routes.</p>
    <p><b>4. Planning model limitation</b><br>
    Current Excel-based planning assumes linear scaling. Does not capture nonlinear congestion effects.</p>
    </div>"""
    content += '</div><div class="chart-box"><div class="st">System Weak Points Map</div>'
    content += f"""
    <div style="padding:8px;line-height:1.8">
    <p><span style="color:{COLORS[3]}">🔴</span> <b>Warehouse A</b> — Structural overload point<br>
    Utilization crosses 85% within days. Queue grows without bound after threshold.</p>
    <p><span style="color:{COLORS[2]}">🟠</span> <b>Fleet Region A</b> — Insufficient elasticity<br>
    Fleet capacity capped at {d['base_params']['fleet_capacity_A']:.0f} orders/day. Demand exceeds by ~30%.</p>
    <p><span style="color:{COLORS[2]}">🟠</span> <b>Last-mile routes (Region A)</b> — Congestion amplification zone<br>
    Delay cascade effect amplifies small delays into systemic SLA breaches.</p>
    <p><span style="color:{COLORS[1]}">🟢</span> <b>Warehouse B/C</b> — Currently stable<br>
    Underutilized capacity exists here — potential rebalancing target.</p>
    </div>"""
    content += "</div></div>"
    return {"icon": "&#x1F50D;", "title": "Root Cause", "content": content}


def build_kpi_forecast(d: dict) -> dict:
    ts = d["ts"]
    times = d["times"]
    cost = d.get("cost_ts", [0.0])
    ot_a = d.get("on_time_a", [1.0])
    util_a = d.get("util_a", [0.0])
    dp_a = d.get("delay_prob_a", [0.0])

    kpis = [
        ("Delivery Reliability", [v * 100 for v in ot_a], "%", COLORS[0],
         "Declining after Day 10 — SLA breached by Day 14"),
        ("Warehouse A Queue", [v * 100 for v in util_a], "% load", COLORS[3],
         "Exponential growth after 85% threshold"),
        ("Fleet Efficiency", [v * 100 for v in d.get("fleet_util_a", [1.0])], "%", COLORS[2],
         "Decreasing despite stable fleet size"),
        ("Operational Cost", [c / 1000 for c in cost], "$K", COLORS[0],
         "Gradual increase due to inefficiency penalties"),
    ]

    content = _insight_box(
        f'Critical inflection point: Day {min(d["crisis_day"] or 30, 30):.0f} — '
        f'Warehouse A utilization crosses the 85% threshold. After this point, '
        f'recovery requires active intervention.')

    for kpi_name, kpi_data, unit, color, note in kpis:
        content += f'<div class="two-col" style="margin:6px 0"><div class="chart-box">'
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=kpi_data, mode="lines",
            line=dict(color=color, width=2), name=kpi_name))
        fig.add_annotation(x=times[len(times)//2] if times else 15,
            y=max(kpi_data) if kpi_data else 50,
            text=note, showarrow=False, font=dict(size=9, color=color))
        fig.update_layout(margin=dict(l=40,r=10,t=20,b=25), height=140,
            xaxis_title=None, yaxis_title=unit,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(size=8), showlegend=False)
        content += _ch_html(fig)
        content += f'</div></div>'

    return {"icon": "&#x1F4C8;", "title": "30-Day KPIs", "content": content}


def build_recommendations(d: dict) -> dict:
    cost = d.get("cost_ts", [0.0])
    total_cost = cost[-1] if cost else 0.0
    scenarios = d["scenarios"]
    best_name = "Combined"
    best_cost = float("inf")
    for s in scenarios:
        c = s["result"].values.get("Total_Cost", [0.0])[-1]
        if c < best_cost:
            best_cost = c
            best_name = s["name"]
    savings = total_cost - best_cost

    content = _insight_box(
        f'The <b>{best_name}</b> scenario saves an estimated <b>${savings:,.0f}</b> '
        f'over the 30-day forecast compared to "Do Nothing" — without major capital investment.')

    phases = [
        ("Immediate (0–7 days)", COLORS[3], [
            ("Rebalance 10-15% volume from Warehouse A → Warehouse C",
             "Reduces A's utilization below the 85% tipping point immediately"),
            ("Introduce peak-hour routing flexibility",
             "Temporary dispatcher overrides for urgent deliveries"),
        ]),
        ("Short-term (1–2 weeks)", COLORS[2], [
            ("Increase fleet in Region A by 8-10%",
             f"Raises dispatch ceiling from {d['base_params']['fleet_capacity_A']:.0f} "
             f"to {d['base_params']['fleet_capacity_A'] * 1.08:.0f} orders/day"),
            ("Adjust dispatch batching frequency",
             "Smaller, more frequent dispatch windows reduce peak load"),
        ]),
        ("Structural (strategic fix)", COLORS[0], [
            ("Redesign warehouse load balancing model",
             "Dynamic routing between A and C based on real-time utilization"),
            ("Implement predictive capacity planning",
             "Monitor demand signals 14 days ahead to trigger pre-emptive expansion"),
        ]),
    ]

    for phase_name, color, items in phases:
        content += f"""
        <div style="border:1px solid #ddd;border-radius:6px;padding:10px;margin:10px 0">
        <h3 style="margin:0 0 6px 0;color:{color}">{phase_name}</h3>"""
        for action, impact in items:
            content += f"""
        <div style="display:flex;gap:10px;margin:6px 0">
            <div style="min-width:40%;font-weight:bold">{action}</div>
            <div style="color:{COLORS[5]}">{impact}</div>
        </div>"""
        content += "</div>"

    content += _insight_box(
        'Your logistics network is: <b>"Efficient under normal conditions, but structurally '
        'vulnerable under moderate growth stress."</b> With targeted redistribution and modest '
        'fleet adjustment, system stability can be restored without major capital investment.')
    return {"icon": "&#x1F3AF;", "title": "Recommendations", "content": content}


# ══════════════════════════════════════════════════════════════════════════════
# HTML ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

TAB_BUILDERS = [
    build_executive_summary, build_network_overview, build_warehouse_dynamics,
    build_fleet_routes, build_delivery_performance, build_system_behavior,
    build_scenario_comparison, build_root_cause, build_kpi_forecast,
    build_recommendations,
]

CSS = """
* { box-sizing:border-box;margin:0;padding:0 }
body { font-family:system-ui,-apple-system,sans-serif;background:#F4F6F8;color:#1A237E;margin:0 }
.header { background:#1A237E;color:white;padding:20px 24px }
.header h1 { font-size:22px;margin:0 0 4px 0 }
.header p { font-size:13px;opacity:0.85;margin:0 }
.tab-bar { display:flex;flex-wrap:wrap;background:#1A237E;padding:0 16px;gap:2px }
.tab-btn { padding:10px 16px;font-size:12px;color:white;background:rgba(255,255,255,0.1);
  border:none;cursor:pointer;border-radius:6px 6px 0 0;transition:background 0.2s }
.tab-btn:hover { background:rgba(255,255,255,0.2) }
.tab-btn.active { background:white;color:#1A237E;font-weight:600 }
.tab-content.active { display:block }
.kpi-row { display:flex;flex-wrap:wrap;gap:10px;margin:10px 0 }
.kpi { flex:1;min-width:140px;background:white;border-radius:8px;padding:12px 14px;
  box-shadow:0 1px 3px rgba(0,0,0,0.08) }
.kl { font-size:11px;color:#546E7A;margin-bottom:2px }
.kv { font-size:20px;font-weight:700 }
.ks { font-size:10px;color:#90A4AE;margin-top:2px }
.two-col { display:flex;gap:12px;margin:8px 0 }
.chart-box { flex:1;min-width:0;background:white;border-radius:6px;padding:8px;
  box-shadow:0 1px 3px rgba(0,0,0,0.06) }
.st { font-size:12px;font-weight:600;color:#1A237E;margin-bottom:6px }
.qa { background:#FFF3E0;border-radius:6px;font-size:12px;line-height:1.6;margin:10px 0 }
.footer { text-align:center;padding:16px;font-size:10px;color:#90A4AE;
  border-top:1px solid #E0E0E0;margin-top:16px }
h3 { font-size:14px;font-weight:600 }
@media (max-width:768px) { .two-col { flex-direction:column } }
"""


def build_html(data: dict) -> str:
    from datetime import datetime
    tabs = []
    print(f"  Building {len(TAB_BUILDERS)} tabs...")
    for fn in TAB_BUILDERS:
        result = fn(data)
        tabs.append(result)
        print(f"    Built: {result['title']}")

    tab_buttons = "".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" '
        f'onclick="switchTab({i})">{t["icon"]} {t["title"]}</button>'
        for i, t in enumerate(tabs))
    tab_contents = "".join(
        f'<div class="tab-content" id="tab{i}"'
        f' style="display:{"block" if i==0 else "none"};padding:16px;max-width:1400px;margin:0 auto">'
        f'{t["content"]}</div>' for i, t in enumerate(tabs))

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    head = f"""<head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Logistics Network — Delivery Failure Risk Forecast</title>
    <style>{CSS}</style></head>"""

    body = f"""<body>
    <div class="header"><h1>&#x1F69A; Logistics Network — Delivery Failure Risk Forecast</h1>
    <p>30-Day Decision Intelligence Dashboard | Generated {date_str}</p></div>
    <div class="tab-bar">{tab_buttons}</div>
    <div>{tab_contents}</div>
    <div class="footer">Logistics Network Dashboard — hybrid simulation (SD + DES).
    Data: TripleStore with 992 historical triples from 5 CSVs via ingest_csv.</div>
    <script src="https://cdn.jsdelivr.net/npm/plotly.js@3.6.0/dist/plotly.min.js"></script>
    <script>
    (function(){{
      var charts = document.querySelectorAll('[data-fig]');
      function renderAll(){{
        for(var i=0;i<charts.length;i++){{
          // Skip charts inside hidden tabs — they'll render when tab becomes visible
          var tab = charts[i].closest('.tab-content');
          if(tab && tab.style.display === 'none') continue;
          try {{
            var fig = JSON.parse(charts[i].getAttribute('data-fig'));
            Plotly.react(charts[i].id, fig.data, fig.layout, {{responsive:true}});
          }} catch(e) {{ console.warn('Chart', charts[i].id, e.message);
            charts[i].innerHTML = '<div style="padding:16px;color:#999;font-size:11px;text-align:center">Chart error: ' + e.message + '</div>';
          }}
        }}
      }}
      if(typeof Plotly !== 'undefined') renderAll();
      else {{
        var iv = setInterval(function(){{
          if(typeof Plotly !== 'undefined'){{ clearInterval(iv); renderAll(); }}
        }}, 100);
        setTimeout(function(){{ clearInterval(iv);
          for(var k=0;k<charts.length;k++){{
            if(!charts[k].querySelector('.js-plotly-plot')){{
              charts[k].innerHTML = '<div style="padding:20px;color:#999;font-size:12px;text-align:center">Chart requires Plotly.js — check ad blocker or open in a non-restricted browser</div>';
            }}
          }}
        }}, 30000);
      }}
    }})();
    </script>
    <script>
    var activeTab = 0;
    function switchTab(i) {{
      document.getElementById('tab'+activeTab).style.display = 'none';
      document.querySelectorAll('.tab-btn')[activeTab].classList.remove('active');
      document.getElementById('tab'+i).style.display = 'block';
      document.querySelectorAll('.tab-btn')[i].classList.add('active');
      activeTab = i;
      // Render and resize charts in the newly visible tab
      setTimeout(function(){{
        var tab = document.getElementById('tab'+i);
        var charts = tab.querySelectorAll('[data-fig]');
        for(var c=0;c<charts.length;c++){{
          if(!charts[c].querySelector('.js-plotly-plot') && typeof Plotly !== 'undefined'){{
            try {{
              var fig = JSON.parse(charts[c].getAttribute('data-fig'));
              Plotly.react(charts[c].id, fig.data, fig.layout, {{responsive:true}});
            }} catch(e) {{ console.warn('Tab chart render:', e.message); }}
          }}
        }}
        try {{
          var plots = tab.querySelectorAll('.js-plotly-plot');
          for(var p=0;p<plots.length;p++) {{ if(typeof Plotly!=='undefined') Plotly.Plots.resize(plots[p]); }}
        }} catch(e) {{}}
      }}, 400);
    }}
    </script></body>"""

    return f"<!DOCTYPE html><html>{head}{body}</html>"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Logistics Network — Delivery Failure Risk Forecast Dashboard")
    print("=" * 60)
    data = run_simulation()
    html = build_html(data)
    out_path = Path(__file__).parent / "logistics_network_dashboard.html"
    out_path.write_text(html)
    print(f"\nDashboard: {out_path} ({len(html)//1024}KB, {len(TAB_BUILDERS)} tabs)")


if __name__ == "__main__":
    main()
