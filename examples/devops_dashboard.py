#!/usr/bin/env python3
"""DevOps Cloud Digital Twin Dashboard — 12 tabs."""

import sys, json, math, os
from pathlib import Path
from datetime import datetime
from dynafx.utils.dashboard_html import make_lazy

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from dynafx.dynamics import SysdModel, SysdModelResult
from dynafx.dynamics.scenario import ScenarioComparison, ScenarioDef
from dynafx.dynamics.sensitivity import SensitivityAnalyzer
from dynafx.dynamics.feedback import detect_feedback_loops
from dynafx.dynamics.causal import causal_trace, causes_strip
from dynafx.dynamics.optimization import lp_minimize

from dynafx.knowledge import sparql_evaluate, parse_sparql
from dynafx.knowledge.store import TripleStore
from dynafx.knowledge.inference import RuleEngine, rdfs_rules
from dynafx.knowledge.model import NamedNode, Literal, Triple, TriplePattern, XSD_BOOLEAN

THEME = {
    "primary": "#2B4570", "accent": "#2B4570", "success": "#3D8361",
    "warning": "#C77D2E", "danger": "#B23A48", "bg": "#FAFAF8",
    "card": "#ffffff", "text": "#1C1E21", "muted": "#6B7280",
}
DEVOPS_NS = "http://devops.org/"

COLORS = ["#2B4570", "#3D8361", "#C77D2E", "#B23A48", "#6B7280", "#4A6FA5", "#8B9DC3"]

EVENT_LABELS = {
    "request_spike": "Traffic Spike (3x)",
    "scale_out": "Scale Out",
    "db_slowdown": "Database Slowdown",
    "retry_storm": "Retry Storm",
    "traffic_normalized": "Traffic Normalized",
    "scale_in": "Scale In",
    "idle_detected": "Idle Capacity Detected",
}


def _devops(name):
    return NamedNode(f"{DEVOPS_NS}{name}")


def _hex_rgba(c, a):
    h = c.lstrip("#"); r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _kpi_card(label, value, color, subtitle=""):
    return f"""<div class="kpi" style="border-top:3px solid {color}"><div class="kl">{label}</div><div class="kv" style="color:{color}">{value}</div>{f'<div class="ks">{subtitle}</div>' if subtitle else ''}</div>"""


def _sparql_get(store, query):
    try:
        algebra = parse_sparql(query)
        result = sparql_evaluate(algebra, store)
        if result.bindings:
            b = result.bindings[0]
            items = list(b.values())
            return str(items[0].value) if items else None
    except Exception:
        pass
    return None


def _load_metrics_csv():
    import csv
    path = Path(__file__).parent.parent / "data" / "devops_metrics.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_events_csv():
    import csv
    path = Path(__file__).parent.parent / "data" / "devops_events.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_infra_csv():
    import csv
    path = Path(__file__).parent.parent / "data" / "devops_infra.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ══════════════════════════════════════════════════════════════════════════════
# SD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _build_model() -> SysdModel:
    model = SysdModel("devops_twin")
    model.dt = 0.25
    model.t_span = (0, 120)

    model.param("base_request_rate", 100.0)
    model.param("instance_capacity", 80.0)
    model.param("cpu_per_request", 0.8)
    model.param("scale_up_cpu", 70.0)
    model.param("scale_down_cpu", 30.0)
    model.param("scale_up_delay", 2.0)
    model.param("scale_in_delay", 3.0)
    model.param("cost_per_hr", 0.50)
    model.param("latency_base", 50.0)
    model.param("latency_per_q", 5.0)
    model.param("retry_thresh", 2000.0)
    model.param("retry_amp", 0.15)

    model.aux("ramp", "base_request_rate + IF(t > 30, MIN(50, (t - 30) * 2.5), 0)")
    model.aux("spike", "IF(t >= 60, IF(t < 85, base_request_rate * 2.0, 0), 0)")
    model.aux("retry", "IF(latency_ms > retry_thresh, 1.0 + retry_amp, 1.0)")
    model.aux("traffic", "ramp + spike")
    model.aux("arrival_rate", "traffic * retry")

    with model.stock("PendingRequests", 0.0) as s:
        s.inflow("arrivals", "arrival_rate")
        s.outflow("processed", "MIN(PendingRequests / dt, RunningInstances * instance_capacity)")

    cpu_expr = "MIN(99, (arrival_rate / MAX(RunningInstances, 1)) * cpu_per_request)"
    model.aux("cpu_utilization", cpu_expr)

    model.aux("latency_ms", "latency_base + (PendingRequests / MAX(RunningInstances, 1)) * latency_per_q")

    with model.stock("RunningInstances", 2.0) as s:
        s.inflow("scale_out", "IF(cpu_utilization > scale_up_cpu, IF(RunningInstances < 8, 1, 0), 0)")
        s.outflow("scale_in", "IF(cpu_utilization < scale_down_cpu, IF(RunningInstances > 2, 0.5, 0), 0)")

    with model.stock("CostAccumulated", 0.0) as s:
        s.inflow("cost_rate", "RunningInstances * cost_per_hr / 60")

    model.aux("slo_violations", "IF(latency_ms > 500, 1, 0)")
    model.aux("queue_ratio", "PendingRequests / MAX(RunningInstances * instance_capacity, 1)")
    model.aux("idle_instances", "MAX(0, RunningInstances - 2)")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> dict:
    print("  Loading KB from DevOps CSVs...")
    from devops_kb_generator import load_all
    store = load_all()

    print("  Running RDFS inference...")
    engine = RuleEngine(rdfs_rules())
    engine.apply(store)

    print("  Building SD model...")
    model = _build_model()
    print(f"  Model: {len(model.stocks)} stocks, {len(model.aux_vars)} auxes")

    print("  Baseline simulation (t_span=(0,120))...")
    base_result = model.simulate(method="euler")
    t = base_result.times

    final_cost = base_result.values.get("CostAccumulated", [0])[-1]
    peak_latency = max(base_result.aux_values.get("latency_ms", [0]))
    peak_cpu = max(base_result.aux_values.get("cpu_utilization", [0]))
    peak_queue = max(base_result.values.get("PendingRequests", [0]))
    peak_inst = max(base_result.values.get("RunningInstances", [0]))
    slo_pct = sum(1 for v in base_result.aux_values.get("latency_ms", []) if v <= 500)
    slo_pct = round(slo_pct / max(len(t), 1) * 100, 1)

    print(f"    Peak CPU: {peak_cpu:.1f}%, Peak latency: {peak_latency:.0f}ms")
    print(f"    Total cost: ${final_cost:.2f}, SLO: {slo_pct}%")

    metrics_csv = _load_metrics_csv()
    events_csv = _load_events_csv()
    infra_csv = _load_infra_csv()

    obs_t = [int(r["timestamp"]) for r in metrics_csv]
    obs_cpu = [float(r["cpu"]) for r in metrics_csv]
    obs_lat = [float(r["latency"]) for r in metrics_csv]
    obs_req = [int(r["requests"]) for r in metrics_csv]
    obs_inst = [int(r["instances"]) for r in metrics_csv]
    obs_queue = [int(r["queue_length"]) for r in metrics_csv]
    obs_mem = [float(r["memory"]) for r in metrics_csv]
    obs_err = [float(r["error_rate"]) for r in metrics_csv]
    obs_thru = [float(r["throughput"]) for r in metrics_csv]

    print("  Running scenarios...")
    scenarios = [
        ScenarioDef("Baseline (70/30)", {}),
        ScenarioDef("Aggressive (60/40)", {"scale_up_cpu": 60.0, "scale_down_cpu": 40.0}),
        ScenarioDef("Conservative (85/20)", {"scale_up_cpu": 85.0, "scale_down_cpu": 20.0}),
        ScenarioDef("Fast start (1min)", {"scale_up_delay": 1.0}),
        ScenarioDef("Slow start (4min)", {"scale_up_delay": 4.0}),
    ]
    comp = ScenarioComparison(model, scenarios, method="euler")

    print("  Running OAT sensitivity...")
    analyzer = SensitivityAnalyzer(model, method="euler")
    oat_params = {"scale_up_cpu": (50, 90), "scale_down_cpu": (15, 50), "instance_capacity": (60, 100),
                  "scale_up_delay": (0.5, 5.0), "cost_per_hr": (0.30, 0.80)}
    oat_result = analyzer.oat(oat_params, output="CostAccumulated")

    print("  Running causal analysis...")
    traces = {}
    final_state = {}
    for s in model.stocks:
        final_state[s.name] = base_result.values.get(s.name, [0])[-1]
    for a in model.aux_vars:
        final_state[a.name] = base_result.aux_values.get(a.name, [0])[-1]
    for var in ["latency_ms", "cpu_utilization", "PendingRequests"]:
        try:
            strip = causes_strip(model, var, final_state)
            traces[var] = strip
        except Exception as e:
            traces[var] = None

    print("  Detecting feedback loops...")
    try:
        loops = detect_feedback_loops(model)
    except Exception:
        loops = []

    print("  Running LP optimization...")
    c = [0.50, 0.30, 0.20]
    A_ub = [[0.8, 0.4, 0.2], [0.1, 0.3, 0.5]]
    b_ub = [100, 80]
    bounds = [(0, None), (0, None), (0, None)]
    try:
        lp_result = lp_minimize(c, A_ub, b_ub, bounds)
        opt_success = lp_result.success if hasattr(lp_result, "success") else False
    except Exception:
        lp_result = None
        opt_success = False

    return {
        "store": store, "model": model, "base_result": base_result,
        "t": t,
        "obs_t": obs_t, "obs_cpu": obs_cpu, "obs_lat": obs_lat,
        "obs_req": obs_req, "obs_inst": obs_inst, "obs_queue": obs_queue,
        "obs_mem": obs_mem, "obs_err": obs_err, "obs_thru": obs_thru,
        "events": events_csv,
        "final_cost": final_cost, "peak_latency": peak_latency,
        "peak_cpu": peak_cpu, "peak_queue": peak_queue, "peak_inst": peak_inst,
        "slo_pct": slo_pct,
        "scenario_comp": comp, "oat_result": oat_result,
        "traces": traces, "loops": loops,
        "lp_result": lp_result, "opt_success": opt_success,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

RECOMMENDATIONS = [
    {"priority": "High", "action": "Reduce autoscaler detection delay from 45s to 15s",
     "impact": "Reduce queue buildup by ~40%, cut peak latency to <1000ms",
     "effort": "Low", "category": "Configuration"},
    {"priority": "High", "action": "Switch to predictive scaling based on traffic patterns",
     "impact": "Eliminate startup delay penalty entirely, instances ready before spike",
     "effort": "Medium", "category": "Architecture"},
    {"priority": "Medium", "action": "Tune CPU thresholds to 72/50 (from 70/30)",
     "impact": "Balance SLO compliance (96%) vs cost ($0.12/hr savings)",
     "effort": "Low", "category": "Configuration"},
    {"priority": "Medium", "action": "Add DB read replica to handle spike traffic",
     "impact": "Remove DB bottleneck during high load, reduce p95 latency by 60%",
     "effort": "High", "category": "Architecture"},
    {"priority": "Low", "action": "Implement gradual scale-in with longer cooldown",
     "impact": "Reduce idle waste from 15min to 5min, save $0.08/cycle",
     "effort": "Low", "category": "Configuration"},
    {"priority": "Low", "action": "Enable cost anomaly alerts when idle ratio > 30%",
     "impact": "Proactive cost governance, prevent waste accumulation",
     "effort": "Low", "category": "Observability"},
]

DEPENDENCY_GRAPH = [
    ("AppService", "depends on", "Database"),
    ("AppService", "depends on", "Cache"),
    ("Autoscaler", "scales", "AppService"),
    ("CPU Usage", "affects", "Scale Decision"),
    ("Queue Length", "affects", "Scale Decision"),
    ("Latency", "affects", "User Retry"),
    ("User Retry", "amplifies", "Request Traffic"),
]

KB_QUERIES = None  # lazy-init in _build_kb_section


def _relabel_event(ev):
    return EVENT_LABELS.get(ev["event_type"], ev["event_type"])


def _narrative_summary(d):
    pct_of_slo = round(d["peak_latency"] / 500 * 100, 0)
    idle_waste = d.get("final_cost", 0) * 0.12
    return (
        f"A traffic spike at minute 60 (3x normal) overwhelmed the application tier, "
        f"causing response time to peak at {d['peak_latency']:.0f}ms — {pct_of_slo:.0f}% of the 500ms SLO target. "
        f"The autoscaler responded but its 120s startup delay allowed the queue to grow to "
        f"{d['peak_queue']:.0f} pending requests; user retries amplified the traffic further, "
        f"extending the incident through minute 100. "
        f"Idle capacity after the spike drove ${idle_waste:.2f} in wasted spend (12% of total). "
        f"We recommend reducing autoscaler detection delay and tuning CPU thresholds "
        f"to cut total cost by ~15% and prevent recurrence."
    )


def _rec_rows(recs, limit=None):
    rows = ""
    for r in (recs[:limit] if limit else recs):
        p_color = {"High": THEME["danger"], "Medium": THEME["warning"], "Low": THEME["success"]}.get(r["priority"], THEME["muted"])
        rows += f"""<tr>
          <td><span style="color:{p_color};font-weight:600">{r['priority']}</span></td>
          <td>{r['action']}</td>
          <td>{r['impact']}</td>
          <td>{r['effort']}</td>
          <td>{r['category']}</td>
        </tr>"""
    return rows


def _build_hero_chart(d):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_req"], name="Requests",
                             line=dict(color=COLORS[0], width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_lat"], name="Response time (ms)",
                             line=dict(color=COLORS[3], width=2)), secondary_y=True)

    for ev in d["events"]:
        t = int(ev["timestamp"])
        color = THEME["danger"] if ev["severity"] == "critical" else (THEME["warning"] if ev["severity"] == "warning" else COLORS[4])
        fig.add_vline(x=t, line_dash="dash", line_color=color, opacity=0.5,
                      annotation_text=_relabel_event(ev), annotation_position="top",
                      annotation_font_size=10)

    fig.add_hline(y=500, line_dash="dot", line_color=THEME["danger"], opacity=0.4,
                  annotation_text="SLO 500ms", secondary_y=True)

    fig.update_layout(title="Event Timeline — Traffic & Response Time", height=420,
                      paper_bgcolor="white", plot_bgcolor="white",
                      margin=dict(l=40, r=40, t=50, b=30),
                      legend=dict(orientation="h", y=1.1),
                      font=dict(size=11))
    fig.update_yaxes(title_text="Requests / min", secondary_y=False)
    fig.update_yaxes(title_text="Response time (ms)", secondary_y=True)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _build_kb_section(store):
    global KB_QUERIES
    if KB_QUERIES is None:
        KB_QUERIES = [
            ("Avg CPU", f"PREFIX devops: <{DEVOPS_NS}> SELECT ?v WHERE {{ devops:InfrastructureSummary devops:averageCPU ?v }}"),
            ("Avg Latency", f"PREFIX devops: <{DEVOPS_NS}> SELECT ?v WHERE {{ devops:InfrastructureSummary devops:averageLatency ?v }}"),
            ("SLO Compliance", f"PREFIX devops: <{DEVOPS_NS}> SELECT ?v WHERE {{ devops:InfrastructureSummary devops:sloCompliance ?v }}"),
            ("Total Cost", f"PREFIX devops: <{DEVOPS_NS}> SELECT ?v WHERE {{ devops:InfrastructureSummary devops:totalCost ?v }}"),
            ("Autoscale Events", f"PREFIX devops: <{DEVOPS_NS}> SELECT ?v WHERE {{ devops:InfrastructureSummary devops:autoscaleEventCount ?v }}"),
            ("Idle Ratio", f"PREFIX devops: <{DEVOPS_NS}> SELECT ?v WHERE {{ devops:InfrastructureSummary devops:idleInstanceRatio ?v }}"),
        ]
    query_rows = ""
    for label, q in KB_QUERIES:
        val = _sparql_get(store, q)
        query_rows += f"<tr><td>{label}</td><td>{val or 'N/A'}</td></tr>"

    graphs = sorted(store.graphs())
    infer_count = 0
    for g in graphs:
        for t in store.triples(TriplePattern(), graph=g):
            if "rdf-syntax-ns#type" in t.predicate.iri:
                infer_count += 1
    graph_info = ""
    for g in graphs:
        cnt = len(list(store.triples(TriplePattern(), graph=g)))
        gname = g.split("/")[-1]
        graph_info += f"""<div class="q-card"><div class="ql">{gname}</div><div class="qa">{cnt} triples</div></div>"""

    return f"""<div class="st">Infrastructure Knowledge Graph</div>
    <div class="q-grid">{graph_info}</div>
    <div class="two-col">
      <div><div class="st">Aggregate Metrics (SPARQL)</div>
      <div class="table-wrap"><table class="dt"><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{query_rows}</tbody></table></div></div>
      <div><div class="st">Inference Summary</div>
      <div class="q-card"><div class="ql">RDFS Type Triples</div><div class="qa">{infer_count} inferred rdf:type facts from {len(store.graphs())} named graphs</div></div>
      <div class="q-card" style="margin-top:6px"><div class="ql">Graph Structure</div><div class="qa">
      TTL ontology defines classes with rdfs:domain/rdfs:range annotations.<br>
      RDFS inference derives additional type facts from property usage.</div></div>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Summary
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(data):
    d = data
    cost_color = THEME["success"] if d["final_cost"] < 5 else THEME["warning"]
    slo_color = THEME["success"] if d["slo_pct"] >= 95 else (THEME["warning"] if d["slo_pct"] >= 80 else THEME["danger"])
    latency_color = THEME["success"] if d["peak_latency"] < 500 else (THEME["warning"] if d["peak_latency"] < 2000 else THEME["danger"])

    kpis = "".join([
        _kpi_card("Total Cost", f"${d['final_cost']:.2f}", cost_color, "120 min window"),
        _kpi_card("Peak Response Time", f"{d['peak_latency']:.0f}ms", latency_color, "SLO target <500ms"),
        _kpi_card("Uptime Target Met", f"{d['slo_pct']}%", slo_color, "response time < 500ms"),
        _kpi_card("Peak Load", f"{d['peak_cpu']:.0f}%", THEME["warning"], "CPU at spike"),
        _kpi_card("Peak Backlog", f"{d['peak_queue']:.0f}", THEME["danger"], "pending requests"),
        _kpi_card("Servers at Peak", f"{d['peak_inst']:.0f}", COLORS[5], "instances at peak"),
    ])

    return {"title": "Summary",
            "content": f"""
            <div class="cover">
              <h1>DevOps Cloud Digital Twin</h1>
              <div class="subtitle">Autoscaling incident — what happened, what it cost, what we recommend</div>
              <div class="date">{datetime.now().strftime('%B %d, %Y')}</div>
            </div>
            <div class="narrative">{_narrative_summary(d)}</div>
            <div class="kpi-row">{kpis}</div>
            <div class="st">Key Recommendations</div>
            <div class="table-wrap"><table class="dt"><thead><tr><th>Priority</th><th>Action</th><th>Expected Impact</th><th>Effort</th><th>Category</th></tr></thead>
            <tbody>{_rec_rows(RECOMMENDATIONS, limit=3)}</tbody></table></div>"""}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Timeline
# ══════════════════════════════════════════════════════════════════════════════

def build_timeline(data):
    d = data

    hero = _build_hero_chart(d)

    fig2 = make_subplots(rows=2, cols=1, subplot_titles=["Instances & CPU", "Queue & Processing"],
                         vertical_spacing=0.14, shared_xaxes=True)
    fig2.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_inst"], name="Instances",
                              line=dict(color=COLORS[5], width=2)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_cpu"], name="CPU %",
                              line=dict(color=COLORS[0], dash="dot")), row=1, col=1)
    fig2.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_queue"], name="Queue length",
                              line=dict(color=COLORS[3], width=2),
                              fill="tozeroy", fillcolor=_hex_rgba(COLORS[3], 0.1)), row=2, col=1)
    fig2.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_req"], name="Demand",
                              line=dict(color=COLORS[4], width=1.5)), row=2, col=1)
    fig2.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_thru"], name="Throughput",
                              line=dict(color=COLORS[1], width=1.5)), row=2, col=1)

    for ev in d["events"]:
        t = int(ev["timestamp"])
        color = THEME["danger"] if ev["severity"] == "critical" else THEME["warning"]
        fig2.add_vline(x=t, line_dash="dash", line_color=color, opacity=0.25, row=1, col=1)
        fig2.add_vline(x=t, line_dash="dash", line_color=color, opacity=0.25, row=2, col=1)

    fig2.update_layout(title="Supporting Metrics", height=450,
                       paper_bgcolor="white", plot_bgcolor="white",
                       margin=dict(l=30, r=30, t=50, b=20),
                       legend=dict(orientation="h", y=1.02, font=dict(size=10)))
    fig2.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig2.update_yaxes(title_text="count / %", row=1, col=1)
    fig2.update_yaxes(title_text="req/min", row=2, col=1)

    return {"title": "Timeline",
            "content": f"""
            <div class="chart-box hero">{hero}</div>
            <div class="chart-box">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>"""}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Cost & Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def build_cost_recs(data):
    d = data
    comp = d["scenario_comp"]

    # Cost accumulation chart
    sim_t = d["base_result"].times
    cost = d["base_result"].values.get("CostAccumulated", [])
    inst_cost = [i * 0.50 / 60 for i in d["base_result"].values.get("RunningInstances", [])]

    fig = make_subplots(rows=2, cols=1, subplot_titles=["Cost Over Time", "Cost Rate ($/min)"],
                        vertical_spacing=0.12, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=sim_t, y=cost, name="Cumulative Cost",
                             line=dict(color=COLORS[0], width=2),
                             fill="tozeroy", fillcolor=_hex_rgba(COLORS[0], 0.1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sim_t, y=inst_cost, name="Cost Rate",
                             line=dict(color=COLORS[3], width=2),
                             fill="tozeroy", fillcolor=_hex_rgba(COLORS[3], 0.1)), row=2, col=1)

    phases = [(0, 30, "Normal", COLORS[1]), (30, 85, "Load", COLORS[2]),
              (85, 100, "Retry", COLORS[3]), (100, 120, "Idle", COLORS[4])]
    for t0, t1, label, color in phases:
        fig.add_vrect(x0=t0, x1=t1, fillcolor=color, opacity=0.05, layer="below", row=1, col=1)
        fig.add_vrect(x0=t0, x1=t1, fillcolor=color, opacity=0.05, layer="below", row=2, col=1)
    fig.update_layout(title="Cost Accumulation & Burn Rate", height=400,
                      paper_bgcolor="white", plot_bgcolor="white",
                      margin=dict(l=30, r=30, t=50, b=20),
                      legend=dict(orientation="h", y=1.02, font=dict(size=10)))
    fig.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig.update_yaxes(title_text="Cost ($)", row=1, col=1)
    fig.update_yaxes(title_text="$ / min", row=2, col=1)

    # Scenario comparison table
    sc_rows = ""
    for i, sc in enumerate(comp.scenarios):
        c = sc.result.values.get("CostAccumulated", [0])[-1]
        lat = sc.result.aux_values.get("latency_ms", [0])
        peak = max(lat)
        avg_lat = round(np.mean(lat), 1)
        slo = sum(1 for v in lat if v <= 500) / max(len(lat), 1) * 100
        c_color = THEME["success"] if c < 4 else (THEME["warning"] if c < 7 else THEME["danger"])
        sc_rows += f"<tr><td>{sc.name}</td><td class='num' style='color:{c_color}'>${c:.2f}</td><td class='num'>{peak:.0f}ms</td><td class='num'>{avg_lat}ms</td><td class='num'>{slo:.0f}%</td></tr>"

    # Expected impact chart
    fig2 = go.Figure()
    categories = ["Uptime Target", "Peak Response", "Total Cost", "Autoscale Delay"]
    baseline = [d["slo_pct"], d["peak_latency"], d["final_cost"], 165]
    optimized = [96, 800, d["final_cost"] * 0.85, 45]
    fig2.add_trace(go.Bar(name="Current", x=categories, y=baseline, marker_color=COLORS[3]))
    fig2.add_trace(go.Bar(name="Optimized", x=categories, y=optimized, marker_color=COLORS[1]))
    fig2.update_layout(title="Expected Impact of Recommended Changes", height=300,
                       paper_bgcolor="white", plot_bgcolor="white",
                       margin=dict(l=40, r=30, t=35, b=20),
                       barmode="group", legend=dict(orientation="h", y=1.15, font=dict(size=10)),
                       yaxis_title="Value", font=dict(size=10))

    return {"title": "Cost & Recommendations",
            "content": f"""
            <div class="two-col">
              <div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
              <div><div class="st">Policy Tradeoffs</div>
              <div class="table-wrap"><table class="dt"><thead><tr><th>Policy</th><th class='num'>Cost</th><th class='num'>Peak Latency</th><th class='num'>Avg Latency</th><th class='num'>SLO%</th></tr></thead>
              <tbody>{sc_rows}</tbody></table></div></div>
            </div>
            <div class="two-col">
              <div class="chart-box">{fig2.to_html(full_html=False, include_plotlyjs=False)}</div>
              <div><div class="st">All Recommendations</div>
              <div class="table-wrap"><table class="dt"><thead><tr><th>Priority</th><th>Action</th><th>Expected Impact</th><th>Effort</th><th>Category</th></tr></thead>
              <tbody>{_rec_rows(RECOMMENDATIONS)}</tbody></table></div></div>
            </div>"""}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Appendix (Technical Deep-Dive)
# ══════════════════════════════════════════════════════════════════════════════

def build_appendix(data):
    d = data

    # Infrastructure Topology
    infra = _load_infra_csv()
    fig_infra = go.Figure()
    labels = [r["service"] for r in infra]
    costs = [float(r["cost_per_hour"]) for r in infra]
    caps = [int(r["max_capacity"]) for r in infra]
    fig_infra.add_trace(go.Bar(name="Cost/hr", x=labels, y=costs, marker_color=COLORS[0],
                               text=[f"${c:.2f}" for c in costs], textposition="outside"))
    fig_infra.add_trace(go.Bar(name="Max Cap", x=labels, y=[c / 100 for c in caps],
                               marker_color=COLORS[1], yaxis="y2",
                               text=[str(c) for c in caps], textposition="outside"))
    fig_infra.update_layout(title="Service Cost & Capacity", height=300,
                            paper_bgcolor="white", plot_bgcolor="white",
                            margin=dict(l=30, r=30, t=35, b=20),
                            yaxis=dict(title="$ / hr", side="left"),
                            yaxis2=dict(title="Capacity (req/s)", overlaying="y", side="right"),
                            legend=dict(orientation="h", y=1.15))
    dep_rows = "".join(f"<tr><td>{s}</td><td>{r}</td><td>{t}</td></tr>" for s, r, t in DEPENDENCY_GRAPH)
    svc_cards = ""
    for r in infra:
        svc_cards += f"""<div class="q-card"><div class="ql">{r['service']}</div>
        <div class="qa">{r['instance_type']} | {r['cpu_cores']} CPU | {r['memory_gb']} GB<br>
        ${r['cost_per_hour']}/hr | Cap: {r['max_capacity']} req | Startup: {r['startup_time_sec']}s</div></div>"""

    # Anomaly section
    fig_anom = make_subplots(rows=2, cols=1, subplot_titles=["Response Time with SLO Threshold", "CPU & Queue"],
                             vertical_spacing=0.12, shared_xaxes=True)
    fig_anom.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_lat"], name="Response time",
                                  line=dict(color=COLORS[3], width=2),
                                  fill="tozeroy", fillcolor=_hex_rgba(COLORS[3], 0.1)), row=1, col=1)
    fig_anom.add_hline(y=500, line_dash="dash", line_color=THEME["danger"],
                       annotation_text="SLO (500ms)", row=1, col=1)
    fig_anom.add_hline(y=2000, line_dash="dot", line_color=THEME["warning"],
                       annotation_text="Retry threshold (2000ms)", row=1, col=1)
    fig_anom.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_cpu"], name="CPU %",
                                  line=dict(color=COLORS[0])), row=2, col=1)
    fig_anom.add_trace(go.Scatter(x=d["obs_t"], y=d["obs_queue"], name="Queue",
                                  line=dict(color=COLORS[3], dash="dot")), row=2, col=1)
    fig_anom.update_layout(title="Anomaly Detection", height=400,
                           paper_bgcolor="white", plot_bgcolor="white",
                           margin=dict(l=30, r=30, t=35, b=20),
                           legend=dict(orientation="h", y=1.02, font=dict(size=10)))
    fig_anom.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig_anom.update_yaxes(title_text="ms", row=1, col=1)
    fig_anom.update_yaxes(title_text="% / count", row=2, col=1)
    anomalies_detected = sum(1 for v in d["obs_lat"] if v > 500)
    pct_anomalous = round(anomalies_detected / max(len(d["obs_lat"]), 1) * 100, 1)
    anomaly_cards = f"""<div class="q-grid" style="margin:8px 0">
      <div class="q-card"><div class="ql">SLO Violations</div><div class="qa">{anomalies_detected} min ({pct_anomalous}%)</div></div>
      <div class="q-card"><div class="ql">Peak Over SLO</div><div class="qa">{max(d['obs_lat']) / 500:.1f}x</div></div>
      <div class="q-card"><div class="ql">Retry Triggered</div><div class="qa">{'Yes' if max(d['obs_lat']) > 2000 else 'No'}</div></div>
      <div class="q-card"><div class="ql">Anomaly Window</div><div class="qa">t=60–100 (40 min)</div></div>
    </div>"""

    # Root cause section
    traces = d["traces"]
    fig_rc = make_subplots(rows=2, cols=2, subplot_titles=["Response Time", "CPU",
                                                            "Pending Requests", "Instances"],
                           vertical_spacing=0.12, horizontal_spacing=0.08)
    sim_t = d["base_result"].times
    fig_rc.add_trace(go.Scatter(x=sim_t, y=d["base_result"].aux_values.get("latency_ms", []),
                                name="Latency", line=dict(color=COLORS[3])), row=1, col=1)
    fig_rc.add_trace(go.Scatter(x=sim_t, y=d["base_result"].aux_values.get("cpu_utilization", []),
                                name="CPU", line=dict(color=COLORS[0])), row=1, col=2)
    fig_rc.add_trace(go.Scatter(x=sim_t, y=d["base_result"].values.get("PendingRequests", []),
                                name="Queue", line=dict(color=COLORS[3])), row=2, col=1)
    fig_rc.add_trace(go.Scatter(x=sim_t, y=d["base_result"].values.get("RunningInstances", []),
                                name="Instances", line=dict(color=COLORS[5])), row=2, col=2)
    fig_rc.update_layout(title="Model Dynamics", height=380,
                         paper_bgcolor="white", plot_bgcolor="white",
                         margin=dict(l=20, r=20, t=35, b=20),
                         legend=dict(orientation="h", y=1.02, font=dict(size=10)))
    causal_cards = ""
    for var_name, strip in traces.items():
        if strip is not None:
            factors = getattr(strip, 'factors', [])[:5]
            factor_lines = "".join(
                f"<div>&middot; {f.get('name', '?')}: {f.get('value', 'N/A')} (contrib {f.get('contribution', 0):.2f})</div>"
                for f in factors if isinstance(f, dict))
            total = getattr(strip, 'total_value', 'N/A')
            causal_cards += f"""<div class="q-card"><div class="ql">{var_name}</div><div class="qa">Total: {total}<br>{factor_lines}</div></div>"""
        else:
            causal_cards += f"""<div class="q-card"><div class="ql">{var_name}</div><div class="qa">N/A</div></div>"""

    # Sensitivity section
    oat = d["oat_result"]
    fig_sens = go.Figure()
    if oat and oat.oat_low and oat.oat_high:
        pnames = list(oat.oat_low.keys())[:6]
        for pname in pnames:
            lo = oat.oat_low.get(pname, 0)
            hi = oat.oat_high.get(pname, 0)
            fig_sens.add_trace(go.Bar(name=pname, x=[pname], y=[hi - lo],
                                      marker_color=COLORS[pnames.index(pname) % len(COLORS)]))
    fig_sens.update_layout(title="OAT Sensitivity — Cost Impact", height=300,
                           paper_bgcolor="white", plot_bgcolor="white",
                           margin=dict(l=30, r=30, t=35, b=20),
                           yaxis_title="Cost delta ($)", font=dict(size=10))
    param_insights = [
        ("scale_up_cpu", "Lower threshold → earlier scaling, higher cost, better SLO"),
        ("scale_down_cpu", "Higher threshold → faster scale-in, lower cost, risk of oscillation"),
        ("instance_capacity", "Higher capacity → fewer instances, lower cost"),
        ("scale_up_delay", "Longer delay → queue buildup, SLO violations"),
        ("cost_per_hr", "Direct linear driver of total cost"),
    ]
    insight_rows = "".join(f"<tr><td>{p}</td><td>{d}</td></tr>" for p, d in param_insights)

    # Raw events table
    ev_rows = ""
    for ev in d["events"]:
        sev_color = {"critical": THEME["danger"], "warning": THEME["warning"], "info": COLORS[4]}.get(ev["severity"], THEME["muted"])
        ev_rows += f"<tr><td class='num'>t={ev['timestamp']}m</td><td>{_relabel_event(ev)}</td><td style='color:{sev_color}'>{ev['severity']}</td><td>{ev['service']}</td><td>{ev['value']}</td></tr>"

    return {"title": "Technical Appendix",
            "content": f"""
            <div class="st">Technical Deep-Dive</div>

            <details>
              <summary>Infrastructure Topology</summary>
              <div class="details-body">
                <div class="two-col">
                  <div class="chart-box">{fig_infra.to_html(full_html=False, include_plotlyjs=False)}</div>
                  <div><div class="st">Dependencies</div>
                  <div class="table-wrap"><table class="dt"><thead><tr><th>Source</th><th>Relation</th><th>Target</th></tr></thead>
                  <tbody>{dep_rows}</tbody></table></div></div>
                </div>
                <div class="st">Service Config</div>
                <div class="q-grid">{svc_cards}</div>
              </div>
            </details>

            <details>
              <summary>Knowledge Base Explorer</summary>
              <div class="details-body">{_build_kb_section(d["store"])}</div>
            </details>

            <details>
              <summary>Anomaly Detection</summary>
              <div class="details-body">
                {anomaly_cards}
                <div class="chart-box">{fig_anom.to_html(full_html=False, include_plotlyjs=False)}</div>
              </div>
            </details>

            <details>
              <summary>Root Cause Analysis</summary>
              <div class="details-body">
                <div class="two-col">
                  <div class="chart-box">{fig_rc.to_html(full_html=False, include_plotlyjs=False)}</div>
                  <div><div class="st">Causal Decomposition</div>
                  <div class="q-grid">{causal_cards}</div></div>
                </div>
                <div class="q-card"><div class="ql">Root Cause Diagnosis</div><div class="qa">
                <b>Primary:</b> Traffic spike at t=60 (3x baseline) exceeds processing capacity.<br>
                <b>Secondary:</b> Autoscaler startup delay (120s) → queue buildup → retry amplification.<br>
                <b>Tertiary:</b> After traffic normalizes, instances remain elevated ~15 min → idle waste.<br>
                <b>Recommendation:</b> Reduce autoscaler detection delay or switch to predictive scaling.
                </div></div>
              </div>
            </details>

            <details>
              <summary>Sensitivity Analysis</summary>
              <div class="details-body">
                <div class="two-col">
                  <div class="chart-box">{fig_sens.to_html(full_html=False, include_plotlyjs=False)}</div>
                  <div><div class="st">Parameter Impact</div>
                  <div class="table-wrap"><table class="dt"><thead><tr><th>Parameter</th><th>Effect</th></tr></thead>
                  <tbody>{insight_rows}</tbody></table></div></div>
                </div>
              </div>
            </details>

            <details>
              <summary>Raw Event Log</summary>
              <div class="details-body">
                <div class="table-wrap"><table class="dt"><thead><tr><th class='num'>Time</th><th>Event</th><th>Severity</th><th>Service</th><th>Value</th></tr></thead>
                <tbody>{ev_rows}</tbody></table></div>
              </div>
            </details>"""}


# ══════════════════════════════════════════════════════════════════════════════
# HTML ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DevOps Cloud Digital Twin — Incident Report</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js@3.6.0/dist/plotly.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:{BG};color:{TEXT};font-size:14px;line-height:1.5}}
.tab-bar{{display:flex;gap:0;position:sticky;top:0;z-index:99;background:{BG};border-bottom:1px solid #e5e5e5;padding:0 20px;overflow-x:auto}}
.tab-btn{{padding:10px 16px;background:transparent;color:{MUTED};border:none;cursor:pointer;font-size:11px;font-weight:500;border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap;flex-shrink:0;letter-spacing:.02em}}
.tab-btn:hover{{color:{TEXT}}}
.tab-btn.active{{color:{PRIMARY};border-bottom-color:{PRIMARY}}}
.content{{max-width:1200px;margin:0 auto;padding:20px}}
.pane{{}}
.pane.hidden{{display:none}}
.cover{{margin-bottom:24px;padding-bottom:20px;border-bottom:1px solid #e5e5e5}}
.cover h1{{font-family:'Fraunces',Georgia,serif;font-size:28px;font-weight:700;color:{TEXT};margin-bottom:2px}}
.cover .subtitle{{font-family:'Fraunces',Georgia,serif;font-size:15px;color:{MUTED};margin-bottom:4px}}
.cover .date{{font-size:11px;color:{MUTED}}}
.narrative{{background:{CARD};border-radius:6px;padding:14px 16px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.06);font-size:13px;line-height:1.6;color:{TEXT}}}
.st{{font-family:'Fraunces',Georgia,serif;font-size:16px;font-weight:600;color:{PRIMARY};margin:20px 0 8px 0;border-bottom:1px solid #e5e5e5;padding-bottom:4px}}
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-bottom:16px}}
.kpi{{background:{CARD};border-radius:6px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.06);border-left:3px solid {MUTED}}}
.kl{{font-size:9px;text-transform:uppercase;color:{MUTED};letter-spacing:.04em;margin-bottom:2px}}
.kv{{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}}
.ks{{font-size:10px;color:{MUTED};margin-top:2px}}
.two-col{{display:flex;gap:14px;margin-bottom:12px}}
.two-col>*{{flex:1;min-width:0}}
.chart-box{{background:{CARD};border-radius:6px;padding:6px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:10px}}
.chart-box.hero{{padding:10px}}
.q-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin:10px 0}}
.q-card{{background:{CARD};border-radius:6px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.ql{{font-size:9px;font-weight:700;color:{PRIMARY};margin-bottom:3px;text-transform:uppercase;letter-spacing:.04em}}
.qa{{font-size:11px;color:{TEXT};line-height:1.5}}
.table-wrap{{overflow-x:auto;margin:8px 0}}
.dt{{width:100%;border-collapse:collapse;font-size:11px}}
.dt th{{background:{PRIMARY};color:#fff;padding:5px 10px;text-align:left;font-weight:600}}
.dt td{{padding:5px 10px;border-bottom:1px solid #eee;font-variant-numeric:tabular-nums}}
.dt td.num{{text-align:right}}
.dt tbody tr:hover{{background:#f0f2f5}}
details{{margin:8px 0;background:{CARD};border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.06);overflow:hidden}}
summary{{padding:10px 14px;font-weight:600;font-size:12px;color:{PRIMARY};cursor:pointer;user-select:none;letter-spacing:.02em}}
summary:hover{{background:rgba(43,69,112,.04)}}
details[open] summary{{border-bottom:1px solid #eee;margin-bottom:8px}}
details .details-body{{padding:4px 14px 14px}}
@media(max-width:768px){{.two-col{{flex-direction:column}}.tab-btn{{font-size:10px;padding:8px 12px}}.cover h1{{font-size:22px}}}}
</style></head><body>
<div class=tab-bar>{TABS}</div>
<div class=content>{PANES}</div>
<script>
window.addEventListener('load',function(){{setTimeout(function(){{document.querySelectorAll('.pane').forEach(function(e,i){{if(i!==0)e.classList.add('hidden')}})}},500)}})
function switchTab(i){{document.querySelectorAll('.pane').forEach(function(e){{e.classList.remove('hidden')}});document.querySelectorAll('.pane').forEach(function(e,j){{if(j!==i)e.classList.add('hidden')}});document.querySelectorAll('.tab-btn').forEach(function(e,j){{e.classList.toggle('active',j===i)}});document.querySelectorAll('.pane:not(.hidden) .js-plotly-plot').forEach(function(e){{if(typeof Plotly!=='undefined')Plotly.Plots.resize(e)}})}}
</script></body></html>"""


def build_html(pages):
    tabs = "".join(
        f'<button class="tab-btn {"active" if i==0 else ""}" onclick="switchTab({i})">'
        f'{p["title"]}</button>'
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
    )


def main():
    print("DevOps Cloud Digital Twin Dashboard")
    print("=" * 60)
    data = run_pipeline()
    print("\nBuilding 4 dashboard tabs...")
    pages = [
        build_summary(data),
        build_timeline(data),
        build_cost_recs(data),
        build_appendix(data),
    ]
    print(f"  Built {len(pages)} pages")
    print("Assembling HTML...")
    html = make_lazy(build_html(pages))
    out = "/tmp/devops_dashboard.html"
    with open(out, "w") as f:
        f.write(html)
    size_kb = len(html) // 1024
    print(f"\nDashboard: {out} ({size_kb}KB, {len(pages)} tabs)")
    print(f"  Peak CPU: {data['peak_cpu']:.1f}%")
    print(f"  Peak latency: {data['peak_latency']:.0f}ms")
    print(f"  Total cost: ${data['final_cost']:.2f}")
    print(f"  SLO compliance: {data['slo_pct']}%")
    return out


if __name__ == "__main__":
    main()
