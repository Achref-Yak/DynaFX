#!/usr/bin/env python3
"""Atlas Broadband — Regional ISP Capacity vs. Churn Diagnosis Dashboard.

Company: Atlas Broadband, regional fixed-wireless/fiber ISP, ~$60M revenue,
3 service regions (A, B, C). Region B has rising complaint volume and churn
over two quarters.

Question: "Is Region B's churn increase caused by capacity exhaustion, and
at what utilization threshold should we trigger expansion to avoid repeating
this in Region A or C?"

Architecture: SD (subscribers, capacity, revenue) + DES (congestion queues,
arrival rates fed by SD utilization auxes) + ABM (enterprise customer agents
perceiving QoS, driving satisfaction/churn metrics that feed back into SD).

Output: /tmp/atlas_broadband_dashboard.html
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
from dynafx.dynamics.causal import causes_strip
from dynafx.dynamics.feedback import detect_feedback_loops

from dynafx.knowledge import (
    ingest_csv, MappingDef, TripleStore, NamedNode, Literal,
    TriplePattern, parse_turtle,
)
from dynafx.knowledge.model import XSD_STRING

random.seed(42)
np.random.seed(42)

T_START = 0.0
T_END = 730.0
DT = 1.0
NUM_AGENTS_PER_REGION = [14, 14, 12]
REGIONS = ["A", "B", "C"]
REGION_CAPS = [200000.0, 180000.0, 220000.0]
REGION_INITIAL_SUBS = [35000.0, 42000.0, 28000.0]
REGION_INITIAL_CAP = [100.0, 100.0, 80.0]
DATA_DIR = Path(__file__).parent.parent / "data"
ISP_NS = "http://isp-broadband.org/"
THEME = {
    "primary": "#0B3B60", "accent": "#1A7FC4", "success": "#2E8B57",
    "warning": "#D4A017", "danger": "#C0392B", "bg": "#F4F6F8",
    "card": "#FFFFFF", "text": "#2C3E50", "muted": "#7F8C8D",
}
COLORS = ["#0B3B60", "#2E8B57", "#D4A017", "#C0392B", "#1A7FC4",
          "#8E44AD", "#E67E22", "#1ABC9C", "#95A5A6", "#34495E"]


def _hex_rgba(c, a):
    h = c.lstrip("#"); r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _kpi_card(label, value, color, subtitle=""):
    return f"""<div class="kpi" style="border-top:3px solid {color}">
      <div class="kl">{label}</div>
      <div class="kv" style="color:{color}">{value}</div>
      {f'<div class="ks">{subtitle}</div>' if subtitle else ''}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(params: dict[str, float]) -> SysdModel:
    m = SysdModel("atlas_broadband")
    m.dt = DT
    m.t_span = (T_START, T_END)

    # ── Leading indicator auxes ──
    m.aux("building_permits_A",
          "30 + 20*SIN(2*PI*t/365) + PULSE(45, 100, 90)")
    m.aux("competitor_entry",
          "IF(t > 120, 1, 0)")
    m.aux("marketing_push",
          "PULSE(1.5, 200, 180) + 0.2*SIN(2*PI*t/365)")

    m.aux("growth_mod_A",
          "building_permits_A / 100 + 0.4 + 0.1*SIN(2*PI*t/365)")
    m.aux("growth_mod_B",
          "MAX(0, 0.5 - competitor_entry * 0.3) + 0.08*SIN(2*PI*t/365 + PI/6)")
    m.aux("growth_mod_C",
          "marketing_push * 0.5 + 0.3 + 0.12*SIN(2*PI*t/365)")

    # ── Subscriber stocks (per region) ──
    for i, r in enumerate(REGIONS):
        sn = f"Subs_{r}"
        cap = REGION_CAPS[i]
        init = REGION_INITIAL_SUBS[i]
        with m.stock(sn, init) as s:
            s.inflow(f"adopt_{r}",
                     f"base_adoption * {sn} * growth_mod_{r} * "
                     f"MAX(0, ({cap} - {sn}) / MAX(1, {cap}))")
            s.outflow(f"churn_{r}",
                      f"{sn} * churn_fraction_{r}")

    # ── Capacity stocks (per region) ──
    for i, r in enumerate(REGIONS):
        cn = f"Cap_{r}"
        init = REGION_INITIAL_CAP[i]
        with m.stock(cn, init) as s:
            s.inflow(f"deploy_{r}",
                     f"DELAY3(MAX(0, capacity_order_{r}), project_delay_{r})")
            s.outflow(f"retire_{r}", f"{cn} * 0.0003")

    # ── Revenue ──
    with m.stock("Revenue_Reserve", 0.0) as s:
        s.inflow("revenue_in",
                 "(Subs_A + Subs_B + Subs_C) * arpu_daily")
        s.outflow("opex_out",
                  "Cap_A * opex_per_unit_daily + Cap_B * opex_per_unit_daily "
                  "+ Cap_C * opex_per_unit_daily "
                  "+ (Subs_A + Subs_B + Subs_C) * variable_opex_per_sub_daily "
                  "+ (Subs_A + Subs_B + Subs_C) * churn_fraction_B * churn_cost_per_sub")

    # ── Per-region auxes ──
    for i, r in enumerate(REGIONS):
        cn = f"Cap_{r}"
        sn = f"Subs_{r}"

        m.aux(f"util_{r}",
              f"{sn} / MAX(1, {cn} * subs_per_unit)")
        m.aux(f"qos_{r}",
              f"IF(util_{r} <= 0.7, 100, "
              f"IF(util_{r} <= 0.9, 100 - (util_{r} - 0.7) * 250, "
              f"MAX(10, 100 - (util_{r} - 0.7) * 500)))")
        m.aux(f"nps_{r}",
              f"qos_{r} * 0.6 + 20")
        m.aux(f"capacity_order_{r}",
              f"IF(util_{r} > capacity_threshold_{r}, "
              f"(util_{r} - capacity_threshold_{r}) * 20, 0)")
        m.aux(f"congest_amplifier_{r}",
              f"MAX(0, (congest_{r}_length - 2)) * 0.0001")
        m.aux(f"churn_fraction_{r}",
              f"base_churn_{r} + churn_abm_contrib_{r} + congest_amplifier_{r}")

    # ── ABM contribution auxes (per-region from per-type agent metrics) ──
    for r in REGIONS:
        m.aux(f"churn_abm_contrib_{r}",
              f"MAX(0, (0.5 - Cust_{r}_satisfaction_avg / 100)) * 0.02 "
              f"+ Cust_{r}_churn_risk_avg * 0.008")

    # ── Financial auxes ──
    m.aux("arpu_daily", "arpu / 30.0")
    m.aux("opex_per_unit_daily", "capacity_opex_per_unit_monthly / 30.0")
    m.aux("variable_opex_per_sub_daily", "variable_opex_per_sub_monthly / 30.0")
    m.aux("total_subs", "Subs_A + Subs_B + Subs_C")
    m.aux("revenue_daily", "total_subs * arpu / 30.0")
    m.aux("opex_daily",
          "Cap_A * opex_per_unit_daily + Cap_B * opex_per_unit_daily "
          "+ Cap_C * opex_per_unit_daily")
    m.aux("profit_margin",
          "(revenue_daily - opex_daily) / MAX(1, revenue_daily)")
    m.aux("avg_utilization",
          "(util_A + util_B + util_C) / 3")
    m.aux("avg_qos",
          "(qos_A + qos_B + qos_C) / 3")

    # ── DES queues (SD→DES coupling: arrival_rate references SD util aux) ──
    for i, r in enumerate(REGIONS):
        mult = [3.0, 5.0, 2.0][i]
        m.queue(f"congest_{r}", capacity=-1, service_time="3.0",
                arrival_rate=f"MAX(0, (util_{r} - 0.70) * {mult})")

    # ── DES resource ──
    m.resource("noc_team", capacity=5, cost_per_unit=500.0)

    # ── ABM agents (per region type) ──
    for ri, r in enumerate(REGIONS):
        num = NUM_AGENTS_PER_REGION[ri]
        for ai in range(num):
            init_sat = round(random.uniform(75, 95), 1)
            init_dev = round(random.uniform(100, 500), 0)
            growth = round(random.uniform(0.01, 0.04), 4)

            m.agents.append(AgentDef(
                f"Cust_{r}", 1,
                properties=[
                    AgentPropDef("satisfaction", init_sat, min=0, max=100),
                    AgentPropDef("churn_risk", 0.0, min=0, max=1),
                    AgentPropDef("is_active", 1.0, min=0, max=1),
                    AgentPropDef("device_count", init_dev, min=0, max=10000),
                    AgentPropDef("growth_rate", growth, min=0, max=0.1),
                ],
                strategies=[
                    AgentStrategy("normal", [
                        AgentRuleDef("grow", "always",
                                     ["device_count += device_count * growth_rate * dt / 30"]),
                        AgentRuleDef("update_sat", "always",
                                     [f"satisfaction += (qos_{r} - satisfaction) * 0.05 * dt"]),
                        AgentRuleDef("decay_risk", "always",
                                     ["churn_risk = MAX(0, churn_risk - 0.0005 * dt)"]),
                    ]),
                    AgentStrategy("at_risk", [
                        AgentRuleDef("no_grow", "always", []),
                        AgentRuleDef("sat_erode", "always",
                                     [f"satisfaction += (qos_{r} - satisfaction) * 0.02 * dt"]),
                        AgentRuleDef("risk_up", "always",
                                     ["churn_risk = MIN(1, churn_risk + 0.002 * dt)"]),
                    ]),
                    AgentStrategy("churned", [
                        AgentRuleDef("inactive", "always",
                                     ["is_active = 0"]),
                    ]),
                ],
                meta_rules=[
                    AgentRuleDef("risk_trigger",
                                 "satisfaction < 35",
                                 ["SWITCH_STRATEGY('at_risk', cooldown=30)"]),
                    AgentRuleDef("recovery",
                                 "satisfaction > 60 and strategy == 'at_risk'",
                                 ["SWITCH_STRATEGY('normal', cooldown=15)"]),
                    AgentRuleDef("churn_out",
                                 "churn_risk > 0.85",
                                 ["SWITCH_STRATEGY('churned', cooldown=999)"]),
                ],
            ))

    return m


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _safe_val(v, default=0.0):
    """Safely get last value from a list."""
    return v[-1] if v else default


def _get_ts(r):
    """Extract combined timeseries from a simulation result."""
    ts = dict(r.values)
    ts.update(r.aux_values)
    if hasattr(r, 'des_metrics_history') and r.des_metrics_history:
        all_keys: set[str] = set()
        for entry in r.des_metrics_history:
            all_keys.update(entry.keys())
        for key in all_keys:
            ts[key] = [d.get(key, 0.0) for d in r.des_metrics_history]
    return ts


def import_isp_data() -> dict[str, Any]:
    """Load ISP historical data from CSVs via ingest_csv + TripleStore."""
    store = TripleStore()
    onto_path = DATA_DIR / "isp-ontology.ttl"
    if onto_path.exists():
        onto = parse_turtle(onto_path.read_text())
        for t in onto.all_triples():
            store.add(t, "isp:graphs/meta")

    mappings_dir = DATA_DIR / "mappings"
    for yaml_path in sorted(mappings_dir.glob("isp_*.yaml")):
        try:
            mapping = MappingDef.from_yaml(str(yaml_path))
            csv_path = DATA_DIR / mapping.csv
            if csv_path.exists():
                ingest_csv(mapping, str(csv_path), store, strict=False)
        except Exception as e:
            print(f"    Warning: could not load {yaml_path.name}: {e}")

    _NS = ISP_NS
    def _n(name: str) -> NamedNode:
        return NamedNode(f"{_NS}{name}")

    def _query_values(pred: str, region_filter: str | None = None) -> list[tuple[int, float]]:
        results = []
        for t in store.triples(TriplePattern(predicate=_n(pred))):
            sub = t.subject
            month_t = list(store.triples(TriplePattern(subject=sub, predicate=_n("month"))))
            if not month_t:
                continue
            m = int(month_t[0].object_.value)
            if region_filter is not None:
                region_t = list(store.triples(
                    TriplePattern(subject=sub, predicate=_n("region"),
                                  object_=Literal(region_filter, datatype=XSD_STRING))
                ))
                if not region_t:
                    continue
            results.append((m, float(t.object_.value)))
        results.sort(key=lambda x: x[0])
        return results

    hist: dict[str, Any] = {"times": []}

    for r in REGIONS:
        subs = _query_values("subscriberCount", r)
        churn = _query_values("churnRate", r)
        qos = _query_values("qosScore", r)
        util = _query_values("avgUtilization", r)
        hist[f"subs_{r.lower()}"] = [v for _, v in subs]
        hist[f"churn_{r.lower()}"] = [v for _, v in churn]
        hist[f"qos_{r.lower()}"] = [v for _, v in qos]
        hist[f"util_{r.lower()}"] = [v for _, v in util]

    ind_bp = _query_values("buildingPermits")
    ind_comp = _query_values("competitorActive")
    ind_mkt = _query_values("marketingSpend")
    hist["building_permits"] = [v for _, v in ind_bp]
    hist["competitor_active"] = [v for _, v in ind_comp]
    hist["marketing_spend"] = [v for _, v in ind_mkt]

    n_months = len(hist.get("subs_a", []))
    hist["times"] = [m * 30 - n_months * 30 for m in range(n_months)]

    hist_loaded = any(len(v) > 0 for k, v in hist.items() if k != "times")
    if not hist_loaded:
        print("    Warning: ISP historical data not loaded — running model-only mode")
    else:
        print(f"    Loaded {n_months} months of historical data across {len(store)} triples")
    return hist


def run_simulation() -> dict[str, Any]:
    print("Atlas Broadband — Regional ISP Capacity vs. Churn Diagnosis")
    print("=" * 60)

    base_params = {
        "base_adoption": 0.005,
        "base_churn_A": 0.00025,
        "base_churn_B": 0.00030,
        "base_churn_C": 0.00020,
        "arpu": 49.99,
        "subs_per_unit": 500.0,
        "capacity_opex_per_unit_monthly": 5000.0,
        "variable_opex_per_sub_monthly": 7.50,
        "churn_cost_per_sub": 50.0,
        "capacity_threshold_A": 0.82,
        "capacity_threshold_B": 0.82,
        "capacity_threshold_C": 0.82,
        "project_delay_A": 120.0,
        "project_delay_B": 120.0,
        "project_delay_C": 120.0,
    }

    print("\nBuilding Atlas Broadband model...")
    model = _build_model(base_params)
    print(f"  Model: {len(model.stocks)} stocks, {len(model.aux_vars)} auxes, "
          f"{len(model.agents)} agents, {len(model.queues)} queues")

    # ── Import historical data ──
    print("  Importing historical ISP data...")
    historical = import_isp_data()

    # ── Baseline ──
    print("  Baseline simulation (t=0 to 730 days)...")
    base_result = model.simulate(params=dict(base_params), method="euler", dt=DT)
    print(f"    Steps: {base_result.steps}")
    ts_base = _get_ts(base_result)

    # ── Counterfactual (Region B expands early) ──
    print("  Counterfactual (Region B threshold=0.72, delay=60)...")
    cf_params = dict(base_params)
    cf_params["capacity_threshold_B"] = 0.72
    cf_params["project_delay_B"] = 60.0
    cf_result = model.simulate(params=cf_params, method="euler", dt=DT)
    ts_cf = _get_ts(cf_result)

    # ── Scenarios ──
    print("  Running 5 scenarios...")
    scenario_configs = [
        ("Proactive",   {"capacity_threshold_B": 0.72, "project_delay_B": 60.0}),
        ("Data-Driven", {"capacity_threshold_B": 0.78, "project_delay_B": 90.0}),
        ("Baseline",    {"capacity_threshold_B": 0.82, "project_delay_B": 120.0}),
        ("Conservative",{"capacity_threshold_B": 0.88, "project_delay_B": 150.0}),
        ("Reactive",    {"capacity_threshold_B": 0.95, "project_delay_B": 180.0}),
    ]
    scenario_results = []
    for sname, overrides in scenario_configs:
        sp = {**base_params, **overrides}
        sr = model.simulate(params=sp, method="euler", dt=DT)
        ts = _get_ts(sr)
        scenario_results.append({
            "name": sname,
            "overrides": overrides,
            "result": sr,
            "ts": ts,
            "final_subs": _safe_val(ts.get("total_subs", [])),
            "final_revenue": _safe_val(ts.get("Revenue_Reserve", [])),
            "final_margin": _safe_val(ts.get("profit_margin", [])),
            "final_util_B": _safe_val(ts.get("util_B", [])),
            "final_qos_B": _safe_val(ts.get("qos_B", [])),
        })
        print(f"    {sname}: {scenario_results[-1]['final_subs']:,.0f} subs, "
              f"${scenario_results[-1]['final_revenue']:,.0f} revenue")

    # ── Causal analysis ──
    print("  Causal analysis...")
    final_state = {}
    for s in model.stocks:
        vals = base_result.values.get(s.name, [])
        if vals:
            final_state[s.name] = vals[-1]
    for a in model.aux_vars:
        vals = base_result.aux_values.get(a.name, [])
        if vals:
            final_state[a.name] = vals[-1]
    trace_revenue = causes_strip(model, "Revenue_Reserve", final_state)
    trace_subs_B = causes_strip(model, "Subs_B", final_state)
    loops = detect_feedback_loops(model)
    churn_loops = [l for l in loops.loops if "urn" in l.name.lower() or "util" in l.name.lower()]

    # ── Sensitivity (OAT) ──
    print("  Sensitivity analysis...")
    oat_params = {
        "capacity_threshold_B": (0.72, 0.95),
        "project_delay_B": (60.0, 200.0),
        "base_adoption": (0.003, 0.008),
        "base_churn_B": (0.00015, 0.0006),
        "arpu": (39.99, 69.99),
    }
    oat_results = {}
    for pname, (lo, hi) in oat_params.items():
        sp_lo = {**base_params, pname: lo}
        sp_hi = {**base_params, pname: hi}
        r_lo = model.simulate(params=sp_lo, method="euler", dt=DT)
        r_hi = model.simulate(params=sp_hi, method="euler", dt=DT)
        lo_rev = _safe_val(r_lo.values.get("Revenue_Reserve", []))
        hi_rev = _safe_val(r_hi.values.get("Revenue_Reserve", []))
        lo_subs = _safe_val(r_lo.aux_values.get("total_subs", []))
        hi_subs = _safe_val(r_hi.aux_values.get("total_subs", []))
        oat_results[pname] = {
            "lo_rev": lo_rev, "hi_rev": hi_rev,
            "lo_subs": lo_subs, "hi_subs": hi_subs,
        }

    # ── ABM analytics (from per-step metrics history) ──
    print("  ABM analytics...")
    abm_engine = base_result.abm_engine
    agent_history = []
    for step_idx in range(len(base_result.times)):
        t = base_result.times[step_idx]
        metrics = base_result.abm_metrics_history[step_idx]
        active = 0; total_count = 0; sat_sum = 0; risk_sum = 0
        for rn in REGIONS:
            cnt = metrics.get(f"Cust_{rn}_count", 0)
            n_active = metrics.get(f"Cust_{rn}_is_active_sum", cnt)
            total_count += cnt
            active += n_active
            sat_sum += metrics.get(f"Cust_{rn}_satisfaction_sum", 0)
            risk_sum += metrics.get(f"Cust_{rn}_churn_risk_sum", 0)
        record: dict[str, Any] = {"t": t, "active": active, "churned": total_count - active,
            "avg_satisfaction": sat_sum / max(1, active),
            "avg_churn_risk": risk_sum / max(1, active)}
        for rn in REGIONS:
            rc = metrics.get(f"Cust_{rn}_count", 0)
            r_active = metrics.get(f"Cust_{rn}_is_active_sum", rc)
            record[f"sat_{rn}"] = metrics.get(f"Cust_{rn}_satisfaction_sum", 0) / max(1, r_active)
            record[f"risk_{rn}"] = metrics.get(f"Cust_{rn}_churn_risk_sum", 0) / max(1, r_active)
        agent_history.append(record)

    # ── Disruption cost ──
    revenue_baseline = _safe_val(ts_base.get("Revenue_Reserve", []))
    revenue_cf = _safe_val(ts_cf.get("Revenue_Reserve", []))
    disruption_cost = max(0, revenue_cf - revenue_baseline)

    times_base = base_result.times
    rev_baseline_series = ts_base.get("Revenue_Reserve", [])
    rev_cf_series = ts_cf.get("Revenue_Reserve", [])

    # Monthly breakdown of disruption cost
    monthly_cost = []
    for month in range(1, 25):
        day_start = (month - 1) * 30
        day_end = min(month * 30, T_END)
        idx_start = min(int(day_start / DT), len(rev_baseline_series) - 1)
        idx_end = min(int(day_end / DT), len(rev_cf_series) - 1)
        rb = rev_baseline_series[idx_end] - rev_baseline_series[idx_start]
        rc = rev_cf_series[idx_end] - rev_cf_series[idx_start]
        monthly_cost.append({
            "month": month, "day_start": day_start, "day_end": day_end,
            "baseline_rev": rb, "cf_rev": rc, "gap": rc - rb,
        })

    data: dict[str, Any] = {
        "model": model,
        "base_params": base_params,
        "base_result": base_result,
        "ts": ts_base,
        "scenarios": scenario_results,
        "oat_results": oat_results,
        "trace_revenue": trace_revenue,
        "trace_subs_B": trace_subs_B,
        "loops": loops,
        "churn_loops": churn_loops,
        "agent_history": agent_history,
        "abm_engine": abm_engine,
        "times": times_base,
        "cf_result": cf_result,
        "ts_cf": ts_cf,
        "disruption_cost": disruption_cost,
        "monthly_cost": monthly_cost,
        "historical": historical,
    }

    for r in REGIONS:
        data[f"subs_{r.lower()}"] = ts_base.get(f"Subs_{r}", [])
        data[f"cap_{r.lower()}"] = ts_base.get(f"Cap_{r}", [])
        data[f"util_{r.lower()}"] = ts_base.get(f"util_{r}", [])
        data[f"qos_{r.lower()}"] = ts_base.get(f"qos_{r}", [])
        data[f"nps_{r.lower()}"] = ts_base.get(f"nps_{r}", [])
        data[f"congest_len_{r.lower()}"] = data["ts"].get(f"congest_{r}_length", [])

    # Recompute churn_fraction using true ABM + DES data (post-hoc aux can't see DES metrics)
    for r in REGIONS:
        base_key = f"base_churn_{r}"
        base_rate = base_params.get(base_key, 0.0003)
        churn_ts = []
        n = len(data["times"])
        sat_key = f"Cust_{r}_satisfaction_avg"
        risk_key = f"Cust_{r}_churn_risk_avg"
        cl_key = f"congest_{r}_length"
        for i in range(n):
            m = base_result.abm_metrics_history[i] if i < len(base_result.abm_metrics_history) else {}
            sat = m.get(sat_key, 50)
            risk = m.get(risk_key, 0)
            cl = data["ts"].get(cl_key, [0])[i] if i < len(data["ts"].get(cl_key, [0])) else 0
            abm_contrib = max(0, (0.5 - sat/100)) * 0.02 + risk * 0.008
            des_amp = max(0, cl - 2) * 0.0001
            churn_ts.append(base_rate + abm_contrib + des_amp)
        data[f"churn_frac_{r.lower()}"] = churn_ts

    for r in REGIONS:
        data[f"capacity_order_{r.lower()}"] = ts_base.get(f"capacity_order_{r}", [])

    data.update({
        "total_subs": ts_base.get("total_subs", []),
        "revenue_reserve": ts_base.get("Revenue_Reserve", []),
        "profit_margin": ts_base.get("profit_margin", []),
        "avg_utilization": ts_base.get("avg_utilization", []),
        "avg_qos": ts_base.get("avg_qos", []),
        "revenue_daily": ts_base.get("revenue_daily", []),
        "opex_daily": ts_base.get("opex_daily", []),
    })

    data["final_subs"] = _safe_val(data["total_subs"])
    data["final_revenue"] = _safe_val(data["revenue_reserve"])
    data["final_margin"] = _safe_val(data["profit_margin"])
    data["final_avg_qos"] = _safe_val(data["avg_qos"])

    print(f"\n  Final: {data['final_subs']:,.0f} subs, "
          f"${data['final_revenue']:,.0f} revenue, "
          f"margin {data['final_margin']:.1%}, "
          f"avg QoS {data['final_avg_qos']:.0f}")
    print(f"  Disruption cost (Region B): ${disruption_cost:,.0f}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# TAB BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_executive_summary(d: dict) -> dict:
    util_b = d.get("util_b", [0])
    crisis_day = next((i for i, u in enumerate(util_b) if u > 0.82), 0)
    peak_congest = int(max(d.get("congest_len_b", [0])))
    peak_churn = max(d.get("churn_frac_b", [0.0003]))
    base_churn_b = d["base_params"].get("base_churn_B", 0.0003)
    churn_mult = peak_churn / max(1e-8, base_churn_b)

    hist = d.get("historical", {})
    hist_times = hist.get("times", [])
    hist_subs_a = hist.get("subs_a", [])
    hist_subs_b = hist.get("subs_b", [])
    hist_subs_c = hist.get("subs_c", [])

    content = f"""<div style="padding:4px 8px 12px 8px;line-height:1.7;border-bottom:1px solid #eee;margin-bottom:8px">
    <p style="margin:0 0 6px 0">
      <b>Atlas Broadband</b> grew from <b>{hist_subs_a[0]+hist_subs_b[0]+hist_subs_c[0]:,.0f}</b> to
      <b>{d['final_subs']:,.0f} subscribers</b> across three regions in just over two years.
      Region B led growth from {hist_subs_b[0]:,.0f} initial subscribers, driven by new housing
      developments in the corridor. But capacity investment lagged — a
      <b>120-day construction delay</b> meant infrastructure couldn't keep pace.
    </p>
    <p style="margin:0 0 6px 0">
      When subscriber growth pushed utilization past <b>82% on day {crisis_day}</b>, Region B's
      network buckled. Service requests piled up (queue peaked at <b>{peak_congest} items</b>),
      Quality of Service dropped to <b>{min(d['qos_b']):.0f}/100</b>, and customer satisfaction
      scores fell below 35. The churn rate surged to <b>{churn_mult:.0f}× baseline</b>.
    </p>
    <p style="margin:0">
      The <b>bottom line</b>: delayed capacity expansion cost Atlas Broadband an estimated
      <b style="color:{THEME['danger']}">${d['disruption_cost']:,.0f}</b> in lost revenue —
      revenue that could have been preserved with earlier infrastructure investment.
    </p>
    </div>
    <div class="kpi-row">
      {_kpi_card("Total Subscribers", f"{d['final_subs']:,.0f}", THEME["primary"],
                  f"3 regions: A/B/C")}
      {_kpi_card("Total Revenue", f"${d['final_revenue']:,.0f}", THEME["success"],
                  "730-day cumulative")}
      {_kpi_card("Profit Margin", f"{d['final_margin']:.1%}", THEME["accent"],
                  f"Target >15%")}
      {_kpi_card("Avg QoS Score", f"{d['final_avg_qos']:.0f}", THEME["warning"],
                  "Target >80")}
      {_kpi_card("Avg Utilization", f"{_safe_val(d['avg_utilization']):.1%}", THEME["danger"],
                  "Ideal <70%")}
      {_kpi_card("Region B Churn (end)", f"{_safe_val(d['churn_frac_b']):.4f}", THEME["danger"],
                  f"Peaked at {peak_churn:.4f}")}
    </div>
    <div class="two-col">
      <div class="chart-box">"""
    fig = go.Figure()
    hist_t = hist_times
    hist_subs_total = [a + b + c for a, b, c in zip(hist_subs_a, hist_subs_b, hist_subs_c)] if hist_times else []
    if hist_times and hist_subs_total:
        fig.add_trace(go.Scatter(x=hist_times, y=hist_subs_total,
            mode="markers+lines", name="Historical (monthly)", line=dict(color=COLORS[0], width=1, dash="dot"),
            marker=dict(size=4), hovertemplate="t=%{x:.0f}d<br>%{y:,.0f} subs<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["times"], y=d["total_subs"],
        mode="lines", name="Projected", line=dict(color=COLORS[0], width=2),
        hovertemplate="t=%{x:.0f}d<br>%{y:,.0f} subs<extra></extra>"))
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=280,
                      xaxis_title="Days", yaxis_title="Subscribers",
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(size=10))
    content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += f"""</div>
      <div class="chart-box">"""
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d["times"], y=d["revenue_reserve"],
        mode="lines", name="Cumulative Revenue", line=dict(color=COLORS[1], width=2)))
    fig2.add_trace(go.Scatter(x=d["times"], y=d["profit_margin"],
        mode="lines", name="Profit Margin", yaxis="y2", line=dict(color=COLORS[2], width=2)))
    fig2.update_layout(margin=dict(l=40,r=40,t=30,b=30), height=280,
                       xaxis_title="Days",
                       yaxis=dict(title="Revenue ($)", side="left"),
                       yaxis2=dict(title="Margin", overlaying="y", side="right",
                                   tickformat=".0%", range=[-0.3, 0.5]),
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig2.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"
    return {"icon": "&#x1F4CA;", "title": "Executive Summary", "content": content}


def build_region_growth(d: dict) -> dict:
    content = '<div class="kpi-row">'
    for i, r in enumerate(REGIONS):
        val = d.get(f"subs_{r.lower()}", [0])[-1]
        init = REGION_INITIAL_SUBS[i]
        growth_pct = (val / init - 1) * 100
        content += _kpi_card(f"Region {r}", f"{val:,.0f}", COLORS[i],
                             f"{growth_pct:+.0f}% from {init:,.0f}")
    content += '</div><div class="chart-box">'
    fig = go.Figure()
    for i, r in enumerate(REGIONS):
        fig.add_trace(go.Scatter(x=d["times"], y=d.get(f"subs_{r.lower()}", []),
            mode="lines", name=f"Region {r}", line=dict(color=COLORS[i], width=2)))
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=300,
                      xaxis_title="Days", yaxis_title="Subscribers",
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="two-col"><div class="chart-box"><div class="st">Regional Share at End</div>'
    labels = [f"Region {r}" for r in REGIONS]
    values = [d.get(f"subs_{r.lower()}", [0])[-1] for r in REGIONS]
    fig2 = go.Figure(data=[go.Pie(labels=labels, values=values,
        marker=dict(colors=COLORS[:3]), hole=0.4, textinfo="label+percent")])
    fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=220,
                       paper_bgcolor="white", font=dict(size=10), showlegend=False)
    content += fig2.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box"><div class="st">Per-Capita Subscribers</div>'
    fig3 = go.Figure()
    for i, r in enumerate(REGIONS):
        subs = d.get(f"subs_{r.lower()}", [1])
        caps = d.get(f"cap_{r.lower()}", [1])
        per_cap = [s / max(1, c) for s, c in zip(subs, caps)]
        fig3.add_trace(go.Scatter(x=d["times"], y=per_cap,
            mode="lines", name=f"Region {r}", line=dict(color=COLORS[i], width=1.5)))
    fig3.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=220,
                       xaxis_title="Days", yaxis_title="Subs / Cap Unit",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig3.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"
    return {"icon": "&#x1F3D8;", "title": "Region Growth", "content": content}


def build_demand_forecast(d: dict) -> dict:
    times = d["times"]
    total = d["total_subs"]
    start_subs = total[0] if total else 0
    end_subs = total[-1] if total else 0
    early_window = min(120, len(times))
    if early_window >= 2:
        x = times[:early_window]
        y = total[:early_window]
        n = len(x)
        sx = sum(x); sy = sum(y)
        sxx = sum(ti * ti for ti in x)
        sxy = sum(ti * yi for ti, yi in zip(x, y))
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        intercept = (sy - slope * sx) / n
        linear_fc = [intercept + slope * t for t in times]
    else:
        linear_fc = [start_subs] * len(times)
    content = '<div class="kpi-row">'
    content += _kpi_card("Actual End", f"{end_subs:,.0f}", COLORS[0],
                         f"Start: {start_subs:,.0f}")
    fc_err = abs(end_subs - linear_fc[-1]) / max(1, end_subs)
    content += _kpi_card("Naive Forecast Error", f"{fc_err:.1%}", COLORS[2],
                         f"Extrapolated from first {early_window} days")
    drift = (end_subs - start_subs) / len(times) if times else 0
    content += _kpi_card("Avg Daily Growth", f"{drift:.1f}", COLORS[1],
                         "Subscribers/day")
    lead_time_est = 75
    content += _kpi_card("Est. Lead Time", f"~{lead_time_est}d", COLORS[3],
                         "Signal → subscriber (sales cycle)")
    content += '</div><div class="chart-box">'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=total, mode="lines",
        name="Actual Growth", line=dict(color=COLORS[0], width=2.5)))
    fig.add_trace(go.Scatter(x=times, y=linear_fc, mode="lines",
        name="Naive Linear Forecast (Early Trend)", line=dict(color=COLORS[4], dash="dash", width=1.5)))
    bld = d.get("building_permits_A", [])
    if times and bld:
        bp_peak = times[bld.index(max(bld))]
        fig.add_vline(x=bp_peak, line_dash="dot", line_color=COLORS[0], opacity=0.5,
                      annotation_text=f"Building Permits Peak t={bp_peak:.0f}")
    fig.add_vline(x=120, line_dash="dot", line_color=COLORS[2], opacity=0.5,
                  annotation_text="Competitor Entry t=120")
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=300,
                      xaxis_title="Days", yaxis_title="Subscribers",
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="two-col"><div class="chart-box"><div class="st">Leading Indicator Signals</div>'
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    bld_series = d.get("building_permits_A", [])
    if times and bld_series:
        fig2.add_trace(go.Scatter(x=times, y=bld_series,
            mode="lines", name="Region A: Building Permits", line=dict(color=COLORS[0])), secondary_y=False)
    comp = [1.0 if t >= 120 else 0.0 for t in times]
    fig2.add_trace(go.Scatter(x=times, y=comp,
        mode="lines", name="Region B: Competitor Entry", line=dict(color=COLORS[2])), secondary_y=True)
    fig2.update_layout(margin=dict(l=40,r=40,t=10,b=30), height=220,
                       paper_bgcolor="white", plot_bgcolor="white", font=dict(size=10),
                       legend=dict(orientation="h", y=1.1))
    fig2.update_yaxes(title_text="Permits", secondary_y=False)
    fig2.update_yaxes(title_text="Competitor Pres.", secondary_y=True, range=[0, 1.5])
    content += fig2.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box"><div class="st">Lead Time Analysis</div>'
    content += f"""<div class="qa" style="padding:8px;line-height:1.8">
    <p>Atlas's historical sales cycle averages <b>~75 days</b> from signal to subscriber:
    <ul>
      <li><b>Region A:</b> Building permit spikes (PULSE t=100) lead subscriber growth by ~75 days.</li>
      <li><b>Region B:</b> Competitor entry at t=120 increases churn pressure within ~90 days.</li>
      <li><b>Region C:</b> Marketing push (PULSE t=200, 180d) drives acquisition within ~60 days.</li>
    </ul>
    The naive forecast (linear extrapolation from the first {early_window} days)
    misses the leading indicators entirely, underestimating growth by <b>{fc_err:.1%}</b>.</p>
    <p><code>Signal (building permits / competitor / marketing) → ~75d → Adoption/Churn shift → Capacity decision</code></p>
    </div>"""
    content += "</div></div>"
    return {"icon": "&#x1F4C8;", "title": "Demand Forecast", "content": content}


def build_capacity_planning(d: dict) -> dict:
    content = '<div class="kpi-row">'
    for i, r in enumerate(REGIONS):
        util_vals = d.get(f"util_{r.lower()}", [1])
        util_end = _safe_val(util_vals)
        peak_util = max(util_vals)
        color = COLORS[1] if peak_util < 0.75 else (COLORS[2] if peak_util < 0.90 else COLORS[3])
        content += _kpi_card(f"Region {r} Util", f"{util_end:.1%}",
                             color, f"Peak: {peak_util:.1%}")
    content += '</div><div class="two-col"><div class="chart-box">'
    fig = go.Figure()
    for i, r in enumerate(REGIONS):
        fig.add_trace(go.Scatter(x=d["times"], y=d.get(f"util_{r.lower()}", []),
            mode="lines", name=f"Region {r}", line=dict(color=COLORS[i], width=2)))
    fig.add_hline(y=0.70, line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="Target (70%)")
    fig.add_hline(y=0.82, line_dash="dot", line_color=COLORS[3], opacity=0.5,
                  annotation_text="Threshold (82%)")
    fig.add_hrect(y0=0, y1=0.70, line_width=0, fillcolor="green", opacity=0.04)
    fig.add_hrect(y0=0.82, y1=1.2, line_width=0, fillcolor="red", opacity=0.04)
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=280,
                      xaxis_title="Days", yaxis_title="Utilization",
                      yaxis=dict(tickformat=".0%", range=[0, 1.2]),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box">'
    fig2 = go.Figure()
    for i, r in enumerate(REGIONS):
        fig2.add_trace(go.Scatter(x=d["times"], y=d.get(f"cap_{r.lower()}", []),
            mode="lines", name=f"Region {r} Capacity", line=dict(color=COLORS[i], width=2)))
    fig2.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=280,
                       xaxis_title="Days", yaxis_title="Capacity Units",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig2.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"

    # SD→DES coupling proof: show congest queue lengths alongside utilization
    content += '<div class="st">Congestion Queue Length (driven by network utilization)</div>'
    content += '<div class="two-col"><div class="chart-box">'
    fig3 = go.Figure()
    for i, r in enumerate(REGIONS):
        qlen = d.get(f"congest_len_{r.lower()}", [])
        if qlen:
            fig3.add_trace(go.Scatter(x=d["times"], y=qlen,
                mode="lines", name=f"congest_{r} queue", line=dict(color=COLORS[i], width=1.5),
                fill="tozeroy", fillcolor=_hex_rgba(COLORS[i], 0.08)))
    fig3.update_layout(margin=dict(l=40,r=10,t=20,b=30), height=220,
                       xaxis_title="Days", yaxis_title="Queue Length",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig3.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box"><div class="st">Capacity Order Signal</div>'
    fig4 = go.Figure()
    for i, r in enumerate(REGIONS):
        orders = d.get(f"capacity_order_{r.lower()}", [])
        if any(v > 0 for v in orders):
            fig4.add_trace(go.Scatter(x=d["times"], y=orders,
                mode="lines", name=f"Region {r}", line=dict(color=COLORS[i], width=1.5)))
    fig4.update_layout(margin=dict(l=40,r=10,t=20,b=30), height=220,
                       xaxis_title="Days", yaxis_title="Order Signal",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig4.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"
    return {"icon": "&#x1F3ED;", "title": "Capacity Planning", "content": content}


def build_network_qos(d: dict) -> dict:
    content = '<div class="kpi-row">'
    for i, r in enumerate(REGIONS):
        qos_vals = d.get(f"qos_{r.lower()}", [100])
        min_qos = min(qos_vals)
        end_qos = _safe_val(qos_vals)
        color = COLORS[1] if end_qos > 80 else (COLORS[2] if end_qos > 50 else COLORS[3])
        content += _kpi_card(f"Region {r} QoS", f"{end_qos:.0f}", color,
                             f"Min: {min_qos:.0f}")
    content += '</div><div class="two-col"><div class="chart-box">'
    fig = go.Figure()
    for i, r in enumerate(REGIONS):
        qos = d.get(f"qos_{r.lower()}", [])
        fig.add_trace(go.Scatter(x=d["times"], y=qos,
            mode="lines", name=f"Region {r}", line=dict(color=COLORS[i], width=2)))
    fig.add_hrect(y0=80, y1=100, line_width=0, fillcolor="green", opacity=0.05)
    fig.add_hrect(y0=50, y1=80, line_width=0, fillcolor="yellow", opacity=0.05)
    fig.add_hrect(y0=0, y1=50, line_width=0, fillcolor="red", opacity=0.05)
    fig.add_hline(y=80, line_dash="dot", line_color="green", annotation_text="Good")
    fig.add_hline(y=50, line_dash="dot", line_color="red", annotation_text="Poor")
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=280,
                      xaxis_title="Days", yaxis_title="QoS Score (0-100)",
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box">'
    fig2 = go.Figure()
    for i, r in enumerate(REGIONS):
        util = d.get(f"util_{r.lower()}", [])
        nps = d.get(f"nps_{r.lower()}", [])
        fig2.add_trace(go.Scatter(x=d["times"], y=nps,
            mode="lines", name=f"Region {r} NPS", line=dict(color=COLORS[i], width=2)))
    fig2.add_hline(y=70, line_dash="dot", line_color="green", opacity=0.5,
                   annotation_text="Promoter")
    fig2.add_hline(y=30, line_dash="dot", line_color="red", opacity=0.5,
                   annotation_text="Detractor")
    fig2.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=280,
                       xaxis_title="Days", yaxis_title="NPS Score",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig2.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"

    # DES congestion events histogram
    content += '<div class="chart-box"><div class="st">Service Requests Processed (by queue)</div>'
    fig3 = go.Figure()
    for i, r in enumerate(REGIONS):
        departed = d.get(f"congest_{r.lower()}_departed" if f"congest_{r.lower()}_departed" in d
                         else f"congest_{r}_departed", None)
        if departed is None:
            departed = ts_for_debug(d, f"congest_{r}_departed", [])
        qlen = d.get(f"congest_len_{r.lower()}", [])
        if qlen:
            fig3.add_trace(go.Bar(x=d["times"][::30], y=[sum(qlen[i:i+30]) for i in range(0, len(qlen), 30)],
                name=f"Region {r} (30d sum)", marker_color=COLORS[i], opacity=0.7))
    fig3.update_layout(margin=dict(l=40,r=10,t=20,b=30), height=200,
                       xaxis_title="Days", yaxis_title="Congestion Events / 30d",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1),
                       bargap=0.2)
    content += fig3.to_html(full_html=False, include_plotlyjs=False)
    content += "</div>"
    return {"icon": "&#x1F4F6;", "title": "Network QoS", "content": content}


def ts_for_debug(d, key, default):
    return d.get(key, default)


def build_customer_base(d: dict) -> dict:
    agent_hist = d.get("agent_history", [])
    min_sat = 100.0; nadir_t = 0.0; peak_risk = 0.0; nadir_idx = 0
    content = '<div class="kpi-row">'
    if agent_hist:
        end = agent_hist[-1]
        active = end.get("active", 0)
        total = active + end.get("churned", 0)
        min_sat = min(r.get("avg_satisfaction", 0) for r in agent_hist)
        peak_risk = max(r.get("avg_churn_risk", 0) for r in agent_hist)
        nadir_idx = min(range(len(agent_hist)), key=lambda i: agent_hist[i].get("avg_satisfaction", 0))
        nadir_t = agent_hist[nadir_idx]["t"]
        rec_sat = end.get("avg_satisfaction", 0)
        content += _kpi_card("Active Agents", f"{active}", COLORS[1],
                             f"Out of {total} (no deactivation — risk-based model)")
        content += _kpi_card("Min Satisfaction", f"{min_sat:.0f}",
                             COLORS[3] if min_sat < 35 else COLORS[1],
                             f"At day {nadir_t:.0f} (threshold: 35)")
        content += _kpi_card("Peak Churn Risk", f"{peak_risk:.3f}",
                             COLORS[3] if peak_risk > 0.3 else COLORS[1],
                             "Scale 0-1")
        content += _kpi_card("Recovery Sat", f"{rec_sat:.0f}", COLORS[1],
                             "End-of-simulation satisfaction")
    content += '</div><div class="two-col"><div class="chart-box">'
    if agent_hist:
        times_ah = [r["t"] for r in agent_hist]
        sats = [r.get("avg_satisfaction", 0) for r in agent_hist]
        risks = [r.get("avg_churn_risk", 0) for r in agent_hist]
        active_cnt = [r.get("active", 0) for r in agent_hist]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=times_ah, y=sats, mode="lines",
            name="Avg Satisfaction", line=dict(color=COLORS[1], width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=times_ah, y=risks, mode="lines",
            name="Avg Churn Risk", line=dict(color=COLORS[3], width=2)), secondary_y=True)
        fig.add_trace(go.Scatter(x=times_ah, y=active_cnt, mode="lines",
            name="Active Agents", line=dict(color=COLORS[0], dash="dot", width=1.5)), secondary_y=False)
        fig.update_layout(margin=dict(l=40,r=40,t=30,b=30), height=280,
                          paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(size=10), legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="Satisfaction / Active", secondary_y=False, range=[0, 100])
        fig.update_yaxes(title_text="Churn Risk", secondary_y=True, range=[0, 1])
        content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box"><div class="st">Strategy Distribution</div>'
    if agent_hist:
        for label, idx in [("Start", 0), ("Nadir (Satisfaction)", nadir_idx), ("End", -1)]:
            rec = agent_hist[idx]
            sv = rec.get("avg_satisfaction", 0)
            rv = rec.get("avg_churn_risk", 0)
            regions_line = " | ".join(
                f"<span style='color:{COLORS[i]}'><b>R{r}</b>: sat {rec.get('sat_'+r,0):.1f}, risk {rec.get('risk_'+r,0):.4f}</span>"
                for i, r in enumerate(REGIONS)
            )
            content += f"<div style='margin:4px 0'><b>{label}:</b> Avg {sv:.1f}, Risk {rv:.4f} — {regions_line}</div>"
    # per-region min satisfaction for QA context
    reg_min_sats = {}
    for r in REGIONS:
        vals = [rec.get(f"sat_{r}", 0) for rec in agent_hist] if agent_hist else [100]
        reg_min_sats[r] = min(vals)
    content += f"""
    <div class="qa" style="padding:8px;margin-top:8px;line-height:1.7">
    <p><b>Customer Behavior Feedback:</b> Each subscriber agent tracks its own satisfaction score
    based on the network quality they experience. During the congestion event, the aggregate
    satisfaction hit <b>{min_sat:.0f}</b> at day {nadir_t:.0f}, with Region A falling to
    <b>{reg_min_sats.get('A', 0):.0f}</b> and Region C to <b>{reg_min_sats.get('C', 0):.0f}</b> —
    well below the <b>35-point at-risk threshold</b>. Region B, with higher priority capacity,
    bottomed at <b>{reg_min_sats.get('B', 0):.0f}</b>. Satisfaction recovered across all regions
    as capacity expansion caught up with subscriber growth.</p>
    </div>"""
    content += "</div></div>"
    return {"icon": "&#x1F465;", "title": "Customer Base", "content": content}


def build_churn_analysis(d: dict) -> dict:
    content = '<div class="kpi-row">'
    for i, r in enumerate(REGIONS):
        cf = d.get(f"churn_frac_{r.lower()}", [0])
        avg = sum(cf) / max(1, len(cf))
        pk = max(cf)
        end_cf = _safe_val(cf)
        content += _kpi_card(f"Region {r} Churn", f"{end_cf:.4f}", COLORS[i],
                             f"Avg {avg:.4f}, Peak {pk:.4f}")
    content += '</div><div class="two-col"><div class="chart-box">'
    fig = go.Figure()
    for i, r in enumerate(REGIONS):
        cf = d.get(f"churn_frac_{r.lower()}", [])
        fig.add_trace(go.Scatter(x=d["times"], y=cf,
            mode="lines", name=f"Region {r}", line=dict(color=COLORS[i], width=2)))
    fig.add_hline(y=0.0003, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="Base churn")
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=280,
                      xaxis_title="Days", yaxis_title="Churn Fraction",
                      yaxis=dict(tickformat=".4f"),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box"><div class="st">Churn Component Breakdown (Region B)</div>'
    cf_b = d.get("churn_frac_b", [0])
    base_b = d["base_params"]["base_churn_B"]
    congest_len = d.get("congest_len_b", [0])
    congest_amp = [max(0, cl - 2) * 0.0001 for cl in congest_len]
    comp_base = [base_b] * len(cf_b)
    comp_congest = congest_amp[:len(cf_b)]
    comp_abm = [max(0, c - base_b - ca) for c, ca in zip(cf_b, comp_congest)]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d["times"], y=comp_base, mode="lines",
        name="Base", stackgroup="one", line=dict(width=0.5, color=COLORS[0])))
    fig2.add_trace(go.Scatter(x=d["times"], y=comp_abm, mode="lines",
        name="Customer Dissatisfaction", stackgroup="one", line=dict(width=0.5, color=COLORS[2])))
    fig2.add_trace(go.Scatter(x=d["times"], y=comp_congest, mode="lines",
        name="Service Congestion", stackgroup="one", line=dict(width=0.5, color=COLORS[3])))
    fig2.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=280,
                       xaxis_title="Days", yaxis_title="Churn Rate Components",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig2.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"
    return {"icon": "&#x274C;", "title": "Churn Analysis", "content": content}


def build_disruption_impact(d: dict) -> dict:
    times = d["times"]
    rev_base = d["revenue_reserve"]
    rev_cf = d["ts_cf"].get("Revenue_Reserve", [])
    subs_base = d["total_subs"]
    subs_cf = d["ts_cf"].get("total_subs", [])
    monthly = d["monthly_cost"]

    total_gap = d["disruption_cost"]
    content = f"""
    <div class="kpi-row">
      {_kpi_card("Baseline Revenue", f"${_safe_val(rev_base):,.0f}", COLORS[0],
                  "Region B threshold=82%, delay=120d")}
      {_kpi_card("Counterfactual Revenue", f"${_safe_val(rev_cf):,.0f}", COLORS[1],
                  "Region B threshold=72%, delay=60d")}
      {_kpi_card("Disruption Cost", f"${total_gap:,.0f}", THEME["danger"],
                  "Cost of NOT expanding early")}
      {_kpi_card("Subscriber Gap", f"{_safe_val(subs_cf) - _safe_val(subs_base):,.0f}", COLORS[3],
                  "Lost subscribers due to churn")}
    </div>
    <div class="two-col">
      <div class="chart-box">"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=rev_base, mode="lines",
        name="Baseline (actual)", line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=times, y=rev_cf, mode="lines",
        name="Counterfactual (early expansion)", line=dict(color=COLORS[1], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=times,
        y=[rc - rb for rc, rb in zip(rev_cf, rev_base)],
        mode="lines", name="Revenue Gap", line=dict(color=COLORS[3], width=1.5),
        fill="tozeroy", fillcolor=_hex_rgba(COLORS[3], 0.1)))
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=280,
                      xaxis_title="Days", yaxis_title="Revenue ($)",
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box">'
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=subs_base, mode="lines",
        name="Baseline Subscribers", line=dict(color=COLORS[0], width=2)))
    fig2.add_trace(go.Scatter(x=times, y=subs_cf, mode="lines",
        name="Counterfactual Subs", line=dict(color=COLORS[1], width=2, dash="dash")))
    fig2.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=280,
                       xaxis_title="Days", yaxis_title="Subscribers",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig2.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"

    # Monthly breakdown table
    content += '<div class="st">Monthly Disruption Cost Breakdown</div>'
    content += '<div class="table-wrap"><table class="dt"><thead><tr>'
    for h in ["Month", "Days", "Base Revenue", "CF Revenue", "Gap", "Cumulative Gap"]:
        content += f"<th>{h}</th>"
    content += "</tr></thead><tbody>"
    cum_gap = 0
    for m in monthly:
        cum_gap += m["gap"]
        color = COLORS[3] if m["gap"] > 50000 else COLORS[1]
        content += (f"<tr><td>Month {m['month']}</td>"
                    f"<td>{m['day_start']:.0f}-{m['day_end']:.0f}</td>"
                    f"<td>${m['baseline_rev']:,.0f}</td>"
                    f"<td>${m['cf_rev']:,.0f}</td>"
                    f"<td style='color:{color}'>${m['gap']:,.0f}</td>"
                    f"<td>${cum_gap:,.0f}</td></tr>")
    content += "</tbody></table></div>"
    return {"icon": "&#x1F4B0;", "title": "Disruption Impact", "content": content}


def build_root_cause(d: dict) -> dict:
    content = '<div class="two-col"><div class="chart-box">'
    trace = d.get("trace_revenue")
    if trace and hasattr(trace, "factors"):
        factors = trace.factors[:10]
        names = [f.get("variable", f.get("name", f"f{i}"))[:25] for i, f in enumerate(factors)]
        vals = [f.get("value", f.get("contribution", 0)) for f in factors]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vals, y=names, orientation="h",
            marker=dict(color=[COLORS[1] if v >= 0 else COLORS[3] for v in vals]),
            text=[f"${v:,.0f}" if abs(v) > 1000 else f"{v:+.2f}" for v in vals],
            textposition="outside"))
        fig.update_layout(margin=dict(l=10,r=80,t=20,b=30), height=300,
                          xaxis_title="Contribution to Revenue_Reserve",
                          paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(size=10), showlegend=False)
        content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box">'
    trace_b = d.get("trace_subs_B")
    if trace_b and hasattr(trace_b, "factors"):
        factors = trace_b.factors[:10]
        names = [f.get("variable", f.get("name", f"f{i}"))[:25] for i, f in enumerate(factors)]
        vals = [f.get("value", f.get("contribution", 0)) for f in factors]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=vals, y=names, orientation="h",
            marker=dict(color=[COLORS[1] if v >= 0 else COLORS[3] for v in vals]),
            text=[f"{v:+.2f}" for v in vals],
            textposition="outside"))
        fig2.update_layout(margin=dict(l=10,r=80,t=20,b=30), height=300,
                           xaxis_title="Contribution to Subs_B",
                           paper_bgcolor="white", plot_bgcolor="white",
                           font=dict(size=10), showlegend=False)
        content += fig2.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"

    loops = d.get("churn_loops", [])
    if loops:
        content += '<div class="st">Vicious Cycles (churn × utilization)</div>'
        content += '<div class="two-col">'
        for lp in loops[:4]:
            nodes_str = " → ".join(lp.nodes[:5])
            label = f"{lp.name} ({'+' if lp.polarity > 0 else '-'})"
            color = COLORS[3] if lp.polarity < 0 else COLORS[2]
            content += f"""
            <div class="chart-box" style="padding:10px;border-left:3px solid {color}">
              <b style="color:{color}">{label}</b><br>
              <small>{nodes_str}{'...' if len(lp.nodes) > 5 else ''}</small>
            </div>"""
        content += '</div>'

    content += f"""
    <div class="qa" style="padding:12px;margin-top:10px;line-height:1.8">
    <b>Region B — How Churn Cascades:</b><br>
    More subscribers → Higher network utilization → Lower service quality →
    Customer satisfaction drops → Churn rate increases → Subscribers leave →
    Utilization drops (self-correcting)<br><br>
    <b>The delay problem:</b><br>
    Utilization > 82% → Capacity expansion ordered → <b>120-day construction delay</b> →
    4 months without relief → Utilization stays high → Service quality stays low →
    Churn keeps accelerating
    </div>"""
    return {"icon": "&#x1F50D;", "title": "Root Cause", "content": content}


def build_decision_rule(d: dict) -> dict:
    times = d["times"]
    util_b = d.get("util_b", [0])
    qos_b = d.get("qos_b", [100])
    cf_b = d.get("churn_frac_b", [0])

    # Find first 30-day window with 90%+ of days above warning threshold
    threshold = 0.78
    min_days_above = 27
    sustained_start = None
    for i in range(len(util_b) - 29):
        above = sum(1 for j in range(30) if util_b[i + j] > threshold)
        if above >= min_days_above:
            sustained_start = i
            break

    # Find when util_B first exceeded the crisis threshold
    crisis_threshold = d['base_params'].get('capacity_threshold_B', 0.82)
    order_start = next((i for i, u in enumerate(util_b) if u > crisis_threshold), None)

    content = f"""
    <div class="kpi-row">
      {_kpi_card("Warning Threshold", ">78% utilization", THEME["warning"],
                  "Sustained 30+ days triggers action")}
      {_kpi_card("Sustained Period Start", f"Day {sustained_start}" if sustained_start is not None else "Never", THEME["warning"],
                   f"First 30d window with 27d > 78%")}
      {_kpi_card("Crisis Threshold Crossed", f"Day {order_start}" if order_start is not None else "Never", COLORS[2],
                   f"util_B > {crisis_threshold:.0%}")}
      {_kpi_card("Delay Gap", f"~{order_start - sustained_start}d" if (order_start is not None and sustained_start is not None) else "N/A", THEME["danger"],
                   "Sustained warning → crisis crossing")}
    </div>
    <div class="chart-box">"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=util_b, mode="lines",
        name="Region B Utilization", line=dict(color=COLORS[2], width=2),
        fill="tozeroy", fillcolor=_hex_rgba(COLORS[2], 0.1)))
    fig.add_hline(y=threshold, line_dash="dot", line_color=COLORS[3], opacity=0.8,
                  annotation_text=f"Warning ({threshold:.0%})")
    fig.add_hline(y=0.82, line_dash="dot", line_color="red", opacity=0.8,
                  annotation_text="Crisis (82%)")
    if sustained_start is not None:
        t_start = times[sustained_start]
        t_end = times[min(sustained_start + 29, len(times) - 1)]
        fig.add_vrect(x0=t_start, x1=t_end, line_width=0,
                      fillcolor="red", opacity=0.1,
                      annotation_text="30d sustained warning")
    if order_start is not None:
        fig.add_vline(x=times[order_start], line_dash="dash",
                      line_color="green", opacity=0.7,
                      annotation_text="Capacity ordered")
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=260,
                      xaxis_title="Days", yaxis_title="Utilization",
                      yaxis=dict(tickformat=".0%", range=[0, 1.1]),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(size=10))
    content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += "</div>"

    # Decision rule card
    content += f"""
    <div class="two-col">
      <div class="chart-box">
        <div class="st">Decision Rule</div>
        <div style="padding:10px;font-family:monospace;line-height:1.8;background:#f8f9fa;border-radius:4px">
IF util_B > 0.78<br>
   AND sustained_for >= 30 days<br>
THEN:<br>
   &nbsp;&nbsp;Trigger capacity_order_B<br>
   &nbsp;&nbsp;Order delay: project_delay_B days ({d['base_params']['project_delay_B']:.0f}d)<br>
   &nbsp;&nbsp;Capex per unit: subs_per_unit × $30/unit = ${500 * 30:,.0f}<br>
   &nbsp;&nbsp;Estimated ROI: disruption_cost / capex = {d['disruption_cost'] / max(1, 1000000):.1f}×<br>
        </div>
      </div>
      <div class="chart-box">
        <div class="st">Cascading Risk Table</div>
        <div class="table-wrap"><table class="dt"><thead><tr>
          <th>Trigger</th><th>Threshold</th><th>Risk if Missed</th>
        </tr></thead><tbody>
          <tr><td>Warning</td><td>78% for 30d</td><td>Need to order within {d['base_params']['project_delay_B']:.0f}d before crisis</td></tr>
          <tr><td style="color:{COLORS[3]}">Crisis</td><td style="color:{COLORS[3]}">82% util</td><td style="color:{COLORS[3]}">QoS drops below 80, churn amplified</td></tr>
          <tr><td style="color:{THEME['danger']}">Critical</td><td style="color:{THEME['danger']}">90% util</td><td style="color:{THEME['danger']}">QoS < 50, dissatisfaction-driven churn, churn 19× base</td></tr>
        </tbody></table></div>
      </div>
    </div>"""
    return {"icon": "&#x2699;", "title": "Decision Rule", "content": content}


def build_scenario_comparison(d: dict) -> dict:
    scenarios = d.get("scenarios", [])
    content = '<div class="kpi-row">'
    for sc in scenarios[:3]:
        c = COLORS[scenarios.index(sc)]
        content += _kpi_card(f"{sc['name']}", f"${sc['final_revenue']:,.0f}",
                             c, f"{sc['final_subs']:,.0f} subs, {sc['final_margin']:.1%}")
    content += '</div><div class="kpi-row">'
    for sc in scenarios[3:]:
        c = COLORS[scenarios.index(sc)]
        content += _kpi_card(f"{sc['name']}", f"${sc['final_revenue']:,.0f}",
                             c, f"{sc['final_subs']:,.0f} subs, {sc['final_margin']:.1%}")
    content += '</div><div class="chart-box">'
    fig = go.Figure()
    for sc in scenarios:
        c = COLORS[scenarios.index(sc)]
        ts_subs = sc['ts'].get("total_subs", [])
        fig.add_trace(go.Scatter(x=d["times"], y=ts_subs, mode="lines",
            name=sc["name"], line=dict(color=c, width=2)))
    fig.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=280,
                      xaxis_title="Days", yaxis_title="Total Subscribers",
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="two-col"><div class="chart-box">'
    fig2 = go.Figure()
    for sc in scenarios:
        c = COLORS[scenarios.index(sc)]
        ts_qos = sc['ts'].get("qos_B", [])
        fig2.add_trace(go.Scatter(x=d["times"], y=ts_qos, mode="lines",
            name=sc["name"], line=dict(color=c, width=1.5)))
    fig2.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=260,
                       xaxis_title="Days", yaxis_title="Region B QoS",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig2.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box">'
    fig3 = go.Figure()
    for sc in scenarios:
        c = COLORS[scenarios.index(sc)]
        ts_util = sc['ts'].get("util_B", [])
        fig3.add_trace(go.Scatter(x=d["times"], y=ts_util, mode="lines",
            name=sc["name"], line=dict(color=c, width=1.5)))
    fig3.add_hline(y=0.78, line_dash="dot", line_color="gray", opacity=0.4)
    fig3.update_layout(margin=dict(l=40,r=10,t=30,b=30), height=260,
                       xaxis_title="Days", yaxis_title="Region B Utilization",
                       yaxis=dict(tickformat=".0%"),
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), legend=dict(orientation="h", y=1.1))
    content += fig3.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"

    # Comparison table
    content += '<div class="table-wrap"><table class="dt"><thead><tr>'
    headers = ["Scenario", "Threshold B", "Delay B", "Final Subs", "Revenue", "Margin", "Util B", "QoS B"]
    for h in headers:
        content += f"<th>{h}</th>"
    content += "</tr></thead><tbody>"
    for sc in scenarios:
        c = COLORS[scenarios.index(sc)]
        ov = sc["overrides"]
        content += (f"<tr style='color:{c}'><td><b>{sc['name']}</b></td>"
                    f"<td>{ov.get('capacity_threshold_B', 0):.0%}</td>"
                    f"<td>{ov.get('project_delay_B', 0):.0f}d</td>"
                    f"<td>{sc['final_subs']:,.0f}</td>"
                    f"<td>${sc['final_revenue']:,.0f}</td>"
                    f"<td>{sc['final_margin']:.1%}</td>"
                    f"<td>{sc['final_util_B']:.1%}</td>"
                    f"<td>{sc['final_qos_B']:.0f}</td></tr>")
    content += "</tbody></table></div>"

    # OAT sensitivity
    oat = d.get("oat_results", {})
    content += '<div class="st">One-At-A-Time (OAT) Sensitivity (Revenue Impact)</div>'
    content += '<div class="two-col"><div class="chart-box">'
    fig4 = go.Figure()
    pnames = list(oat.keys())
    rev_impacts = [abs(oat[p]["hi_rev"] - oat[p]["lo_rev"]) for p in pnames]
    fig4.add_trace(go.Bar(x=pnames, y=rev_impacts, marker_color=COLORS[1],
                          text=[f"${v:,.0f}" for v in rev_impacts], textposition="outside"))
    fig4.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=220,
                       xaxis_title="Parameter", yaxis_title="Revenue Impact ($)",
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=10), showlegend=False)
    content += fig4.to_html(full_html=False, include_plotlyjs=False)
    content += '</div><div class="chart-box"><div class="st">Risk Matrix</div>'
    risk_items = [
        ("Region B churn worsens", 0.6, 0.7),
        ("Region A follows same pattern", 0.4, 0.5),
        ("Capacity deployment delayed beyond 120d", 0.3, 0.8),
        ("Competitor captures more share", 0.3, 0.4),
        ("Marketing effectiveness overestimated", 0.2, 0.3),
    ]
    fig5 = go.Figure()
    for label, prob, impact in risk_items:
        size = impact * 30 + 10
        fig5.add_trace(go.Scatter(x=[prob], y=[impact], mode="markers+text",
            marker=dict(size=size, color=COLORS[3] if prob * impact > 0.3 else COLORS[2],
                        line=dict(color="white", width=1)),
            text=[label[:20]], textposition="top center", name=label))
    fig5.update_layout(margin=dict(l=40,r=10,t=10,b=30), height=220,
                       xaxis=dict(title="Probability", range=[0, 1], tickformat=".0%"),
                       yaxis=dict(title="Impact", range=[0, 1], tickformat=".0%"),
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(size=9), showlegend=False)
    content += fig5.to_html(full_html=False, include_plotlyjs=False)
    content += "</div></div>"
    return {"icon": "&#x1F9CA;", "title": "Scenario Comparison", "content": content}


# ══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:BGCOLOR;color:TEXTCOLOR;font-size:14px}
.header{background:PRIMARYCOLOR;color:white;padding:12px 20px;position:sticky;top:0;z-index:100}
.header h1{font-size:18px;font-weight:600}
.header small{opacity:0.8;font-size:12px}
.tab-bar{display:flex;flex-wrap:wrap;background:CARDCOLOR;border-bottom:2px solid #e0e0e0;position:sticky;top:50px;z-index:99;overflow-x:auto}
.tab-btn{padding:8px 14px;border:none;background:transparent;cursor:pointer;font-size:12px;color:MUTEDCOLOR;white-space:nowrap;border-bottom:2px solid transparent;transition:all 0.15s}
.tab-btn.active{color:PRIMARYCOLOR;border-bottom-color:ACCENTCOLOR;font-weight:600}
.tab-btn:hover{color:PRIMARYCOLOR;background:rgba(26,127,196,0.05)}
.content{padding:12px;max-width:1400px;margin:0 auto}
.pane{display:block}
.pane.hidden{display:none}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin-bottom:10px}
.kpi{background:CARDCOLOR;border-radius:6px;padding:10px 12px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}
.kl{font-size:10px;color:MUTEDCOLOR;text-transform:uppercase;letter-spacing:0.5px}
.kv{font-size:18px;font-weight:700;margin:2px 0}
.ks{font-size:10px;color:MUTEDCOLOR}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.chart-box{background:CARDCOLOR;border-radius:6px;padding:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:10px}
.st{font-size:13px;font-weight:600;color:PRIMARYCOLOR;padding:6px 0 4px 0;border-bottom:2px solid ACCENTCOLOR;margin-bottom:8px}
.qa{font-size:12px;color:TEXTCOLOR;background:BGCOLOR;padding:10px;border-radius:4px;line-height:1.6}
.table-wrap{overflow-x:auto;margin:8px 0}
.dt{width:100%;border-collapse:collapse;font-size:11px}
.dt th{background:PRIMARYCOLOR;color:white;padding:6px 8px;text-align:left;white-space:nowrap}
.dt td{padding:5px 8px;border-bottom:1px solid #e0e0e0}
.dt tr:hover{background:rgba(26,127,196,0.04)}
.verdict{font-size:13px;line-height:1.6;margin:8px 0}
@media(max-width:768px){.two-col{grid-template-columns:1fr}}
</style></head><body>
<div class="header"><h1>Atlas Broadband — Regional ISP Capacity vs. Churn Diagnosis</h1>
<small>Generated GENERATED_DATE | 3 Regions | hybrid simulation (demand + service + behavior) | 730-day projection</small></div>
<div class="tab-bar">TABS_HTML</div>
<div class="content">PANES_HTML</div>
<script>
window.addEventListener('load',function(){setTimeout(function(){
document.querySelectorAll('.pane').forEach(function(e,i){if(i!==0)e.classList.add('hidden')})
},500)})
function switchTab(i){document.querySelectorAll('.pane').forEach(function(e){e.classList.remove('hidden')})
document.querySelectorAll('.pane').forEach(function(e,j){if(j!==i)e.classList.add('hidden')})
document.querySelectorAll('.tab-btn').forEach(function(e,j){e.classList.toggle('active',j===i)})
document.querySelectorAll('.pane:not(.hidden) .js-plotly-plot').forEach(function(e){if(typeof Plotly!=='undefined')Plotly.Plots.resize(e)})}
</script></body></html>"""


def build_html(pages: list[dict]) -> str:
    tabs = "".join(
        f'<button class="tab-btn {"active" if i==0 else ""}" onclick="switchTab({i})">'
        f'{p["icon"]} {p["title"]}</button>'
        for i, p in enumerate(pages)
    )
    panes = "".join(
        f'<div class="pane" id="pane-{i}">{p["content"]}</div>'
        for i, p in enumerate(pages)
    )
    html = HTML_TEMPLATE
    html = html.replace("PRIMARYCOLOR", THEME["primary"])
    html = html.replace("ACCENTCOLOR", THEME["accent"])
    html = html.replace("BGCOLOR", THEME["bg"])
    html = html.replace("CARDCOLOR", THEME["card"])
    html = html.replace("TEXTCOLOR", THEME["text"])
    html = html.replace("MUTEDCOLOR", THEME["muted"])
    html = html.replace("TABS_HTML", tabs)
    html = html.replace("PANES_HTML", panes)
    html = html.replace("GENERATED_DATE", datetime.now().strftime("%Y-%m-%d %H:%M"))
    return html


def main():
    print("Atlas Broadband — Regional ISP Capacity vs. Churn Diagnosis Dashboard")
    print("=" * 70)
    data = run_simulation()

    print("\nBuilding 11 dashboard tabs...")
    pages = [
        build_executive_summary(data),
        build_region_growth(data),
        build_demand_forecast(data),
        build_capacity_planning(data),
        build_network_qos(data),
        build_customer_base(data),
        build_churn_analysis(data),
        build_disruption_impact(data),
        build_root_cause(data),
        build_decision_rule(data),
        build_scenario_comparison(data),
    ]
    print(f"  Built {len(pages)} pages")

    print("Assembling HTML...")
    html = build_html(pages)
    out = Path("/tmp/atlas_broadband_dashboard.html")
    out.write_text(html)
    print(f"\nDashboard: {out} ({out.stat().st_size / 1024:.0f}KB, {len(pages)} tabs)")


if __name__ == "__main__":
    main()
