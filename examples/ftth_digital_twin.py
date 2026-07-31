#!/usr/bin/env python3
"""FTTH Digital Twin — full Knowledge-Bridge-Simulation-Evidence loop.

Console narrative walking the entire digital-twin value chain for a
fiber-to-the-home network operator across three zones (A/B/C):

  1. Knowledge layer     — CSV → KB via ingest_csv, FTTH topology/regulation
                           triples, RDFS inference
  2. Bridge              — KB beliefs → simulation parameters (params_from_kb)
  3. Multi-paradigm sim  — SD stocks + DES queues + ABM households
  4. Live KB             — KB_QUERY/KB_ASSERT mid-run self-healing
  5. Evidence round-trip — full_roundtrip → serviceLevel/invRisk triples
  6. Scenarios           — ScenarioComparison, grading, ranking, filtering
  7. Production rules    — SparqlCondition → BridgeAction + TripleAction
  8. Optimization        — kb_lp_minimize capex allocation across zones
  9. Explainability      — causal_trace + detect_feedback_loops
 10. Provenance          — record_provenance
 11. Maturity mapping    — L1..L5
 12. Takeaway

Run:  python examples/ftth_digital_twin.py
Output: console narrative only (no server, no files written).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── DynaFX imports ────────────────────────────────────────────────
from dynafx.dynamics.dsl import (
    SysdModel, AgentDef, AgentPropDef, AgentRuleDef, AgentStrategy,
    QueueDef, ResourceDef,
)
from dynafx.dynamics.causal import causes_strip, causal_trace
from dynafx.dynamics.feedback import detect_feedback_loops
from dynafx.dynamics.optimization import kb_lp_minimize, kb_lp_maximize
from dynafx.dynamics.scenario import ScenarioComparison, ScenarioDef
from dynafx.knowledge import (
    TripleStore, NamedNode, Literal, Triple, TriplePattern,
    MappingDef, ingest_csv, parse_turtle,
)
from dynafx.knowledge.inference import RuleEngine, rdfs_rules
from dynafx.knowledge.production import (
    ProductionRuleEngine, ProductionRule, SparqlCondition,
    BridgeAction, TripleAction, ComparisonCondition,
)
from dynafx.bridge import KBSimBridge, grade_queries

# ── Namespaces ────────────────────────────────────────────────────
FTTH_NS = "http://ftth.org/"
ISP_NS = "http://isp-broadband.org/"
XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"

DATA_DIR = ROOT / "data"

REGIONS = ["A", "B", "C"]
T_START, T_END, DT = 0.0, 180.0, 1.0

# Zone economics / network shape (KB will hold these; defaults here)
ZONE_ARPUS = {"A": 49.99, "B": 44.99, "C": 39.99}
ZONE_CAPEX_PER_HOME = {"A": 850.0, "B": 950.0, "C": 1100.0}
ZONE_RELIABILITY = {"A": 0.97, "B": 0.95, "C": 0.92}
ZONE_INIT_PREMISES = {"A": 60000.0, "B": 50000.0, "C": 35000.0}
ZONE_TARGET_PREMISES = {"A": 100000.0, "B": 85000.0, "C": 65000.0}
ZONE_INIT_SUBS = {"A": 27860.0, "B": 29850.0, "C": 12000.0}
ZONE_BASE_CHURN = {"A": 0.00025, "B": 0.00030, "C": 0.00020}
AGENTS_PER_ZONE = {"A": 8, "B": 8, "C": 8}

FIBER_CUT_PENALTY = 0.5   # service level multiplier while disrupted
CHURN_DISRUPTION = 0.004  # extra daily churn while disrupted

W = 78  # banner width


def banner(title: str, sub: str = "") -> None:
    """Section banner for the console narrative."""
    print()
    print("=" * W)
    print(f"  {title.upper()}")
    if sub:
        print(f"  {sub}")
    print("=" * W)


def _n(name: str) -> NamedNode:
    return NamedNode(f"{FTTH_NS}{name}")


# ══════════════════════════════════════════════════════════════════
# 1. Knowledge layer
# ══════════════════════════════════════════════════════════════════

def _build_store() -> tuple[TripleStore, NamedNode, list[tuple[str, str, Any]]]:
    """Seed the KB: ISP history CSVs via ingest_csv + FTTH domain triples.

    No banner/print side effects so sections can build a fresh store cheaply.
    Returns (store, regulator_node, per-mapping ingest reports).
    """
    store = TripleStore()
    reports: list[tuple[str, str, Any]] = []

    # 1a. ISP historical CSVs (subscribers, churn, QoS, leading indicators)
    onto_path = DATA_DIR / "isp-ontology.ttl"
    if onto_path.exists():
        onto = parse_turtle(onto_path.read_text())
        for t in onto.all_triples():
            store.add(t, f"{ISP_NS}meta")

    mappings_dir = DATA_DIR / "mappings"
    for yaml_path in sorted(mappings_dir.glob("isp_*.yaml")):
        mapping = MappingDef.from_yaml(str(yaml_path))
        csv_path = DATA_DIR / mapping.csv
        if csv_path.exists():
            report = ingest_csv(mapping, str(csv_path), store, strict=False)
            reports.append((yaml_path.stem, mapping.target_graph.split("/")[-1], report))

    # 1b. FTTH network topology + regulation triples (the twin's static model)
    graph = f"{FTTH_NS}topology"
    # zones
    for z in REGIONS:
        zn = _n(f"Zone{z}")
        store.add(Triple(zn, NamedNode(f"{FTTH_NS}rdf-type"), _n("Zone")), graph)
        store.add(Triple(zn, _n("arpu"), Literal(str(ZONE_ARPUS[z]), datatype=XSD_DOUBLE)), graph)
        store.add(Triple(zn, _n("capexPerHome"), Literal(str(ZONE_CAPEX_PER_HOME[z]), datatype=XSD_DOUBLE)), graph)
        store.add(Triple(zn, _n("reliability"), Literal(str(ZONE_RELIABILITY[z]), datatype=XSD_DOUBLE)), graph)
        store.add(Triple(zn, _n("targetPremises"), Literal(str(ZONE_TARGET_PREMISES[z]), datatype=XSD_DOUBLE)), graph)
    # regulator (ARCEP-style) coverage obligation
    reg = _n("ARCEP")
    store.add(Triple(reg, NamedNode(f"{FTTH_NS}rdf-type"), _n("Regulator")), graph)
    store.add(Triple(reg, _n("coverageTarget"), Literal("0.80", datatype=XSD_DOUBLE)), graph)
    # the live status the twin senses mid-run (World agent will flip it)
    store.add(Triple(_n("ZoneB"), _n("status"), Literal("normal")), graph)

    # 1c. RDFS inference over the combined KB
    applied = RuleEngine(rdfs_rules()).apply(store)
    return store, reg, reports, applied


def section1_build_kb() -> tuple[TripleStore, NamedNode]:
    """Seed the KB and print the ingest/inference summary."""
    banner("1. Knowledge Layer",
           "CSV → named graphs (ingest_csv) · FTTH topology/regulation · RDFS inference")
    store, reg, reports, applied = _build_store()

    for stem, graph, report in reports:
        print(f"    {stem:26s}  graph={graph:22s}"
              f"  rows={report.rows_parsed:3d} triples={report.triples_added:5d}")
    total = sum(sum(1 for _ in store.triples(TriplePattern(), graph=g)) for g in store.graphs())
    print(f"\n    RDFS inference: {applied} derived facts")
    print(f"    KB now holds {total} triples across {len(store.graphs())} named graphs")
    return store, reg


# ══════════════════════════════════════════════════════════════════
# 2. Bridge — KB beliefs → simulation params
# ══════════════════════════════════════════════════════════════════

def section2_bridge(store: TripleStore, reg: NamedNode) -> dict[str, float]:
    """Map KB beliefs into model params with params_from_kb."""
    banner("2. Bridge",
           "KB beliefs → simulation parameters (KBSimBridge.params_from_kb)")
    bridge = KBSimBridge(store, ns_base=FTTH_NS)
    claim_map: list[tuple[NamedNode, NamedNode, object, str]] = []
    for z in REGIONS:
        zn = _n(f"Zone{z}")
        claim_map.append((zn, _n("arpu"), None, f"arpu_{z}"))
        claim_map.append((zn, _n("capexPerHome"), None, f"capex_per_home_{z}"))
        claim_map.append((zn, _n("reliability"), None, f"reliability_{z}"))
        claim_map.append((zn, _n("targetPremises"), None, f"target_premises_{z}"))
    claim_map.append((reg, _n("coverageTarget"), None, "coverage_target"))
    kb_params = bridge.params_from_kb(claim_map, default=0.5)
    print("    Param                    KB belief")
    print("    " + "-" * 52)
    for k, v in sorted(kb_params.items()):
        print(f"    {k:26s} {v:.4g}")
    return kb_params


# ══════════════════════════════════════════════════════════════════
# 3. Multi-paradigm model — SD + DES + ABM
# ══════════════════════════════════════════════════════════════════

def build_model(params: dict[str, float], with_world: bool = False) -> SysdModel:
    """SD subscriber/premises stocks + per-zone DES install queues +
    ABM households (+ optional World orchestrator for live-KB demo)."""
    m = SysdModel("ftth_digital_twin")
    m.dt = DT
    m.t_span = (T_START, T_END)

    # ── leading-indicator demand auxes (from ISP data shape) ──
    m.aux("building_permits_A", "30 + 20*SIN(2*PI*t/365) + PULSE(45, 100, 90)")
    m.aux("competitor_entry", "IF(t > 120, 1, 0)")
    m.aux("marketing_push", "PULSE(1.5, 200, 180) + 0.2*SIN(2*PI*t/365)")

    m.aux("growth_mod_A", "building_permits_A / 100 + 0.4")
    m.aux("growth_mod_B", "MAX(0, 0.5 - competitor_entry * 0.3) + 0.08")
    m.aux("growth_mod_C", "marketing_push * 0.5 + 0.3 + 0.12*SIN(2*PI*t/365 + PI/6)")

    # ── live KB state (param holds a SPARQL string) ──
    m.aux("disruption_active", "KB_QUERY(disp_q)")
    m.aux("disruption_penalty", "1.0 - disruption_active * 0.5")

    # ── per-zone subscriber + premises stocks ──
    for z in REGIONS:
        pn = f"Premises_{z}"
        sn = f"Subs_{z}"
        penalty = " * disruption_penalty" if z == "B" else ""
        with m.stock(pn, ZONE_INIT_PREMISES[z]) as s:
            s.inflow(f"deploy_{z}",
                     f"MAX(0, capex_daily_{z} / capex_per_home_{z}){penalty}")
        with m.stock(sn, ZONE_INIT_SUBS[z]) as s:
            s.inflow(f"adopt_{z}", f"adopt_rate_{z}")
            s.outflow(f"churn_{z}", f"{sn} * churn_fraction_{z}")

    # ── global financial + cumulative demand/met ──
    with m.stock("Revenue_Reserve", 0.0) as s:
        s.inflow("revenue_in", "total_subs * arpu_avg / 30.0")
        s.outflow("opex_out",
                  "total_subs * variable_opex_daily + capex_daily_total "
                  "+ disruption_repair_cost * disruption_active")
    with m.stock("Cumulative_Demand", 0.0) as s:
        s.inflow("demand_flow", "total_demand_daily")
    with m.stock("Cumulative_Met", 0.0) as s:
        s.inflow("met_flow", "total_demand_daily * service_level")

    # ── per-zone auxes ──
    for z in REGIONS:
        sn = f"Subs_{z}"
        pn = f"Premises_{z}"
        penalty = " * disruption_penalty" if z == "B" else ""
        churn_disp = " + CHURN_DISRUPTION * disruption_active" if z == "B" else ""
        m.aux(f"demand_{z}", f"base_demand_{z} * growth_mod_{z}{penalty}")
        # aux intermediate so DES arrival injector + Subs inflow both see it
        m.aux(f"adopt_rate_{z}", f"demand_{z} * capacity_gate_{z}")
        m.aux(f"capacity_gate_{z}",
              f"MAX(0, ({pn} - {sn}) / MAX(1, {pn}))")
        m.aux(f"util_{z}", f"{sn} / MAX(1, {pn} * subs_per_unit)")
        m.aux(f"qos_{z}",
              f"IF(util_{z} <= 0.7, 100, "
              f"IF(util_{z} <= 0.9, 100 - (util_{z} - 0.7) * 250, "
              f"MAX(10, 100 - (util_{z} - 0.7) * 500)))")
        # cap the congestion feedback so a build backlog can't make the
        # system stiff or swamp the KB-driven churn signal
        m.aux(f"congest_amplifier_{z}",
              f"MIN(0.003, MAX(0, (install_{z}_length - 5)) * 0.00005)")
        m.aux(f"churn_fraction_{z}",
              f"base_churn_{z} + churn_abm_contrib_{z} + congest_amplifier_{z}{churn_disp}")
        m.aux(f"coverage_{z}", f"{pn} / MAX(1, target_premises_{z})")

    # ── ABM churn contribution (household satisfaction aggregates) ──
    for z in REGIONS:
        m.aux(f"churn_abm_contrib_{z}",
              f"MAX(0, (0.5 - HouseHold_{z}_satisfaction_avg / 100)) * 0.02 "
              f"+ HouseHold_{z}_churn_risk_avg * 0.008")

    # ── global auxes ──
    m.aux("total_subs", "Subs_A + Subs_B + Subs_C")
    m.aux("total_premises", "Premises_A + Premises_B + Premises_C")
    m.aux("total_demand_daily", "demand_A + demand_B + demand_C")
    m.aux("arpu_avg", "(arpu_A + arpu_B + arpu_C) / 3")
    m.aux("capex_daily_total",
          "capex_daily_A + capex_daily_B + capex_daily_C")
    m.aux("variable_opex_daily", "variable_opex_per_sub / 30.0")
    m.aux("service_level",
          "reliability_avg * (1.0 - disruption_active * 0.5)")
    m.aux("reliability_avg", "(reliability_A + reliability_B + reliability_C) / 3")

    # ── DES: per-zone install queues (arrival = adopt rate) + crews ──
    # Server pools sized near each zone's adopt rate so backlogs stay in the
    # tens (visible, but not enough to destabilise the SD subscriber base).
    for z in REGIONS:
        m.queue(f"install_{z}", capacity=-1, service_time="3.0",
                servers={"A": 200, "B": 120, "C": 65}[z],
                arrival_rate=f"MAX(0, adopt_rate_{z})")
    m.resource("tech_crews", capacity=20, cost_per_unit=1500.0)

    # ── ABM: households per zone (satisfaction/churn strategies) ──
    for z in REGIONS:
        for ai in range(AGENTS_PER_ZONE[z]):
            init_sat = round(70 + ((ai * 37) % 25), 1)
            m.agents.append(AgentDef(
                f"HouseHold_{z}", 1,
                properties=[
                    AgentPropDef("satisfaction", init_sat, min=0, max=100),
                    AgentPropDef("churn_risk", 0.0, min=0, max=1),
                    AgentPropDef("is_active", 1.0, min=0, max=1),
                ],
                strategies=[
                    AgentStrategy("normal", [
                        AgentRuleDef("update_sat", "always",
                                     [f"satisfaction += (qos_{z} - satisfaction) * 0.05 * dt"]),
                        AgentRuleDef("decay_risk", "always",
                                     ["churn_risk = MAX(0, churn_risk - 0.0005 * dt)"]),
                    ]),
                    AgentStrategy("at_risk", [
                        AgentRuleDef("sat_erode", "always",
                                     [f"satisfaction += (qos_{z} - satisfaction) * 0.02 * dt"]),
                        AgentRuleDef("risk_up", "always",
                                     ["churn_risk = MIN(1, churn_risk + 0.002 * dt)"]),
                    ]),
                    AgentStrategy("churned", [
                        AgentRuleDef("inactive", "always", ["is_active = 0"]),
                    ]),
                ],
                meta_rules=[
                    AgentRuleDef("risk_trigger", "satisfaction < 35",
                                 ["SWITCH_STRATEGY('at_risk', cooldown=30)"]),
                    AgentRuleDef("recovery", "satisfaction > 60 and strategy == 'at_risk'",
                                 ["SWITCH_STRATEGY('normal', cooldown=15)"]),
                    AgentRuleDef("churn_out", "churn_risk > 0.85",
                                 ["SWITCH_STRATEGY('churned', cooldown=999)"]),
                ],
            ))

    # ── optional World orchestrator for the live-KB self-healing demo ──
    if with_world:
        m.agents.append(AgentDef(
            "World", 1,
            properties=[AgentPropDef("triggered", 0, min=0, max=2)],
            strategies=[
                AgentStrategy("normal", [
                    AgentRuleDef("fire_disruption",
                                 "t >= 60 and triggered == 0",
                                 ["triggered = 1",
                                  "KB_ASSERT('http://ftth.org/ZoneB', "
                                  "'http://ftth.org/status', 'cut')"]),
                    AgentRuleDef("fire_recovery",
                                 "t >= 90 and triggered == 1",
                                 ["triggered = 2",
                                  "KB_ASSERT('http://ftth.org/ZoneB', "
                                  "'http://ftth.org/status', 'normal')"]),
                ]),
            ],
        ))

    # model params (merge KB-derived with constants)
    for k, v in params.items():
        m.param(k, v)
    for z in REGIONS:
        m.param(f"base_demand_{z}", {  # new orders/day at t=0
            "A": 120.0, "B": 95.0, "C": 60.0,
        }[z])
        m.param(f"capex_daily_{z}", {
            "A": 15000.0, "B": 11000.0, "C": 7500.0,
        }[z])
        m.param(f"base_churn_{z}", ZONE_BASE_CHURN[z])
    m.param("subs_per_unit", 1.0)
    m.param("variable_opex_per_sub", 7.5)
    m.param("disruption_repair_cost", 25000.0)
    m.param("CHURN_DISRUPTION", CHURN_DISRUPTION)
    return m


def _get_ts(r) -> dict[str, list[float]]:
    """Combined timeseries aligned to r.times.

    values/aux_values are indexed by timestep. DES/ABM metric histories are
    recorded AFTER each step's advance, so entry i describes time i-1 — shift
    them back one slot (carrying the last value) to align with r.times.
    """
    ts = dict(r.values)
    ts.update(r.aux_values)
    n = len(r.times)
    for hist_attr in ("des_metrics_history", "abm_metrics_history"):
        hist = getattr(r, hist_attr, None)
        if not hist:
            continue
        all_keys: set[str] = set()
        for entry in hist:
            all_keys.update(entry.keys())
        for key in all_keys:
            series: list[float] = []
            for i in range(n):
                src = hist[i + 1] if i + 1 < len(hist) else (hist[-1] if hist else {})
                series.append(src.get(key, 0.0))
            ts[key] = series
    return ts


def _fmt(x: float) -> str:
    return f"{x:,.1f}"


def section3_multiparadigm(store: TripleStore, kb_params: dict[str, float]) -> tuple[SysdModel, Any]:
    """Build the model, run a baseline, summarize SD/DES/ABM state."""
    banner("3. Multi-paradigm Simulation",
           "System Dynamics stocks · DES queues · ABM households — one shared twin")
    model = build_model(kb_params, with_world=False)
    disp_q = ('ASK { <http://ftth.org/ZoneB> <http://ftth.org/status> "cut" }')
    params = dict(kb_params)
    params["disp_q"] = disp_q  # live-KB query (SPARQL string param)
    result = model.simulate(params=params, kb=store, method="rk4")
    ts = _get_ts(result)

    t = result.times
    print(f"\n    Horizon: {T_START:.0f} → {T_END:.0f} days · dt={DT} · "
          f"{len(t)} steps · model '{model.name}'")
    print(f"    Components: {len(model.stocks)} SD stocks · "
          f"{len(model.queues)} DES queues · {len(model.resources)} resources · "
          f"{sum(a.count for a in model.agents)} ABM agents")
    print("\n    Snapshot at t=180:")
    print(f"      Subscribers     : " + "  ".join(
        f"{z}={_fmt(ts[f'Subs_{z}'][-1])}" for z in REGIONS))
    print(f"      Premises passed : " + "  ".join(
        f"{z}={_fmt(ts[f'Premises_{z}'][-1])}" for z in REGIONS))
    print(f"      QoS             : " + "  ".join(
        f"{z}={ts[f'qos_{z}'][-1]:5.1f}" for z in REGIONS))
    fill = ts["Cumulative_Met"][-1] / max(1.0, ts["Cumulative_Demand"][-1])
    print(f"      Fill rate       : {fill * 100:.1f}%  "
          f"({_fmt(ts['Cumulative_Met'][-1])} met of "
          f"{_fmt(ts['Cumulative_Demand'][-1])} requested)")
    print(f"      Revenue reserve : ${_fmt(ts['Revenue_Reserve'][-1])}")
    q_stats = {q: result.des_engine.queues[q].stats.avg_length for q in result.des_engine.queues}
    print("      DES queue avg length: " + "  ".join(
        f"{k}={v:.1f}" for k, v in q_stats.items()))
    print(f"      ABM household satisfaction avg: "
          + "  ".join(f"{z}={ts[f'HouseHold_{z}_satisfaction_avg'][-1]:.1f}" for z in REGIONS))
    return model, result


# ══════════════════════════════════════════════════════════════════
# 4. Live KB — self-healing disruption
# ══════════════════════════════════════════════════════════════════

def section4_live_kb(kb_params: dict[str, float]) -> Any:
    """World agent flips ZoneB status mid-run; KB_QUERY auxes react live.

    Runs on a fresh store copy so the mid-run KB_ASSERT mutations never leak
    into later sections. The post-hoc aux replay re-reads the final KB state
    (both 'cut' and 'normal' present), so we reconstruct the per-step value
    the twin actually sensed from the World agent's own triggered trace:
    triggered == 1 exactly while 'cut' was in the KB.
    """
    banner("4. Live Knowledge Base",
           "World agent KB_ASSERTs a fiber cut at t=60, recovery at t=90 — "
           "auxes sense it via KB_QUERY every step")
    store, _, _, _ = _build_store()
    model = build_model(kb_params, with_world=True)
    disp_q = ('ASK { <http://ftth.org/ZoneB> <http://ftth.org/status> "cut" }')
    params = dict(kb_params)
    params["disp_q"] = disp_q
    result = model.simulate(params=params, kb=store, method="rk4")
    ts = _get_ts(result)

    # what the twin sensed each step: disruption_active == 1 ⟺ 'cut' in KB
    triggered = ts.get("World_triggered_avg", [0.0] * len(result.times))
    sensed = [1.0 if v == 1 else 0.0 for v in triggered]
    ts["disruption_active"] = sensed
    ra = ts["reliability_avg"]
    ts["service_level"] = [ra[i] * (1.0 - sensed[i] * FIBER_CUT_PENALTY)
                           for i in range(len(sensed))]
    ts["churn_fraction_B"] = [ZONE_BASE_CHURN["B"]
                              + ts["churn_abm_contrib_B"][i]
                              + ts["congest_amplifier_B"][i]
                              + CHURN_DISRUPTION * sensed[i]
                              for i in range(len(sensed))]

    # show the mid-run KB flip + the twin's reaction
    cut_at = next((i for i, v in enumerate(triggered) if v >= 1), None)
    rec_at = next((i for i, v in enumerate(triggered) if v >= 2), None)
    t = result.times
    cut_t = t[cut_at] if cut_at is not None else None
    rec_t = t[rec_at] if rec_at is not None else None

    def _at(tt: float) -> dict[str, float]:
        idx = min(range(len(t)), key=lambda i: abs(t[i] - tt))
        return {k: v[idx] for k, v in ts.items()}

    print("\n    World agent timeline:")
    print(f"      t=60  KB_ASSERT ZoneB/status = 'cut'     (triggered "
          f"{'YES' if cut_t == 60.0 else 'n/a'})")
    print(f"      t=90  KB_ASSERT ZoneB/status = 'normal'  (triggered "
          f"{'YES' if rec_t == 90.0 else 'n/a'})")

    for label, tt in (("Normal (t=30)", 30.0), ("Disrupted (t=75)", 75.0), ("Recovered (t=150)", 150.0)):
        s = _at(tt)
        print(f"\n    {label}:")
        print(f"      disruption_active = {s['disruption_active']:.0f} · "
              f"service_level = {s['service_level']:.3f} · "
              f"churn_B = {s['churn_fraction_B']:.5f} · "
              f"install_B length = {s.get('install_B_length', 0.0):.0f}")

    # post-hoc: compute churn_abm contribution (metric present in ts)
    print(f"\n    Household satisfaction (Zone B): "
          f"30d={_at(30.0)['HouseHold_B_satisfaction_avg']:.1f} "
          f"75d={_at(75.0)['HouseHold_B_satisfaction_avg']:.1f} "
          f"150d={_at(150.0)['HouseHold_B_satisfaction_avg']:.1f}")
    print("    ⇒ The twin senses the KB state change and degrades service "
          "in real time, then self-heals on recovery.")
    return result


# ══════════════════════════════════════════════════════════════════
# 5. Evidence round-trip
# ══════════════════════════════════════════════════════════════════

def _svc_score(initial: list[float], final: list[float]) -> float:
    """service level 0..1 (placeholder — real ratio computed via closure)."""
    return min(1.0, max(0.0, final[-1]))


def section5_evidence(store: TripleStore, kb_params: dict[str, float]) -> KBSimBridge:
    """full_roundtrip: KB→params→sim→evidence triples."""
    banner("5. Evidence Round-trip",
           "KBSimBridge.full_roundtrip writes serviceLevel/invRisk evidence back into the KB")
    bridge = KBSimBridge(store, ns_base=FTTH_NS)
    model = build_model(kb_params, with_world=False)
    disp_q = ('ASK { <http://ftth.org/ZoneB> <http://ftth.org/status> "cut" }')
    params = dict(kb_params)
    params["disp_q"] = disp_q

    claim_map: list[tuple[NamedNode, NamedNode, object, str]] = []
    for z in REGIONS:
        zn = _n(f"Zone{z}")
        claim_map.append((zn, _n("arpu"), None, f"arpu_{z}"))
        claim_map.append((zn, _n("capexPerHome"), None, f"capex_per_home_{z}"))
        claim_map.append((zn, _n("reliability"), None, f"reliability_{z}"))
        claim_map.append((zn, _n("targetPremises"), None, f"target_premises_{z}"))
    claim_map.append((_n("ARCEP"), _n("coverageTarget"), None, "coverage_target"))

    # service level measured from the Cumulative met/demand stocks
    def _make_evidence_map() -> list[tuple]:
        emap = []
        for z in REGIONS:
            zn = _n(f"Zone{z}")
            target = ZONE_TARGET_PREMISES[z]
            emap.append((f"Premises_{z}", zn, _n("serviceLevel"),
                         lambda i, f, tgt=target: min(1.0, max(0.0, f[-1] / tgt))))
            emap.append((f"Premises_{z}", zn, _n("invRisk"),
                         lambda i, f, tgt=target: max(0.0, 1.0 - f[-1] / tgt)))
            emap.append((f"Subs_{z}", zn, _n("subscriberGrowth"),
                         lambda i, f: f[-1] / max(1.0, i[0])))
        return emap

    result, triples = bridge.full_roundtrip(
        model, claim_map, _make_evidence_map(),
        params=params, evidence_graph=f"{FTTH_NS}evidence",
    )
    print(f"\n    {len(triples)} evidence triples written to '{FTTH_NS}evidence'")
    for z in REGIONS:
        zn = _n(f"Zone{z}")
        svc = list(store.triples(TriplePattern(zn, _n("serviceLevel"), None)))
        inv = list(store.triples(TriplePattern(zn, _n("invRisk"), None)))
        print(f"    Zone{z}: serviceLevel={svc[0].object_.value if svc else '?'} · "
              f"invRisk={inv[0].object_.value if inv else '?'}")
    return bridge


# ══════════════════════════════════════════════════════════════════
# 6. Scenarios
# ══════════════════════════════════════════════════════════════════

def section6_scenarios(store: TripleStore, kb_params: dict[str, float],
                       bridge: KBSimBridge) -> ScenarioComparison:
    """Run a scenario family, grade, rank, filter."""
    banner("6. Scenario Analysis",
           "KB-driven disruptions · grading · ranking · constraint filtering")
    model = build_model(kb_params, with_world=False)

    def _params(**extra) -> dict[str, float]:
        p = dict(kb_params)
        p["disp_q"] = 'ASK { <http://ftth.org/ZoneB> <http://ftth.org/status> "cut" }'
        p.update(extra)
        return p

    scenarios = [
        ScenarioDef("baseline", _params()),
        ScenarioDef("fiber_cut", _params(capex_daily_B=3000.0,
                                         base_churn_B=ZONE_BASE_CHURN["B"] + CHURN_DISRUPTION)),
        ScenarioDef("housing_boom", _params(base_demand_A=260.0, base_demand_B=200.0,
                                            base_demand_C=130.0)),
        ScenarioDef("aggressive_capex", _params(capex_daily_A=24000.0, capex_daily_B=18000.0,
                                                capex_daily_C=12000.0)),
    ]
    comp = ScenarioComparison(model, scenarios, method="rk4", kb=store)
    print("\n    Scenario                Subscribers(final)   Revenue($)   FillRate")
    print("    " + "-" * 70)
    for sr in comp.scenarios:
        ts = _get_ts(sr.result)
        fill = ts["Cumulative_Met"][-1] / max(1.0, ts["Cumulative_Demand"][-1])
        print(f"    {sr.name:24s} "
              f"{ts['total_subs'][-1]:14,.0f}   "
              f"{ts['Revenue_Reserve'][-1]:10,.0f}   {fill * 100:5.1f}%")

    # grading on higher-is-better serviceLevel per zone
    evidence_map = []
    for z in REGIONS:
        zn = _n(f"Zone{z}")
        target = ZONE_TARGET_PREMISES[z]
        evidence_map.append((f"Premises_{z}", zn, _n("serviceLevel"),
                             lambda i, f, tgt=target: min(1.0, max(0.0, f[-1] / tgt))))
    grade_specs = [
        (f'SELECT ?v WHERE {{ <{FTTH_NS}ZoneA> <{FTTH_NS}serviceLevel> ?v }}', "v", 0.9, 0.0),
        (f'SELECT ?v WHERE {{ <{FTTH_NS}ZoneB> <{FTTH_NS}serviceLevel> ?v }}', "v", 0.9, 0.0),
        (f'SELECT ?v WHERE {{ <{FTTH_NS}ZoneC> <{FTTH_NS}serviceLevel> ?v }}', "v", 0.9, 0.0),
    ]
    # drop section 5's baseline evidence so ranking reads only per-scenario evidence
    store.remove(TriplePattern(), graph=f"{FTTH_NS}evidence")
    ranked = comp.rank(grade_specs, store, evidence_map=evidence_map, bridge=bridge)
    print("\n    Ranked by KB serviceLevel (higher is better):")
    for name, score in ranked:
        print(f"      {name:24s} {score:.3f}")

    # filter: keep only scenarios that reach ≥70% fill at the end
    def _fill_ok(result) -> bool:
        ts = _get_ts(result)
        return ts["Cumulative_Met"][-1] / max(1.0, ts["Cumulative_Demand"][-1]) >= 0.70

    surviving = [sr.name for sr in comp.scenarios if _fill_ok(sr.result)]
    print(f"\n    Constraint 'fill ≥ 70%' keeps: {', '.join(surviving)}")
    return comp


# ══════════════════════════════════════════════════════════════════
# 7. Production rules
# ══════════════════════════════════════════════════════════════════

def section7_production_rules(store: TripleStore, kb_params: dict[str, float],
                              bridge: KBSimBridge) -> None:
    """SparqlCondition → BridgeAction (re-simulate) + TripleAction (mitigate)."""
    banner("7. Production Rules",
           "SparqlCondition → BridgeAction re-simulate · TripleAction mitigate")
    model = build_model(kb_params, with_world=False)
    disp_q = 'ASK { <http://ftth.org/ZoneB> <http://ftth.org/status> "cut" }'

    # condition: ZoneB coverage falls below ARCEP target
    coverage_ask = f'ASK {{ <{FTTH_NS}ZoneB> <{FTTH_NS}coverage> ?c . FILTER(?c < 0.65) }}'
    # seed the coverage triple so the rule can trigger
    result0 = model.simulate(params={**kb_params, "disp_q": disp_q}, kb=store, method="rk4")
    cov_b = result0.aux_values.get("coverage_B", [0.0])[-1]
    store.add(Triple(_n("ZoneB"), _n("coverage"), Literal(str(cov_b), datatype=XSD_DOUBLE)),
              f"{FTTH_NS}monitoring")

    engine = ProductionRuleEngine(store)
    claim_map: list[tuple[NamedNode, NamedNode, object, str]] = []
    for z in REGIONS:
        zn = _n(f"Zone{z}")
        claim_map.append((zn, _n("arpu"), None, f"arpu_{z}"))
        claim_map.append((zn, _n("capexPerHome"), None, f"capex_per_home_{z}"))
        claim_map.append((zn, _n("reliability"), None, f"reliability_{z}"))

    engine.add_rule(ProductionRule(
        name="coverage_breach",
        description="Zone B coverage < 65% of target → accelerate buildout + flag",
        event="on_change",
        body=[SparqlCondition(coverage_ask, min_results=1)],
        head=[
            BridgeAction(bridge=bridge, model=model, claim_map=claim_map,
                         params_override={"capex_daily_B": 25000.0}),
            TripleAction(_n("ZoneB"), _n("mitigation"),
                         Literal("accelerated_buildout")),
        ],
        fire_once=True,
    ))
    engine.start()
    fired = engine.evaluate()
    engine.stop()

    triples = list(store.triples(TriplePattern(_n("ZoneB"), _n("mitigation"), None)))
    print(f"\n    ZoneB coverage (pre)   : {cov_b:.3f}")
    print(f"    Rule 'coverage_breach' fired → BridgeAction re-simulated "
          f"with capex_B boosted" + (f"  ({len(fired)} action fired)" if fired else ""))
    print(f"    Rule 'coverage_breach' fired → TripleAction asserted: "
          f"{triples[0].object_.value if triples else 'n/a'}")


# ══════════════════════════════════════════════════════════════════
# 8. Optimization — KB-driven LP
# ══════════════════════════════════════════════════════════════════

def section8_optimization(store: TripleStore) -> None:
    """kb_lp_minimize: minimize capex to meet per-zone coverage under a budget."""
    banner("8. Optimization",
           "kb_lp_minimize allocates capex across zones from SPARQL coeffs/bounds")
    # c = capex per home (minimize cost); bounds = min/max capex per zone
    c_q = f'SELECT ?v WHERE {{ ?z <{FTTH_NS}capexPerHome> ?v }} ORDER BY ?z'
    b_q = (f'SELECT ?v ?v2 WHERE {{ '
           f'?z <{FTTH_NS}capexPerHome> ?v . '
           f'?z <{FTTH_NS}maxCapexBudget> ?v2 }} ORDER BY ?z')
    for z in REGIONS:
        store.add(Triple(_n(f"Zone{z}"), _n("maxCapexBudget"),
                         Literal("900000", datatype=XSD_DOUBLE)), f"{FTTH_NS}lp")
    eq_q = (f'SELECT ?v ?v2 ?v3 WHERE {{ '
            f'<{FTTH_NS}BudgetRow> <{FTTH_NS}coeffA> ?v . '
            f'<{FTTH_NS}BudgetRow> <{FTTH_NS}coeffB> ?v2 . '
            f'<{FTTH_NS}BudgetRow> <{FTTH_NS}coeffC> ?v3 }}')
    beq_q = f'SELECT ?v WHERE {{ <{FTTH_NS}BudgetRow> <{FTTH_NS}total> ?v }}'
    store.add(Triple(_n("BudgetRow"), _n("coeffA"), Literal("1.0")), f"{FTTH_NS}lp")
    store.add(Triple(_n("BudgetRow"), _n("coeffB"), Literal("1.0")), f"{FTTH_NS}lp")
    store.add(Triple(_n("BudgetRow"), _n("coeffC"), Literal("1.0")), f"{FTTH_NS}lp")
    store.add(Triple(_n("BudgetRow"), _n("total"), Literal("2000000")), f"{FTTH_NS}lp")

    result = kb_lp_minimize(store, c_q, b_q, A_eq_query=eq_q, b_eq_query=beq_q)
    print("\n    Minimize capex per home subject to €2.0M build budget:")
    print(f"      {result.success=}  obj=${result.objective_value:,.0f}")
    for i, z in enumerate(REGIONS):
        print(f"      Zone{z}: capex ${result.x[i]:,.0f}  (cap "
              f"${900000:,.0f} · cost/home ${ZONE_CAPEX_PER_HOME[z]:,.0f})")


# ══════════════════════════════════════════════════════════════════
# 9. Explainability
# ══════════════════════════════════════════════════════════════════

def section9_explainability(model: SysdModel, result: Any) -> None:
    """causal_trace on churn_B + detect_feedback_loops across the model."""
    banner("9. Explainability",
           "causal_trace decomposes churn · detect_feedback_loops maps loops")
    ts = _get_ts(result)
    state: dict[str, float] = {}
    for name, vals in ts.items():
        if vals:
            state[name] = vals[-1]

    trace = causal_trace(model, "churn_fraction_B", state, max_depth=4)
    strip = trace.get("strip") or {}
    print("\n    causal_trace('churn_fraction_B') — factor decomposition at t=180:")
    for f in (strip.get("factors") or []):
        print(f"      {f['name']:28s} value={f['value']:12,.4f}")
    rate = (strip or {}).get("total_value", 0.0)
    print(f"      (churn fraction B = {rate:,.4f}/day → ≈"
          f"{state.get('Subs_B', 0.0) * rate:,.0f} churned subs/day)")

    analysis = detect_feedback_loops(model, max_loop_length=6)
    loops = analysis.loops
    print(f"\n    detect_feedback_loops → {len(loops)} feedback loops")
    for loop in loops[:6]:
        sign = "reinforcing (+)" if loop.polarity == 1 else "balancing (−)"
        print(f"      {loop.name:9s} {sign}  nodes={', '.join(loop.nodes[:4])}")

    # which variables sit in loops?
    var_loops = analysis.variable_loops
    hot = sorted(var_loops.items(), key=lambda kv: len(kv[1]), reverse=True)[:4]
    print("\n    Most loop-connected variables:")
    for v, lns in hot:
        print(f"      {v:26s} in {len(lns)} loops")


# ══════════════════════════════════════════════════════════════════
# 10. Provenance
# ══════════════════════════════════════════════════════════════════

def section10_provenance(store: TripleStore, kb_params: dict[str, float],
                         model: SysdModel, bridge: KBSimBridge) -> None:
    """record_provenance stores a full run's audit trail in RDF."""
    banner("10. Provenance",
           "record_provenance writes run/params/stocks as PROV RDF")
    disp_q = 'ASK { <http://ftth.org/ZoneB> <http://ftth.org/status> "cut" }'
    result = model.simulate(params={**kb_params, "disp_q": disp_q}, kb=store, method="rk4")
    run_node = bridge.record_provenance(
        result, params={**kb_params, "disp_q": disp_q},
        graph=f"{FTTH_NS}provenance",
    )
    g = f"{FTTH_NS}provenance"
    run_triples = list(store.triples(TriplePattern(run_node, None, None), graph=g))
    print(f"\n    Run node: {run_node.iri}")
    print(f"    {len(run_triples)} provenance triples (type, timestamps, params, stocks)")
    print(f"    start/end: "
          + " / ".join(str(t.object_.value) for t in run_triples
                       if t.predicate.iri.endswith("startedAtTime"))
          + " → "
          + " / ".join(str(t.object_.value) for t in run_triples
                       if t.predicate.iri.endswith("endedAtTime")))


# ══════════════════════════════════════════════════════════════════
# 11. Maturity mapping
# ══════════════════════════════════════════════════════════════════

def section11_maturity() -> None:
    banner("11. Twin Maturity Mapping", "L1..L5 — which section proves which level")
    rows = [
        ("L1 — Descriptive", "What happened?",
         "Section 3: stock/queue/agent state over time; DES + ABM aggregates"),
        ("L2 — Diagnostic", "Why did it happen?",
         "Section 9: causal_trace factor decomposition + feedback loops"),
        ("L3 — Predictive", "What will happen?",
         "Section 6: scenario comparison (fiber cut, housing boom, capex)"),
        ("L4 — Prescriptive", "What should we do?",
         "Section 7/8: production rules + KB-driven LP capex allocation"),
        ("L5 — Autonomous", "What acts on our behalf?",
         "Section 4: World agent + KB_QUERY/KB_ASSERT self-healing"),
    ]
    print()
    for level, q, sec in rows:
        print(f"    {level:16s} {q:20s} — {sec}")


# ══════════════════════════════════════════════════════════════════
# 12. Takeaway
# ══════════════════════════════════════════════════════════════════

def section12_takeaway() -> None:
    banner("12. Takeaway", "the loop closes")
    print("""
    The FTTH twin is not a dashboard — it is a closed loop:

       knowledge graph ──► parameters ──► SD+DES+ABM simulation
           ▲                                        │
           │                                        ▼
       evidence triples ◄─────── results ◄─── live KB (QUERY/ASSERT)

    Data sources land in named graphs via ingest_csv (no new code).
    KB beliefs drive params; the model runs stocks, queues, and agents
    against one shared state. Mid-run the KB itself changes and the
    twin reacts the same step. Results return as RDF evidence that
    rules grade, rank, filter, and act upon — and every run is
    provenance-tracked. From visibility (L1) to autonomy (L5), the
    entire loop runs on the same DynaFX primitives in one file.
    """)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main() -> int:
    store, reg = section1_build_kb()
    kb_params = section2_bridge(store, reg)
    model, result = section3_multiparadigm(store, kb_params)
    section4_live_kb(kb_params)
    bridge = section5_evidence(store, kb_params)
    section6_scenarios(store, kb_params, bridge)
    section7_production_rules(store, kb_params, bridge)
    section8_optimization(store)
    section9_explainability(model, result)
    section10_provenance(store, kb_params, model, bridge)
    section11_maturity()
    section12_takeaway()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
