#!/usr/bin/env python3
"""Global Solar EPC Supply Chain Digital Twin.

A living digital twin of a solar EPC enterprise that continuously reasons
about its supply chain during a typhoon-induced port closure disruption.
The twin spans the full L1->L5 decision spectrum:

  L1 Sense    - ingest 7 EPC enterprise CSVs into named graphs + RDFS inference
  L2 Assemble - pull KB facts into simulation params (params_from_kb)
  L3 Model    - SD + ABM + DES model of the global supply chain
  L4 Live     - baseline run, inject typhoon disruption via KB flag,
                ABM agents write live KB triples mid-run
  L5 Decide   - evidence round-trip, scenario grading/ranking/filtering,
                production rules, LP mitigation allocation, causal trace,
                feedback loops, provenance, maturity mapping

Built on the KB + KBSimBridge + ProductionRules + ScenarioComparison +
SensitivityAnalyzer + lp_minimize stack. Run:  python examples/global_solar_epc_twin.py
"""

import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dynafx.dynamics import parse_sysd_file
from dynafx.dynamics.causal import causes_strip
from dynafx.dynamics.feedback import detect_feedback_loops
from dynafx.dynamics.scenario import ScenarioComparison, ScenarioDef
from dynafx.dynamics.sensitivity import SensitivityAnalyzer
from dynafx.dynamics.optimization import kb_lp_minimize

from dynafx.bridge import KBSimBridge
from dynafx.knowledge import sparql_evaluate, parse_sparql
from dynafx.knowledge.store import TripleStore
from dynafx.knowledge.inference import InferencePattern, RuleEngine, rdfs_rules
from dynafx.knowledge.model import (
    NamedNode, Literal, Triple, TriplePattern,
    XSD_BOOLEAN, XSD_DOUBLE, XSD_INTEGER,
)
from dynafx.knowledge.production import (
    ProductionRule, ProductionRuleEngine, TripleCondition,
    ComparisonCondition, LogAction, TripleAction,
)
from dynafx.knowledge.ingest_csv import ingest_csv, load_all_mappings

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = DATA_DIR / "models" / "global_solar_epc.sysd"
MAPPINGS_DIR = DATA_DIR / "mappings"
ONTOLOGY_FILE = DATA_DIR / "epc-ontology.ttl"

EPC_NS = "http://epc.org/"
G_PROJECTS = "http://epc.org/graphs/projects"
G_SUPPLIERS = "http://epc.org/graphs/suppliers"
G_LOGISTICS = "http://epc.org/graphs/logistics"
G_WORKFORCE = "http://epc.org/graphs/workforce"
G_META = "http://epc.org/graphs/meta"
G_EVIDENCE = "http://epc.org/graphs/evidence"

DISRUPTION_Q = f"PREFIX epc: <{EPC_NS}> ASK {{ epc:GlobalDisruption epc:active true }}"
SUPPLIER_Q = f"PREFIX epc: <{EPC_NS}> SELECT ?v WHERE {{ epc:Portfolio epc:aggregateSupplierReliability ?v }}"
PROJECTS_Q = f"PREFIX epc: <{EPC_NS}> SELECT ?v WHERE {{ epc:Portfolio epc:projectsAtRisk ?v }}"


def _epc(name: str) -> NamedNode:
    return NamedNode(f"{EPC_NS}{name}")


def _lit_num(val, dtype=XSD_DOUBLE) -> Literal:
    return Literal(str(val), datatype=dtype)


def _lit_bool(val: bool) -> Literal:
    return Literal("true" if val else "false", datatype=XSD_BOOLEAN)


def _sparql_val(store: TripleStore, query: str) -> float:
    try:
        ast = parse_sparql(query)
        r = sparql_evaluate(ast, store)
        if hasattr(r, "bindings") and r.bindings and r.bindings[0]:
            items = list(r.bindings[0].values())
            if items:
                return float(items[0].value)
        if getattr(r, "cardinality", 0) > 0:
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def _disruption_active(store: TripleStore) -> bool:
    """Check the KB disruption flag via direct scan (ASK boolean matching is literal-typed)."""
    for t in store.triples(TriplePattern(subject=_epc("GlobalDisruption"),
                                         predicate=_epc("active"))):
        if t.object_.value == "true":
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 1. SENSE — enterprise KB from CSVs
# ══════════════════════════════════════════════════════════════════════════════

def sense() -> TripleStore:
    print("=" * 78)
    print("1. SENSE  — ingest 7 EPC enterprise CSVs into named graphs")
    print("=" * 78)

    store = TripleStore()

    # Ontology first (meta graph)
    from dynafx.knowledge.turtle import parse_turtle
    if ONTOLOGY_FILE.exists():
        onto = parse_turtle(ONTOLOGY_FILE.read_text())
        for triple in onto.all_triples():
            store.add(triple, G_META)
        print(f"  Ontology loaded into {G_META}")

    # Enterprise CSVs via YAML mappings
    for md in load_all_mappings(MAPPINGS_DIR):
        if not md.csv.startswith("epc_"):
            continue
        csv_path = DATA_DIR / md.csv
        report = ingest_csv(md, str(csv_path), store, strict=False)
        print(f"  {md.csv:<26} -> {report.triples_added:>5} triples ({report.rows_parsed} rows)")

    # RDFS inference (domain/range class derivation)
    engine = RuleEngine(rdfs_rules())
    engine.apply(store)
    print(f"  RDFS inference applied ({len(store.graphs())} named graphs)")

    # Portfolio aggregates from enterprise CSVs (SPARQL has no GROUP BY)
    import csv as _csv
    rows_s = list(_csv.DictReader(open(DATA_DIR / "epc_suppliers.csv", newline="")))
    rows_p = list(_csv.DictReader(open(DATA_DIR / "epc_projects.csv", newline="")))
    rows_c = list(_csv.DictReader(open(DATA_DIR / "epc_containers.csv", newline="")))
    portfolio, disruption = _epc("Portfolio"), _epc("GlobalDisruption")
    store.add(Triple(portfolio, _epc("type"), _epc("Portfolio")), G_META)
    store.add(Triple(disruption, _epc("type"), _epc("Disruption")), G_META)

    avg_rel = round(mean(float(r["reliability"]) for r in rows_s), 3)
    at_risk = sum(1 for r in rows_p if r["status"] in ("at_risk", "delayed"))
    active = sum(1 for r in rows_p if r["status"] == "active")
    total_mw = round(sum(float(r["capacity_mw"]) for r in rows_p), 1)
    in_transit = sum(1 for r in rows_c if r["status"] == "in_transit")
    store.add(Triple(portfolio, _epc("aggregateSupplierReliability"), _lit_num(avg_rel)), G_META)
    store.add(Triple(portfolio, _epc("projectsAtRisk"), _lit_num(at_risk, XSD_INTEGER)), G_META)
    store.add(Triple(portfolio, _epc("activeProjects"), _lit_num(active, XSD_INTEGER)), G_META)
    store.add(Triple(portfolio, _epc("totalCapacityMW"), _lit_num(total_mw)), G_META)
    store.add(Triple(portfolio, _epc("containersInTransit"), _lit_num(in_transit, XSD_INTEGER)), G_META)
    store.add(Triple(disruption, _epc("active"), _lit_bool(False)), G_META)

    print(f"  Portfolio: {active} active projects / {at_risk} at risk / "
          f"{total_mw} MW / {in_transit} containers in transit")
    print(f"  Aggregate supplier reliability: {avg_rel:.3f}")
    return store


# ══════════════════════════════════════════════════════════════════════════════
# 2. ASSEMBLE — KB -> simulation params
# ══════════════════════════════════════════════════════════════════════════════

def assemble(store: TripleStore) -> dict:
    print("\n" + "=" * 78)
    print("2. ASSEMBLE — map KB facts to simulation params")
    print("=" * 78)
    bridge = KBSimBridge(store)
    claim_map = [
        (_epc("Portfolio"), _epc("aggregateSupplierReliability"), None, "kb_supplier_reliability"),
        (_epc("Portfolio"), _epc("projectsAtRisk"), None, "kb_projects_at_risk"),
        (_epc("Portfolio"), _epc("activeProjects"), None, "kb_active_projects"),
        (_epc("Portfolio"), _epc("containersInTransit"), None, "kb_in_transit"),
        (_epc("Portfolio"), _epc("totalCapacityMW"), None, "kb_total_mw"),
    ]
    kb_params = bridge.params_from_kb(claim_map, default=0.0, exclude_graphs=set())
    for name, val in kb_params.items():
        print(f"  {name:<24} <- {val}")
    return kb_params


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL — parse SD+ABM+DES twin
# ══════════════════════════════════════════════════════════════════════════════

def model_twin() -> object:
    print("\n" + "=" * 78)
    print("3. MODEL  — global solar EPC supply chain (SD + ABM + DES)")
    print("=" * 78)
    model = parse_sysd_file(str(MODEL_PATH))
    flows = sum(len(s.flows) for s in model.stocks)
    print(f"  {len(model.stocks)} stocks, {len(model.aux_vars)} auxes, {flows} flows")
    des = getattr(model, "des_specs", None)
    if des:
        print(f"  DES: {len(des.queues)} queues + {len(des.resources)} resources")
    abm = getattr(model, "agent_defs", None)
    if abm:
        print(f"  ABM: {len(abm)} agent definitions")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. LIVE — baseline, typhoon injection, live KB writes
# ══════════════════════════════════════════════════════════════════════════════

def live(store: TripleStore, model) -> tuple:
    print("\n" + "=" * 78)
    print("4. LIVE  — baseline run + typhoon disruption via KB flag")
    print("=" * 78)
    base_params = {
        "disruption_q": DISRUPTION_Q,
        "supplier_q": SUPPLIER_Q,
        "projects_q": PROJECTS_Q,
    }

    # Baseline (no disruption)
    base = model.simulate(params=base_params, kb=store, method="euler", dt=1.0)
    base_profit = base.values["Portfolio_Revenue"][-1] - base.values["Portfolio_Cost"][-1]
    base_completion = base.aux_values.get("completion_pct", [0])[-1]
    print(f"  Baseline     completion {base_completion:.1%}  profit ${base_profit:,.0f}K")

    # Typhoon hits: flip the GlobalDisruption flag in the meta graph
    print("  Typhoon makes landfall at the Asian chokepoint port...")
    disruption = _epc("GlobalDisruption")
    for t in store.triples(TriplePattern(subject=disruption, predicate=_epc("active")), graph=G_META):
        store.remove(t, graph=G_META)
    store.add(Triple(disruption, _epc("active"), _lit_bool(True)), G_META)
    active = _disruption_active(store)
    print(f"  KB disruption flag now active: {active}")

    disrupt_params = {**base_params, "disruption_start_day": 180.0,
                      "disruption_duration": 30.0, "disruption_severity": 0.85}
    dis = model.simulate(params=disrupt_params, kb=store, method="euler", dt=1.0)
    dis_profit = dis.values["Portfolio_Revenue"][-1] - dis.values["Portfolio_Cost"][-1]
    dis_completion = dis.aux_values.get("completion_pct", [0])[-1]
    impact = dis_profit - base_profit
    print(f"  Disrupted    completion {dis_completion:.1%}  profit ${dis_profit:,.0f}K  (impact ${impact:,.0f}K)")

    # ABM agents wrote live KB triples mid-run
    rels = list(store.triples(TriplePattern(subject=_epc("Supplier"), predicate=_epc("reliability"))))
    pens = list(store.triples(TriplePattern(subject=_epc("ProjectManager"), predicate=_epc("penaltyRisk"))))
    print(f"  Live ABM KB writes: {len(rels)} supplier reliability triples, "
          f"{len(pens)} project-manager penalty triples")
    if dis.abm_engine:
        m = dis.abm_engine.get_metrics()
        print(f"  ABM avg reliability {m.get('Supplier_reliability_avg', 0):.3f}, "
              f"penalty risk {m.get('ProjectManager_penalty_risk_avg', 0):.3f}")
    if dis.des_engine:
        stats = dis.des_engine.get_all_stats()
        qnames = sorted(k for k in stats if isinstance(stats[k], dict) and "utilization" in stats[k])
        print(f"  DES queues monitored: {', '.join(qnames[:5])}")
    return base, dis, base_params


# ══════════════════════════════════════════════════════════════════════════════
# 5. LEARN — simulation -> evidence triples
# ══════════════════════════════════════════════════════════════════════════════

def learn(store: TripleStore, bridge: KBSimBridge, dis) -> dict:
    print("\n" + "=" * 78)
    print("5. LEARN  — simulation results round-trip into the KB")
    print("=" * 78)
    evidence_map = [
        ("Portfolio_Revenue", _epc("Portfolio"), _epc("revenueK"),
         lambda i, f: round(f[-1] / 1000.0, 3)),
        ("Portfolio_Cost", _epc("Portfolio"), _epc("costK"),
         lambda i, f: round(f[-1] / 1000.0, 3)),
        ("Penalty_Accrual", _epc("Portfolio"), _epc("penaltyK"),
         lambda i, f: round(f[-1] / 1000.0, 3)),
        ("Global_Panel_Supply", _epc("Portfolio"), _epc("supplyBuffer"),
         lambda i, f: round(f[-1], 1)),
    ]
    triples = bridge.evidence_from_result(dis, evidence_map, graph=G_EVIDENCE)
    completion = dis.aux_values.get("completion_pct", [0])[-1]
    health = dis.aux_values.get("portfolio_health", [0])[-1]
    triples.append(Triple(_epc("Portfolio"), _epc("completionPct"), _lit_num(round(completion, 4))))
    triples.append(Triple(_epc("Portfolio"), _epc("portfolioHealth"), _lit_num(round(health, 4))))
    for t in triples:
        store.add(t, graph=G_EVIDENCE)
    print(f"  Wrote {len(triples)} evidence triples -> {G_EVIDENCE}")
    for t in triples[:6]:
        print(f"    {t.subject.iri.rsplit('/', 1)[-1]}.{t.predicate.iri.rsplit('/', 1)[-1]} = {t.object_.value}")
    return {"completion": completion, "health": health}


# ══════════════════════════════════════════════════════════════════════════════
# 6. PREDICT — scenario comparison + sensitivity (fresh-store grading)
# ══════════════════════════════════════════════════════════════════════════════

def predict(model, base_params, store: TripleStore) -> ScenarioComparison:
    print("\n" + "=" * 78)
    print("6. PREDICT — scenario comparison across disruption profiles")
    print("=" * 78)
    scenarios = [
        ScenarioDef("Baseline", {**base_params}),
        ScenarioDef("Moderate", {**base_params, "disruption_start_day": 180, "disruption_duration": 60, "disruption_severity": 0.7}),
        ScenarioDef("Severe", {**base_params, "disruption_start_day": 180, "disruption_duration": 90, "disruption_severity": 0.95}),
        ScenarioDef("Late Disruption", {**base_params, "disruption_start_day": 270, "disruption_duration": 45, "disruption_severity": 0.85}),
        ScenarioDef("Extended Recovery", {**base_params, "disruption_start_day": 150, "disruption_duration": 120, "disruption_severity": 0.8}),
        ScenarioDef("Smooth Recovery", {**base_params, "disruption_start_day": 180, "disruption_duration": 30, "disruption_severity": 0.5}),
    ]
    sc = ScenarioComparison(model, scenarios, method="euler", dt=1.0, kb=store)
    for sr in sc.scenarios:
        profit = (sr.result.values["Portfolio_Revenue"][-1] -
                  sr.result.values["Portfolio_Cost"][-1])
        completion = sr.result.aux_values.get("completion_pct", [0])[-1]
        print(f"  {sr.name:<20} profit ${profit:>12,.0f}K  completion {completion:.1%}")
    return sc


def predict_grading(sc: ScenarioComparison, store: TripleStore) -> None:
    print("\n  Scenario grading / ranking (fresh-store evidence):")
    fresh = TripleStore()
    bridge = KBSimBridge(store)
    # scoring fns return 0..1 where HIGHER is better
    evidence_map = [
        ("Portfolio_Revenue", _epc("Portfolio"), _epc("revenueScore"),
         lambda i, f: min(1.0, f[-1] / 1_200_000.0)),
        ("Penalty_Accrual", _epc("Portfolio"), _epc("penaltyScore"),
         lambda i, f: max(0.0, 1.0 - f[-1] / 500_000.0)),
    ]
    grade_specs = [
        (f"SELECT ?v WHERE {{ <{EPC_NS}Portfolio> <{EPC_NS}revenueScore> ?v }}", "v", 0.8, 0.2),
        (f"SELECT ?v WHERE {{ <{EPC_NS}Portfolio> <{EPC_NS}penaltyScore> ?v }}", "v", 0.8, 0.2),
    ]
    ranked = sc.rank(grade_specs, fresh, evidence_map=evidence_map, bridge=bridge)
    for name, score in ranked:
        print(f"    {name:<20} grade {score:.3f}")

    # FILTER keeps only scenarios that meet a completion bar
    print("  FILTER completion>75%:")
    keep_q = f"ASK {{ <{EPC_NS}Portfolio> <{EPC_NS}completionPct> ?v FILTER(?v > 0.75) }}"
    pre = len(sc.scenarios)
    surviving = []
    for sr in sc.scenarios:
        completion = sr.result.aux_values.get("completion_pct", [0])[-1]
        fresh.add(Triple(_epc("Portfolio"), _epc("completionPct"), _lit_num(round(completion, 4))),
                  graph="_filter_temp")
        ast = parse_sparql(keep_q)
        qr = sparql_evaluate(ast, fresh)
        passed = qr.cardinality > 0
        fresh.remove(TriplePattern(_epc("Portfolio"), _epc("completionPct")), graph="_filter_temp")
        print(f"      {'KEEP' if passed else 'drop'}  {sr.name:<20} completion {completion:.1%}")
        if passed:
            surviving.append(sr)
    print(f"    {len(surviving)}/{pre} scenarios pass the filter")


# ══════════════════════════════════════════════════════════════════════════════
# 7. ACT — production rules on evidence
# ══════════════════════════════════════════════════════════════════════════════

def act(store: TripleStore, severe) -> list:
    print("\n" + "=" * 78)
    print("7. ACT    — production rules detect portfolio risk")
    print("=" * 78)
    # Rules evaluate against an isolated evidence snapshot (no cross-graph leakage)
    snapshot = TripleStore()
    severe_completion = severe.aux_values.get("completion_pct", [0])[-1]
    snapshot.add(Triple(_epc("Portfolio"), _epc("completionPct"),
                        _lit_num(round(severe_completion, 4))), "evidence")
    eng = ProductionRuleEngine(snapshot)

    at_risk = ProductionRule(
        name="portfolio-at-risk",
        description="Portfolio completion below target requires mitigation",
        body=[
            TripleCondition(InferencePattern(subject=_epc("Portfolio"),
                                             predicate=_epc("completionPct"), object_="?v")),
            ComparisonCondition("?v", "<", 0.75),
        ],
        head=[
            LogAction("Portfolio completion below 75% - mitigation required"),
            TripleAction(_epc("Portfolio"), _epc("requiresMitigation"),
                         Literal(1.0, datatype=XSD_DOUBLE)),
        ],
        fire_once=False,
    )
    on_track = ProductionRule(
        name="portfolio-on-track",
        description="Portfolio healthy",
        body=[
            TripleCondition(InferencePattern(subject=_epc("Portfolio"),
                                             predicate=_epc("completionPct"), object_="?v")),
            ComparisonCondition("?v", ">=", 0.75),
        ],
        head=[LogAction("Portfolio completion at or above target")],
        fire_once=False,
    )
    eng.add_rule(at_risk)
    eng.add_rule(on_track)
    results = eng.evaluate()
    fired = [a.action_type for a in results if a.success]
    print(f"  Severe scenario completion {severe_completion:.1%} -> fired actions: {fired}")
    print(f"  risk rule {eng._fired_count.get('portfolio-at-risk', 0)}x, "
          f"on-track rule {eng._fired_count.get('portfolio-on-track', 0)}x")
    mit = list(snapshot.triples(InferencePattern(subject=_epc("Portfolio"),
                                                 predicate=_epc("requiresMitigation"))))
    print(f"  requiresMitigation triples: {len(mit)}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 8. OPTIMIZE — LP mitigation allocation from KB
# ══════════════════════════════════════════════════════════════════════════════

def optimize(store: TripleStore) -> None:
    print("\n" + "=" * 78)
    print("8. OPTIMIZE — LP mitigation budget allocation (KB-driven)")
    print("=" * 78)
    LP_NS = "http://epc.org/lp/mitigation/"
    def _lp(name): return NamedNode(f"{LP_NS}{name}")
    g = "_lp"

    # Three levers: port_capacity boost, crew boost, buffer stock
    levers = [("port_capacity", 1.0, 3), ("crew_boost", 0.8, 4), ("buffer_stock", 0.6, 5)]
    for i, (name, cost, hi) in enumerate(levers):
        store.add(Triple(_lp(f"obj_{name}"), _lp("coeff"), _lit_num(cost)), g)
        store.add(Triple(_lp(f"b_{name}"), _lp("rowIndex"), _lit_num(i, XSD_INTEGER)), g)
        store.add(Triple(_lp(f"b_{name}"), _lp("lo"), _lit_num(0)), g)
        store.add(Triple(_lp(f"b_{name}"), _lp("hi"), _lit_num(hi)), g)

    # budget row: 1.0*x0 + 0.8*x1 + 0.6*x2 <= 2.0
    store.add(Triple(_lp("budget"), _lp("rowIndex"), _lit_num(0, XSD_INTEGER)), g)
    for j, (_, cost, _) in enumerate(levers):
        store.add(Triple(_lp("budget"), _lp(f"c{j}"), _lit_num(cost)), g)
    store.add(Triple(_lp("budget"), _lp("rhs"), _lit_num(2.0)), g)
    # min-gain row: 0.6*x0 + 0.4*x1 + 0.3*x2 >= 1.0  => -0.6x0 -0.4x1 -0.3x2 <= -1.0
    gains = [0.6, 0.4, 0.3]
    store.add(Triple(_lp("gain"), _lp("rowIndex"), _lit_num(1, XSD_INTEGER)), g)
    for j, gr in enumerate(gains):
        store.add(Triple(_lp("gain"), _lp(f"c{j}"), _lit_num(-gr)), g)
    store.add(Triple(_lp("gain"), _lp("rhs"), _lit_num(-1.0)), g)

    c_q = f"SELECT ?v WHERE {{ ?o <{LP_NS}coeff> ?v . }}"
    bounds_q = (f"SELECT ?lo ?hi WHERE {{ ?b <{LP_NS}rowIndex> ?i . "
                f"?b <{LP_NS}lo> ?lo . ?b <{LP_NS}hi> ?hi . }} ORDER BY ?i")
    a_q = (f"SELECT ?v0 ?v1 ?v2 WHERE {{ ?r <{LP_NS}rowIndex> ?i . "
           f"?r <{LP_NS}c0> ?v0 . ?r <{LP_NS}c1> ?v1 . ?r <{LP_NS}c2> ?v2 . }} ORDER BY ?i")
    b_q = f"SELECT ?v WHERE {{ ?r <{LP_NS}rowIndex> ?i . ?r <{LP_NS}rhs> ?v . }} ORDER BY ?i"

    res = kb_lp_minimize(store, c_q, bounds_q, a_q, b_q, var_count=3)
    names = [name for name, _, _ in levers]
    print(f"  LP solve success={res.success} objective=${res.objective_value:.2f}K")
    for name, x in zip(names, res.x):
        print(f"    {name:<16} x = {x:.2f}")
    print("  Constraint: budget <= $2.0K, minimum completion gain >= 1.0%")
    print("  (spend the full budget on the cheapest lever per unit of gain)")


# ══════════════════════════════════════════════════════════════════════════════
# 9. DIAGNOSE — causal strip + feedback loops
# ══════════════════════════════════════════════════════════════════════════════

def diagnose(model, base) -> None:
    print("\n" + "=" * 78)
    print("9. DIAGNOSE — causal anatomy + feedback loops")
    print("=" * 78)
    final = {}
    for s in model.stocks:
        vals = base.values.get(s.name, [])
        if vals:
            final[s.name] = vals[-1]
    for a in model.aux_vars:
        vals = base.aux_values.get(a.name, [])
        if vals:
            final[a.name] = vals[-1]
    strip = causes_strip(model, "Portfolio_Revenue", final)
    print(f"  Portfolio_Revenue driven by {len(strip.factors)} factors:")
    for f in sorted(strip.factors, key=lambda d: -abs(d.get("value", 0)))[:5]:
        print(f"    {f.get('name'):<24} {f.get('value', 0):+.4f}")
    loops = detect_feedback_loops(model)
    print(f"  Feedback loops detected: {len(loops.loops)}")
    for lp in loops.loops:
        print(f"    {lp.name}  ({lp.polarity})")


# ══════════════════════════════════════════════════════════════════════════════
# 10. PROVENANCE — record the run as RDF
# ══════════════════════════════════════════════════════════════════════════════

def provenance(store: TripleStore, bridge: KBSimBridge, dis, params: dict) -> None:
    print("\n" + "=" * 78)
    print("10. PROVENANCE — audit trail of the disrupted run")
    print("=" * 78)
    run = bridge.record_provenance(
        dis, params=params, graph="provenance",
        extra_annotations=[
            Triple(_epc("GlobalDisruption"), _epc("active"), _lit_bool(True)),
            Triple(_epc("Portfolio"), _epc("completionPct"),
                   _lit_num(round(dis.aux_values.get("completion_pct", [0])[-1], 4))),
        ],
    )
    n = len(list(store.triples(InferencePattern(subject=run), graph="provenance")))
    print(f"  Run entity {run.iri.rsplit('/', 1)[-1]} with {n} provenance triples")


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAP — L1->L5 maturity ladder
# ══════════════════════════════════════════════════════════════════════════════

def maturity() -> None:
    print("\n" + "=" * 78)
    print("11. MAP    — digital twin maturity ladder")
    print("=" * 78)
    ladder = [
        ("L1 Sense", "Named-graph enterprise KB + RDFS inference"),
        ("L2 Assemble", "KB facts -> simulation params"),
        ("L3 Model", "SD stocks + DES queues + ABM agents"),
        ("L4 Live", "KB-driven disruption + live ABM KB writes"),
        ("L5 Decide", "Evidence, scenarios, rules, LP, causal, provenance"),
    ]
    for level, desc in ladder:
        print(f"    {level:<10} {desc}")
    print("    This twin implements all five levels end-to-end.")


# ══════════════════════════════════════════════════════════════════════════════
# 12. TAKEAWAY
# ══════════════════════════════════════════════════════════════════════════════

def takeaway(store: TripleStore, dis, impact: float) -> None:
    print("\n" + "=" * 78)
    print("12. TAKEAWAY — executive summary")
    print("=" * 78)
    rel = _sparql_val(store, SUPPLIER_Q)
    risk = _sparql_val(store, PROJECTS_Q)
    print(f"  A typhoon-induced port closure of 30 days costs ${abs(impact):,.0f}K in profit "
          f"with supplier reliability at {rel:.2f} and {int(risk)} projects at risk.")
    print("  The twin turned static CSVs into a closed loop: enterprise facts drive")
    print("  the simulation, results return as evidence, rules flag risk, and LP")
    print("  allocates a mitigation budget - all in one living knowledge graph.")
    print(f"  (evidence graph: {G_EVIDENCE}; disruption flag: {_disruption_active(store)})")


def main() -> int:
    store = sense()
    assemble(store)
    model = model_twin()
    base, dis, base_params = live(store, model)
    bridge = KBSimBridge(store)
    evidence = learn(store, bridge, dis)
    sc = predict(model, base_params, store)
    predict_grading(sc, store)
    severe = sc.get("Severe")
    act(store, severe.result if severe else dis)
    optimize(store)
    diagnose(model, base)
    provenance(store, bridge, dis, {**base_params, "disruption_start_day": 180.0})
    maturity()
    impact = (dis.values["Portfolio_Revenue"][-1] - dis.values["Portfolio_Cost"][-1]) - \
             (base.values["Portfolio_Revenue"][-1] - base.values["Portfolio_Cost"][-1])
    takeaway(store, dis, impact)
    print("\nDigital twin example complete. exit 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
