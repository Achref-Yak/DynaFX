#!/usr/bin/env python3
"""
KB Showcase — 6 KB↔Simulation Integration Patterns
====================================================

A self-contained demonstration of every KB integration pattern in DynaFX:

  1. KB seeding + RDFS inference
  2. Pre-flight:  KBSimBridge.params_from_kb()
  3. Mid-flight:  KB_QUERY builtins in model expressions
  4. Post-flight: KBSimBridge.evidence_from_result()
  5. Closed-loop: ClosedLoopReasoner (simulate → grade → nudge → re-simulate)
  6. KB-constrained optimization: kb_lp_minimize / kb_calibrate

Output: /tmp/kb_showcase_dashboard.html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dynafx import (
    KBSimBridge,
    ClosedLoopReasoner,
    ReasoningPass,
    TripleStore,
    parse_sysd_file,
    kb_lp_minimize,
)
from dynafx.knowledge.model import NamedNode, Literal, Triple, TriplePattern
from dynafx.knowledge.inference import RuleEngine, rdfs_rules
from dynafx.knowledge.production import (
    ProductionRule,
    TripleCondition,
    TripleAction,
    ProductionRuleEngine,
)

NS = "http://sc.org/"
S = lambda n: NamedNode(f"{NS}{n}")
P = lambda n: NamedNode(f"{NS}{n}")

OUTPUT_PATH = "/tmp/kb_showcase_dashboard.html"


def _seed_kb() -> TripleStore:
    """Seed the KB with disruption intelligence, supplier data, contracts."""
    store = TripleStore()

    store.add(Triple(
        S("GlobalDisruption"), P("active"), Literal("false"),
    ), graph="disruption")
    store.add(Triple(
        S("GlobalDisruption"), P("severity"), Literal("0.7"),
    ), graph="disruption")

    store.add(Triple(
        S("Supplier_A"), P("reliability"), Literal("0.92"),
    ), graph="suppliers")
    store.add(Triple(
        S("Supplier_B"), P("reliability"), Literal("0.65"),
    ), graph="suppliers")
    store.add(Triple(
        S("Supplier_A"), P("region"), Literal("asia"),
    ), graph="suppliers")
    store.add(Triple(
        S("Supplier_B"), P("region"), Literal("europe"),
    ), graph="suppliers")

    store.add(Triple(
        S("Contract_A"), P("supplier"), S("Supplier_A"),
    ), graph="contracts")
    store.add(Triple(
        S("Contract_A"), P("safetyStock"), Literal("300"),
    ), graph="contracts")

    store.add(Triple(
        S("CurrentState"), P("hasStatus"), Literal("normal"),
    ), graph="scenarios")

    store.add(Triple(
        S("Portfolio"), P("type"), S("Portfolio"),
    ), graph="meta")
    store.add(Triple(
        S("GlobalDisruption"), P("type"), S("Disruption"),
    ), graph="meta")
    store.add(Triple(
        S("Supplier_A"), P("type"), S("Supplier"),
    ), graph="meta")
    store.add(Triple(
        S("Supplier_B"), P("type"), S("Supplier"),
    ), graph="meta")

    return store


def _rdfs_hierarchy(store: TripleStore) -> None:
    """Add RDFS class hierarchy and run inference."""
    RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    RDFS_SUBCLASS = NamedNode("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    RDFS_DOMAIN = NamedNode("http://www.w3.org/2000/01/rdf-schema#domain")
    RDFS_RANGE = NamedNode("http://www.w3.org/2000/01/rdf-schema#range")

    store.add(Triple(S("Entity"), RDFS_SUBCLASS, S("Thing")), graph="ontology")
    store.add(Triple(S("Supplier"), RDFS_SUBCLASS, S("Entity")), graph="ontology")
    store.add(Triple(S("Disruption"), RDFS_SUBCLASS, S("Event")), graph="ontology")
    store.add(Triple(P("reliability"), RDFS_DOMAIN, S("Supplier")), graph="ontology")
    store.add(Triple(P("reliability"), RDFS_RANGE, S("NumericRating")), graph="ontology")

    RuleEngine(rdfs_rules()).apply(store)


def _fill_rate_score(init: list[float], final: list[float]) -> float:
    if not init or not final:
        return 0.0
    cum_demand = max(0.001, final[0] - init[0])
    cum_met = max(0.0, final[-1] - init[-1])
    return min(1.0, cum_met / cum_demand) * 0.8 + 0.1


def _inventory_risk_score(init: list[float], final: list[float]) -> float:
    if not init:
        return 0.5
    avg = sum(init + final) / len(init + final)
    safety = 300.0
    return 0.9 if avg >= safety else max(0.1, avg / safety * 0.8)


def _grade_update(grades: dict[str, float], kb_store: TripleStore) -> dict:
    sev = 0.0
    for t in kb_store.triples(TriplePattern(S("GlobalDisruption"), P("severity"), None), graph="disruption"):
        if hasattr(t.object_, "value"):
            try:
                sev = float(t.object_.value)
            except (ValueError, TypeError):
                sev = 0.0

    fill_ok = all(v >= 0.5 for k, v in grades.items() if "fill" in k.lower())
    if sev > 0.3 or not fill_ok:
        return {"safety_stock": 400.0, "expedite_factor": 1.5, "recovery_active": 1.0}
    return {}


def build_dashboard(results, passes, ev_triples, lp_result) -> str:
    """Build interactive Plotly dashboard showing demo results."""
    titles = ["Pass 1: Baseline", "Pass 2: Disruption", "Pass 3: Recovery"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Retailer Inventory", "Fill Rate",
            "KB Evidence Triples", "Optimization",
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "table"}, {"type": "table"}]],
    )
    colors = ["#2ca02c", "#d62728", "#1f77b4"]

    for i, res in enumerate(results):
        t = res.times
        inv = res.values.get("Retailer_Inventory", [])
        dem = res.values.get("Cumulative_Demand", [1])
        met = res.values.get("Cumulative_Met", [0])
        fill = [m / max(d, 0.001) for m, d in zip(met, dem)] if met and dem else []
        label = titles[i] if i < len(titles) else f"Pass {i+1}"

        if inv:
            fig.add_trace(
                go.Scatter(x=list(t[:len(inv)]), y=inv, mode="lines",
                           name=f"{label} — Inventory",
                           line=dict(color=colors[i % len(colors)])),
                row=1, col=1,
            )
        if fill:
            fig.add_trace(
                go.Scatter(x=list(t[:len(fill)]), y=fill, mode="lines",
                           name=f"{label} — Fill Rate",
                           line=dict(color=colors[i % len(colors)], dash="dot")),
                row=1, col=2,
            )

    evidence_headers = ["Subject", "Predicate", "Value", "Belief"]
    evidence_rows = []
    for t in ev_triples:
        subj = str(t.subject).split("/")[-1][:20]
        pred = str(t.predicate).split("/")[-1][:20]
        val = str(getattr(t.object_, "value", str(t.object_)))[:10]
        bel = "1.0"
        evidence_rows.append([subj, pred, val, bel])

    fig.add_trace(
        go.Table(
            header=dict(values=evidence_headers, align="left"),
            cells=dict(values=list(zip(*evidence_rows)) if evidence_rows else [[""]*4],
                       align="left"),
        ),
        row=2, col=1,
    )

    opt_rows = [
        ["Objective", "3*x0 + 1*x1"],
        ["x0 (optimal)", f"{lp_result.x[0]:.2f}"],
        ["x1 (optimal)", f"{lp_result.x[1]:.2f}"],
        ["Objective Value", f"{lp_result.objective_value:.1f}"],
        ["Success", str(lp_result.success)],
    ] if lp_result else [["No LP result", ""]]
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], align="left"),
            cells=dict(values=list(zip(*opt_rows)), align="left"),
        ),
        row=2, col=2,
    )

    fig.update_layout(
        title_text=f"KB Showcase — {len(results)} Passes, {len(ev_triples)} Evidence Triples",
        height=800, showlegend=True,
    )

    html = fig.to_html(include_plotlyjs=True, full_html=True)
    return html


def main():
    print("=" * 70)
    print("  DynaFX KB Integration Showcase — 6 Patterns in One Script")
    print("=" * 70)

    model_path = Path(__file__).resolve().parent.parent / "models" / "sc_bridge_demo.sysd"
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)
    model = parse_sysd_file(str(model_path))

    DISRUPTION_Q = "ASK { <http://sc.org/CurrentState> <http://sc.org/hasStatus> \"disrupted\" }"
    NORMAL_Q = "ASK { <http://sc.org/CurrentState> <http://sc.org/hasStatus> \"normal\" }"

    # ═══ PATTERN 1: Seed KB + RDFS inference ═══
    print("\n1) Seeding TripleStore with disruption, supplier, contract data...")
    store = _seed_kb()
    _rdfs_hierarchy(store)
    print(f"   → {len(list(store.all_triples()))} triples across {len(store.graphs())} named graphs")

    print("   → RDFS inference applied: Supplier, Disruption types derived")

    # ═══ PATTERN 2: Pre-flight params_from_kb ═══
    print("\n2) Pre-flight: params_from_kb(claim_map) extracting KB beliefs...")
    bridge = KBSimBridge(store)

    claim_map = [
        (S("GlobalDisruption"), P("severity"), None, "disruption_severity"),
        (S("Supplier_A"), P("reliability"), None, "supplier_reliability"),
        (S("Contract_A"), P("safetyStock"), None, "contract_safety_stock"),
    ]
    params_raw = bridge.params_from_kb(claim_map)
    params_int = bridge.params_from_kb(claim_map, type_coerce={"contract_safety_stock": "int"})
    print(f"   → Raw (belief-weighted): disruption_severity={params_raw.get('disruption_severity'):.2f}, "
          f"supplier_reliability={params_raw.get('supplier_reliability'):.2f}, "
          f"safety_stock={params_raw.get('contract_safety_stock'):.2f}")
    print(f"   → type_coerce=int:       safety_stock={params_int.get('contract_safety_stock')} "
          f"(type={type(params_int.get('contract_safety_stock')).__name__})")

    # ═══ PATTERN 3+4+5: Mid-flight KB_QUERY, Post-flight evidence, Closed-loop ═══
    print("\n3) Running ClosedLoopReasoner with 3 passes...")
    print("   Pass 1: Baseline (no disruption)")
    print("   Pass 2: Disruption (severity=0.7 injected into KB)")
    print("   Pass 3: Recovery (grade_update activates KB-stored policies)")

    pass1 = ReasoningPass(
        name="baseline",
        claim_map=[],
        evidence_map=[
            ("Cumulative_Met", S("NormalOps"), P("fillRateObserved"), _fill_rate_score),
            ("Retailer_Inventory", S("NormalOps"), P("inventoryRisk"), _inventory_risk_score),
        ],
        params_override={
            "disruption_severity": 0.0,
            "safety_stock": 300.0,
            "recovery_active": 0.0,
            "expedite_factor": 1.0,
            "disruption_q": DISRUPTION_Q,
            "normal_q": NORMAL_Q,
        },
    )

    pass2 = ReasoningPass(
        name="disruption",
        claim_map=[
            (S("GlobalDisruption"), P("severity"), None, "disruption_severity"),
        ],
        evidence_map=[
            ("Cumulative_Met", S("Disrupted"), P("fillRateObserved"), _fill_rate_score),
            ("Retailer_Inventory", S("Disrupted"), P("inventoryRisk"), _inventory_risk_score),
        ],
        params_override={
            "safety_stock": 300.0,
            "recovery_active": 0.0,
            "expedite_factor": 1.0,
            "disruption_q": DISRUPTION_Q,
            "normal_q": NORMAL_Q,
            "escalation_threshold": 1.0,
        },
        grade_update=_grade_update,
    )

    pass3 = ReasoningPass(
        name="recovery",
        claim_map=[],
        evidence_map=[
            ("Cumulative_Met", S("Recovery"), P("fillRateObserved"), _fill_rate_score),
            ("Retailer_Inventory", S("Recovery"), P("inventoryRisk"), _inventory_risk_score),
        ],
        params_override={
            "disruption_q": DISRUPTION_Q,
            "normal_q": NORMAL_Q,
            "escalation_threshold": 1.0,
        },
    )

    # Inject disruption into KB before pass 2
    store.add(Triple(
        S("GlobalDisruption"), P("active"), Literal("true"),
    ), graph="disruption")
    store.add(Triple(
        S("CurrentState"), P("hasStatus"), Literal("disrupted"),
    ), graph="scenarios")

    reasoner = ClosedLoopReasoner(
        bridge, model,
        passes=[pass1, pass2, pass3],
        evidence_graph="simulation",
        provenance_graph="provenance",
    )
    cl_result = reasoner.run()

    print("\n   ── Pipeline Results ──")
    for i, (rp, res) in enumerate(zip(cl_result.passes, cl_result.results)):
        final_fill = res.values.get("Cumulative_Met", [0])[-1]
        final_demand = res.values.get("Cumulative_Demand", [1])[-1]
        fill = final_fill / max(final_demand, 0.001)
        inv = res.values.get("Retailer_Inventory", [0])[-1]
        n_agents = len(res.abm_engine.instances) if res.abm_engine and hasattr(res.abm_engine, 'instances') else 0
        print(f"     Pass {i+1} ({rp.name:>10s}):  "
              f"fill_rate={fill:.3f}  retailer_inv={inv:.0f}  agents={n_agents}")

    ev_triples = list(store.triples_in_graph("simulation"))
    print(f"\n   → Evidence triples added to KB: {len(ev_triples)}")

    # ═══ PATTERN 6: KB-Constrained Optimization ═══
    print("\n4) KB-constrained optimization...")
    opt_store = TripleStore()
    COEFF = P("coeff")
    BOUND = P("bound")
    opt_store.add(Triple(S("c0"), COEFF, Literal("3.0")), graph="opt")
    opt_store.add(Triple(S("c1"), COEFF, Literal("1.0")), graph="opt")
    opt_store.add(Triple(S("b0"), BOUND, Literal("0.0")), graph="opt")
    opt_store.add(Triple(S("b1"), BOUND, Literal("0.0")), graph="opt")

    c_q = f"SELECT ?v WHERE {{ ?s <{COEFF.iri}> ?v }} ORDER BY ?s"
    b_q = f"SELECT ?v WHERE {{ ?s <{BOUND.iri}> ?v }} ORDER BY ?s"

    lp_result = kb_lp_minimize(opt_store, c_q, b_q, var_count=2)
    print(f"   → LP minimize 3*x0 + 1*x1:  x=[{lp_result.x[0]:.2f}, {lp_result.x[1]:.2f}]  "
          f"obj={lp_result.objective_value:.1f}  success={lp_result.success}")

    # ═══ Generate Dashboard ═══
    html = build_dashboard(cl_result.results, cl_result.passes, ev_triples, lp_result)
    Path(OUTPUT_PATH).write_text(html)
    print(f"\nDashboard saved to: {OUTPUT_PATH}")

    # ═══ Summary ═══
    print("\n" + "=" * 70)
    print("  Patterns Demonstrated")
    print("=" * 70)
    print(
        "  1. KB seeding + RDFS inference      — TripleStore with named graphs,\n"
        "                                         RDFS class hierarchy inference\n"
        "  2. params_from_kb (Pre-flight)       — Extract KB beliefs as sim params\n"
        "  3. KB_QUERY (Mid-flight)             — ABM agents query KB via SPARQL\n"
        "  4. evidence_from_result (Post-flight)— Write sim outcomes as KB triples\n"
        "  5. ClosedLoopReasoner                — 3-pass simulate→grade→nudge cycle\n"
        "  6. KB-constrained optimization       — lp_minimize reads from SPARQL"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
