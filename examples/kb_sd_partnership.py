#!/usr/bin/env python3
"""
KB + SD Partnership — a knowledge-driven System Dynamics digital twin
=====================================================================
Demonstrates the KB and SD working as complementary layers, not competitors:

  KB (what is true)  ->  KBSimBridge  ->  SysdModel (how it evolves)
       ^                                      |
       |                                      v
       +-------------------  evidence triples <+

  1. KB seeding + RDFS inference          — the world model
  2. Pre-flight: params_from_kb           — where do the numbers come from?
  3. full_roundtrip (KB -> Sim -> KB)     — the circular pipeline
  4. Scenario management with KB grading  — KB keeps assumptions + results
  5. Policy reasoning                     — carbon cap -> simulate impact
  6. Provenance                           — every run recorded as RDF

Run:  python examples/kb_sd_partnership.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dynafx import KBSimBridge, TripleStore, parse_sysd
from dynafx.dynamics import ScenarioComparison, ScenarioDef
from dynafx.knowledge.inference import RuleEngine, rdfs_rules
from dynafx.knowledge.model import Literal, NamedNode, Triple, TriplePattern
from dynafx.knowledge.production import (
    BridgeAction,
    ProductionRule,
    ProductionRuleEngine,
    SparqlCondition,
    TripleAction,
)

NS = "http://sd.org/"
S = lambda n: NamedNode(f"{NS}{n}")
P = lambda n: NamedNode(f"{NS}{n}")

PORT_OPEN_Q = f'ASK {{ <{NS}Port_X> <{NS}status> "open" }}'
PORT_CLOSED_Q = f'ASK {{ <{NS}Port_X> <{NS}status> "closed" }}'

# ── SD model: params come from the KB, KB_QUERY reads live context ──

MODEL = """
model 'KB-SD Partnership'
  dt 0.25
  from 0 to 90

  // Params — numeric auxes overridden at runtime by KBSimBridge
  aux demand_rate: 100.0
  aux supplier_reliability: 0.9
  aux lead_time: 5.0
  aux max_capacity: 200.0
  aux carbon_factor: 1.0

  // Live KB context, re-read at every timestep (1.0 open, 0.0 closed)
  aux port_open: KB_QUERY(port_q)
  aux disruption_factor: 0.6 + 0.4 * port_open
  aux capacity: max_capacity * supplier_reliability * carbon_factor * disruption_factor
  aux demand: demand_rate * (1.0 + (1.0 - port_open) * 0.3)

  // Inventory policy (order-up-to)
  aux safety_stock: 500.0
  aux order_rate: MAX(0, demand + (safety_stock - Inventory) / lead_time)
  aux production: MIN(capacity, order_rate)
  aux shipments: MIN(Inventory / dt, demand)
  aux stockout: MAX(0, demand - shipments)
  aux fill_rate: Cumulative_Met / MAX(Cumulative_Demand, 0.001)

  stock Inventory: 500
    + production
    - shipments

  stock Cumulative_Demand: 0
    + demand

  stock Cumulative_Met: 0
    + shipments
"""


# ── 1. KB seeding + RDFS inference ────────────────────────────────

def seed_kb() -> TripleStore:
    store = TripleStore()

    store.add(Triple(S("Supplier_A"), P("reliability"), Literal("0.92")), graph="suppliers")
    store.add(Triple(S("Supplier_B"), P("reliability"), Literal("0.65")), graph="suppliers")
    store.add(Triple(S("Supplier_A"), P("region"), Literal("asia")), graph="suppliers")

    store.add(Triple(S("Port_X"), P("status"), Literal("open")), graph="ports")
    store.add(Triple(S("Port_X"), P("capacityPerDay"), Literal("2000")), graph="ports")

    store.add(Triple(S("Contract_A"), P("maxCapacity"), Literal("200")), graph="contracts")

    store.add(Triple(S("Policy"), P("carbonActive"), Literal("false")), graph="policy")

    # RDFS ontology: reliability is a property OF a Supplier
    RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    RDFS_SUBCLASS = NamedNode("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    RDFS_DOMAIN = NamedNode("http://www.w3.org/2000/01/rdf-schema#domain")
    RDFS_RANGE = NamedNode("http://www.w3.org/2000/01/rdf-schema#range")

    store.add(Triple(S("Entity"), RDFS_SUBCLASS, S("Thing")), graph="ontology")
    store.add(Triple(S("Supplier"), RDFS_SUBCLASS, S("Entity")), graph="ontology")
    store.add(Triple(S("Disruption"), RDFS_SUBCLASS, S("Event")), graph="ontology")
    store.add(Triple(P("reliability"), RDFS_DOMAIN, S("Supplier")), graph="ontology")
    store.add(Triple(P("reliability"), RDFS_RANGE, S("NumericRating")), graph="ontology")

    store.add(Triple(S("Supplier_A"), RDF_TYPE, S("Supplier")), graph="meta")
    store.add(Triple(S("Supplier_B"), RDF_TYPE, S("Supplier")), graph="meta")
    store.add(Triple(S("Port_X"), RDF_TYPE, S("Port")), graph="meta")

    RuleEngine(rdfs_rules()).apply(store)
    return store


# ── Evidence scoring fns (receive initial/final slices of a stock) ─

def inventory_risk(init: list[float], final: list[float]) -> float:
    safety = 500.0
    avg_inv = sum(final) / len(final) if final else 0.0
    return max(0.0, min(1.0, (safety - avg_inv) / safety))


def service_level(init: list[float], final: list[float]) -> float:
    safety = 500.0
    avg_inv = sum(final) / len(final) if final else 0.0
    return max(0.0, min(1.0, avg_inv / safety))


def _fmt(val, nd=3):
    return f"{val:.{nd}f}"


def _fill(result) -> float:
    return result.values["Cumulative_Met"][-1] / max(1, result.values["Cumulative_Demand"][-1])


def main():
    print("=" * 72)
    print("  KB + SD Partnership — knowledge-driven System Dynamics twin")
    print("=" * 72)

    model = parse_sysd(MODEL)

    # ── 1. Seed KB + RDFS inference ──────────────────────────────
    store = seed_kb()
    bridge = KBSimBridge(store)
    n_triples = len(list(store.all_triples()))
    print(f"\n[1] KB seeded + RDFS inference applied: {n_triples} triples across "
          f"{len(store.graphs())} named graphs")

    # ── 2. Pre-flight: KB facts -> SD parameters ─────────────────
    print("\n[2] params_from_kb — the numbers come from the KB, not the analyst:")
    claim_map = [
        (S("Supplier_A"), P("reliability"), None, "supplier_reliability"),
        (S("Contract_A"), P("maxCapacity"), None, "max_capacity"),
    ]
    params = bridge.params_from_kb(claim_map, type_coerce={"max_capacity": "int"})
    for k, v in params.items():
        print(f"      {k:>22s} = {v}  ({type(v).__name__})")

    # ── 3. full_roundtrip: KB -> Sim -> Evidence -> KB ──────────
    print("\n[3] full_roundtrip — KB -> Sim -> evidence -> KB")
    evidence_map = [
        ("Inventory", S("Sim"), P("serviceLevel"), service_level),
        ("Inventory", S("Sim"), P("invRisk"), inventory_risk),
    ]
    result, ev_triples = bridge.full_roundtrip(
        model,
        claim_map,
        evidence_map,
        params={"port_q": PORT_OPEN_Q},
        evidence_graph="simulation",
    )
    print(f"      simulated with KB params: fill_rate={_fmt(_fill(result))}, "
          f"final_inventory={result.values['Inventory'][-1]:.0f}")
    for t in ev_triples:
        print(f"      evidence -> <{getattr(t.subject, 'iri', t.subject).split('/')[-1]}> "
              f"<{getattr(t.predicate, 'iri', t.predicate).split('/')[-1]}> "
              f"{getattr(t.object_, 'value', t.object_):.3f}")

    # ── 4. Scenario management with KB grading ──────────────────
    print("\n[4] ScenarioComparison — KB keeps assumptions + grades results")
    comp = ScenarioComparison(
        model,
        [
            ScenarioDef("normal", {"port_q": PORT_OPEN_Q, "demand_rate": 100.0}),
            ScenarioDef("port_closed", {"port_q": PORT_CLOSED_Q, "demand_rate": 100.0}),
            ScenarioDef("expedite", {"port_q": PORT_CLOSED_Q, "demand_rate": 180.0}),
        ],
        kb=store,
    )

    # Higher-is-better metric (service level) so descending rank = best first
    grade_specs = [
        (f'SELECT ?v WHERE {{ <{NS}Sim> <{NS}serviceLevel> ?v }}', "v", 0.5, 0.1),
    ]
    grades = comp.grade_scenarios(grade_specs, store, evidence_map=evidence_map, bridge=bridge)

    header = f"      {'Scenario':<14s} {'Fill Rate':>10s} {'Svc Lvl':>9s} {'Inv Risk':>9s}"
    print(header)
    print("      " + "-" * (len(header) - 6))
    for sr in comp.scenarios:
        name = sr.name
        fill = _fill(sr.result)
        inv_risk = inventory_risk([], sr.result.values["Inventory"][-len(sr.result.values["Inventory"]) // 10:])
        svc = next(iter(grades[name].values()))
        print(f"      {name:<14s} {fill:>10.2f} {svc:>9.3f} {inv_risk:>9.3f}")

    # rank() sorts descending by score -> use a higher-is-better grade
    ranking = comp.rank(grade_specs, store, evidence_map=evidence_map, bridge=bridge)
    print(f"      ranked by service level (higher = better):")
    for i, (name, score) in enumerate(ranking, 1):
        print(f"        {i}. {name} (svc_lvl={score:.3f})")

    # Same risk, different fill: expedite raises demand but capacity is the
    # binding constraint (port closed), so it cannot improve service.
    print("      note: expedite doesn't help — port closure caps capacity, not demand")

    # Strict cutoff: keep only genuinely low-risk scenarios
    kept = comp.filter(
        store,
        [f'ASK {{ <{NS}Sim> <{NS}invRisk> ?v FILTER(?v < 0.5) }}'],
        evidence_map=evidence_map, bridge=bridge,
    )
    print(f"      survived ASK filter (invRisk < 0.5): {[s.name for s in kept.scenarios]}")

    # ── 5. Policy reasoning: carbon cap -> simulate impact ──────
    print("\n[5] ProductionRuleEngine — 'when should we react?'")
    engine = ProductionRuleEngine(store)
    carbon_rule = ProductionRule(
        name="carbon_policy",
        description="Carbon cap active -> re-simulate with 40% capacity",
        body=[SparqlCondition(f'ASK {{ <{NS}Policy> <{NS}carbonActive> "true" }}')],
        head=[
            BridgeAction(
                bridge=bridge,
                model=model,
                claim_map=[(S("Contract_A"), P("maxCapacity"), None, "max_capacity")],
                params_override={"carbon_factor": 0.4, "port_q": PORT_OPEN_Q},
            ),
            TripleAction(subject=S("Policy"), predicate=P("impactRecorded"),
                         object_=Literal("true"), graph="policy"),
        ],
        max_fires=1,
    )
    engine.add_rule(carbon_rule)

    store.add(Triple(S("Policy"), P("carbonActive"), Literal("true")), graph="policy")
    fired = engine.evaluate()
    for ar in fired:
        res = ar.output.get("result")
        if res is None:
            continue
        cap_fill = _fill(res)
        print(f"      {ar.action_type}: carbon-constrained sim fill_rate={_fmt(cap_fill)} "
              f"(vs {_fmt(_fill(result))} unconstrained)")

    impacted = list(store.triples(
        TriplePattern(S("Policy"), P("impactRecorded"), None), graph="policy"))
    print(f"      KB now records: Policy impactRecorded={'true' if impacted else 'false'}")

    # ── 6. Provenance ───────────────────────────────────────────
    run_node = bridge.record_provenance(result, params={**params, "port_q": "ASK open"})
    prov_count = len(list(store.triples_in_graph("provenance")))
    print(f"\n[6] Provenance: run <{run_node}> recorded, {prov_count} triples "
          f"in 'provenance' graph")

    print("\n" + "=" * 72)
    print("  KB and SD stay in their lanes: KB = facts, SD = dynamics,")
    print("  bridge = translation layer.")
    print("=" * 72)


if __name__ == "__main__":
    main()
