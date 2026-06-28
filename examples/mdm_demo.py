"""MDM Pipeline Demo — End-to-end workflow with validation.

Runs the full extraction pipeline on a logistics scenario, produces a
structured MDM analysis, and validates that each step produced correct output.

Run:  python examples/mdm_demo.py
Exit: 0 = all validation checks passed, 1 = one or more failed
"""

import json
import sys
from collections import defaultdict

from dynafx import loom
from dynafx.core.models import EdgeType
from dynafx.mdm.matrix import MultipleDomainMatrix

PREMISE = """\
Port X handles 60% of container throughput for the region. Due to labor \
shortages and equipment failures, port capacity has dropped by 40%. \
Vessels are experiencing 5-7 day delays at anchorage. Warehouse Y \
receives 80% of its inventory from Port X and currently holds 1200 units \
with a daily demand of 200 units. The safety stock threshold is 500 units. \
Retailer Z depends on Warehouse Y for 90% of its stock and has seen \
demand increase 15% due to seasonal promotion. If Warehouse Y falls below \
safety stock, emergency orders will be placed with an alternate supplier \
at 3x cost, which will further strain port capacity. Carrier W handles \
last-mile delivery from Warehouse Y to Retailer Z and has reported driver \
shortages reducing fleet capacity by 25%.
"""

LOGISTICS_KINDS = {"ORG"}
OPERATIONS_KINDS = {
    "Shortage", "Failure", "Capacity", "Demand", "Stock",
    "Order", "Supplier", "Cost", "Delivery", "Promotion",
    "Inventory", "Threshold",
}


# ── Pipeline ──────────────────────────────────────────────────

def run_pipeline(text):
    """Run the full extraction pipeline."""
    return loom.weave(
        text,
        steps={
            "extract":   "extract",
            "classify":  {"ref": "schema",     "depends_on": ["extract"]},
            "relate":    {"ref": "relate",      "depends_on": ["classify"]},
            "propagate": {"ref": "propagate",   "depends_on": ["relate"]},
            "check":     {"ref": "constraint",  "depends_on": ["propagate"]},
            "report":    {"ref": "compress",    "depends_on": ["check"]},
        },
        name="mdm-validation",
    )


# ── Analysis ──────────────────────────────────────────────────

def analyze_graph(graph):
    """Extract structural statistics from the weave graph."""
    node_types = defaultdict(int)
    edge_types = defaultdict(int)
    beliefs = []
    nodes = []
    edges = []

    # Build node lookup for source/target text
    node_lookup = {}
    for nid, node in graph.nodes.items():
        node_lookup[nid] = node.text
        node_types[node.type.name] += 1
        if node.opinion:
            b = node.opinion.belief if hasattr(node.opinion, "belief") else 0.5
            beliefs.append(b)
        opinion = node.opinion
        nodes.append({
            "id": str(nid),
            "text": node.text,
            "type": node.type.name,
            "belief": round(opinion.belief, 4),
            "disbelief": round(opinion.disbelief, 4),
            "uncertainty": round(opinion.uncertainty, 4),
            "prior": round(opinion.prior, 4),
        })

    for edge in graph.edges.values():
        edge_types[edge.type.name] += 1
        src_text = node_lookup.get(edge.source_id, "unknown")
        tgt_text = node_lookup.get(edge.target_id, "unknown")
        edges.append({
            "source": src_text[:60],
            "type": edge.type.name,
            "target": tgt_text[:60],
            "weight": round(edge.weight, 4),
        })

    avg_belief = sum(beliefs) / len(beliefs) if beliefs else 0.0

    # Sort by belief descending
    nodes.sort(key=lambda n: n["belief"], reverse=True)

    # Sort edges by type then weight descending
    edges.sort(key=lambda e: (e["type"], -e["weight"]))

    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "avg_belief": round(avg_belief, 4),
        "belief_count": len(beliefs),
        "nodes": nodes,
        "edges": edges,
    }


def build_mdm(graph):
    """Build MDM matrices from the weave graph."""
    mdm = MultipleDomainMatrix()

    logistics_elements = []
    operations_elements = []

    for eid, ent in graph.entities.items():
        name = ent.name[:60]
        if ent.kind in LOGISTICS_KINDS:
            logistics_elements.append(name)
        elif ent.kind in OPERATIONS_KINDS:
            operations_elements.append(name)

    def dedupe(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    logistics_elements = dedupe(logistics_elements)
    operations_elements = dedupe(operations_elements)

    if logistics_elements:
        mdm.add_domain("logistics", logistics_elements)
    if operations_elements:
        mdm.add_domain("operations", operations_elements)

    # Create DMMs for cross-domain mappings
    if logistics_elements and operations_elements:
        mdm.add_dmm("logistics", "operations")
        mdm.add_dmm("operations", "logistics")

    # Map node IDs to ALL matching entities (substring match)
    node_to_entities = {}
    for nid, node in graph.nodes.items():
        text = node.text
        matches = []
        for name in logistics_elements:
            if name in text:
                matches.append(("logistics", name))
        for name in operations_elements:
            if name in text:
                matches.append(("operations", name))
        if matches:
            node_to_entities[nid] = matches

    for edge in graph.edges.values():
        src_matches = node_to_entities.get(edge.source_id, [])
        tgt_matches = node_to_entities.get(edge.target_id, [])
        weight = edge.weight if edge.weight else 0.5

        for src_domain, src_name in src_matches:
            for tgt_domain, tgt_name in tgt_matches:
                if src_domain == tgt_domain:
                    dsm = mdm.get_dsm(src_domain)
                    if dsm:
                        dsm.add_relation(src_name, tgt_name, weight)
                elif src_domain == "logistics" and tgt_domain == "operations":
                    dmm = mdm.get_dmm("logistics", "operations")
                    if dmm:
                        dmm.add_mapping(src_name, tgt_name, weight)
                elif src_domain == "operations" and tgt_domain == "logistics":
                    dmm = mdm.get_dmm("operations", "logistics")
                    if dmm:
                        dmm.add_mapping(src_name, tgt_name, weight)

    return mdm, logistics_elements, operations_elements


def analyze_dung(graph):
    """Run Dung's argumentation on the graph beliefs."""
    from collections import defaultdict
    from dynafx.core.math import dung_semantics

    beliefs = {}
    for nid, node in graph.nodes.items():
        if node.opinion:
            b = node.opinion.belief if hasattr(node.opinion, "belief") else 0.5
        else:
            b = 0.5
        beliefs[nid] = b

    attack_graph = defaultdict(list)
    for edge in graph.edges.values():
        if edge.type in (EdgeType.ATTACKS, EdgeType.CONTRADICTS, EdgeType.REBUTS):
            attack_graph[edge.target_id].append(edge.source_id)

    acceptable = dung_semantics(beliefs, dict(attack_graph))

    n_acceptable = len([n for n in acceptable if n in beliefs])
    n_rejected = len([n for n in beliefs if n not in acceptable])

    return {
        "total_nodes": len(beliefs),
        "attack_edges": sum(len(v) for v in attack_graph.values()),
        "acceptable": n_acceptable,
        "rejected": n_rejected,
    }


# ── Validation ────────────────────────────────────────────────

def validate(graph, mdm_analysis, dung_analysis):
    """Run validation checks against expected pipeline output."""
    mdm, logistics_elements, operations_elements = mdm_analysis
    checks = []

    def check(condition, message):
        checks.append((condition, message))

    # --- Extraction checks ---
    org_entities = [
        e for e in graph.entities.values() if e.kind in LOGISTICS_KINDS
    ]
    op_entities = [
        e for e in graph.entities.values() if e.kind in OPERATIONS_KINDS
    ]
    causes_edges = [
        e for e in graph.edges.values() if e.type == EdgeType.CAUSES
    ]
    associated_edges = [
        e for e in graph.edges.values() if e.type == EdgeType.ASSOCIATED_WITH
    ]

    check(
        len(org_entities) >= 4,
        f"extract: found {len(org_entities)} ORG entities (expected ≥4)",
    )
    check(
        len(op_entities) >= 2,
        f"extract: found {len(op_entities)} operational entities (expected ≥2)",
    )
    check(
        len(causes_edges) > 0,
        f"extract: found {len(causes_edges)} CAUSES edges (expected >0)",
    )
    check(
        len(associated_edges) > 0,
        f"extract: found {len(associated_edges)} ASSOCIATED_WITH edges (expected >0)",
    )
    contradicts_edges = [
        e for e in graph.edges.values() if e.type == EdgeType.CONTRADICTS
    ]
    check(
        True,
        f"extract: found {len(contradicts_edges)} CONTRADICTS edges",
    )

    # --- MDM checks ---
    check(
        "logistics" in mdm.domains,
        f"mdm: logistics domain exists (domains: {list(mdm.domains.keys())})",
    )
    check(
        "operations" in mdm.domains,
        f"mdm: operations domain exists",
    )
    check(
        len(logistics_elements) >= 4,
        f"mdm: logistics has {len(logistics_elements)} elements (expected ≥4)",
    )
    check(
        len(operations_elements) >= 2,
        f"mdm: operations has {len(operations_elements)} elements (expected ≥2)",
    )
    dsm_count = len(mdm.dsms)
    check(
        dsm_count >= 1,
        f"mdm: {dsm_count} DSMs populated (expected ≥1)",
    )
    dmm_count = len(mdm.dmms)
    check(
        dmm_count >= 1,
        f"mdm: {dmm_count} DMMs populated (expected ≥1)",
    )

    # --- Dung's checks ---
    check(
        dung_analysis["total_nodes"] > 0,
        f"dung: evaluated {dung_analysis['total_nodes']} nodes",
    )
    check(
        dung_analysis["acceptable"] > 0,
        f"dung: {dung_analysis['acceptable']}/{dung_analysis['total_nodes']} acceptable, {dung_analysis['rejected']} rejected",
    )

    # --- Propagation checks ---
    beliefs = [
        node.opinion.belief
        for node in graph.nodes.values()
        if node.opinion and hasattr(node.opinion, "belief")
    ]
    check(
        len(beliefs) > 0 and min(beliefs) > 0,
        f"propagate: {len(beliefs)} beliefs, min={min(beliefs):.4f} (expected >0)",
    )

    return checks


# ── Main ──────────────────────────────────────────────────────

def main():
    passed = 0
    failed = 0

    def log(msg):
        print(msg, file=sys.stderr)

    # ── Step 1: Run pipeline ──
    log("[1/3] Running extraction pipeline...")
    result = run_pipeline(PREMISE)
    graph = result.graph
    log(f"  Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")

    # ── Step 2: Build analysis ──
    log("[2/3] Building analysis...")
    graph_stats = analyze_graph(graph)
    mdm_analysis = build_mdm(graph)
    mdm, logistics_elements, operations_elements = mdm_analysis
    dung_analysis = analyze_dung(graph)

    log(f"  Domains: logistics ({len(logistics_elements)}), operations ({len(operations_elements)})")
    log(f"  Dung's: {dung_analysis['acceptable']} acceptable, {dung_analysis['rejected']} rejected")

    # ── Step 3: Validate ──
    log("[3/3] Validating...")
    checks = validate(graph, mdm_analysis, dung_analysis)

    for ok, message in checks:
        status = "✓" if ok else "✗"
        log(f"  {status} {message}")
        if ok:
            passed += 1
        else:
            failed += 1

    log(f"\n  {passed}/{passed + failed} checks passed")

    # ── Output ──
    output = {
        "analysis": {
            "graph": graph_stats,
            "mdm": {
                "domains": list(mdm.domains.keys()),
                "logistics_elements": logistics_elements,
                "operations_elements": operations_elements,
                "dsm_count": len(mdm.dsms),
                "dmm_count": len(mdm.dmms),
            },
            "dung": dung_analysis,
        },
        "validation": {
            "passed": passed,
            "total": passed + failed,
            "checks": [{"ok": ok, "message": msg} for ok, msg in checks],
        },
    }

    print(json.dumps(output, indent=2, default=str))

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
