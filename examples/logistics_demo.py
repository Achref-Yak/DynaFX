"""Logistics Supply Chain Disruption — Practical Demo.

Demonstrates the reasoning engine on a port congestion scenario:
  1. Text extraction → world model (nodes, edges, entities)
  2. Systems thinking → feedback loops, leverage points, archetypes
  3. Stock-flow analysis → inventory dynamics
  4. What-if simulation → supplier failure scenario
  5. Causal chain analysis → longest paths, cycle detection
  6. Risk attitude modeling → conservative vs aggressive planners

Run:  python examples/logistics_demo.py | python -m json.tool
"""

import json
import sys
from collections import defaultdict
from cognitive_engine import loom
from cognitive_engine.core.models import Graph, Node, NodeType, EdgeType, Opinion
from cognitive_engine.core.state import State
from cognitive_engine.core.math import (
    cross_domain_edge_density,
    causal_chain_depth,
    feedback_loop_count,
    context_similarity,
    risk_adjusted_belief,
    stakeholder_utility,
    aggregate_stakeholder_beliefs,
)
from cognitive_engine.operators.systems import (
    FeedbackLoopDetector,
    LeveragePointScorer,
    SystemArchetypeClassifier,
)
from cognitive_engine.operators.stock_flow import StockFlowOperator
from cognitive_engine.operators.simulate import SimulateOperator
from cognitive_engine.analysis import build_verifiable_summary


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


def analyze_graph(graph: Graph) -> dict:
    """Analyze graph structure and opinion distributions."""
    node_types = defaultdict(int)
    edge_types = defaultdict(int)
    beliefs = []

    for node in graph.nodes.values():
        node_types[node.type.name] += 1
        if node.opinion:
            b = node.opinion.belief if isinstance(node.opinion, Opinion) else node.opinion[0]
            beliefs.append(b)

    for edge in graph.edges.values():
        edge_types[edge.type.name] += 1

    avg_belief = sum(beliefs) / len(beliefs) if beliefs else 0.0
    belief_variance = (
        sum((b - avg_belief) ** 2 for b in beliefs) / len(beliefs) if beliefs else 0.0
    )

    return {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "avg_belief": round(avg_belief, 4),
        "belief_variance": round(belief_variance, 6),
    }


def analyze_systems(graph: Graph) -> dict:
    """Run systems thinking analysis: loops, leverage points, archetypes."""
    # Feedback loops
    loop_detector = FeedbackLoopDetector()
    state_loops = loop_detector(State(graph=graph))
    loops_data = state_loops.metadata.get("feedback_loops", {})
    loops = loops_data.get("loops", []) if isinstance(loops_data, dict) else []

    loops_out = []
    for loop in loops:
        loops_out.append({
            "type": loop.get("loop_type", "unknown"),
            "strength": round(loop.get("strength", 0), 4),
            "node_count": len(loop.get("nodes", [])),
            "edge_types": loop.get("edge_types", []),
            "description": loop.get("description", ""),
        })

    # Leverage points
    leverage_scorer = LeveragePointScorer()
    state_leverage = leverage_scorer(State(graph=graph))
    lev_data = state_leverage.metadata.get("leverage_points", {})
    leverage = lev_data.get("points", []) if isinstance(lev_data, dict) else []

    leverage_out = []
    for lp in leverage[:5]:
        leverage_out.append({
            "text": lp.get("text", "")[:60],
            "score": round(lp.get("score", 0), 4),
            "reason": lp.get("reason", ""),
            "in_degree": lp.get("in_degree", 0),
            "out_degree": lp.get("out_degree", 0),
            "betweenness": round(lp.get("betweenness", 0), 4),
        })

    # Archetypes
    archetype_classifier = SystemArchetypeClassifier()
    state_arch = archetype_classifier(State(graph=graph))
    arch_data = state_arch.metadata.get("system_archetypes", {})
    archetypes = arch_data.get("archetypes", []) if isinstance(arch_data, dict) else []

    arch_out = []
    for arch in archetypes:
        arch_out.append({
            "name": arch.get("name", ""),
            "confidence": round(arch.get("confidence", 0), 4),
            "description": arch.get("description", ""),
        })

    return {
        "feedback_loops": loops_out,
        "leverage_points": leverage_out,
        "archetypes": arch_out,
    }


def analyze_stock_flow(graph: Graph) -> dict:
    """Run stock-flow analysis on the graph."""
    sf_op = StockFlowOperator()
    state_sf = sf_op(State(graph=graph))
    sf = state_sf.metadata.get("stock_flow", {})

    # Extract key metrics
    stocks = sf.get("stocks", [])
    accumulations = sf.get("accumulations", {})
    doubling_times = sf.get("doubling_times", {})

    stocks_out = []
    for s in sf.get("stocks", [])[:5]:
        stocks_out.append({
            "node_id": s.get("node_id", "")[:8],
            "text": s.get("text", "")[:60],
            "confidence": round(s.get("confidence", 0), 4),
            "in_degree": s.get("in_degree", 0),
            "out_degree": s.get("out_degree", 0),
        })

    accum_out = []
    for a in sf.get("accumulations", [])[:5]:
        accum_out.append({
            "node_id": a.get("node_id", "")[:8],
            "net": round(a.get("net_accumulation", 0), 4),
            "inflow": round(a.get("total_inflow", 0), 4),
            "outflow": round(a.get("total_outflow", 0), 4),
            "growing": a.get("growing", False),
        })

    dt_out = []
    for d in sf.get("doubling_times", [])[:5]:
        dt_out.append({
            "node_id": d.get("node_id", "")[:8],
            "doubling_time": round(d.get("doubling_time", 0), 2),
        })

    return {
        "stocks": stocks_out,
        "accumulations": accum_out,
        "doubling_times": dt_out,
        "total_flows": sf.get("total_flows", 0),
    }


def run_simulation(graph: Graph) -> dict:
    """Run what-if simulation: what if port capacity drops 40%."""
    sim_op = SimulateOperator()

    # Find the port node
    port_id = None
    for nid, node in graph.nodes.items():
        if "port" in node.text.lower():
            port_id = nid
            break

    if port_id is None:
        return {"status": "no_port_node_found"}

    # Simulate: reduce port belief to 0.3 (capacity crisis)
    modifications = {str(port_id): {"belief": 0.3}}
    state_sim = sim_op(State(graph=graph), modifications=modifications)
    sim_result = state_sim.metadata.get("simulation", {})

    return {
        "scenario": "Port capacity drops 40%",
        "target_node": str(port_id),
        "original_belief": round(graph.nodes[port_id].opinion.belief, 4)
            if graph.nodes[port_id].opinion else None,
        "simulated_belief": 0.3,
        "objective_change": round(sim_result.get("objective_change", 0), 4),
        "violations": sim_result.get("violations", 0),
        "status": sim_result.get("status", "completed"),
    }


def analyze_causal_chains(graph: Graph) -> dict:
    """Analyze causal chain depth and feedback cycles."""
    depth = causal_chain_depth(graph.nodes, graph.edges)
    cycles = feedback_loop_count(graph.nodes, graph.edges)

    # Cross-domain density (treat each node type as a community)
    communities = {}
    for nid, node in graph.nodes.items():
        communities[nid] = hash(node.type.name) % 100
    density = cross_domain_edge_density(graph.nodes, graph.edges, communities)

    return {
        "causal_chain_depth": depth,
        "feedback_cycles": cycles,
        "cross_domain_edge_density": round(density, 4),
    }


def analyze_risk(graph: Graph) -> dict:
    """Analyze risk attitudes across stakeholder perspectives."""
    # Find key nodes
    nodes_by_type = defaultdict(list)
    for nid, node in graph.nodes.items():
        nodes_by_type[node.type.name].append((nid, node))

    # Risk-adjusted beliefs for different stakeholder types
    conservative = risk_adjusted_belief(0.5, alpha=2.0)  # risk-averse
    neutral = risk_adjusted_belief(0.5, alpha=1.0)       # risk-neutral
    aggressive = risk_adjusted_belief(0.5, alpha=0.5)    # risk-seeking

    # Utility for a loss scenario (e.g., stockout)
    loss_utility = stakeholder_utility(0.2, alpha=1.0)
    gain_utility = stakeholder_utility(0.8, alpha=1.0)

    # Aggregate beliefs from evidence nodes
    evidence_beliefs = [
        (node.opinion.belief if node.opinion else 0.5, 1.0)
        for nid, node in nodes_by_type.get("EVIDENCE", [])
    ]
    aggregated = aggregate_stakeholder_beliefs(evidence_beliefs)

    return {
        "risk_profiles": {
            "conservative_alpha2": round(conservative, 4),
            "neutral_alpha1": round(neutral, 4),
            "aggressive_alpha05": round(aggressive, 4),
        },
        "utility": {
            "loss_0.2": round(loss_utility, 4),
            "gain_0.8": round(gain_utility, 4),
            "loss_aversion_ratio": round(abs(loss_utility) / max(gain_utility, 0.001), 2),
        },
        "evidence_aggregation": {
            "count": len(evidence_beliefs),
            "aggregated_belief": round(aggregated, 4),
        },
    }


def main():
    print("=" * 70, file=sys.stderr)
    print("COGNITIVE ENGINE: Logistics Supply Chain Disruption Demo", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # ── Step 1: Run the extraction pipeline ────────────────────────
    print("\n[1/4] Running extraction pipeline...", file=sys.stderr)
    result = loom.weave(
        PREMISE,
        steps={
            "extract":   "extract",
            "classify":  {"ref": "schema",     "depends_on": ["extract"]},
            "relate":    {"ref": "relate",      "depends_on": ["classify"]},
            "propagate": {"ref": "propagate",   "depends_on": ["relate"]},
            "check":     {"ref": "constraint",  "depends_on": ["propagate"]},
            "report":    {"ref": "compress",    "depends_on": ["check"]},
        },
        name="logistics-disruption",
    )
    graph = result.graph
    print(f"  Extracted {len(graph.nodes)} nodes, {len(graph.edges)} edges", file=sys.stderr)

    # ── Step 2: Analyze graph structure ────────────────────────────
    print("[2/4] Analyzing graph structure...", file=sys.stderr)
    graph_analysis = analyze_graph(graph)
    print(f"  Node types: {graph_analysis['node_types']}", file=sys.stderr)
    print(f"  Edge types: {graph_analysis['edge_types']}", file=sys.stderr)
    print(f"  Avg belief: {graph_analysis['avg_belief']:.3f}", file=sys.stderr)

    # ── Step 3: Systems + stock-flow + causal analysis ─────────────
    print("[3/4] Running systems thinking + stock-flow analysis...", file=sys.stderr)
    systems_analysis = analyze_systems(graph)
    stock_flow = analyze_stock_flow(graph)
    causal_analysis = analyze_causal_chains(graph)

    print(f"  Feedback loops: {len(systems_analysis['feedback_loops'])}", file=sys.stderr)
    print(f"  Leverage points: {len(systems_analysis['leverage_points'])}", file=sys.stderr)
    print(f"  Archetypes: {[a['name'] for a in systems_analysis['archetypes']]}", file=sys.stderr)
    print(f"  Causal chain depth: {causal_analysis['causal_chain_depth']}", file=sys.stderr)
    print(f"  Feedback cycles: {causal_analysis['feedback_cycles']}", file=sys.stderr)

    # ── Step 4: Simulation + risk analysis ─────────────────────────
    print("[4/4] Running simulation + risk analysis...", file=sys.stderr)
    simulation = run_simulation(graph)
    risk_analysis = analyze_risk(graph)

    print(f"  Simulation: {simulation.get('status', 'unknown')}", file=sys.stderr)
    print(f"  Objective change: {simulation.get('objective_change', 'N/A')}", file=sys.stderr)

    # ── Build output ───────────────────────────────────────────────
    print("\n" + "=" * 70, file=sys.stderr)
    print("Outputting analysis as JSON...", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    output = {
        "scenario": "Port Congestion Supply Chain Disruption",
        "input_text": PREMISE.strip(),
        "graph_analysis": graph_analysis,
        "systems_analysis": systems_analysis,
        "stock_flow_analysis": stock_flow,
        "causal_analysis": causal_analysis,
        "simulation": simulation,
        "risk_analysis": risk_analysis,
        "pipeline_result": result.to_dict(),
    }

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
