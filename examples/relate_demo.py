"""Standalone demo of the relate operator.  Run:  python examples/relate_demo.py"""
import json
from cognitive_engine.core.models import Graph, Node, NodeType, Opinion, EdgeType
from cognitive_engine.core.state import State
from cognitive_engine.operators.relate import RelateOperator


def main():
    op = RelateOperator()

    # ── Two nodes with contradictory numeric values ────────────────
    # The epistemic opposition axiom should create a CONTRADICTS edge
    # because the ratio 14/60 ≈ 0.23 is well below the 0.5 threshold.
    state = State(graph=Graph(source_text="demo"))
    n1 = Node(text="Lease clause 14 requires 60 days notice", type=NodeType.CLAIM)
    n2 = Node(text="Tenant gave 14 days notice",              type=NodeType.EVIDENCE)
    state.graph.nodes[n1.id] = n1
    state.graph.nodes[n2.id] = n2

    result = op(state, max_edges_per_node=5)

    for e in result.graph.edges.values():
        src = result.graph.nodes[e.source_id].text[:45]
        tgt = result.graph.nodes[e.target_id].text[:45]
        print(f"  {e.type.name:15s}  ({e.weight:.2f})  {src}  →  {tgt}")

    print(f"\nCreated {len(result.graph.edges)} edge(s)")
    print(f"Log: {result.history[-1].description}")

    # ── Four nodes to see multiple edges ───────────────────────────
    state2 = State(graph=Graph(source_text="demo2"))
    a = Node(text="The contract requires 30 days notice",  type=NodeType.CLAIM)
    b = Node(text="Notice period is 30 days per clause 5", type=NodeType.EVIDENCE)
    c = Node(text="Tenant only gave 14 days",              type=NodeType.COUNTERCLAIM)
    d = Node(text="Landlord accepted late notice in writing", type=NodeType.EVIDENCE)
    for n in [a, b, c, d]:
        state2.graph.nodes[n.id] = n

    result2 = op(state2, max_edges_per_node=3)

    print("\n── Multi-node demo ──")
    for e in result2.graph.edges.values():
        src = result2.graph.nodes[e.source_id].text[:40]
        tgt = result2.graph.nodes[e.target_id].text[:40]
        print(f"  {e.type.name:15s}  ({e.weight:.2f})  {src}  →  {tgt}")

    print(f"\nCreated {len(result2.graph.edges)} edge(s) across 4 nodes")


if __name__ == "__main__":
    main()
