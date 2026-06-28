"""Example: Manual system decomposition via the Python API.

Demonstrates:
  - Acyclic causal graph (server pipeline) — no feedback loops
  - Cyclic causal graph (population) — balancing loop detection
  - Cross-domain dependency deduction

Usage:
    python examples/higraph_and_emergence.py | jq
"""

import json
import os
import sys

os.environ["TQDM_DISABLE"] = "1"

from dynafx.core.decomposer import SystemDecomposer
from dynafx.operators.detect_emergence import detect_feedback_loops


def build_pipeline() -> dict:
    """Acyclic graph — server failure cascades through the system."""
    d = SystemDecomposer(name="Server Pipeline")

    d.add_node("server", type="ENTITY", partition="technical")
    d.add_node("pipeline", type="PROCESS", partition="technical")
    d.add_node("queue_processor", type="PROCESS", partition="technical",
               parent="pipeline")
    d.add_node("database_connection", type="ENTITY", partition="technical")
    d.add_node("redundancy", type="RESOURCE", partition="technical",
               confidence=0.15)
    d.add_node("budget_constraint", type="CONSTRAINT", partition="managerial")
    d.add_node("backup_hardware", type="RESOURCE", partition="managerial")

    d.add_edge("server", "pipeline", "CAUSES", polarity=-1)
    d.add_edge("queue_processor", "database_connection", "DEPENDS")
    d.add_edge("pipeline", "redundancy", "DEPENDS")
    d.add_edge("budget_constraint", "backup_hardware", "CAUSES", polarity=-1)

    d.detect()
    loops = detect_feedback_loops(d.graph)
    deduced = d.graph.metadata.get("deduced_dependencies", {})

    return {
        "name": "Server Pipeline (acyclic)",
        "decomposition": d.summary(),
        "feedback_loops": [l.to_dict() for l in loops],
        "deduced_dependencies": deduced,
    }


def build_population() -> dict:
    """Cyclic graph — births and deaths form a feedback structure."""
    d = SystemDecomposer(name="Population Model")

    d.add_node("population", type="STOCK")
    d.add_node("births", type="FLOW")
    d.add_node("deaths", type="FLOW")
    d.add_node("birth_rate", type="VARIABLE")
    d.add_node("death_rate", type="VARIABLE")

    d.add_edge("births", "population", "CAUSES", polarity=+1)
    d.add_edge("population", "deaths", "CAUSES", polarity=+1)
    d.add_edge("deaths", "population", "CAUSES", polarity=-1)
    d.add_edge("birth_rate", "births", "CAUSES", polarity=+1)
    d.add_edge("death_rate", "deaths", "CAUSES", polarity=+1)

    loops = detect_feedback_loops(d.graph)
    deduced = d.graph.metadata.get("deduced_dependencies", {})

    return {
        "name": "Population Model (cyclic)",
        "decomposition": d.summary(),
        "feedback_loops": [
            {
                "type": l.loop_type,
                "gain": l.gain_sign,
                "nodes": [d.graph.nodes[nid].text[:30] for nid in l.nodes if nid in d.graph.nodes],
                "edge_count": l.edge_count,
            }
            for l in loops
        ],
        "deduced_dependencies": deduced,
    }


def main():
    result = {
        "graphs": [build_pipeline(), build_population()],
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stderr.close()
