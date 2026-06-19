"""Advanced multi-scenario reasoning with graph analysis and evidence tracing.

This example demonstrates:
  1. Complex reasoning over multiple interacting claims
  2. Direct graph manipulation and analysis
  3. Subjective Logic opinion computation
  4. Reasoning mode configuration (ARGUMENT, CAUSAL, CONDITIONAL, ANALOGY)
  5. Graph traversal and evidence chain analysis
  6. Constraint checking and violation detection
  7. Opinion propagation across reasoning graphs

Run:  python examples/advanced_reasoning_demo.py | python -m json.tool
"""

import json
import sys
from cognitive_engine import loom
from cognitive_engine.core.models import Graph, Node, NodeType, EdgeType, Opinion
from cognitive_engine.core.math import bayes_rule
from cognitive_engine.analysis import (
    find_evidence_chains,
    classify_evidence,
    generate_label,
    build_verifiable_summary,
)
from collections import defaultdict


def validate_dag(steps: dict) -> None:
    """Validate that all DAG steps reference known operators and dependencies."""
    known_ops = {"extract", "schema", "relate", "propagate", "constraint", "compress"}
    
    for step_name, config in steps.items():
        ref = config if isinstance(config, str) else config.get("ref")
        deps = config.get("depends_on", []) if isinstance(config, dict) else []
        
        for dep in deps:
            if dep not in steps:
                raise ValueError(f"Step '{step_name}' depends on undefined step '{dep}'")
        
        if ref not in known_ops:
            raise ValueError(f"Step '{step_name}': unknown operator '{ref}'. Known: {known_ops}")
        
        print(f"  ✓ Step '{step_name}' → '{ref}' (deps: {deps})", file=sys.stderr)


COMPLEX_SCENARIO = """\
The Company's Data Protection Policy (Policy v2.1, effective March 15, 2024) requires that \
all employee personal data be encrypted at rest. The policy was approved by the Board of Directors \
on March 10, 2024, and must be enforced by April 1, 2024. Clause 3.2 states: "Failure to encrypt \
sensitive personal data constitutes a material breach subject to regulatory fines up to €20 million."

Last week, our IT audit discovered that the HR database containing employee salary information \
(approximately 2,000 records) is stored unencrypted on the main application server. The IT director \
Bob confirmed the finding in a meeting on June 10, 2024. However, Bob also noted that this server \
is physically isolated in a secure data center with restricted access, mitigating some risk. The \
European Data Protection Board's 2023 guidance states that physical isolation can reduce encryption \
requirements under specific conditions, particularly when combined with access controls.

Our legal counsel Sarah recommends immediate remediation to avoid regulatory action. However, the \
CFO estimates encryption would require $250,000 in infrastructure upgrades and 6 weeks of implementation, \
during which time the database cannot be accessed. A counterargument is that we could use temporary \
data masking as a stopgap measure, reducing exposure to €20 million in fines while we implement full \
encryption. The precedent set in GDPR enforcement cases shows that demonstrated remediation efforts \
can result in penalty reductions of 40-70%.

Additionally, employee consent forms signed in 2022 only mention "industry standard" security, not \
specific encryption. This ambiguity means regulatory interpretation is uncertain. However, Regulation \
2024/1689 (published May 2024) explicitly clarifies that "industry standard" now implies encryption \
at rest, retroactively applying to 2022 contracts.

The deadline for compliance with Policy v2.1 is April 1, 2024 (already passed), but the remediation \
deadline is July 15, 2024 (45 days from now). The probability of regulatory inspection in the next \
90 days is estimated at 15% based on recent sector trends.
"""


def analyze_graph_structure(graph: Graph) -> dict:
    """Analyze graph connectivity, opinion propagation, and edge distributions."""
    analysis = {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "node_types": defaultdict(int),
        "edge_types": defaultdict(int),
        "incoming_edges": defaultdict(list),
        "outgoing_edges": defaultdict(list),
        "node_opinions": {},
    }
    
    for node_id, node in graph.nodes.items():
        analysis["node_types"][node.type.name] += 1
        if hasattr(node, 'opinion') and node.opinion:
            analysis["node_opinions"][str(node_id)] = {
                "type": node.type.name,
                "text": node.text[:60] + "..." if len(node.text) > 60 else node.text,
                "belief": node.opinion.belief if isinstance(node.opinion, Opinion) else node.opinion[0],
                "disbelief": node.opinion.disbelief if isinstance(node.opinion, Opinion) else node.opinion[1],
                "uncertainty": node.opinion.uncertainty if isinstance(node.opinion, Opinion) else node.opinion[2],
            }
    
    for edge_id, edge in graph.edges.items():
        analysis["edge_types"][edge.type.name] += 1
        analysis["incoming_edges"][edge.target_id].append({
            "source": str(edge.source_id),
            "type": edge.type.name,
            "weight": getattr(edge, 'weight', None),
        })
        analysis["outgoing_edges"][edge.source_id].append({
            "target": str(edge.target_id),
            "type": edge.type.name,
            "weight": getattr(edge, 'weight', None),
        })
    
    return {
        "node_count": analysis["node_count"],
        "edge_count": analysis["edge_count"],
        "node_types": dict(analysis["node_types"]),
        "edge_types": dict(analysis["edge_types"]),
        "node_opinions": analysis["node_opinions"],
        "avg_incoming_edges": sum(len(v) for v in analysis["incoming_edges"].values()) / max(len(analysis["incoming_edges"]), 1),
        "avg_outgoing_edges": sum(len(v) for v in analysis["outgoing_edges"].values()) / max(len(analysis["outgoing_edges"]), 1),
    }


def apply_bayesian_reasoning(graph: Graph) -> dict:
    """Apply Bayes' rule to update claim beliefs based on evidence.
    
    Uses: P(Claim|Evidence) = P(Evidence|Claim) × P(Claim) / P(Evidence)
    
    Tracks intermediate calculations and evidence node beliefs to explain posterior shifts.
    """
    bayesian_updates = {}
    
    for node_id, node in graph.nodes.items():
        if node.type in [NodeType.CLAIM, NodeType.HYPOTHESIS] and node.opinion:
            # Find supporting and attacking evidence
            supporting_evidence = []
            attacking_evidence = []
            
            for edge_id, edge in graph.edges.items():
                target_node = graph.nodes.get(edge.target_id)
                if edge.target_id == node_id and target_node:
                    if edge.type in [EdgeType.SUPPORTS, EdgeType.INFERS, EdgeType.JUSTIFIES]:
                        supporting_evidence.append(target_node)
                    elif edge.type in [EdgeType.ATTACKS, EdgeType.CONTRADICTS, EdgeType.REBUTS]:
                        attacking_evidence.append(target_node)
            
            if supporting_evidence:
                # Compute likelihood and evidence probability from supporting nodes
                prior = node.opinion.belief  # P(Claim)
                
                # P(Evidence|Claim) = use edge weights as confidence signal (not count-based heuristic)
                edge_weights = []
                for eid, e in graph.edges.items():
                    if e.source_id == node_id and e.target_id in [id(se) for se in supporting_evidence]:
                        edge_weights.append(getattr(e, 'weight', 0.75))
                likelihood = min(0.95, sum(edge_weights) / max(len(edge_weights), 1)) if edge_weights else 0.75
                
                # P(Evidence) = average belief of supporting evidence nodes
                supporting_beliefs = [e.opinion.belief if e.opinion else 0.5 
                                    for e in supporting_evidence]
                evidence_prob = sum(supporting_beliefs) / len(supporting_beliefs)
                
                # Apply Bayes' rule
                posterior = bayes_rule(likelihood, prior, max(evidence_prob, 0.01))
                
                # Track attacking evidence strengths
                attacking_disbeliefs = []
                if attacking_evidence:
                    attacking_disbeliefs = [e.opinion.disbelief if e.opinion else 0.5 
                                          for e in attacking_evidence]
                    attack_strength = sum(attacking_disbeliefs) / len(attacking_disbeliefs)
                    posterior *= (1.0 - attack_strength)
                
                bayesian_updates[str(node_id)] = {
                    "claim": node.text[:60],
                    "prior_belief": round(prior, 3),
                    "likelihood": round(likelihood, 3),
                    "evidence_prob": round(evidence_prob, 3),
                    "posterior_before_penalty": round(bayes_rule(likelihood, prior, max(evidence_prob, 0.01)), 3),
                    "posterior_belief": round(posterior, 3),
                    "supporting_evidence_count": len(supporting_evidence),
                    "supporting_evidence_beliefs": [round(b, 3) for b in supporting_beliefs],
                    "attacking_evidence_count": len(attacking_evidence),
                    "attacking_evidence_disbeliefs": [round(d, 3) for d in attacking_disbeliefs],
                    "attack_penalty_strength": round(sum(attacking_disbeliefs) / len(attacking_disbeliefs), 3) if attacking_disbeliefs else 0.0,
                    "belief_change": round(posterior - prior, 3),
                }
    
    return bayesian_updates


def main():
    """Run the complex reasoning pipeline with graph analysis."""
    
    print("=" * 80, file=sys.stderr)
    print("COGNITIVE ENGINE: Advanced Multi-Scenario Reasoning Demo", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    
    # Step 1: Validate and execute the reasoning pipeline
    print("\n[1/5] Validating reasoning DAG...", file=sys.stderr)
    steps = {
        "extract":     "extract",
        "classify":    {"ref": "schema",     "depends_on": ["extract"]},
        "relate":      {"ref": "relate",      "depends_on": ["classify"]},
        "propagate":   {"ref": "propagate",   "depends_on": ["relate"]},
        "check":       {"ref": "constraint",  "depends_on": ["propagate"]},
        "compress":    {"ref": "compress",    "depends_on": ["check"]},
    }
    validate_dag(steps)
    
    print("\n[2/5] Running full reasoning pipeline...", file=sys.stderr)
    result = loom.weave(
        COMPLEX_SCENARIO,
        steps=steps,
        name="advanced-policy-reasoning",
    )
    
    # Step 2: Analyze the resulting graph
    print("[3/5] Analyzing graph structure and opinions...", file=sys.stderr)
    graph = result.graph
    analysis = analyze_graph_structure(graph)
    
    print(f"\n  Nodes extracted:          {analysis['node_count']}", file=sys.stderr)
    print(f"  Edges created:            {analysis['edge_count']}", file=sys.stderr)
    print(f"  Node type distribution:   {analysis['node_types']}", file=sys.stderr)
    print(f"  Edge type distribution:   {analysis['edge_types']}", file=sys.stderr)
    print(f"  Avg incoming edges:       {analysis['avg_incoming_edges']:.2f}", file=sys.stderr)
    print(f"  Avg outgoing edges:       {analysis['avg_outgoing_edges']:.2f}", file=sys.stderr)
    
    # Step 3: Identify evidence chains with bidirectional traversal
    print("\n[4/5] Tracing evidence chains (bidirectional)...", file=sys.stderr)
    chains = find_evidence_chains(graph, max_depth=3)

    # Classify evidence for the primary claim
    primary_claim_id = None
    for nid, node in graph.nodes.items():
        if node.type == NodeType.CLAIM and "Data Protection Policy" in node.text:
            primary_claim_id = nid
            break

    evidence_class = None
    if primary_claim_id:
        evidence_class = classify_evidence(graph, primary_claim_id, max_depth=3)

    print(f"\n  Identified {len(chains)} evidence chains", file=sys.stderr)
    if evidence_class:
        print(f"  Supporting:    {len(evidence_class.supporting)}", file=sys.stderr)
        print(f"  Weakening:     {len(evidence_class.weakening)}", file=sys.stderr)
        print(f"  Contradicting: {len(evidence_class.contradicting)}", file=sys.stderr)
        print(f"  Contextual:    {len(evidence_class.contextual)}", file=sys.stderr)
        for i, chain in enumerate(chains[:5], 1):
            print(f"    {i}. [{chain.classification}] {chain.label}: {chain.evidence_text[:60]}", file=sys.stderr)
    
    # Step 4: Apply Bayesian reasoning
    print("\n[5/5] Applying Bayesian belief updating (edge-weight based)...", file=sys.stderr)
    bayesian_updates = apply_bayesian_reasoning(graph)
    
    print(f"\n  Updated {len(bayesian_updates)} claims with posterior beliefs", file=sys.stderr)
    for i, (node_id, update) in enumerate(list(bayesian_updates.items())[:3], 1):  # Show first 3
        print(f"    {i}. Claim: {update['claim']}", file=sys.stderr)
        print(f"       Prior P(H): {update['prior_belief']:.3f}", file=sys.stderr)
        print(f"       Likelihood P(E|H): {update['likelihood']:.3f}", file=sys.stderr)
        print(f"       Evidence P(E): {update['evidence_prob']:.3f}", file=sys.stderr)
        print(f"       → Supporting evidence beliefs: {update['supporting_evidence_beliefs']}", file=sys.stderr)
        print(f"       Posterior (before penalty): {update['posterior_before_penalty']:.3f}", file=sys.stderr)
        if update['attacking_evidence_count'] > 0:
            print(f"       Attacking evidence disbeliefs: {update['attacking_evidence_disbeliefs']}", file=sys.stderr)
            print(f"       Attack penalty strength: {update['attack_penalty_strength']:.3f}", file=sys.stderr)
        print(f"       Final Posterior P(H|E): {update['posterior_belief']:.3f} (Δ {update['belief_change']:+.3f})", file=sys.stderr)
        print()
    
    # Generate comprehensive output
    # Strip heavy metadata from full_graph to reduce output size
    weave_dict = result.to_dict()
    weave_dict.get("metadata", {}).pop("constraint_beliefs", None)
    weave_dict.get("metadata", {}).pop("beliefs", None)
    weave_dict.get("metadata", {}).pop("truth_values", None)
    weave_dict.get("metadata", {}).pop("constraint_violations", None)
    weave_dict.get("metadata", {}).pop("compressed_chain", None)
    weave_dict.get("metadata", {}).pop("compression_summary", None)
    # Build classified evidence output
    evidence_output = {
        "supporting": [
            {"label": c.label, "text": c.evidence_text, "belief": round(c.evidence_belief, 3), "edge_type": c.edge_type}
            for c in chains if c.classification == "supporting"
        ],
        "weakening": [
            {"label": c.label, "text": c.evidence_text, "belief": round(c.evidence_belief, 3), "edge_type": c.edge_type}
            for c in chains if c.classification == "weakening"
        ],
        "contradicting": [
            {"label": c.label, "text": c.evidence_text, "belief": round(c.evidence_belief, 3), "edge_type": c.edge_type}
            for c in chains if c.classification == "contradicting"
        ],
        "contextual": [
            {"label": c.label, "text": c.evidence_text, "belief": round(c.evidence_belief, 3), "edge_type": c.edge_type}
            for c in chains if c.classification == "contextual"
        ],
    }

    # Build verifiable summary (audit-grade structured ledger)
    verifiable_summary = None
    if primary_claim_id:
        verifiable_summary = build_verifiable_summary(graph, primary_claim_id)

    output = {
        "metadata": {
            "scenario": "Data Protection Policy Compliance",
            "text_length": len(COMPLEX_SCENARIO),
            "processing_mode": "full-pipeline",
        },
        "verifiable_summary": verifiable_summary.to_dict() if verifiable_summary else None,
        "graph_analysis": {
            "node_count": analysis["node_count"],
            "edge_count": analysis["edge_count"],
            "node_types": analysis["node_types"],
            "edge_types": analysis["edge_types"],
            "connectivity": {
                "avg_incoming_edges": analysis["avg_incoming_edges"],
                "avg_outgoing_edges": analysis["avg_outgoing_edges"],
            }
        },
        "evidence": evidence_output,
        "bayesian_reasoning": {
            "method": "Bayes' rule: P(H|E) = P(E|H) × P(H) / P(E), with penalty for attacking evidence",
            "claims_updated": len(bayesian_updates),
            "sample_updates": list(bayesian_updates.values())[:5],
        },
        "full_graph": weave_dict,
    }
    
    print("\n" + "=" * 80, file=sys.stderr)
    print("Outputting complete analysis as JSON...", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
