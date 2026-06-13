import networkx as nx

from cognitive_engine.core.models import Graph, Violation, Severity
from cognitive_engine.reason.product_logic import validate_categories as product_logic_check


def level_mapping_check(graph: Graph) -> list[Violation]:
    if not graph.nodes:
        return []
    nx_graph = nx.DiGraph()
    for node_id in graph.nodes:
        nx_graph.add_node(node_id.hex)
    for edge in graph.edges:
        nx_graph.add_edge(edge.source_id.hex, edge.target_id.hex)

    violations: list[Violation] = []
    try:
        list(nx.topological_sort(nx_graph))
    except nx.NetworkXUnfeasible:
        for component in nx.strongly_connected_components(nx_graph):
            if len(component) > 1 or (
                len(component) == 1
                and nx_graph.has_edge(list(component)[0], list(component)[0])
            ):
                violations.append(
                    Violation(
                        type="CYCLE_DETECTED",
                        severity=Severity.ERROR,
                        description=f"Cycle detected between nodes: {list(component)}",
                        node_id=None,
                    )
                )
        return violations

    level_map: dict[str, int] = {}
    for pos, node_id in enumerate(nx.topological_sort(nx_graph)):
        level_map[node_id] = pos
    for edge in graph.edges:
        src_lvl = level_map.get(edge.source_id.hex, 0)
        tgt_lvl = level_map.get(edge.target_id.hex, 0)
        if src_lvl >= tgt_lvl:
            violations.append(
                Violation(
                    type="LEVEL_VIOLATION",
                    severity=Severity.WARNING,
                    description=f"Edge from level {src_lvl} to level {tgt_lvl} "
                    f"violates strict monotonicity",
                    edge_id=edge.id,
                )
            )
    return violations


def opinion_invariant_check(graph: Graph) -> list[Violation]:
    violations: list[Violation] = []
    for node in graph.nodes.values():
        b, d, u, a = node.opinion
        total = b + d + u
        if abs(total - 1.0) > 0.01:
            violations.append(
                Violation(
                    type="OPINION_INVARIANT",
                    severity=Severity.ERROR,
                    description=f"Node {node.id.hex[:8]} opinion b+d+u={total:.3f} != 1",
                    node_id=node.id,
                )
            )
        if u < -0.001:
            violations.append(
                Violation(
                    type="NEGATIVE_UNCERTAINTY",
                    severity=Severity.ERROR,
                    description=f"Node {node.id.hex[:8]} has negative uncertainty {u:.3f}",
                    node_id=node.id,
                )
            )
    for edge in graph.edges:
        b, d, u, a = edge.opinion
        total = b + d + u
        if abs(total - 1.0) > 0.01:
            violations.append(
                Violation(
                    type="OPINION_INVARIANT",
                    severity=Severity.ERROR,
                    description=f"Edge {edge.id.hex[:8]} opinion b+d+u={total:.3f} != 1",
                    edge_id=edge.id,
                )
            )
    return violations


def validate_all(graph: Graph) -> list[Violation]:
    violations: list[Violation] = []
    violations.extend(product_logic_check(graph))
    violations.extend(level_mapping_check(graph))
    violations.extend(opinion_invariant_check(graph))
    return violations


def format_violations(violations: list[Violation]) -> str:
    if not violations:
        return "No violations found."
    lines = [f"[{v.severity.name}] {v.type}: {v.description}" for v in violations]
    return "\n".join(lines)
