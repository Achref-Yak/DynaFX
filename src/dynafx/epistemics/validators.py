import networkx as nx

from dynafx.core.models import Graph, Node, Violation, Severity
from dynafx.epistemics.product_logic import validate_categories as product_logic_check


def _build_digraph(graph: Graph) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    for node_id in graph.nodes:
        nx_graph.add_node(node_id.hex)
    for edge in graph.edges.values():
        nx_graph.add_edge(edge.source_id.hex, edge.target_id.hex)
    return nx_graph


def _summarize_node(node_id: str, nodes: dict[str, Node]) -> str:
    node = nodes.get(node_id)
    if node is None:
        return f"{node_id[:8]}"
    text = node.text[:60].replace("\n", " ")
    return f"'{text}' [{node.type.name}]"


def _detect_cycles(
    nx_graph: nx.DiGraph, nodes: dict[str, Node],
) -> list[Violation]:
    try:
        list(nx.topological_sort(nx_graph))
        return []
    except nx.NetworkXUnfeasible:
        violations: list[Violation] = []
        for component in nx.strongly_connected_components(nx_graph):
            comp_list = list(component)
            if len(comp_list) > 1 or nx_graph.has_edge(comp_list[0], comp_list[0]):
                samples = [_summarize_node(nid, nodes) for nid in list(component)[:3]]
                rest = f" … and {len(component) - 3} more" if len(component) > 3 else ""
                violations.append(
                    Violation(
                        type="CYCLE_DETECTED",
                        severity=Severity.ERROR,
                        description=f"Cycle detected ({len(component)} nodes): "
                        f"{'; '.join(samples)}{rest}",
                        node_id=None,
                    )
                )
        return violations


def level_mapping_check(graph: Graph) -> list[Violation]:
    if not graph.nodes:
        return []
    nx_graph = _build_digraph(graph)

    nodes_hex = {n.id.hex: n for n in graph.nodes.values()}
    cycle_violations = _detect_cycles(nx_graph, nodes_hex)
    if cycle_violations:
        return cycle_violations

    violations: list[Violation] = []
    level_map = {node_id: pos for pos, node_id in enumerate(nx.topological_sort(nx_graph))}
    for edge in graph.edges.values():
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
    for edge in graph.edges.values():
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
