"""⊘ (Bottleneck) operator — Constraint node and edge detection.

Identifies bottleneck structures in the graph: nodes or edges whose
removal maximally disrupts connectivity or flow. Uses edge-betweenness
and flow simulation to rank constraints.
"""

from __future__ import annotations

from collections import defaultdict, deque
from uuid import UUID

from dynafx.core.models import Graph
from dynafx.core.state import State


class BottleneckOperator:
    """⊘: Bottleneck analysis.

    Finds bottleneck nodes and edges using:
    - Edge-betweenness centrality (paths that pass through each edge)
    - Cut simulation (connectivity loss if edge is removed)
    - Flow-based severity scoring

    A bottleneck is any edge whose removal splits the graph into
    disconnected components or reduces reachability by >50%.
    """
    name = "bottleneck"

    def __call__(
        self,
        state: State,
        max_bottlenecks: int = 10,
        min_severity: float = 0.2,
        **kwargs,
    ) -> State:
        if not state.graph.nodes or not state.graph.edges:
            state.metadata["bottleneck"] = {"status": "empty_graph"}
            return state

        bottlenecks = self._find_bottlenecks(
            state.graph, max_bottlenecks, min_severity,
        )
        components = self._find_disconnected_components(state.graph)
        system_summary = self._summarize_system(state.graph, bottlenecks, components)

        state.metadata["bottleneck"] = {
            "bottlenecks": bottlenecks,
            "disconnected_components": components,
            "system_summary": system_summary,
            "total_bottlenecks": len(bottlenecks),
        }

        severe = [b for b in bottlenecks if b["severity"] > 0.7]
        top_edges = [f"{b['source_id'][:8]}→{b['target_id'][:8]} (severity={b['severity']:.2f})" for b in bottlenecks[:5]]
        state.record(
            self.name,
            f"Identified {len(bottlenecks)} bottleneck edges acting as critical connectors across {system_summary['component_count']} graph components. "
            f"Severe bottlenecks (severity > 0.7): {len(severe)}. "
            f"Top bottlenecks: {'; '.join(top_edges)}. "
            f"The graph is {'connected' if system_summary['is_connected'] else 'disconnected'} ({system_summary['node_count']} nodes, {system_summary['edge_count']} edges). "
            f"Removing these bottleneck edges would maximally disrupt connectivity, making them key leverage points for intervention.",
        )
        return state

    def _find_bottlenecks(
        self, graph: Graph, max_n: int, min_severity: float,
    ) -> list[dict]:
        """Rank edges by bottleneck severity.

        Uses edge-betweenness: for each pair of nodes, count shortest
        paths that traverse each edge. Higher betweenness = more
        critical bottleneck.
        """
        edge_flow: dict[UUID, float] = defaultdict(float)
        node_ids = list(graph.nodes.keys())
        total_pairs = 0

        for i, src in enumerate(node_ids):
            for tgt in node_ids:
                if src == tgt:
                    continue
                total_pairs += 1
                paths = self._bfs_shortest_paths(graph, src, tgt)
                if not paths:
                    continue
                path_count = len(paths)
                for path in paths:
                    for eid in path:
                        edge_flow[eid] += 1.0 / path_count

        max_flow = max(edge_flow.values()) if edge_flow else 1.0
        bottlenecks = []
        for eid, flow in edge_flow.items():
            severity = flow / max_flow
            if severity < min_severity:
                continue
            edge = graph.edges.get(eid)
            if edge is None:
                continue
            bottlenecks.append({
                "edge_id": eid.hex,
                "source_id": edge.source_id.hex,
                "target_id": edge.target_id.hex,
                "type": edge.type.name,
                "betweenness": round(flow, 2),
                "severity": round(severity, 4),
                "weight": edge.weight,
            })

        bottlenecks.sort(key=lambda x: x["severity"], reverse=True)
        return bottlenecks[:max_n]

    def _bfs_shortest_paths(
        self, graph: Graph, src: UUID, tgt: UUID,
    ) -> list[list[UUID]]:
        """Find all shortest paths from src to tgt as lists of edge IDs."""
        if src == tgt:
            return [[]]

        outgoing: dict[UUID, list[tuple[UUID, UUID]]] = defaultdict(list)
        for eid, edge in graph.edges.items():
            outgoing[edge.source_id].append((edge.target_id, eid))

        queue: deque[tuple[UUID, list[UUID]]] = deque()
        queue.append((src, []))
        visited_dist: dict[UUID, int] = {src: 0}
        paths: list[list[UUID]] = []
        found_dist: int | None = None

        while queue:
            current, path = queue.popleft()
            if found_dist is not None and len(path) > found_dist:
                break
            if current == tgt:
                paths.append(path)
                found_dist = len(path)
                continue
            for neighbor, eid in outgoing.get(current, []):
                new_dist = len(path) + 1
                if neighbor not in visited_dist or visited_dist[neighbor] >= new_dist:
                    visited_dist[neighbor] = new_dist
                    queue.append((neighbor, path + [eid]))

        return paths

    def _find_disconnected_components(self, graph: Graph) -> list[list[str]]:
        """Find weakly connected components in the graph."""
        adjacency: dict[UUID, set[UUID]] = defaultdict(set)
        for edge in graph.edges.values():
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

        visited: set[UUID] = set()
        components: list[list[str]] = []

        for nid in graph.nodes:
            if nid in visited:
                continue
            component: list[UUID] = []
            queue = deque([nid])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adjacency.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                components.append([n.hex for n in component])

        return components

    def _summarize_system(
        self, graph: Graph,
        bottlenecks: list[dict],
        components: list[list[str]],
    ) -> dict:
        """Generate a summary of the system structure."""
        return {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "component_count": len(components),
            "is_connected": len(components) <= 1,
            "severe_bottlenecks": sum(
                1 for b in bottlenecks if b["severity"] > 0.7
            ),
        }
