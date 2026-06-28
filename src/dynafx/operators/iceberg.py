"""🧊 (Iceberg) operator — Event→Pattern→Structure→Mental Model decomposition.

Decomposes the reasoning graph into the four layers of the Iceberg
Model:
1. Events — visible occurrences (leaf nodes, low abstraction)
2. Patterns — recurring structures (repeated edge motifs)
3. Structure — deep causal architecture (strongly-connected components)
4. Mental Models — belief systems (opinion distributions)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional
from uuid import UUID

from dynafx.core.models import Graph, NodeType
from dynafx.core.state import State


class IcebergOperator:
    """🧊: Iceberg decomposition.

    Maps the graph onto the four iceberg layers:

    Layer 1 — Events:
        Leaf nodes with no children; observable facts.

    Layer 2 — Patterns:
        Nodes with recurring edge motifs (e.g., multiple edges of same
        type). Detected via edge-type frequency analysis.

    Layer 3 — Structure:
        Strongly-connected components and feedback loops that form the
        deep architecture of the system.

    Layer 4 — Mental Models:
        Nodes with high belief (opinion[0] > 0.7) acting as axiomatic
        assumptions, plus nodes with high uncertainty as open questions.
    """
    name = "iceberg"

    def __call__(
        self,
        state: State,
        belief_threshold: float = 0.7,
        **kwargs,
    ) -> State:
        if not state.graph.nodes:
            state.metadata["iceberg"] = {"status": "empty_graph"}
            return state

        events = self._detect_events(state.graph)
        patterns = self._detect_patterns(state.graph)
        structures = self._detect_structures(state.graph)
        mental_models = self._detect_mental_models(state.graph, belief_threshold)

        state.metadata["iceberg"] = {
            "layers": {
                "events": events,
                "patterns": patterns,
                "structure": structures,
                "mental_models": mental_models,
            },
            "layer_counts": {
                "events": len(events),
                "patterns": len(patterns),
                "structure": len(structures),
                "mental_models": len(mental_models),
            },
        }

        top_events = [e["text"][:40] for e in events[:3]]
        top_patterns = [f"{p['edge_type']} (x{p['occurrences']})" for p in patterns[:3]]
        top_structures = [f"SCC of {s['size']} nodes" for s in structures[:3]]
        top_mental = [f"'{m['text'][:30]}' ({m['role']})" for m in mental_models[:5]]
        state.record(
            self.name,
            f"Iceberg decomposition across four layers of systemic depth. "
            f"Layer 1 — Events: {len(events)} observable surface facts. Examples: {'; '.join(top_events)}. "
            f"Layer 2 — Patterns: {len(patterns)} recurring edge motifs. Examples: {'; '.join(top_patterns)}. "
            f"Layer 3 — Structure: {len(structures)} deep causal architectures (SCCs). Examples: {'; '.join(top_structures)}. "
            f"Layer 4 — Mental Models: {len(mental_models)} belief-anchored assumptions. Examples: {'; '.join(top_mental)}. "
            f"The deepest layer (mental models) drives the shallowest (events) — surfacing these exposes root causes.",
        )
        return state

    def _detect_events(self, graph: Graph) -> list[dict]:
        """Layer 1: Find leaf nodes (observable events).

        An event node has no outgoing edges (or only outgoing to
        other events) and low abstraction level.
        """
        events = []
        has_children: dict[UUID, bool] = defaultdict(bool)
        for edge in graph.edges.values():
            has_children[edge.source_id] = True

        for nid, node in graph.nodes.items():
            if not has_children.get(nid, False):
                b, d, u, a = node.opinion
                events.append({
                    "node_id": nid.hex,
                    "text": node.text[:80],
                    "type": node.type.name,
                    "belief": b,
                    "abstraction_level": node.abstraction_level,
                })

        events.sort(key=lambda x: x["abstraction_level"])
        return events

    def _detect_patterns(self, graph: Graph) -> list[dict]:
        """Layer 2: Find recurring edge motifs."""
        edge_type_counts: dict[str, int] = defaultdict(int)
        for edge in graph.edges.values():
            edge_type_counts[edge.type.name] += 1

        patterns = []
        for type_name, count in edge_type_counts.items():
            if count >= 2:
                patterns.append({
                    "edge_type": type_name,
                    "occurrences": count,
                    "frequency": round(count / max(len(graph.edges), 1), 4),
                })

        patterns.sort(key=lambda x: x["occurrences"], reverse=True)
        return patterns

    def _detect_structures(self, graph: Graph) -> list[dict]:
        """Layer 3: Find strongly-connected components (deep structure).

        Uses Tarjan's algorithm to find SCCs.
        """
        index_counter = 0
        stack: list[UUID] = []
        lowlink: dict[UUID, int] = {}
        index: dict[UUID, int] = {}
        on_stack: set[UUID] = set()
        sccs: list[list[UUID]] = []

        def strongconnect(node_id: UUID) -> None:
            nonlocal index_counter
            index[node_id] = index_counter
            lowlink[node_id] = index_counter
            index_counter += 1
            stack.append(node_id)
            on_stack.add(node_id)

            for edge in graph.edges.values():
                if edge.source_id != node_id:
                    continue
                if edge.target_id not in index:
                    strongconnect(edge.target_id)
                    lowlink[node_id] = min(lowlink[node_id], lowlink[edge.target_id])
                elif edge.target_id in on_stack:
                    lowlink[node_id] = min(lowlink[node_id], index[edge.target_id])

            if lowlink[node_id] == index[node_id]:
                scc: list[UUID] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == node_id:
                        break
                if len(scc) > 1:
                    sccs.append(scc)

        for nid in graph.nodes:
            if nid not in index:
                strongconnect(nid)

        structures = []
        for scc in sccs:
            internal_edges = sum(
                1 for e in graph.edges.values()
                if e.source_id in scc and e.target_id in scc
            )
            outgoing_edges = sum(
                1 for e in graph.edges.values()
                if e.source_id in scc and e.target_id not in scc
            )
            incoming_edges = sum(
                1 for e in graph.edges.values()
                if e.target_id in scc and e.source_id not in scc
            )
            structures.append({
                "size": len(scc),
                "node_ids": [n.hex for n in scc],
                "internal_edges": internal_edges,
                "outgoing_edges": outgoing_edges,
                "incoming_edges": incoming_edges,
            })

        structures.sort(key=lambda x: x["size"], reverse=True)
        return structures

    def _detect_mental_models(
        self, graph: Graph, belief_threshold: float,
    ) -> list[dict]:
        """Layer 4: Find belief-anchored nodes (mental models).

        Mental models are nodes with:
        - High belief (strong conviction / axiom)
        - High uncertainty (open question / unknown)
        - Low abstraction_level but high connectivity (core concepts)
        """
        models = []
        for nid, node in graph.nodes.items():
            b, d, u, a = node.opinion
            degree = sum(
                1 for e in graph.edges.values()
                if e.source_id == nid or e.target_id == nid
            )

            if b >= belief_threshold:
                models.append({
                    "node_id": nid.hex,
                    "text": node.text[:80],
                    "type": node.type.name,
                    "role": "axiom",
                    "belief": b,
                    "degree": degree,
                })

            if u >= 0.6:
                models.append({
                    "node_id": nid.hex,
                    "text": node.text[:80],
                    "type": node.type.name,
                    "role": "open_question",
                    "uncertainty": u,
                    "degree": degree,
                })

            if degree > 3 and node.abstraction_level <= 1:
                models.append({
                    "node_id": nid.hex,
                    "text": node.text[:80],
                    "type": node.type.name,
                    "role": "core_concept",
                    "degree": degree,
                    "belief": b,
                })

        models.sort(key=lambda x: x.get("belief", x.get("degree", 0)), reverse=True)
        return models
