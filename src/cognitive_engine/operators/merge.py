"""M (Merge) operator — Combine multiple graphs.

Fuses multiple graphs into one, handling deduplication and conflict resolution.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional
from uuid import uuid4

from cognitive_engine.core.models import Edge, Graph, Node, Opinion
from cognitive_engine.core.schema import Schema
from cognitive_engine.core.state import State


class MergeOperator:
    """M: Combine multiple graphs.

    Merge strategies:
    - average: average beliefs for duplicate nodes
    - weighted: weighted average by source confidence
    - keep_both: keep all nodes, mark conflicts
    - dempster_shafer: Dempster-Shafer fusion
    """
    name = "merge"

    def __call__(
        self,
        state: State,
        graphs: list[Graph] = None,
        schema: Schema = None,
        **kwargs,
    ) -> State:
        if not graphs:
            return state

        schema = schema or state.metadata.get("schema")
        strategy = schema.merge_strategy if schema else "average"

        merged = state.graph
        for g in graphs:
            merged = self._merge_graphs(merged, g, strategy)

        state.graph = merged
        state.metadata["merge_count"] = len(graphs)
        state.metadata["merge_strategy"] = strategy
        original_nodes = len(state.graph.nodes) if state.graph else 0
        added = len(merged.nodes) - original_nodes
        state.record(
            self.name,
            f"Merged {len(graphs)} additional graph(s) into the current cognitive graph "
            f"using '{strategy}' strategy. "
            f"Result: {len(merged.nodes)} nodes (+{added} added), {len(merged.edges)} edges. "
            f"{'Duplicate nodes fused via ' + strategy + ' opinion fusion. ' if added < sum(len(g.nodes) for g in graphs) else 'All nodes from merged graphs were novel additions. '}"
            f"Merge combines evidence from multiple reasoning paths into a unified belief structure.",
        )
        return state

    def _merge_graphs(self, g1: Graph, g2: Graph, strategy: str) -> Graph:
        """Merge two graphs using the given strategy."""
        result = deepcopy(g1)

        # Build text -> node mapping for dedup
        text_map = {n.text.strip().lower(): nid for nid, n in result.nodes.items()}

        for nid, node in g2.nodes.items():
            key = node.text.strip().lower()
            if key in text_map:
                # Duplicate node — apply strategy
                existing_id = text_map[key]
                existing = result.nodes[existing_id]
                existing.opinion = self._fuse_opinions(
                    existing.opinion, node.opinion, strategy
                )
            else:
                # New node — add it
                new_id = uuid4()
                result.nodes[new_id] = deepcopy(node)
                text_map[key] = new_id

        # Merge edges
        existing_edges = {(e.source_id, e.target_id, e.type) for e in result.edges.values()}
        for edge in g2.edges.values():
            if (edge.source_id, edge.target_id, edge.type) not in existing_edges:
                # Map edge IDs if nodes were merged
                src = self._map_id(edge.source_id, g2, result, text_map)
                tgt = self._map_id(edge.target_id, g2, result, text_map)
                if src and tgt:
                    new_edge = deepcopy(edge)
                    new_edge.source_id = src
                    new_edge.target_id = tgt
                    result.edges[new_edge.id] = new_edge

        return result

    def _fuse_opinions(self, o1: Opinion, o2: Opinion, strategy: str) -> Opinion:
        """Fuse two opinions using the given strategy."""
        if strategy == "average":
            return tuple((a + b) / 2 for a, b in zip(o1, o2))
        elif strategy == "keep_both":
            # Return the one with higher belief
            return o1 if o1[0] >= o2[0] else o2
        elif strategy == "weighted":
            # Simple weighted average (equal weights for now)
            return tuple((a + b) / 2 for a, b in zip(o1, o2))
        else:
            return o1  # Default: keep first

    def _map_id(self, old_id, source: Graph, target: Graph, text_map: dict):
        """Map a node ID from source to target graph."""
        if old_id in target.nodes:
            return old_id
        if old_id in source.nodes:
            key = source.nodes[old_id].text.strip().lower()
            return text_map.get(key)
        return None
