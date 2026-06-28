"""κ (Compress) operator — Summarize graph.

Reduces graph complexity while preserving key structure.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from uuid import UUID

from dynafx.core.models import Graph
from dynafx.core.state import State


class CompressOperator:
    """κ: Summarize graph.

    Identifies and extracts:
    - Dominant inference chain (funnel)
    - Key nodes by belief strength
    - Weak links in the chain
    """
    name = "compress"

    def __call__(
        self,
        state: State,
        max_chain_length: int = 10,
        **kwargs,
    ) -> State:
        graph = state.graph
        if not graph.nodes:
            return state

        # Find dominant root-to-leaf chain
        chain = self._find_dominant_chain(graph, max_chain_length)

        # Compute summary stats
        beliefs = [graph.nodes[nid].opinion[0] for nid in chain if nid in graph.nodes]
        summary = {
            "chain_length": len(chain),
            "min_belief": min(beliefs) if beliefs else 0.0,
            "max_belief": max(beliefs) if beliefs else 0.0,
            "avg_belief": sum(beliefs) / len(beliefs) if beliefs else 0.0,
            "weak_links": [
                {"id": nid.hex, "belief": graph.nodes[nid].opinion[0]}
                for nid in chain
                if nid in graph.nodes and graph.nodes[nid].opinion[0] < 0.3
            ],
        }

        state.metadata["compressed_chain"] = [
            {
                "id": nid.hex,
                "text": graph.nodes[nid].text[:100],
                "belief": graph.nodes[nid].opinion[0],
            }
            for nid in chain
            if nid in graph.nodes
        ]
        state.metadata["compression_summary"] = summary

        chain_texts = [summary["text"][:40] for summary in state.metadata.get("compressed_chain", [])[:5]]
        weak_links = summary.get("weak_links", [])
        state.record(
            self.name,
            f"Compressed the cognitive graph into a dominant inference chain of {len(chain)} propositions. "
            f"Narrative thread: {' → '.join(chain_texts) + ('...' if len(chain) > 5 else '')}. "
            f"Belief range along chain: [{summary['min_belief']:.2f}, {summary['max_belief']:.2f}], average: {summary['avg_belief']:.2f}. "
            f"Weak links (belief < 0.3): {len(weak_links)} — these are points where the inference chain is fragile and may need reinforcement. "
            f"The compressed chain represents the strongest root-to-leaf reasoning path in the graph.",
        )
        return state

    def _find_dominant_chain(self, graph: Graph, max_length: int) -> list[UUID]:
        """Find the strongest root-to-leaf inference path."""
        # Build adjacency map
        children: dict[UUID, list[tuple[UUID, float]]] = {}
        targets = set()
        for edge in graph.edges.values():
            children.setdefault(edge.source_id, []).append(
                (edge.target_id, edge.opinion[0])
            )
            targets.add(edge.target_id)

        # Find root (source not target, or highest belief)
        roots = [nid for nid in graph.nodes if nid not in targets]
        if not roots:
            roots = list(graph.nodes.keys())
        root = max(roots, key=lambda nid: graph.nodes[nid].opinion[0])

        # Build chain by following strongest edges
        chain = [root]
        visited = {root}
        current = root
        for _ in range(max_length - 1):
            if current not in children:
                break
            # Pick child with strongest edge
            candidates = [(cid, w) for cid, w in children[current] if cid not in visited]
            if not candidates:
                break
            next_id, _ = max(candidates, key=lambda x: x[1])
            chain.append(next_id)
            visited.add(next_id)
            current = next_id

        return chain
