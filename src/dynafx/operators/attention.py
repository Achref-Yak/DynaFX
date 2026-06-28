"""π (Attention) operator — Select relevant subgraph.

Filters the graph to focus on relevant nodes/edges.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional
from uuid import UUID

from dynafx.core.models import Graph, NodeType
from dynafx.core.state import State


class AttentionOperator:
    """π: Select relevant subgraph.

    Filters nodes by:
    - node_type: only keep nodes of given type(s)
    - threshold: only keep nodes with belief >= threshold
    - min_degree: only keep nodes with at least N edges
    - concept: filter by semantic similarity to a concept (uses embeddings)
    """
    name = "attention"

    def __call__(
        self,
        state: State,
        node_type: str | list[str] = None,
        threshold: float = None,
        min_degree: int = None,
        concept: str = None,
        concept_threshold: float = 0.5,
        **kwargs,
    ) -> State:
        graph = state.graph
        keep_ids = set(graph.nodes.keys())

        # Filter by node type
        if node_type is not None:
            if isinstance(node_type, str):
                node_type = [node_type]
            type_set = {NodeType[t.upper()] for t in node_type}
            keep_ids = {nid for nid in keep_ids if graph.nodes[nid].type in type_set}

        # Filter by belief threshold
        if threshold is not None:
            keep_ids = {
                nid for nid in keep_ids
                if graph.nodes[nid].opinion[0] >= threshold
            }

        # Filter by minimum degree
        if min_degree is not None:
            degree = {nid: 0 for nid in keep_ids}
            for edge in graph.edges.values():
                if edge.source_id in keep_ids:
                    degree[edge.source_id] = degree.get(edge.source_id, 0) + 1
                if edge.target_id in keep_ids:
                    degree[edge.target_id] = degree.get(edge.target_id, 0) + 1
            keep_ids = {nid for nid, d in degree.items() if d >= min_degree}

        # Filter by concept similarity
        if concept is not None:
            keep_ids = self._filter_by_concept(graph, keep_ids, concept, concept_threshold)

        # Build filtered graph
        filtered = deepcopy(graph)
        filtered.nodes = {nid: graph.nodes[nid] for nid in keep_ids}
        filtered.edges = {
            e.id: e for e in graph.edges.values()
            if e.source_id in keep_ids and e.target_id in keep_ids
        }

        state.graph = filtered
        state.metadata["attention_filter"] = {
            "node_type": node_type,
            "threshold": threshold,
            "min_degree": min_degree,
            "concept": concept,
            "concept_threshold": concept_threshold,
            "original_nodes": len(graph.nodes),
            "filtered_nodes": len(filtered.nodes),
        }
        kept_labels = [graph.nodes[nid].text[:60] for nid in sorted(keep_ids, key=lambda x: graph.nodes[x].opinion[0], reverse=True)[:5]]
        dropped = len(graph.nodes) - len(filtered.nodes)
        state.record(
            self.name,
            f"Focused attention on {len(filtered.nodes)} of {len(graph.nodes)} propositions. "
            f"Retained {len(filtered.nodes)} nodes and {len(filtered.edges)} edges after filtering. "
            f"{'Dropped ' + str(dropped) + ' low-salience nodes. ' if dropped else 'All nodes were relevant, no filtering applied. '}"
            f"{'Concept filter: ' + concept + '. ' if concept else ''}"
            f"{'Belief threshold: ' + str(threshold) + '. ' if threshold else ''}"
            f"{'Min degree: ' + str(min_degree) + '. ' if min_degree else ''}"
            f"Top propositions by salience: {'; '.join(kept_labels)}.",
        )
        return state

    def _filter_by_concept(
        self,
        graph: Graph,
        keep_ids: set[UUID],
        concept: str,
        threshold: float,
    ) -> set[UUID]:
        """Filter nodes by semantic similarity to a concept."""
        from dynafx.core.embeddings import EmbeddingModel
        model = EmbeddingModel.get_instance()
        concept_emb = model.encode(concept)

        filtered_ids = set()
        for nid in keep_ids:
            node = graph.nodes[nid]
            emb = node.embedding
            if emb is None:
                emb = model.encode(node.text)
                node.embedding = emb

            sim = EmbeddingModel.cosine_similarity(concept_emb, emb)
            if sim >= threshold:
                filtered_ids.add(nid)

        return filtered_ids
