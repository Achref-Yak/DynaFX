"""AnalogyOperator — Transfer structure between domains.

Deterministic analogy via structural alignment and mapping.
No LLM needed — structure is transferred from aligned graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Edge, EdgeType, Graph, Node
from cognitive_engine.core.state import State


@dataclass
class AnalogyMapping:
    """A mapping between source and target nodes."""
    source_id: UUID
    target_id: UUID
    source_text: str
    target_text: str
    similarity: float


@dataclass
class InferredEdge:
    """An edge inferred by analogy from the source graph."""
    source_id: UUID
    target_id: UUID
    source_text: str
    target_text: str
    analogy_from: str  # e.g., "Heart→Pump"
    edge_type: str


class AnalogyOperator:
    """Analogy: Transfer structure between domains.

    Core mechanism:
        1. Extract structure from source graph (nodes + edges)
        2. Extract structure from target graph
        3. Align by embedding similarity
        4. Map: if A→B in source, and A'≈A, B'≈B in target,
           then infer A'→B' in target
        5. Output inferred edges

    Example:
        Source: Heart → Body (causes)
        Target: Brain ≈ Heart, Mind ≈ Body
        → Inferred: Brain → Mind (causes)
    """
    name = "analogy"

    def __call__(
        self,
        state: State,
        source_graph: Graph = None,
        target_graph: Graph = None,
        similarity_threshold: float = 0.6,
        max_mappings: int = 20,
        **kwargs,
    ) -> State:
        # Get graphs
        if source_graph is None:
            other_states = state.metadata.get("other_states", [])
            if not other_states:
                state.metadata["analogy_result"] = {
                    "mappings": [],
                    "inferred_edges": [],
                    "total_mappings": 0,
                    "total_inferred": 0,
                }
                state.record(self.name, "Analogy: no source graph")
                return state
            source_graph = state.graph
            target_graph = other_states[-1].graph
        elif target_graph is None:
            target_graph = state.graph

        if not source_graph.nodes or not target_graph.nodes:
            state.metadata["analogy_result"] = {
                "mappings": [],
                "inferred_edges": [],
                "total_mappings": 0,
                "total_inferred": 0,
            }
            state.record(self.name, "Analogy: no nodes to map")
            return state

        # Align nodes by embedding similarity
        mappings = self._align_nodes(source_graph, target_graph, similarity_threshold)
        mappings = mappings[:max_mappings]

        # Build mapping lookup
        source_to_target = {m.source_id: m.target_id for m in mappings}
        target_to_source = {m.target_id: m.source_id for m in mappings}

        # Transfer structure: infer edges in target
        inferred = self._transfer_structure(
            source_graph, target_graph, source_to_target, target_to_source
        )

        # Build result
        state.metadata["analogy_result"] = {
            "mappings": [
                {
                    "source_id": str(m.source_id),
                    "target_id": str(m.target_id),
                    "source_text": m.source_text,
                    "target_text": m.target_text,
                    "similarity": m.similarity,
                }
                for m in mappings
            ],
            "inferred_edges": [
                {
                    "source_id": str(e.source_id),
                    "target_id": str(e.target_id),
                    "source_text": e.source_text,
                    "target_text": e.target_text,
                    "analogy_from": e.analogy_from,
                    "edge_type": e.edge_type,
                }
                for e in inferred
            ],
            "total_mappings": len(mappings),
            "total_inferred": len(inferred),
        }

        top_maps = [f"'{m.source_text[:30]}' ↔ '{m.target_text[:30]}' (sim={m.similarity:.2f})" for m in mappings[:3]]
        top_inferred = [f"'{e.source_text[:20]}'→'{e.target_text[:20]}' from '{e.analogy_from}'" for e in inferred[:3]]
        state.record(
            self.name,
            f"Transferred structure between domains via analogical mapping. "
            f"Found {len(mappings)} cross-domain node correspondences. "
            f"Inferred {len(inferred)} new edges by analogical transfer. "
            f"Key mappings: {'; '.join(top_maps)}. "
            f"{'Inferred relations: ' + '; '.join(top_inferred) + '. ' if top_inferred else ''}"
            f"Analogy uses source-graph structure to predict missing edges in the target domain.",
        )
        return state

    def _align_nodes(
        self,
        source: Graph,
        target: Graph,
        threshold: float,
    ) -> list[AnalogyMapping]:
        """Align nodes by embedding similarity."""
        from cognitive_engine.core.embeddings import EmbeddingModel
        model = EmbeddingModel.get_instance()

        mappings = []
        used_target = set()

        for src_id, src_node in source.nodes.items():
            src_emb = src_node.embedding
            if src_emb is None:
                src_emb = model.encode(src_node.text)
                src_node.embedding = src_emb

            best_target = None
            best_sim = -1.0

            for tgt_id, tgt_node in target.nodes.items():
                if tgt_id in used_target:
                    continue

                tgt_emb = tgt_node.embedding
                if tgt_emb is None:
                    tgt_emb = model.encode(tgt_node.text)
                    tgt_node.embedding = tgt_emb

                sim = EmbeddingModel.cosine_similarity(src_emb, tgt_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_target = tgt_id

            if best_target and best_sim >= threshold:
                mappings.append(AnalogyMapping(
                    source_id=src_id,
                    target_id=best_target,
                    source_text=src_node.text,
                    target_text=target.nodes[best_target].text,
                    similarity=best_sim,
                ))
                used_target.add(best_target)

        return mappings

    def _transfer_structure(
        self,
        source: Graph,
        target: Graph,
        source_to_target: dict[UUID, UUID],
        target_to_source: dict[UUID, UUID],
    ) -> list[InferredEdge]:
        """Transfer edges from source to target via mappings."""
        inferred = []

        for edge in source.edges.values():
            # Map source edge to target
            tgt_source = source_to_target.get(edge.source_id)
            tgt_target = source_to_target.get(edge.target_id)

            if tgt_source is None or tgt_target is None:
                continue

            # Check if edge already exists in target
            edge_exists = any(
                e.source_id == tgt_source and e.target_id == tgt_target
                for e in target.edges.values()
            )

            if not edge_exists:
                src_text = source.nodes.get(edge.source_id, Node(text="")).text
                tgt_text = target.nodes.get(tgt_target, Node(text="")).text

                # Find the source edge text for provenance
                src_edge_text = ""
                if edge.source_id in source.nodes and edge.target_id in source.nodes:
                    src_edge_text = (
                        f"{source.nodes[edge.source_id].text[:30]}→"
                        f"{source.nodes[edge.target_id].text[:30]}"
                    )

                inferred.append(InferredEdge(
                    source_id=tgt_source,
                    target_id=tgt_target,
                    source_text=source.nodes.get(edge.source_id, Node(text="")).text[:50],
                    target_text=tgt_text[:50],
                    analogy_from=src_edge_text,
                    edge_type=edge.type.name,
                ))

        return inferred
