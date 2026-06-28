"""D (Compare) operator — Structured diff between two graphs.

Finds shared/unique concepts, belief deltas, and structural changes.
Uses embeddings for semantic concept matching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from dynafx.core.models import Graph, NodeType, EdgeType, Node, Edge
from dynafx.core.state import State


@dataclass
class ConceptMatch:
    """A matched concept between two graphs."""
    node_a: UUID
    node_b: UUID
    text_a: str
    text_b: str
    similarity: float


@dataclass
class BeliefDelta:
    """Change in belief for a concept between two graphs."""
    node_id: UUID
    text: str
    belief_a: float
    belief_b: float
    delta: float


@dataclass
class GraphDiff:
    """Structured diff between two graphs."""
    shared_concepts: list[ConceptMatch] = field(default_factory=list)
    unique_to_a: list[UUID] = field(default_factory=list)
    unique_to_b: list[UUID] = field(default_factory=list)
    belief_deltas: list[BeliefDelta] = field(default_factory=list)
    added_edges: list[Edge] = field(default_factory=list)
    removed_edges: list[Edge] = field(default_factory=list)
    score: float = 0.0


class CompareOperator:
    """D: Compare two graphs.

    Finds:
    - Shared concepts (via embedding similarity)
    - Unique concepts (only in one graph)
    - Belief deltas (change in belief for shared concepts)
    - Structural changes (added/removed edges)
    """
    name = "compare"

    def __call__(
        self,
        state: State,
        graph_a: Graph = None,
        graph_b: Graph = None,
        similarity_threshold: float = 0.7,
        **kwargs,
    ) -> State:
        if graph_a is None:
            graph_a = state.graph
        if graph_b is None:
            other_states = state.metadata.get("other_states", [])
            if not other_states:
                state.metadata["compare_result"] = {
                    "shared_concepts": [],
                    "unique_to_a": [],
                    "unique_to_b": [],
                    "belief_deltas": [],
                    "added_edges": 0,
                    "removed_edges": 0,
                    "score": 0.0,
                }
                state.metadata["compare_summary"] = {
                    "shared": 0,
                    "unique_a": 0,
                    "unique_b": 0,
                    "belief_deltas": 0,
                    "score": 0.0,
                }
                state.record(self.name, "No second graph to compare")
                return state
            graph_b = other_states[-1].graph

        diff = self._compare(graph_a, graph_b, similarity_threshold)

        state.metadata["compare_result"] = {
            "shared_concepts": [
                {"node_a": str(m.node_a), "node_b": str(m.node_b),
                 "text_a": m.text_a, "text_b": m.text_b, "similarity": m.similarity}
                for m in diff.shared_concepts
            ],
            "unique_to_a": [
                {"node_id": str(uid), "text": graph_a.nodes[uid].text}
                for uid in diff.unique_to_a if uid in graph_a.nodes
            ],
            "unique_to_b": [
                {"node_id": str(uid), "text": graph_b.nodes[uid].text}
                for uid in diff.unique_to_b if uid in graph_b.nodes
            ],
            "belief_deltas": [
                {"node_id": str(d.node_id), "text": d.text,
                 "belief_a": d.belief_a, "belief_b": d.belief_b, "delta": d.delta}
                for d in diff.belief_deltas
            ],
            "added_edges": len(diff.added_edges),
            "removed_edges": len(diff.removed_edges),
            "score": diff.score,
        }
        state.metadata["compare_summary"] = {
            "shared": len(diff.shared_concepts),
            "unique_a": len(diff.unique_to_a),
            "unique_b": len(diff.unique_to_b),
            "belief_deltas": len(diff.belief_deltas),
            "score": diff.score,
        }
        shared_texts = [f"'{m.text_a[:30]}' ↔ '{m.text_b[:30]}'" for m in diff.shared_concepts[:3]]
        top_deltas = [f"'{d.text[:30]}': {d.belief_a:.2f}→{d.belief_b:.2f} (δ={d.delta:+.2f})" for d in diff.belief_deltas[:3]]
        unique_b_texts = [f"'{n['text'][:30]}'" for n in state.metadata.get("compare_result", {}).get("unique_to_b", [])[:3]]
        state.record(
            self.name,
            f"Compared two graph snapshots: {len(diff.shared_concepts)} shared propositions, "
            f"{len(diff.unique_to_a)} unique to current, {len(diff.unique_to_b)} unique to prior. "
            f"Belief changes: {len(diff.belief_deltas)} propositions shifted (Δ>0.01). "
            f"Structural changes: {len(diff.added_edges)} edges added, {len(diff.removed_edges)} removed. "
            f"{'Shared: ' + '; '.join(shared_texts) + '. ' if shared_texts else ''}"
            f"{'Top belief deltas: ' + '; '.join(top_deltas) + '. ' if top_deltas else ''}"
            f"Comparison score: {diff.score:.3f} (1.0 = identical graphs).",
        )
        return state

    def _compare(
        self, graph_a: Graph, graph_b: Graph, threshold: float,
    ) -> GraphDiff:
        diff = GraphDiff()

        from dynafx.core.embeddings import EmbeddingModel
        model = EmbeddingModel.get_instance()

        nodes_a = list(graph_a.nodes.items())
        nodes_b = list(graph_b.nodes.items())

        if not nodes_a or not nodes_b:
            diff.unique_to_a = list(graph_a.nodes.keys())
            diff.unique_to_b = list(graph_b.nodes.keys())
            return diff

        matched_b: set[UUID] = set()

        for nid_a, node_a in nodes_a:
            emb_a = node_a.embedding
            if emb_a is None:
                texts_a = [node_a.text for _, node_a in nodes_a]
                embs_a = model.encode_batch(texts_a)
                for (_, n), e in zip(nodes_a, embs_a):
                    n.embedding = e
                emb_a = node_a.embedding

            best_sim = -1.0
            best_match: Optional[ConceptMatch] = None

            for nid_b, node_b in nodes_b:
                if nid_b in matched_b:
                    continue
                emb_b = node_b.embedding
                if emb_b is None:
                    emb_b = model.encode(node_b.text)
                    node_b.embedding = emb_b

                sim = EmbeddingModel.cosine_similarity(emb_a, emb_b)
                if sim > best_sim:
                    best_sim = sim
                    best_match = ConceptMatch(
                        node_a=nid_a,
                        node_b=nid_b,
                        text_a=node_a.text,
                        text_b=node_b.text,
                        similarity=sim,
                    )

            if best_match and best_sim >= threshold:
                diff.shared_concepts.append(best_match)
                matched_b.add(best_match.node_b)

                belief_a = graph_a.nodes[nid_a].opinion[0]
                belief_b = graph_b.nodes[best_match.node_b].opinion[0]
                delta = belief_b - belief_a
                if abs(delta) > 0.01:
                    diff.belief_deltas.append(BeliefDelta(
                        node_id=nid_a,
                        text=node_a.text,
                        belief_a=belief_a,
                        belief_b=belief_b,
                        delta=delta,
                    ))
            else:
                diff.unique_to_a.append(nid_a)

        for nid_b, _ in nodes_b:
            if nid_b not in matched_b:
                diff.unique_to_b.append(nid_b)

        edges_a = {(e.source_id, e.target_id, e.type): e for e in graph_a.edges.values()}
        edges_b = {(e.source_id, e.target_id, e.type): e for e in graph_b.edges.values()}

        id_map = {m.node_b: m.node_a for m in diff.shared_concepts}

        for key, edge_b in edges_b.items():
            mapped = (id_map.get(key[0], key[0]), id_map.get(key[1], key[1]), key[2])
            if mapped not in edges_a:
                diff.added_edges.append(edge_b)

        for key, edge_a in edges_a.items():
            mapped = (id_map.get(key[0], key[0]), id_map.get(key[1], key[1]), key[2])
            if mapped not in edges_b:
                diff.removed_edges.append(edge_a)

        total = max(len(graph_a.nodes), len(graph_b.nodes), 1)
        shared_score = len(diff.shared_concepts) / total
        delta_penalty = sum(abs(d.delta) for d in diff.belief_deltas) / max(len(diff.belief_deltas), 1)
        diff.score = shared_score * (1.0 - delta_penalty * 0.5)

        return diff
