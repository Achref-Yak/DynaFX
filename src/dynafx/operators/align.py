"""AlignOperator — Semantic concept alignment across multiple graphs.

Matches concepts across graphs using embedding similarity and produces
a unified alignment mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from dynafx.core.models import Graph, NodeType
from dynafx.core.state import State


@dataclass
class Alignment:
    """A set of matched concepts across multiple graphs."""
    group_id: int
    nodes: dict[str, UUID] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    centroid: Optional[list[float]] = None


@dataclass
class AlignmentResult:
    """Result of aligning multiple graphs."""
    alignments: list[Alignment] = field(default_factory=list)
    unmatched: dict[str, list[UUID]] = field(default_factory=dict)
    graph_ids: list[str] = field(default_factory=list)
    coverage: float = 0.0


class AlignOperator:
    """Align concepts across multiple graphs using embedding similarity.

    For each graph, matches nodes to existing alignment groups or creates
    new groups. Produces a unified alignment mapping.
    """
    name = "align"

    def __call__(
        self,
        state: State,
        graphs: dict[str, Graph] = None,
        similarity_threshold: float = 0.7,
        **kwargs,
    ) -> State:
        if graphs is None:
            other_states = state.metadata.get("other_states", [])
            if not other_states:
                state.metadata["alignment_result"] = {
                    "alignments": [],
                    "unmatched": {},
                    "graph_ids": [],
                    "coverage": 0.0,
                }
                state.metadata["alignment_summary"] = {
                    "groups": 0,
                    "total_nodes": 0,
                    "coverage": 0.0,
                }
                state.record(self.name, "No graphs to align")
                return state
            graphs = {}
            for i, s in enumerate(other_states):
                graphs[f"graph_{i}"] = s.graph
            graphs["current"] = state.graph

        result = self._align(graphs, similarity_threshold)

        state.metadata["alignment_result"] = {
            "alignments": [
                {"group_id": a.group_id,
                 "nodes": {k: str(v) for k, v in a.nodes.items()},
                 "texts": a.texts,
                 "centroid_len": len(a.centroid) if a.centroid else 0}
                for a in result.alignments
            ],
            "unmatched": {k: [str(uid) for uid in v] for k, v in result.unmatched.items()},
            "graph_ids": result.graph_ids,
            "coverage": result.coverage,
        }
        state.metadata["alignment_summary"] = {
            "groups": len(result.alignments),
            "total_nodes": sum(len(a.nodes) for a in result.alignments),
            "coverage": result.coverage,
        }
        group_sizes = [f"group_{a.group_id}: {len(a.nodes)} nodes" for a in result.alignments[:5]]
        graph_ids = result.graph_ids
        state.record(
            self.name,
            f"Aligned concepts across {len(graph_ids)} graph snapshots using embedding similarity. "
            f"Formed {len(result.alignments)} alignment groups with {result.coverage:.1%} coverage "
            f"({sum(len(a.nodes) for a in result.alignments)} matched nodes of {sum(len(g.nodes) for g in graphs.values())} total). "
            f"Groups: {'; '.join(group_sizes)}. "
            f"Unmatched nodes per graph: {', '.join(f'{gid}: {len(um)}' for gid, um in result.unmatched.items())}. "
            f"Alignment enables cross-graph comparison, merging, and temporal tracking.",
        )
        return state

    def _align(
        self, graphs: dict[str, Graph], threshold: float,
    ) -> AlignmentResult:
        from dynafx.core.embeddings import EmbeddingModel
        model = EmbeddingModel.get_instance()

        result = AlignmentResult(graph_ids=list(graphs.keys()))
        alignments: list[Alignment] = []
        group_counter = 0

        for graph_id, graph in graphs.items():
            unmatched_ids = []

            for nid, node in graph.nodes.items():
                emb = node.embedding
                if emb is None:
                    emb = model.encode(node.text)
                    node.embedding = emb

                best_group: Optional[Alignment] = None
                best_sim = -1.0

                for alignment in alignments:
                    if graph_id in alignment.nodes:
                        continue
                    centroid = alignment.centroid
                    if centroid is None:
                        continue
                    sim = EmbeddingModel.cosine_similarity(emb, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_group = alignment

                if best_group and best_sim >= threshold:
                    best_group.nodes[graph_id] = nid
                    best_group.texts[graph_id] = node.text
                    best_group.embeddings[graph_id] = emb
                    best_group.centroid = self._compute_centroid(
                        list(best_group.embeddings.values())
                    )
                else:
                    new_group = Alignment(
                        group_id=group_counter,
                        nodes={graph_id: nid},
                        texts={graph_id: node.text},
                        embeddings={graph_id: emb},
                        centroid=emb,
                    )
                    alignments.append(new_group)
                    group_counter += 1
                    unmatched_ids.append(nid)

            result.unmatched[graph_id] = unmatched_ids

        result.alignments = alignments

        total_possible = sum(len(g.nodes) for g in graphs.values())
        total_matched = sum(len(a.nodes) for a in alignments)
        result.coverage = total_matched / max(total_possible, 1)

        return result

    @staticmethod
    def _compute_centroid(vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        centroid = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                centroid[i] += vec[i]
        n = len(vectors)
        return [x / n for x in centroid]
