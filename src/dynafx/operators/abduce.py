"""𝒜 (Abduce) operator — Infer best explanation from structured causes.

Deterministic abduction via reverse traversal of causal edges.
No LLM needed — hypotheses are retrieved from structured knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from dynafx.core.models import EdgeType, Graph, Node
from dynafx.core.state import State


@dataclass
class Hypothesis:
    """A candidate explanation for an observation."""
    node_id: UUID
    text: str
    score: float
    edge_type: str
    provenance: str = "causal_graph"


class AbductionOperator:
    """𝒜: Infer best explanation from structured causes.

    Core mechanism:
        1. Find observation node (by text match or embedding similarity)
        2. Reverse traverse: find all edges pointing TO the observation
        3. Score each source node as a hypothesis
        4. Optionally expand via schema templates
        5. Return scored hypotheses

    Example:
        Observation: WetGrass
        WetGrass ← Rain
        WetGrass ← Sprinkler
        → Hypotheses: [Rain, Sprinkler]
    """
    name = "abduce"

    def __call__(
        self,
        state: State,
        observation: str = None,
        observation_node_id: str = None,
        max_hypotheses: int = 10,
        edge_types: list[str] = None,
        schema_expand: bool = False,
        embedding_retrieve: bool = True,
        similarity_threshold: float = 0.5,
        **kwargs,
    ) -> State:
        graph = state.graph
        if not graph.nodes:
            state.metadata["abduction_result"] = {
                "observation": observation or "",
                "hypotheses": [],
                "total_candidates": 0,
            }
            return state

        # Resolve observation node
        obs_node = self._find_observation_node(
            graph, observation, observation_node_id, similarity_threshold
        )
        if obs_node is None:
            state.metadata["abduction_result"] = {
                "observation": observation or "",
                "hypotheses": [],
                "total_candidates": 0,
            }
            state.record(self.name, f"No observation node found for: {observation}")
            return state

        # Resolve edge types to traverse
        traverse_types = self._resolve_edge_types(edge_types)

        # Reverse traversal — find all nodes that point TO the observation
        hypotheses = self._reverse_traverse(graph, obs_node.id, traverse_types)

        # Score hypotheses
        hypotheses = self._score_hypotheses(hypotheses, graph)

        # Optional schema expansion
        if schema_expand:
            hypotheses = self._schema_expand(hypotheses, graph, state)

        # Sort by score descending
        hypotheses.sort(key=lambda h: h.score, reverse=True)

        # Limit results
        hypotheses = hypotheses[:max_hypotheses]

        # Build result
        obs_text = observation or obs_node.text
        state.metadata["abduction_result"] = {
            "observation": obs_text,
            "observation_node_id": str(obs_node.id),
            "hypotheses": [
                {
                    "node_id": str(h.node_id),
                    "text": h.text,
                    "score": h.score,
                    "edge_type": h.edge_type,
                    "provenance": h.provenance,
                }
                for h in hypotheses
            ],
            "total_candidates": len(hypotheses),
        }

        hyp_texts = [f"{h.text[:40]} (score={h.score:.2f}, edge={h.edge_type})" for h in hypotheses[:5]]
        best = hypotheses[0] if hypotheses else None
        state.record(
            self.name,
            f"Performed backward abduction from observation '{obs_text[:60]}...' — inferring best explanations via reverse traversal of causal edges. "
            f"Generated {len(hypotheses)} candidate hypotheses ranked by node belief × edge confidence. "
            f"{'Best explanation: ' + best.text[:60] + ' (score=' + str(best.score) + '). ' if best else ''}"
            f"Top hypotheses: {'; '.join(hyp_texts)}. "
            f"Abduction identifies the most likely causes given the current graph structure.",
        )
        return state

    def _find_observation_node(
        self,
        graph: Graph,
        observation: str,
        observation_node_id: str,
        threshold: float,
    ) -> Optional[Node]:
        """Find the observation node by ID, text similarity, or auto-detect."""
        # Direct UUID match
        if observation_node_id:
            from uuid import UUID as _UUID
            try:
                nid = _UUID(observation_node_id)
                if nid in graph.nodes:
                    return graph.nodes[nid]
            except ValueError:
                pass

        # Text exact match
        if observation:
            for node in graph.nodes.values():
                if node.text.lower().strip() == observation.lower().strip():
                    return node

        # Embedding similarity
        if observation:
            from dynafx.core.embeddings import EmbeddingModel
            model = EmbeddingModel.get_instance()
            obs_emb = model.encode(observation)

            best_node = None
            best_sim = -1.0

            for node in graph.nodes.values():
                emb = node.embedding
                if emb is None:
                    emb = model.encode(node.text)
                    node.embedding = emb

                sim = EmbeddingModel.cosine_similarity(obs_emb, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_node = node

            if best_sim >= threshold:
                return best_node

            # If observation was provided but didn't match, return None
            return None

        # Auto-detect only when no observation provided at all
        return self._auto_detect_observation(graph)

    def _resolve_edge_types(self, edge_types: list[str] | None) -> set[EdgeType]:
        """Resolve edge type names to EdgeType enum values."""
        if edge_types is None:
            return {EdgeType.CAUSES, EdgeType.INFERS, EdgeType.SUPPORTS}

        type_set = set()
        for name in edge_types:
            name_upper = name.upper()
            if hasattr(EdgeType, name_upper):
                type_set.add(EdgeType[name_upper])
        return type_set or {EdgeType.CAUSES, EdgeType.INFERS, EdgeType.SUPPORTS}

    def _auto_detect_observation(self, graph: Graph) -> Optional[Node]:
        """Auto-detect observation node: node with highest in-degree."""
        if not graph.nodes:
            return None

        in_degree = {nid: 0 for nid in graph.nodes}
        for edge in graph.edges.values():
            if edge.target_id in in_degree:
                in_degree[edge.target_id] += 1

        if not any(in_degree.values()):
            return list(graph.nodes.values())[0]

        max_node = max(in_degree, key=in_degree.get)
        return graph.nodes[max_node]

    def _reverse_traverse(
        self,
        graph: Graph,
        observation_id: UUID,
        edge_types: set[EdgeType],
    ) -> list[Hypothesis]:
        """Find all nodes that point TO the observation via causal edges."""
        hypotheses = []

        for edge in graph.edges.values():
            if edge.target_id == observation_id and edge.type in edge_types:
                if edge.source_id in graph.nodes:
                    source_node = graph.nodes[edge.source_id]
                    hypotheses.append(Hypothesis(
                        node_id=edge.source_id,
                        text=source_node.text,
                        score=1.0,  # base score, will be refined
                        edge_type=edge.type.name,
                        provenance="causal_graph",
                    ))

        return hypotheses

    def _score_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        graph: Graph,
    ) -> list[Hypothesis]:
        """Score hypotheses by edge belief and node belief."""
        for h in hypotheses:
            node = graph.nodes.get(h.node_id)
            if node is None:
                h.score = 0.0
                continue

            # Node belief (how confident we are in this cause)
            # opinion = [b, u, bel, pls] where bel is belief mass
            node_belief = node.opinion[2] if len(node.opinion) > 2 else node.opinion[0]

            # Edge belief (how strong the causal link is)
            edge_belief = 0.5
            for edge in graph.edges.values():
                if edge.source_id == h.node_id and edge.target_id in graph.nodes:
                    edge_belief = edge.opinion[2] if len(edge.opinion) > 2 else edge.opinion[0]
                    break

            # Combined score
            h.score = 0.6 * node_belief + 0.4 * edge_belief

        return hypotheses

    def _schema_expand(
        self,
        hypotheses: list[Hypothesis],
        graph: Graph,
        state: State,
    ) -> list[Hypothesis]:
        """Expand hypotheses via schema templates."""
        # Schema expansion adds category information
        for h in hypotheses:
            node = graph.nodes.get(h.node_id)
            if node:
                h.provenance = f"causal_graph:{node.type.name}"
        return hypotheses
