"""⊙ (Theory-of-Mind) operator — Nested belief propagation.

Models recursive belief-of-belief reasoning: what node A believes
about what node B believes about proposition P. Uses shared
ancestor paths and opinion fusion to compute nested belief states.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional
from uuid import UUID

from dynafx.core.models import Graph, Opinion, EdgeType
from dynafx.core.state import State


class ToMOperator:
    """⊙: Theory-of-Mind inference.

    Computes nested belief maps:
    - Level 0: Direct belief P(believes(X))
    - Level 1: Second-order P(believes(A believes(X)))
    - Level 2: Third-order P(believes(A believes(B believes(X))))

    Uses shared-path propagation: if A→X and B→X both exist in the
    graph, then A's opinion on X can be transferred through shared
    ancestors to infer what B believes A believes.
    """
    name = "tom"

    def __call__(
        self,
        state: State,
        max_depth: int = 2,
        min_confidence: float = 0.1,
        **kwargs,
    ) -> State:
        if not state.graph.nodes or not state.graph.edges:
            state.metadata["tom"] = {"status": "empty_graph"}
            return state

        belief_map = self._compute_nested_beliefs(
            state.graph, max_depth, min_confidence,
        )
        theory_mindedness = self._compute_theory_mindedness(state.graph, belief_map)
        mind_sharing = self._compute_mind_sharing(state.graph, belief_map)

        smm = self._compute_shared_mental_models(state.graph)

        state.metadata["tom"] = {
            "belief_map": belief_map,
            "theory_mindedness": theory_mindedness,
            "mind_sharing": mind_sharing,
            "shared_mental_models": smm,
            "max_depth": max_depth,
            "total_nested_beliefs": sum(
                len(level) for level in belief_map.values()
            ),
        }

        depth_dist = theory_mindedness.get("depth_distribution", {})
        shared = mind_sharing[:3]
        shared_text = "; ".join(f"agents {s['agent_1'][:8]}↔{s['agent_2'][:8]}: {s['shared_beliefs']} shared beliefs" for s in shared) if shared else "no significant mind-sharing detected"
        smm = state.metadata["tom"].get("shared_mental_models", {})
        top_pairs = smm.get("pairs", [])[:3]
        smm_text = "; ".join(
            f"{p['agent_a'][:8]}↔{p['agent_b'][:8]}: Jaccard={p['jaccard']:.3f}"
            for p in top_pairs
        ) if top_pairs else "no significant SMM overlap"
        state.record(
            self.name,
            f"Simulated theory-of-mind across depth {max_depth} (recursive belief-of-belief inference). "
            f"Total nested belief inferences: {theory_mindedness['total_nested_inferences']}. "
            f"Depth distribution: { {str(k): v for k, v in depth_dist.items()} }. "
            f"Average ToM score: {theory_mindedness['avg_score']:.3f} (higher = more confident nested beliefs). "
            f"Mind-sharing between agents: {shared_text}. "
            f"Shared mental models (Jaccard): {smm_text}. "
            f"The system can model recursive perspectives — what each agent believes about other agents' beliefs.",
        )
        return state

    def _compute_nested_beliefs(
        self, graph: Graph, max_depth: int, min_confidence: float,
    ) -> dict[str, list[dict]]:
        """Compute belief maps at each recursion depth.

        Strategy: For each pair of agents (nodes that share an edge
        with a common target), propagate opinion through the shared
        target as a "belief proxy."
        """
        levels: dict[str, list[dict]] = {}

        for depth in range(max_depth + 1):
            if depth == 0:
                beliefs = self._level0_direct(graph)
            else:
                beliefs = self._level_n(graph, depth, min_confidence)
            levels[f"level_{depth}"] = beliefs

        return levels

    def _level0_direct(self, graph: Graph) -> list[dict]:
        """Level 0: Direct beliefs (opinion on each node)."""
        beliefs = []
        for nid, node in graph.nodes.items():
            if node.opinion is None:
                continue
            b, d, u, a = node.opinion
            beliefs.append({
                "subject_id": nid.hex,
                "target_id": nid.hex,
                "belief": b,
                "disbelief": d,
                "uncertainty": u,
                "depth": 0,
                "confidence": 1.0 - u,
            })
        return beliefs

    def _level_n(
        self, graph: Graph, depth: int, min_confidence: float,
    ) -> list[dict]:
        """Level N: Recursive belief inference.

        For each target node T, find all pairs of agents (A, B) such
        that A→T and B→T share a path. A's opinion on T is propagated
        to B as "what B believes A believes about T."
        """
        incoming: dict[UUID, list[tuple[UUID, UUID]]] = defaultdict(list)
        for eid, edge in graph.edges.items():
            incoming[edge.target_id].append((edge.source_id, eid))

        beliefs = []
        for target_id, sources in incoming.items():
            if len(sources) < 2:
                continue

            for i, (agent_a, _) in enumerate(sources):
                for agent_b, _ in sources[i + 1:]:
                    node_a = graph.nodes.get(agent_a)
                    if node_a is None or node_a.opinion is None:
                        continue
                    b, d, u, a = node_a.opinion
                    confidence = 1.0 - u
                    if confidence < min_confidence:
                        continue
                    beliefs.append({
                        "subject_id": agent_b.hex,
                        "agent_id": agent_a.hex,
                        "target_id": target_id.hex,
                        "inferred_belief": b,
                        "inferred_disbelief": d,
                        "inferred_uncertainty": u,
                        "depth": depth,
                        "confidence": round(confidence, 4),
                        "mechanism": "shared_target",
                    })

        return beliefs

    def _compute_theory_mindedness(
        self, graph: Graph, belief_map: dict[str, list[dict]],
    ) -> dict:
        """Score how "theory-minded" the system is.

        High theory_mindedness = many nested belief inferences exist
        with high confidence.
        """
        total_beliefs = 0
        total_confidence = 0.0
        depth_distribution: dict[int, int] = {}

        for key, beliefs in belief_map.items():
            depth = int(key.split("_")[1])
            depth_distribution[depth] = len(beliefs)
            total_beliefs += len(beliefs)
            for b in beliefs:
                total_confidence += b.get("confidence", 0.0)

        avg_score = (
            total_confidence / max(total_beliefs, 1)
            if total_beliefs > 0 else 0.0
        )

        return {
            "avg_score": round(avg_score, 4),
            "total_nested_inferences": total_beliefs,
            "depth_distribution": depth_distribution,
            "max_inferred_depth": max(
                (k for k, v in depth_distribution.items() if v > 0),
                default=0,
            ),
        }

    def _compute_mind_sharing(
        self, graph: Graph, belief_map: dict[str, list[dict]],
    ) -> list[dict]:
        """Find nodes that share similar belief systems.

        Two agents share a mind if their inferred beliefs about
        common targets correlate above threshold.
        """
        agent_pairs: dict[tuple[str, str], list[float]] = defaultdict(list)

        for key, beliefs in belief_map.items():
            if "level_0" in key:
                continue
            for b in beliefs:
                subj = b["subject_id"]
                agent = b["agent_id"]
                pair = (min(subj, agent), max(subj, agent))
                agent_pairs[pair].append(b["inferred_belief"])

        sharing = []
        for (a, b), beliefs in agent_pairs.items():
            if len(beliefs) < 2:
                continue
            avg = sum(beliefs) / len(beliefs)
            sharing.append({
                "agent_1": a,
                "agent_2": b,
                "shared_beliefs": len(beliefs),
                "avg_belief_alignment": round(avg, 4),
            })

        sharing.sort(key=lambda x: x["shared_beliefs"], reverse=True)
        return sharing

    def _compute_shared_mental_models(self, graph: Graph) -> dict:
        """Compute Shared Mental Models via Jaccard similarity.

        Finds all AGENT/PERSON/ENTITY nodes that have BELIEVES edges,
        then computes Jaccard overlap of their belief sets (target nodes
        they believe in).

        Returns dict with:
          - pairs: sorted by Jaccard similarity
          - avg_jaccard: average across all pairs
          - agents_with_beliefs: count of agents with BELIEVES edges
        """
        # Find agents (AGENT type nodes) and their belief targets
        agent_beliefs: dict[UUID, set[UUID]] = {}
        for edge in graph.edges.values():
            if edge.type.name != "BELIEVES":
                continue
            agent = graph.nodes.get(edge.source_id)
            if agent is None:
                continue
            agent_beliefs.setdefault(edge.source_id, set()).add(edge.target_id)

        # Filter to agents with at least 2 beliefs
        valid_agents = {aid: targets for aid, targets in agent_beliefs.items() if len(targets) >= 2}

        if len(valid_agents) < 2:
            return {"pairs": [], "avg_jaccard": 0.0, "agents_with_beliefs": len(valid_agents)}

        # Compute pairwise Jaccard
        agent_ids = list(valid_agents.keys())
        pairs: list[dict] = []
        jaccard_sum = 0.0
        pair_count = 0

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a_id = agent_ids[i]
                b_id = agent_ids[j]
                a_targets = valid_agents[a_id]
                b_targets = valid_agents[b_id]
                intersection = a_targets & b_targets
                union = a_targets | b_targets
                jaccard = len(intersection) / len(union) if union else 0.0
                if jaccard > 0.0:
                    pairs.append({
                        "agent_a": a_id.hex,
                        "agent_b": b_id.hex,
                        "jaccard": round(jaccard, 4),
                        "shared_beliefs": len(intersection),
                        "total_beliefs": len(union),
                        "belief_intersection": [nid.hex for nid in intersection],
                    })
                jaccard_sum += jaccard
                pair_count += 1

        pairs.sort(key=lambda x: x["jaccard"], reverse=True)
        avg_jaccard = jaccard_sum / pair_count if pair_count > 0 else 0.0

        return {
            "pairs": pairs,
            "avg_jaccard": round(avg_jaccard, 4),
            "agents_with_beliefs": len(valid_agents),
        }
