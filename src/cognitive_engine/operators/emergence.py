"""⋔ (Emergence) operator — Macro-pattern detection.

Detects emergent properties: system-level patterns not present in
individual nodes. Clusters the graph by semantic/structural similarity
and scores each cluster for emergent behavior (consensus, topology,
information gain).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph
from cognitive_engine.core.state import State


class EmergenceOperator:
    """⋔: Emergence detection.

    Finds emergent patterns by:
    1. Clustering nodes by type + opinion similarity
    2. Computing cluster-level properties (consensus belief,
       topological density, internal vs external edge ratio)
    3. Scoring emergence as the information gain of cluster-level
       properties over individual-level properties

    Higher emergence score = the cluster exhibits behavior that
    cannot be predicted from its individual nodes alone.
    """
    name = "emergence"

    def __call__(
        self,
        state: State,
        min_cluster_size: int = 3,
        similarity_threshold: float = 0.3,
        **kwargs,
    ) -> State:
        if not state.graph.nodes:
            state.metadata["emergence"] = {"status": "empty_graph"}
            return state

        clusters = self._cluster_nodes(state.graph, similarity_threshold)
        analyzed = self._analyze_clusters(state.graph, clusters, min_cluster_size)
        macro_patterns = self._detect_macro_patterns(analyzed)
        emergence_scores = self._score_emergence(analyzed)

        state.metadata["emergence"] = {
            "clusters": analyzed,
            "macro_patterns": macro_patterns,
            "emergence_scores": emergence_scores,
            "total_clusters": len(analyzed),
            "total_emergent": sum(
                1 for s in emergence_scores if s["emergence_score"] > 0.5
            ),
        }

        emergent_count = sum(
            1 for s in emergence_scores if s["emergence_score"] > 0.5
        )
        cluster_details = []
        for c in analyzed[:3]:
            macro = [m["pattern"] for m in macro_patterns if m["cluster"] == c["cluster_name"]]
            cluster_details.append(f"{c['cluster_name']}: {c['size']} nodes, density={c['density']:.2f}, belief={c['avg_belief']:.2f}" + (f", patterns: {macro}" if macro else ""))
        top_emergent = [f"{s['cluster']}: emergence_score={s['emergence_score']:.3f}" for s in emergence_scores[:3] if s["emergence_score"] > 0.5]
        state.record(
            self.name,
            f"Analyzed the graph for emergent structure: found {len(analyzed)} clusters of tightly-connected propositions. "
            f"{emergent_count} clusters exhibit emergent behavior (score > 0.5). "
            f"Macro-patterns detected: {len(macro_patterns)} ({', '.join(m['pattern'] for m in macro_patterns) if macro_patterns else 'none'}). "
            f"Cluster details: {'; '.join(cluster_details)}. "
            f"{'Emergent clusters: ' + '; '.join(top_emergent) + '. ' if top_emergent else ''}"
            f"Emergence means the cluster exhibits properties not predictable from individual nodes alone.",
        )
        return state

    def _cluster_nodes(
        self, graph: Graph, threshold: float,
    ) -> dict[str, list[UUID]]:
        """Group nodes by type consensus and opinion alignment.

        Two nodes are clustered if they share the same type or their
        opinions differ by less than threshold on belief.
        """
        type_clusters: dict[str, list[UUID]] = defaultdict(list)
        for nid, node in graph.nodes.items():
            type_clusters[node.type.name].append(nid)

        merged: dict[str, list[UUID]] = {}
        cluster_idx = 0
        for nid, node in graph.nodes.items():
            assigned = False
            for key, members in merged.items():
                representative = graph.nodes.get(members[0])
                if representative is None:
                    continue
                b1, _, _, _ = node.opinion
                b2, _, _, _ = representative.opinion
                if abs(b1 - b2) < threshold:
                    members.append(nid)
                    assigned = True
                    break
            if not assigned:
                merged[f"cluster_{cluster_idx}"] = [nid]
                cluster_idx += 1

        return merged

    def _analyze_clusters(
        self, graph: Graph, clusters: dict[str, list[UUID]],
        min_size: int,
    ) -> list[dict]:
        """Compute properties for each cluster."""
        results = []
        for name, members in clusters.items():
            if len(members) < min_size:
                continue

            internal_edges = 0
            external_edges = 0
            beliefs = []
            uncertainties = []
            member_set = set(members)

            for edge in graph.edges.values():
                src_in = edge.source_id in member_set
                tgt_in = edge.target_id in member_set
                if src_in and tgt_in:
                    internal_edges += 1
                elif src_in or tgt_in:
                    external_edges += 1

            for nid in members:
                node = graph.nodes.get(nid)
                if node is None:
                    continue
                b, d, u, a = node.opinion
                beliefs.append(b)
                uncertainties.append(u)

            avg_belief = sum(beliefs) / len(beliefs) if beliefs else 0.0
            avg_uncertainty = (
                sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
            )
            density = (
                internal_edges / max(len(members) * (len(members) - 1) / 2, 1)
                if len(members) > 1 else 0.0
            )

            results.append({
                "cluster_name": name,
                "size": len(members),
                "avg_belief": round(avg_belief, 4),
                "avg_uncertainty": round(avg_uncertainty, 4),
                "internal_edges": internal_edges,
                "external_edges": external_edges,
                "density": round(density, 4),
                "is_dense": density > 0.5,
                "member_ids": [n.hex for n in members],
            })

        results.sort(key=lambda x: x["size"], reverse=True)
        return results

    def _detect_macro_patterns(self, clusters: list[dict]) -> list[dict]:
        """Identify emergent macro-level patterns."""
        patterns = []
        for cluster in clusters:
            if cluster["is_dense"] and cluster["avg_belief"] > 0.7:
                patterns.append({
                    "pattern": "Consensus Block",
                    "cluster": cluster["cluster_name"],
                    "description": "Densely connected cluster with strong belief agreement",
                    "confidence": round(cluster["avg_belief"] * cluster["density"], 4),
                })
            if cluster["external_edges"] > cluster["internal_edges"] * 2:
                patterns.append({
                    "pattern": "Hub Interface",
                    "cluster": cluster["cluster_name"],
                    "description": "Cluster acts as bridge between different regions",
                    "confidence": round(
                        cluster["external_edges"] / max(cluster["internal_edges"] + cluster["external_edges"], 1),
                        4,
                    ),
                })
            if cluster["avg_uncertainty"] > 0.6:
                patterns.append({
                    "pattern": "Ambiguous Region",
                    "cluster": cluster["cluster_name"],
                    "description": "High uncertainty suggests emergent ambiguity",
                    "confidence": round(cluster["avg_uncertainty"], 4),
                })
        return patterns

    def _score_emergence(self, clusters: list[dict]) -> list[dict]:
        """Score each cluster for emergent properties.

        Emergence score = combination of:
        - Density (structural emergence)
        - Belief consensus (informational emergence)
        - Internal/external edge ratio (boundary emergence)
        """
        scores = []
        for cluster in clusters:
            if cluster["size"] < 2:
                continue
            structural = cluster["density"]
            informational = 1.0 - cluster["avg_uncertainty"]
            total_edges = cluster["internal_edges"] + cluster["external_edges"]
            boundary = (
                cluster["internal_edges"] / max(total_edges, 1)
                if total_edges > 0 else 0.0
            )
            emergence = structural * 0.3 + informational * 0.4 + boundary * 0.3

            scores.append({
                "cluster": cluster["cluster_name"],
                "emergence_score": round(emergence, 4),
                "structural_emergence": round(structural, 4),
                "informational_emergence": round(informational, 4),
                "boundary_emergence": round(boundary, 4),
            })

        scores.sort(key=lambda x: x["emergence_score"], reverse=True)
        return scores
