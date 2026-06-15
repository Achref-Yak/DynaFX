"""Systems Thinking Operators — Feedback loops, leverage points, archetypes.

Implements first-class operators for systems thinking analysis:
- FeedbackLoopDetector: Find reinforcing/balancing loops
- LeveragePointScorer: Identify high-leverage intervention points
- SystemArchetypeClassifier: Classify system archetypes
- CausalSCM: Structural causal model operations

All operators follow the standard interface: (State, **kwargs) -> State
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import EdgeType, Graph, Node
from cognitive_engine.core.state import State


@dataclass
class FeedbackLoop:
    """A detected feedback loop."""
    nodes: list[UUID]
    edge_types: list[str]
    loop_type: str  # "reinforcing" or "balancing"
    strength: float
    description: str


@dataclass
class LeveragePoint:
    """A scored leverage point."""
    node_id: UUID
    text: str
    score: float
    reason: str
    in_degree: int
    out_degree: int
    betweenness: float


@dataclass
class SystemArchetype:
    """A classified system archetype."""
    name: str
    confidence: float
    nodes: list[UUID]
    description: str
    intervention建议: str


class FeedbackLoopDetector:
    """Detect reinforcing and balancing feedback loops in causal graphs.

    Reinforcing loops: A → B → C → A (amplifies change)
    Balancing loops: A → B → C → A with negation (stabilizes)

    Usage:
        op = FeedbackLoopDetector()
        result = op(state)
        loops = result.metadata["feedback_loops"]
    """
    name = "feedback_loops"

    def __call__(self, state: State, **kwargs) -> State:
        graph = state.graph
        loops = self._detect_loops(graph)

        state.metadata["feedback_loops"] = {
            "loops": [
                {
                    "nodes": [str(nid) for nid in loop.nodes],
                    "edge_types": loop.edge_types,
                    "loop_type": loop.loop_type,
                    "strength": loop.strength,
                    "description": loop.description,
                }
                for loop in loops
            ],
            "total_loops": len(loops),
            "reinforcing": sum(1 for l in loops if l.loop_type == "reinforcing"),
            "balancing": sum(1 for l in loops if l.loop_type == "balancing"),
        }

        reinforcing = sum(1 for l in loops if l.loop_type == "reinforcing")
        balancing = sum(1 for l in loops if l.loop_type == "balancing")
        top_loops = [f"{l.loop_type} loop of {len(l.nodes)} nodes (strength={l.strength:.2f})" for l in loops[:5]]
        state.record(
            self.name,
            f"Detected {len(loops)} feedback loops in the causal graph: {reinforcing} reinforcing (self-amplifying), "
            f"{balancing} balancing (stabilizing). "
            f"Top loops: {'; '.join(top_loops)}. "
            f"Reinforcing loops drive exponential growth or collapse; balancing loops resist change and maintain equilibrium. "
            f"Understanding the loop structure reveals systemic behavior drivers.",
        )
        return state

    def _detect_loops(self, graph: Graph) -> list[FeedbackLoop]:
        """Detect all cycles in the graph."""
        loops = []
        visited = set()
        rec_stack = set()

        adj = defaultdict(list)
        for edge in graph.edges.values():
            adj[edge.source_id].append((edge.target_id, edge.type))

        def dfs(node_id: UUID, path: list, edge_types: list):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor, edge_type in adj[node_id]:
                if neighbor not in visited:
                    edge_types.append(edge_type.name)
                    dfs(neighbor, path, edge_types)
                    edge_types.pop()
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:]
                    cycle_edges = edge_types[cycle_start:]

                    loop_type = self._classify_loop(cycle, cycle_edges, graph)
                    strength = self._compute_strength(cycle, graph)

                    loops.append(FeedbackLoop(
                        nodes=cycle,
                        edge_types=cycle_edges,
                        loop_type=loop_type,
                        strength=strength,
                        description=f"Cycle: {' → '.join(str(n)[:8] for n in cycle)}",
                    ))

            path.pop()
            rec_stack.remove(node_id)

        for node_id in graph.nodes:
            if node_id not in visited:
                dfs(node_id, [], [])

        return loops

    def _classify_loop(
        self,
        cycle: list[UUID],
        edge_types: list[str],
        graph: Graph,
    ) -> str:
        """Classify loop as reinforcing or balancing."""
        negation_count = 0
        for et in edge_types:
            if et in ("ATTACKS", "CONTRADICTS", "REBUTS"):
                negation_count += 1

        return "balancing" if negation_count % 2 == 1 else "reinforcing"

    def _compute_strength(self, cycle: list[UUID], graph: Graph) -> float:
        """Compute loop strength from edge beliefs."""
        beliefs = []
        for edge in graph.edges.values():
            if edge.source_id in cycle and edge.target_id in cycle:
                beliefs.append(edge.opinion[2] if len(edge.opinion) > 2 else 0.5)

        return sum(beliefs) / len(beliefs) if beliefs else 0.5


class LeveragePointScorer:
    """Identify high-leverage intervention points in a system.

    Leverage points are nodes where small changes produce large effects.
    Scored by: centrality, connectivity, causal influence.

    Usage:
        op = LeveragePointScorer()
        result = op(state)
        points = result.metadata["leverage_points"]
    """
    name = "leverage_points"

    def __call__(
        self,
        state: State,
        max_points: int = 10,
        min_score: float = 0.3,
        **kwargs,
    ) -> State:
        graph = state.graph
        points = self._score_leverage_points(graph, max_points, min_score)

        state.metadata["leverage_points"] = {
            "points": [
                {
                    "node_id": str(p.node_id),
                    "text": p.text,
                    "score": p.score,
                    "reason": p.reason,
                    "in_degree": p.in_degree,
                    "out_degree": p.out_degree,
                    "betweenness": p.betweenness,
                }
                for p in points
            ],
            "total_points": len(points),
        }

        top_points = [f"'{p.text[:40]}' (score={p.score:.2f}, reason: {p.reason})" for p in points[:5]]
        state.record(
            self.name,
            f"Identified {len(points)} leverage points — nodes where small interventions produce large systemic effects. "
            f"Top leverage points: {'; '.join(top_points)}. "
            f"Scored by a weighted combination of in-degree (converging influence), out-degree (diverging influence), "
            f"and betweenness centrality (information flow control). "
            f"The highest-leverage node controls the most critical information pathways in the system.",
        )
        return state

    def _score_leverage_points(
        self,
        graph: Graph,
        max_points: int,
        min_score: float,
    ) -> list[LeveragePoint]:
        """Score nodes as leverage points."""
        if not graph.nodes:
            return []

        in_degree = {nid: 0 for nid in graph.nodes}
        out_degree = {nid: 0 for nid in graph.nodes}

        for edge in graph.edges.values():
            if edge.target_id in in_degree:
                in_degree[edge.target_id] += 1
            if edge.source_id in out_degree:
                out_degree[edge.source_id] += 1

        max_in = max(in_degree.values()) if in_degree else 1
        max_out = max(out_degree.values()) if out_degree else 1

        all_betweenness = self._estimate_betweenness(graph)

        points = []
        for nid, node in graph.nodes.items():
            in_norm = in_degree[nid] / max_in if max_in > 0 else 0
            out_norm = out_degree[nid] / max_out if max_out > 0 else 0

            betweenness = all_betweenness.get(nid, 0.0)

            score = 0.4 * in_norm + 0.3 * out_norm + 0.3 * betweenness

            if score >= min_score:
                reason = self._explain_leverage(in_norm, out_norm, betweenness)
                points.append(LeveragePoint(
                    node_id=nid,
                    text=node.text[:100],
                    score=score,
                    reason=reason,
                    in_degree=in_degree[nid],
                    out_degree=out_degree[nid],
                    betweenness=betweenness,
                ))

        points.sort(key=lambda p: p.score, reverse=True)
        return points[:max_points]

    def _estimate_betweenness(self, graph: Graph) -> dict[UUID, float]:
        """Compute betweenness centrality using Brandes' algorithm (directed)."""
        nodes = list(graph.nodes.keys())
        node_set = set(nodes)
        n = len(nodes)
        if n <= 2:
            return {v: 0.0 for v in nodes}

        adj = defaultdict(list)
        for edge in graph.edges.values():
            adj[edge.source_id].append(edge.target_id)

        CB = {v: 0.0 for v in nodes}

        for s in nodes:
            S: list[UUID] = []
            P: dict[UUID, list[UUID]] = {w: [] for w in nodes}
            sigma = {w: 0.0 for w in nodes}
            sigma[s] = 1.0
            d = {w: -1 for w in nodes}
            d[s] = 0

            Q: list[UUID] = [s]
            while Q:
                v = Q.pop(0)
                S.append(v)
                for w in adj.get(v, []):
                    if w not in node_set:
                        continue
                    if d[w] < 0:
                        Q.append(w)
                        d[w] = d[v] + 1
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v]
                        P[w].append(v)

            delta = {w: 0.0 for w in nodes}
            while S:
                w = S.pop()
                for v in P[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    CB[w] += delta[w]

        norm = (n - 1) * (n - 2)
        if norm > 0:
            for v in nodes:
                CB[v] /= norm

        return CB

    def _explain_leverage(
        self,
        in_norm: float,
        out_norm: float,
        betweenness: float,
    ) -> str:
        """Explain why this is a leverage point."""
        if betweenness > 0.5:
            return "High betweenness — controls information flow"
        if in_norm > 0.7:
            return "High in-degree — many influences converge here"
        if out_norm > 0.7:
            return "High out-degree — influences many other nodes"
        return "Balanced connectivity — moderate influence"


class SystemArchetypeClassifier:
    """Classify system archetypes from causal structure.

    Common archetypes:
    - Fixes that Fail
    - Shifting the Burden
    - Tragedy of the Commons
    - Limits to Growth
    - Success to the Successful

    Usage:
        op = SystemArchetypeClassifier()
        result = op(state)
        archetypes = result.metadata["system_archetypes"]
    """
    name = "system_archetypes"

    def __call__(self, state: State, **kwargs) -> State:
        graph = state.graph
        archetypes = self._classify_archetypes(graph)

        state.metadata["system_archetypes"] = {
            "archetypes": [
                {
                    "name": a.name,
                    "confidence": a.confidence,
                    "nodes": [str(nid) for nid in a.nodes],
                    "description": a.description,
                    "intervention建议": a.intervention建议,
                }
                for a in archetypes
            ],
            "total_archetypes": len(archetypes),
        }

        arch_texts = [f"'{a.name}' (confidence={a.confidence:.2f}): {a.description}" for a in archetypes]
        state.record(
            self.name,
            f"Classified {len(archetypes)} system archetypes from the causal graph structure. "
            f"Archetypes identified: {'; '.join(arch_texts)}. "
            f"Each archetype represents a recurring systemic pattern with known intervention strategies. "
            f"{'Dominant archetype: ' + archetypes[0].name + ' — ' + archetypes[0].intervention建议 + '.' if archetypes else ''} "
            f"Archetype classification helps select appropriate systems-thinking interventions.",
        )
        return state

    def _classify_archetypes(self, graph: Graph) -> list[SystemArchetype]:
        """Classify archetypes based on graph structure."""
        archetypes = []

        loops = self._find_loops(graph)
        if loops:
            archetypes.append(SystemArchetype(
                name="Feedback Dominated",
                confidence=0.8,
                nodes=[nid for loop in loops for nid in loop],
                description="System dominated by feedback loops",
                intervention建议="Identify loop type and break or amplify",
            ))

        chains = self._find_long_chains(graph)
        if chains:
            archetypes.append(SystemArchetype(
                name="Delay Dominated",
                confidence=0.7,
                nodes=[nid for chain in chains for nid in chain],
                description="System with long causal chains (delays)",
                intervention建议="Look for delay reduction opportunities",
            ))

        hubs = self._find_hubs(graph)
        if hubs:
            archetypes.append(SystemArchetype(
                name="Centralized Control",
                confidence=0.75,
                nodes=hubs,
                description="System with central controlling nodes",
                intervention建议="Consider decentralization or redundancy",
            ))

        return archetypes

    def _find_loops(self, graph: Graph) -> list[list[UUID]]:
        """Find feedback loops."""
        loops = []
        adj = defaultdict(list)
        for edge in graph.edges.values():
            adj[edge.source_id].append(edge.target_id)

        visited = set()
        for start in graph.nodes:
            if start in visited:
                continue
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                for neighbor in adj[node]:
                    if neighbor == start and len(path) > 2:
                        loops.append(list(path))
                    elif neighbor not in visited and neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
                visited.add(node)

        return loops[:5]

    def _find_long_chains(self, graph: Graph) -> list[list[UUID]]:
        """Find long causal chains (>3 nodes)."""
        chains = []
        adj = defaultdict(list)
        for edge in graph.edges.values():
            adj[edge.source_id].append(edge.target_id)

        def dfs(node, path, visited):
            if len(path) > 3:
                chains.append(list(path))
            for neighbor in adj[node]:
                if neighbor not in visited:
                    dfs(neighbor, path + [neighbor], visited | {neighbor})

        for node in graph.nodes:
            dfs(node, [node], {node})

        return chains[:5]

    def _find_hubs(self, graph: Graph) -> list[UUID]:
        """Find hub nodes (high degree)."""
        degree = defaultdict(int)
        for edge in graph.edges.values():
            degree[edge.source_id] += 1
            degree[edge.target_id] += 1

        if not degree:
            return []

        avg_degree = sum(degree.values()) / len(degree)
        return [nid for nid, d in degree.items() if d > avg_degree * 1.5]


class CausalSCM:
    """Structural Causal Model operations.

    Implements:
    - do-operator (intervention)
    - Counterfactual reasoning
    - Backdoor path detection
    - Causal effect estimation

    Usage:
        op = CausalSCM()
        result = op(state, intervention={"node_id": "value"})
        counterfactuals = result.metadata["causal_scm"]
    """
    name = "causal_scm"

    def __call__(
        self,
        state: State,
        intervention: dict = None,
        counterfactual: dict = None,
        **kwargs,
    ) -> State:
        graph = state.graph
        result = {}

        if intervention:
            result["intervention"] = self._do_intervention(graph, intervention)

        if counterfactual:
            result["counterfactual"] = self._counterfactual(graph, counterfactual)

        result["backdoor_paths"] = self._find_backdoor_paths(graph)
        result["causal_effects"] = self._estimate_causal_effects(graph)

        state.metadata["causal_scm"] = result

        effects = result.get("causal_effects", {})
        backdoors = result.get("backdoor_paths", [])
        interventions = result.get("intervention", {})
        counterfactuals = result.get("counterfactual", {})
        effect_text = "; ".join(f"{e['source'][:8]}→{e['target'][:8]} (strength={e['effect_strength']:.2f})" for e in list(effects.values())[:5])
        state.record(
            self.name,
            f"Built a structural causal model (SCM) with {len(result)} analysis components. "
            f"Causal effect estimates ({len(effects)} edges): {effect_text or 'none'}. "
            f"Backdoor paths found: {len(backdoors)} potential confounders. "
            f"{'Interventions simulated: ' + str(len(interventions)) + '. ' if interventions else ''}"
            f"{'Counterfactuals evaluated: ' + str(len(counterfactuals)) + '. ' if counterfactuals else ''}"
            f"SCM identifies causal pathways and enables do-operator intervention analysis.",
        )
        return state

    def _do_intervention(
        self,
        graph: Graph,
        intervention: dict,
    ) -> dict:
        """Simulate intervention do(X=x)."""
        results = {}
        for node_id_str, value in intervention.items():
            node_id = UUID(node_id_str) if len(node_id_str) == 32 else node_id_str
            if node_id in graph.nodes:
                affected = self._propagate_intervention(graph, node_id, value)
                results[node_id_str] = {
                    "intervened_value": value,
                    "affected_nodes": [str(nid) for nid in affected],
                    "effect_size": len(affected) / len(graph.nodes) if graph.nodes else 0,
                }
        return results

    def _propagate_intervention(
        self,
        graph: Graph,
        node_id: UUID,
        value: float,
    ) -> set[UUID]:
        """Propagate intervention through causal graph."""
        affected = set()
        queue = [node_id]

        while queue:
            current = queue.pop(0)
            if current in affected:
                continue
            affected.add(current)

            for edge in graph.edges.values():
                if edge.source_id == current and edge.target_id not in affected:
                    if edge.type in (EdgeType.CAUSES, EdgeType.INFERS):
                        queue.append(edge.target_id)

        return affected

    def _counterfactual(
        self,
        graph: Graph,
        counterfactual: dict,
    ) -> dict:
        """Reason about counterfactuals."""
        results = {}
        for node_id_str, hypothetical_value in counterfactual.items():
            node_id = UUID(node_id_str) if len(node_id_str) == 32 else node_id_str
            if node_id in graph.nodes:
                original = graph.nodes[node_id].opinion[2] if graph.nodes[node_id].opinion else 0.5
                results[node_id_str] = {
                    "original_value": original,
                    "hypothetical_value": hypothetical_value,
                    "difference": hypothetical_value - original,
                }
        return results

    def _find_backdoor_paths(self, graph: Graph) -> list[list[UUID]]:
        """Find backdoor paths (confounders)."""
        paths = []
        adj = defaultdict(list)
        for edge in graph.edges.values():
            adj[edge.source_id].append(edge.target_id)
            adj[edge.target_id].append(edge.source_id)

        for node in graph.nodes:
            visited = {node}
            stack = [(node, [node])]
            while stack:
                current, path = stack.pop()
                for neighbor in adj[current]:
                    if neighbor == node and len(path) > 2:
                        paths.append(list(path))
                    elif neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
                        visited.add(neighbor)

        return paths[:10]

    def _estimate_causal_effects(self, graph: Graph) -> dict:
        """Estimate causal effects between connected nodes."""
        effects = {}
        for edge in graph.edges.values():
            if edge.type in (EdgeType.CAUSES, EdgeType.INFERS):
                key = f"{edge.source_id}_{edge.target_id}"
                belief = edge.opinion[2] if len(edge.opinion) > 2 else 0.5
                effects[key] = {
                    "source": str(edge.source_id),
                    "target": str(edge.target_id),
                    "effect_strength": belief,
                    "edge_type": edge.type.name,
                }
        return effects
