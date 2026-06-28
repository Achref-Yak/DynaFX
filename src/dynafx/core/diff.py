from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from dynafx.core.models import BfoCategory, Graph, Node, Edge, Opinion
from dynafx.core.state import State
from dynafx.core.math import convergence_norm, graph_distance


@dataclass
class NodeSnapshot:
    id: str
    type: str
    text: str
    opinion: list[float]
    category: int
    abstract_level: int
    metadata: dict
    bfo_category: Optional[str] = None


@dataclass
class EdgeSnapshot:
    id: str
    source_id: str
    target_id: str
    type: str
    weight: float
    opinion: list[float]


@dataclass
class NodeChange:
    id: str
    opinion_before: list[float]
    opinion_after: list[float]
    type_before: str
    type_after: str
    category_before: int
    category_after: int
    text: str


@dataclass
class EdgeChange:
    id: str
    source_id: str
    target_id: str
    type: str
    weight_before: float
    weight_after: float


@dataclass
class ConflictInfo:
    node_id: str
    text: str
    attacking_nodes: list[str]
    attack_types: list[str]


@dataclass
class CycleDiff:
    step_id: str
    cycle_number: int
    nodes_added: list[NodeSnapshot] = field(default_factory=list)
    nodes_removed: list[str] = field(default_factory=list)
    nodes_modified: list[NodeChange] = field(default_factory=list)
    edges_added: list[EdgeSnapshot] = field(default_factory=list)
    edges_removed: list[str] = field(default_factory=list)
    edges_modified: list[EdgeChange] = field(default_factory=list)
    contradictions: list[ConflictInfo] = field(default_factory=list)
    convergence_delta: float = 0.0
    opinion_shifts: dict[str, tuple[list[float], list[float]]] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "cycle_number": self.cycle_number,
            "nodes_added": [vars(n) for n in self.nodes_added],
            "nodes_removed": self.nodes_removed,
            "nodes_modified": [vars(n) for n in self.nodes_modified],
            "edges_added": [vars(e) for e in self.edges_added],
            "edges_removed": self.edges_removed,
            "edges_modified": [vars(e) for e in self.edges_modified],
            "contradictions": [vars(c) for c in self.contradictions],
            "convergence_delta": self.convergence_delta,
            "opinion_shifts": {k: [list(a), list(b)] for k, (a, b) in self.opinion_shifts.items()},
            "summary": self.summary,
        }

    def to_compact_dict(self) -> dict:
        """Compact serialization: IDs + counts only, no full objects."""
        result: dict[str, Any] = {
            "step_id": self.step_id,
            "cycle_number": self.cycle_number,
            "summary": self.summary,
            "convergence_delta": self.convergence_delta,
        }
        if self.nodes_added:
            result["nodes_added"] = [
                {"id": n.id, "type": n.type, "text": n.text[:80],
                 "opinion": [round(x, 3) for x in n.opinion]}
                for n in self.nodes_added
            ]
        if self.nodes_removed:
            result["nodes_removed"] = self.nodes_removed
        if self.nodes_modified:
            result["nodes_modified"] = [
                {"id": n.id, "type_before": n.type_before, "type_after": n.type_after}
                for n in self.nodes_modified
            ]
        if self.edges_added:
            result["edges_added"] = [
                {"id": e.id, "type": e.type, "source_id": e.source_id, "target_id": e.target_id}
                for e in self.edges_added
            ]
        if self.edges_removed:
            result["edges_removed"] = self.edges_removed
        if self.edges_modified:
            result["edges_modified"] = [
                {"id": e.id, "type": e.type, "weight_before": e.weight_before, "weight_after": e.weight_after}
                for e in self.edges_modified
            ]
        if self.contradictions:
            seen = set()
            unique = []
            for c in self.contradictions:
                if c.node_id not in seen:
                    seen.add(c.node_id)
                    unique.append({"node_id": c.node_id, "text": c.text[:80], "attack_types": c.attack_types})
            result["contradictions"] = unique
        if self.opinion_shifts:
            shifts = {}
            for k, (a, b) in self.opinion_shifts.items():
                if a != b:
                    shifts[k] = {
                        "before": [round(x, 3) for x in a],
                        "after": [round(x, 3) for x in b],
                    }
            if shifts:
                result["opinion_shifts"] = shifts
        return result


def _snapshot_node(node: Node) -> NodeSnapshot:
    op = node.opinion or Opinion()
    return NodeSnapshot(
        id=node.id.hex,
        type=node.type.name,
        text=node.text,
        opinion=[op.belief, op.disbelief, op.uncertainty, op.prior],
        category=node.category,
        abstract_level=node.abstraction_level,
        metadata=dict(node.metadata),
        bfo_category=node.bfo_category.name if node.bfo_category else None,
    )


def _snapshot_edge(edge: Edge) -> EdgeSnapshot:
    op = edge.opinion or Opinion()
    return EdgeSnapshot(
        id=edge.id.hex,
        source_id=edge.source_id.hex,
        target_id=edge.target_id.hex,
        type=edge.type.name,
        weight=edge.weight,
        opinion=[op.belief, op.disbelief, op.uncertainty, op.prior],
    )


def compute_diff(before: State, after: State, step_id: str = "", cycle: int = 0) -> CycleDiff:
    bg = before.graph
    ag = after.graph

    before_ids = set(bg.nodes)
    after_ids = set(ag.nodes)

    removed_ids = before_ids - after_ids
    added_ids = after_ids - before_ids
    common_ids = before_ids & after_ids

    removed_edges = {e.id.hex for e in bg.edges.values()} - {e.id.hex for e in ag.edges.values()}
    added_edges = {e.id.hex for e in ag.edges.values()} - {e.id.hex for e in bg.edges.values()}
    common_edge_ids = {e.id.hex for e in bg.edges.values()} & {e.id.hex for e in ag.edges.values()}

    diff = CycleDiff(step_id=step_id, cycle_number=cycle)

    for nid in added_ids:
        diff.nodes_added.append(_snapshot_node(ag.nodes[nid]))

    diff.nodes_removed = [nid.hex for nid in removed_ids]

    opinion_shifts: dict[str, tuple[list[float], list[float]]] = {}
    for nid in common_ids:
        bn = bg.nodes[nid]
        an = ag.nodes[nid]
        bop = bn.opinion or Opinion()
        aop = an.opinion or Opinion()
        bb = [bop.belief, bop.disbelief, bop.uncertainty, bop.prior]
        ab = [aop.belief, aop.disbelief, aop.uncertainty, aop.prior]

        if bb != ab or bn.type != an.type or bn.category != an.category:
            diff.nodes_modified.append(NodeChange(
                id=nid.hex,
                opinion_before=bb,
                opinion_after=ab,
                type_before=bn.type.name,
                type_after=an.type.name,
                category_before=bn.category,
                category_after=an.category,
                text=an.text,
            ))
            opinion_shifts[nid.hex] = (bb, ab)

    diff.opinion_shifts = opinion_shifts

    be_map = {e.id.hex: e for e in bg.edges.values()}
    ae_map = {e.id.hex: e for e in ag.edges.values()}

    for eid in added_edges:
        e = ae_map[eid]
        diff.edges_added.append(_snapshot_edge(e))

    diff.edges_removed = list(removed_edges)

    for eid in common_edge_ids:
        be = be_map[eid]
        ae = ae_map[eid]
        if be.weight != ae.weight or be.type != ae.type:
            diff.edges_modified.append(EdgeChange(
                id=eid,
                source_id=ae.source_id.hex,
                target_id=ae.target_id.hex,
                type=ae.type.name,
                weight_before=be.weight,
                weight_after=ae.weight,
            ))

    attack_types = {"ATTACKS", "CONTRADICTS", "REBUTS"}
    for e in ag.edges.values():
        if e.type.name in attack_types and e.target_id.hex not in added_ids:
            if e.target_id in ag.nodes:
                diff.contradictions.append(ConflictInfo(
                    node_id=e.target_id.hex,
                    text=ag.nodes[e.target_id].text[:80],
                    attacking_nodes=[e.source_id.hex],
                    attack_types=[e.type.name],
                ))

    g_dist = graph_distance(
        before_ids, after_ids,
        list(bg.edges.values()), list(ag.edges.values()),
        _beliefs(before), _beliefs(after),
    )
    diff.convergence_delta = convergence_norm(
        graph_distance=g_dist,
        attention_distance=g_dist,
        hidden_distance=0.0,
        operator_change=0.0 if step_id == "tick" else 1.0,
    )

    parts = []
    if diff.nodes_added:
        parts.append(f"+{len(diff.nodes_added)} nodes")
    if diff.nodes_removed:
        parts.append(f"-{len(diff.nodes_removed)} nodes")
    if diff.edges_added:
        parts.append(f"+{len(diff.edges_added)} edges")
    if diff.nodes_modified:
        parts.append(f"~{len(diff.nodes_modified)} opinions")
    if diff.contradictions:
        parts.append(f"{len(diff.contradictions)} contradictions")
    diff.summary = f"{step_id}: {', '.join(parts)}" if parts else f"{step_id}: no changes"

    return diff


def _beliefs(state: State) -> dict[UUID, float]:
    result = {}
    for nid, node in state.graph.nodes.items():
        op = node.opinion or Opinion()
        result[nid] = op.belief + op.prior * op.uncertainty
    return result
