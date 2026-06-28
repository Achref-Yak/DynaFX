"""Graph analysis building blocks for verifiable reasoning summaries.

Provides:
    - traverse: bidirectional BFS from a root node
    - classify_evidence: categorize paths by edge type (supporting/weakening/contradicting/contextual)
    - generate_label: auto-generate short labels for nodes
    - find_evidence_chains: evidence chain finder using traversal and classification
    - build_verifiable_summary: audit-grade structured ledger of graph evidence
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from dynafx.core.models import (
    Edge,
    EdgeType,
    Graph,
    Node,
    NodeType,
    Opinion,
)


# ── Edge type categories ────────────────────────────────────────

_SUPPORTING = frozenset({
    EdgeType.SUPPORTS, EdgeType.INFERS, EdgeType.JUSTIFIES,
    EdgeType.ENABLES, EdgeType.DIRECT,
})
_ATTACKING = frozenset({
    EdgeType.ATTACKS, EdgeType.REBUTS, EdgeType.CONTRADICTS,
})
_CONTEXTUAL = frozenset({
    EdgeType.ASSOCIATED_WITH, EdgeType.CITES, EdgeType.PART_OF,
    EdgeType.HAS_ATTRIBUTE, EdgeType.LOCATED_AT, EdgeType.EMPLOYED_BY,
    EdgeType.CONTACT_OF, EdgeType.SIMILAR, EdgeType.EVIDENCE,
    EdgeType.CAUSES, EdgeType.TEMPORAL, EdgeType.FLOWS_TO,
    EdgeType.QUALIFIES, EdgeType.DEPENDS, EdgeType.SUPPORT,
    EdgeType.HAS_GOAL, EdgeType.INTENDS, EdgeType.KNOWS,
    EdgeType.COMMUNICATED, EdgeType.PREFERS, EdgeType.USES,
    EdgeType.PRODUCES, EdgeType.CONSUMES, EdgeType.TRANSFORMS,
})

# Node types that carry meaningful evidence
_EVIDENCE_TYPES = frozenset({
    NodeType.CLAIM, NodeType.AXIOM, NodeType.EVIDENCE, NodeType.COUNTERCLAIM,
    NodeType.OBSERVATION, NodeType.HYPOTHESIS,
})

# All types included in facts (evidence + world-model + entity for grounding)
_FACT_TYPES = _EVIDENCE_TYPES | frozenset({
    NodeType.ENTITY, NodeType.AGENT, NodeType.PROCESS,
    NodeType.STATE, NodeType.PROPERTY, NodeType.GOAL,
    NodeType.BELIEF, NodeType.KNOWLEDGE, NodeType.DECISION, NodeType.ACTION,
})


# ── 1. Bidirectional traversal ──────────────────────────────────

@dataclass
class PathEntry:
    """A single step in a traversal path."""
    node_id: UUID
    node_type: str
    text: str
    edge_type: Optional[str] = None
    direction: str = "outgoing"  # "outgoing" or "incoming"
    belief: float = 0.0


@dataclass
class TraversalResult:
    """Result of bidirectional traversal from a root node."""
    root_id: UUID
    root_text: str
    root_type: str
    root_belief: float
    paths: list[list[PathEntry]] = field(default_factory=list)


def traverse(
    graph: Graph,
    root_id: UUID,
    max_depth: int = 3,
    include_contextual: bool = False,
) -> TraversalResult:
    """Bidirectional BFS from root_id, following both incoming and outgoing edges.

    Args:
        graph: The graph to traverse.
        root_id: Starting node.
        max_depth: Maximum hops from root.
        include_contextual: If True, follow ASSOCIATED_WITH and other contextual edges.
            If False, only follow SUPPORTS/INFERS/JUSTIFIES/ATTACKS/REBUTS/CONTRADICTS.
    """
    root = graph.nodes[root_id]
    result = TraversalResult(
        root_id=root_id,
        root_text=root.text,
        root_type=root.type.name,
        root_belief=(root.opinion or Opinion()).belief,
    )

    reasoning_edges = _SUPPORTING | _ATTACKING
    allowed = reasoning_edges if not include_contextual else reasoning_edges | _CONTEXTUAL

    visited: set[UUID] = {root_id}
    # Each queue entry: (current_node_id, path_so_far)
    queue: list[tuple[UUID, list[PathEntry]]] = [
        (root_id, [PathEntry(
            node_id=root_id,
            node_type=root.type.name,
            text=root.text,
            belief=(root.opinion or Opinion()).belief,
        )])
    ]

    while queue:
        current_id, path = queue.pop(0)
        if len(path) > max_depth:
            continue

        # Outgoing edges
        for edge in graph.edges.values():
            if edge.source_id != current_id:
                continue
            if edge.type not in allowed:
                continue
            target_id = edge.target_id
            if target_id not in visited:
                visited.add(target_id)
                target = graph.nodes.get(target_id)
                if target is None:
                    continue
                entry = PathEntry(
                    node_id=target_id,
                    node_type=target.type.name,
                    text=target.text,
                    edge_type=edge.type.name,
                    direction="outgoing",
                    belief=(target.opinion or Opinion()).belief,
                )
                new_path = path + [entry]
                queue.append((target_id, new_path))
                if len(new_path) > 1:
                    result.paths.append(new_path)

        # Incoming edges
        for edge in graph.edges.values():
            if edge.target_id != current_id:
                continue
            if edge.type not in allowed:
                continue
            source_id = edge.source_id
            if source_id not in visited:
                visited.add(source_id)
                source = graph.nodes.get(source_id)
                if source is None:
                    continue
                entry = PathEntry(
                    node_id=source_id,
                    node_type=source.type.name,
                    text=source.text,
                    edge_type=edge.type.name,
                    direction="incoming",
                    belief=(source.opinion or Opinion()).belief,
                )
                new_path = path + [entry]
                queue.append((source_id, new_path))
                if len(new_path) > 1:
                    result.paths.append(new_path)

    return result


# ── 2. Evidence classification ──────────────────────────────────

@dataclass
class EvidenceItem:
    """A single piece of evidence with its classification."""
    path: list[PathEntry]
    label: str
    text: str
    belief: float
    edge_type: str
    direction: str
    classification: str  # "supporting", "weakening", "contradicting", "contextual"


@dataclass
class EvidenceClassification:
    """Classified evidence relative to a root claim."""
    root_text: str
    root_belief: float
    supporting: list[EvidenceItem] = field(default_factory=list)
    weakening: list[EvidenceItem] = field(default_factory=list)
    contradicting: list[EvidenceItem] = field(default_factory=list)
    contextual: list[EvidenceItem] = field(default_factory=list)


def classify_evidence(
    graph: Graph,
    root_id: UUID,
    max_depth: int = 3,
    include_contextual: bool = False,
) -> EvidenceClassification:
    """Traverse from root and classify all discovered evidence by edge type.

    Filters to CLAIM/AXIOM/EVIDENCE/COUNTERCLAIM nodes only.
    Deduplicates by terminal node ID.

    Classification logic (deterministic, edge-based):
        - SUPPORTS/INFERS/JUSTIFIES outgoing = supporting
        - SUPPORTS/INFERS/JUSTIFIES incoming = supporting (someone supports root)
        - ATTACKS/REBUTS/CONTRADICTS outgoing = weakening (root attacks something)
        - ATTACKS/REBUTS/CONTRADICTS incoming = contradicting (something attacks root)
    """
    traversal = traverse(graph, root_id, max_depth, include_contextual)
    result = EvidenceClassification(
        root_text=traversal.root_text,
        root_belief=traversal.root_belief,
    )

    seen_terminal_ids: set[UUID] = set()

    for path in traversal.paths:
        if len(path) < 2:
            continue

        # The evidence is the last node in the path
        evidence = path[-1]
        terminal_node = graph.nodes.get(evidence.node_id)

        # Filter: only evidence-bearing node types
        if terminal_node is None or terminal_node.type not in _EVIDENCE_TYPES:
            continue

        # Deduplicate: one entry per terminal node
        if evidence.node_id in seen_terminal_ids:
            continue
        seen_terminal_ids.add(evidence.node_id)

        edge_type_str = evidence.edge_type or "UNKNOWN"
        edge_type = EdgeType[edge_type_str] if edge_type_str in EdgeType.__members__ else None

        # Classify based on edge type and direction
        if edge_type in _SUPPORTING:
            classification = "supporting"
        elif edge_type in _ATTACKING:
            if evidence.direction == "outgoing":
                classification = "weakening"
            else:
                classification = "contradicting"
        else:
            classification = "contextual"

        label = generate_label(terminal_node, edge_type_str)

        item = EvidenceItem(
            path=path,
            label=label,
            text=evidence.text,
            belief=evidence.belief,
            edge_type=edge_type_str,
            direction=evidence.direction,
            classification=classification,
        )

        if classification == "supporting":
            result.supporting.append(item)
        elif classification == "weakening":
            result.weakening.append(item)
        elif classification == "contradicting":
            result.contradicting.append(item)
        else:
            result.contextual.append(item)

    return result


# ── 3. Evidence label generator ─────────────────────────────────

def generate_label(
    node: Node | None,
    edge_type: str = "",
    max_words: int = 8,
) -> str:
    """Auto-generate a short label for a node.

    Priority:
        1. Node metadata["concept"] if present
        2. Node metadata["entity_kind"] if present
        3. First N words of node text
    """
    if node is None:
        return "evidence"

    # Try concept metadata
    concept = node.metadata.get("concept")
    if concept:
        return concept.replace("_", " ").title()

    # Try entity kind
    entity_kind = node.metadata.get("entity_kind")
    if entity_kind:
        return entity_kind

    # Fall back to first N words
    words = node.text.split()[:max_words]
    label = " ".join(words)
    if len(node.text.split()) > max_words:
        label += "..."
    return label


# ── 4. Evidence chains ──────────────────────────────────────────

@dataclass
class EvidenceChain:
    """A single evidence chain with full text and classification."""
    root_text: str
    root_belief: float
    chain_length: int
    edge_type: str
    direction: str
    evidence_text: str
    evidence_belief: float
    classification: str  # "supporting", "weakening", "contradicting", "contextual"
    label: str


def find_evidence_chains(
    graph: Graph,
    root_id: UUID | None = None,
    max_depth: int = 3,
    include_contextual: bool = False,
) -> list[EvidenceChain]:
    """Find and classify all evidence chains from root nodes.

    If root_id is None, finds all CLAIM and HYPOTHESIS nodes and traces from each.
    Returns full text (not truncated), classified by position relative to root.
    """
    roots: list[UUID] = []
    if root_id is not None:
        roots = [root_id]
    else:
        for nid, node in graph.nodes.items():
            if node.type in (NodeType.CLAIM, NodeType.HYPOTHESIS):
                roots.append(nid)

    chains: list[EvidenceChain] = []

    for rid in roots:
        classification = classify_evidence(graph, rid, max_depth, include_contextual)
        root_node = graph.nodes[rid]

        for item in classification.supporting:
            chains.append(EvidenceChain(
                root_text=root_node.text,
                root_belief=(root_node.opinion or Opinion()).belief,
                chain_length=len(item.path),
                edge_type=item.edge_type,
                direction=item.direction,
                evidence_text=item.text,
                evidence_belief=item.belief,
                classification="supporting",
                label=item.label,
            ))

        for item in classification.weakening:
            chains.append(EvidenceChain(
                root_text=root_node.text,
                root_belief=(root_node.opinion or Opinion()).belief,
                chain_length=len(item.path),
                edge_type=item.edge_type,
                direction=item.direction,
                evidence_text=item.text,
                evidence_belief=item.belief,
                classification="weakening",
                label=item.label,
            ))

        for item in classification.contradicting:
            chains.append(EvidenceChain(
                root_text=root_node.text,
                root_belief=(root_node.opinion or Opinion()).belief,
                chain_length=len(item.path),
                edge_type=item.edge_type,
                direction=item.direction,
                evidence_text=item.text,
                evidence_belief=item.belief,
                classification="contradicting",
                label=item.label,
            ))

        for item in classification.contextual:
            chains.append(EvidenceChain(
                root_text=root_node.text,
                root_belief=(root_node.opinion or Opinion()).belief,
                chain_length=len(item.path),
                edge_type=item.edge_type,
                direction=item.direction,
                evidence_text=item.text,
                evidence_belief=item.belief,
                classification="contextual",
                label=item.label,
            ))

    return chains


# ── 5. Verifiable summary ───────────────────────────────────────

@dataclass
class Fact:
    """A raw node-grounded statement."""
    node_id: str
    text: str
    belief: float
    belief_tier: str  # "high" (>=0.7), "medium" (0.3-0.7), "low" (<0.3), "uninitialized" (0.0)


@dataclass
class EdgeRecord:
    """An explicit edge between two nodes."""
    source_node_id: str
    target_node_id: str
    edge_type: str
    source_belief: float
    target_belief: float


@dataclass
class GraphAggregates:
    """Deterministic graph statistics."""
    node_count_by_type: dict[str, int]
    edge_count_by_type: dict[str, int]
    max_belief_node: Fact | None
    min_belief_node: Fact | None
    max_depth_path: int
    avg_incoming_edges: float
    avg_outgoing_edges: float
    support_edge_count: int
    contradict_edge_count: int
    edge_polarity_balance: float


@dataclass
class VerifiableSummary:
    """Audit-grade structured ledger of graph evidence.

    Every field is directly traceable to graph nodes, edges, or
    deterministic aggregates. No inference, no heuristics, no
    semantic guessing.
    """
    root_id: str
    facts: list[Fact]
    supports: list[EdgeRecord]
    contradictions: list[EdgeRecord]
    aggregates: GraphAggregates

    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "facts": [
                {"node_id": f.node_id, "text": f.text, "belief": round(f.belief, 3),
                 "belief_tier": f.belief_tier}
                for f in self.facts
            ],
            "supports": [
                {"source_node_id": s.source_node_id, "target_node_id": s.target_node_id,
                 "edge_type": s.edge_type, "source_belief": round(s.source_belief, 3),
                 "target_belief": round(s.target_belief, 3)}
                for s in self.supports
            ],
            "contradictions": [
                {"source_node_id": c.source_node_id, "target_node_id": c.target_node_id,
                 "edge_type": c.edge_type, "source_belief": round(c.source_belief, 3),
                 "target_belief": round(c.target_belief, 3)}
                for c in self.contradictions
            ],
            "aggregates": {
                "node_count_by_type": self.aggregates.node_count_by_type,
                "edge_count_by_type": self.aggregates.edge_count_by_type,
                "max_belief_node": {
                    "node_id": self.aggregates.max_belief_node.node_id,
                    "text": self.aggregates.max_belief_node.text,
                    "belief": round(self.aggregates.max_belief_node.belief, 3),
                    "belief_tier": self.aggregates.max_belief_node.belief_tier,
                } if self.aggregates.max_belief_node else None,
                "min_belief_node": {
                    "node_id": self.aggregates.min_belief_node.node_id,
                    "text": self.aggregates.min_belief_node.text,
                    "belief": round(self.aggregates.min_belief_node.belief, 3),
                    "belief_tier": self.aggregates.min_belief_node.belief_tier,
                } if self.aggregates.min_belief_node else None,
                "max_depth_path": self.aggregates.max_depth_path,
                "avg_incoming_edges": round(self.aggregates.avg_incoming_edges, 2),
                "avg_outgoing_edges": round(self.aggregates.avg_outgoing_edges, 2),
                "support_edge_count": self.aggregates.support_edge_count,
                "contradict_edge_count": self.aggregates.contradict_edge_count,
                "edge_polarity_balance": round(self.aggregates.edge_polarity_balance, 3),
            },
        }


def _compute_belief_tier(belief: float) -> str:
    """Classify belief into a deterministic tier."""
    if belief == 0.0:
        return "uninitialized"
    if belief >= 0.7:
        return "high"
    if belief >= 0.3:
        return "medium"
    return "low"


def _compute_aggregates(graph: Graph) -> GraphAggregates:
    """Compute deterministic graph statistics."""
    # Node counts by type
    node_types = Counter(n.type.name for n in graph.nodes.values())

    # Edge counts by type
    edge_types = Counter(e.type.name for e in graph.edges.values())

    # Max/min belief nodes (only from _FACT_TYPES)
    max_node: Fact | None = None
    min_node: Fact | None = None
    for nid, node in graph.nodes.items():
        if node.type not in _FACT_TYPES:
            continue
        op = node.opinion or Opinion()
        belief = op.belief
        fact = Fact(
            node_id=nid.hex,
            text=node.text[:80],
            belief=belief,
            belief_tier=_compute_belief_tier(belief),
        )
        if max_node is None or belief > max_node.belief:
            max_node = fact
        if min_node is None or belief < min_node.belief:
            min_node = fact

    # Max depth: BFS from each node to find longest shortest path
    max_depth = 0
    for nid in graph.nodes:
        visited: set[UUID] = {nid}
        queue: list[tuple[UUID, int]] = [(nid, 0)]
        while queue:
            current, depth = queue.pop(0)
            for edge in graph.edges.values():
                neighbor = None
                if edge.source_id == current and edge.target_id not in visited:
                    neighbor = edge.target_id
                elif edge.target_id == current and edge.source_id not in visited:
                    neighbor = edge.source_id
                if neighbor is not None and neighbor in graph.nodes:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    if depth + 1 > max_depth:
                        max_depth = depth + 1

    # Average incoming/outgoing edges
    incoming_count = Counter()
    outgoing_count = Counter()
    for edge in graph.edges.values():
        outgoing_count[edge.source_id] += 1
        incoming_count[edge.target_id] += 1

    n_nodes = len(graph.nodes) or 1
    avg_in = sum(incoming_count.values()) / n_nodes
    avg_out = sum(outgoing_count.values()) / n_nodes

    # Support/contradict edge counts and polarity balance
    support_count = sum(1 for e in graph.edges.values() if e.type in _SUPPORTING)
    contradict_count = sum(1 for e in graph.edges.values() if e.type in _ATTACKING)
    total_polar = support_count + contradict_count
    polarity = support_count / total_polar if total_polar > 0 else 0.0

    return GraphAggregates(
        node_count_by_type=dict(node_types),
        edge_count_by_type=dict(edge_types),
        max_belief_node=max_node,
        min_belief_node=min_node,
        max_depth_path=max_depth,
        avg_incoming_edges=avg_in,
        avg_outgoing_edges=avg_out,
        support_edge_count=support_count,
        contradict_edge_count=contradict_count,
        edge_polarity_balance=polarity,
    )


def build_verifiable_summary(
    graph: Graph,
    root_id: UUID,
    max_depth: int = 3,
) -> VerifiableSummary:
    """Build an audit-grade verifiable summary of graph evidence.

    Every output statement maps to explicit graph structure:
        - facts: node-grounded statements (CLAIM/AXIOM/EVIDENCE/ENTITY)
        - supports: explicit SUPPORTS-type edges
        - contradictions: explicit ATTACKS/REBUTS/CONTRADICTS edges
        - aggregates: deterministic graph statistics
    """
    # Facts: extract from all _FACT_TYPES nodes
    facts: list[Fact] = []
    for nid, node in graph.nodes.items():
        if node.type not in _FACT_TYPES:
            continue
        op = node.opinion or Opinion()
        text = node.text.strip()
        if not text:
            continue
        facts.append(Fact(
            node_id=nid.hex,
            text=text,
            belief=op.belief,
            belief_tier=_compute_belief_tier(op.belief),
        ))
    # Sort by belief descending
    facts.sort(key=lambda f: f.belief, reverse=True)

    # Supports: explicit SUPPORTS-type edges (exclude entity nodes)
    supports: list[EdgeRecord] = []
    for edge in graph.edges.values():
        if edge.type not in _SUPPORTING:
            continue
        source_node = graph.nodes.get(edge.source_id)
        target_node = graph.nodes.get(edge.target_id)
        if source_node is None or target_node is None:
            continue
        if source_node.type == NodeType.ENTITY or target_node.type == NodeType.ENTITY:
            continue
        supports.append(EdgeRecord(
            source_node_id=edge.source_id.hex,
            target_node_id=edge.target_id.hex,
            edge_type=edge.type.name,
            source_belief=(source_node.opinion or Opinion()).belief,
            target_belief=(target_node.opinion or Opinion()).belief,
        ))

    # Contradictions: explicit ATTACKS/REBUTS/CONTRADICTS edges (exclude entity nodes)
    contradictions: list[EdgeRecord] = []
    for edge in graph.edges.values():
        if edge.type not in _ATTACKING:
            continue
        source_node = graph.nodes.get(edge.source_id)
        target_node = graph.nodes.get(edge.target_id)
        if source_node is None or target_node is None:
            continue
        if source_node.type == NodeType.ENTITY or target_node.type == NodeType.ENTITY:
            continue
        contradictions.append(EdgeRecord(
            source_node_id=edge.source_id.hex,
            target_node_id=edge.target_id.hex,
            edge_type=edge.type.name,
            source_belief=(source_node.opinion or Opinion()).belief,
            target_belief=(target_node.opinion or Opinion()).belief,
        ))

    # Aggregates
    aggregates = _compute_aggregates(graph)

    return VerifiableSummary(
        root_id=root_id.hex,
        facts=facts,
        supports=supports,
        contradictions=contradictions,
        aggregates=aggregates,
    )
