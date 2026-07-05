"""System modeling operators — read and write layer for G→S transformation.

Read operators are query-only — they never modify state.
Write operators are the only things that touch the graph.
Every call attaches provenance (who proposed, reasoning, confidence).
retract creates new trace entry, never deletes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from dynafx.core.models import (
    Edge,
    EdgeType,
    Graph,
    Node,
    NodeType,
    Opinion,
    Parameter,
)

logger = logging.getLogger(__name__)


class Role(Enum):
    """System dynamics roles for nodes."""
    STOCK = "stock"
    FLOW = "flow"
    AUXILIARY = "auxiliary"
    INPUT = "input"
    OUTPUT = "output"
    CONSTANT = "constant"


@dataclass(frozen=True)
class TraceEntry:
    """Immutable trace entry for provenance tracking."""
    action_id: str
    timestamp: float
    action_type: str  # "create_node", "create_edge", "set_role", "set_parameter", "merge_nodes", "retract"
    target_id: Optional[UUID]
    proposer: str  # who/what proposed this action
    reasoning: str  # what source text or reasoning motivated it
    confidence: float  # what confidence it was proposed with
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleAssignment:
    """Role assignment with provenance."""
    node_id: UUID
    role: Role
    opinion: Opinion
    action_id: str


# ── Read operators ──────────────────────────────────────────────

def list_nodes(graph: Graph) -> list[Node]:
    """List all nodes in the graph. Query-only, never modifies state."""
    return list(graph.nodes.values())


def get_node(graph: Graph, node_id: UUID) -> Optional[Node]:
    """Get a node by ID. Query-only, never modifies state."""
    return graph.nodes.get(node_id)


def get_edge(graph: Graph, edge_id: UUID) -> Optional[Edge]:
    """Get an edge by ID. Query-only, never modifies state."""
    return graph.edges.get(edge_id)


def query_contested(graph: Graph) -> list[RoleAssignment]:
    """Query nodes with contested roles (resolved: None). Query-only, never modifies state."""
    contested = []
    for node in graph.nodes.values():
        if node.type == NodeType.ENTITY:
            continue
        # Check if node has role assignment with unresolved conflict
        if hasattr(node, 'metadata') and node.metadata.get('role_opinion') is not None:
            opinion = node.metadata['role_opinion']
            if isinstance(opinion, Opinion) and opinion.uncertainty > 0.5:
                contested.append(RoleAssignment(
                    node_id=node.id,
                    role=Role(node.metadata.get('role', 'auxiliary')),
                    opinion=opinion,
                    action_id=node.metadata.get('role_action_id', ''),
                ))
    return contested


def query_by_role(graph: Graph, role: str) -> list[Node]:
    """Query nodes by role. Query-only, never modifies state."""
    result = []
    for node in graph.nodes.values():
        if node.type == NodeType.ENTITY:
            continue
        if hasattr(node, 'metadata') and node.metadata.get('role') == role:
            result.append(node)
    return result


def get_trace_history(graph: Graph, node_id: UUID) -> list[TraceEntry]:
    """Get trace history for a node. Query-only, never modifies state."""
    node = graph.nodes.get(node_id)
    if node is None:
        return []

    history = []
    # Reconstruct trace from node metadata
    if hasattr(node, 'metadata'):
        # Check for trace entries in metadata
        trace_entries = node.metadata.get('trace_history', [])
        for entry in trace_entries:
            if isinstance(entry, dict):
                history.append(TraceEntry(
                    action_id=entry.get('action_id', ''),
                    timestamp=entry.get('timestamp', 0.0),
                    action_type=entry.get('action_type', ''),
                    target_id=node_id,
                    proposer=entry.get('proposer', ''),
                    reasoning=entry.get('reasoning', ''),
                    confidence=entry.get('confidence', 0.0),
                    metadata=entry.get('metadata', {}),
                ))
    return history


# ── Write operators ─────────────────────────────────────────────

def create_node(
    graph: Graph,
    name: str,
    candidate_roles: dict[str, float],
    proposer: str = "human",
    reasoning: str = "",
    confidence: float = 0.5,
) -> UUID:
    """Create a new node with candidate roles. Write operator, modifies graph."""
    node_id = uuid4()
    action_id = str(uuid4())

    # Determine node type based on candidate roles
    node_type = NodeType.CONCEPT
    if any(role in candidate_roles for role in ["stock", "flow"]):
        node_type = NodeType.CONCEPT  # Will be refined by set_role

    node = Node(
        id=node_id,
        type=node_type,
        text=name,
        metadata={
            "candidate_roles": candidate_roles,
            "role": None,
            "role_opinion": None,
            "role_action_id": None,
            "trace_history": [{
                "action_id": action_id,
                "timestamp": __import__('time').time(),
                "action_type": "create_node",
                "proposer": proposer,
                "reasoning": reasoning,
                "confidence": confidence,
                "metadata": {"candidate_roles": candidate_roles},
            }],
        },
    )
    graph.nodes[node_id] = node

    logger.info("Created node %s: %s", node_id, name)
    return node_id


def create_edge(
    graph: Graph,
    source: UUID,
    target: UUID,
    polarity: str,
    confidence: float,
    proposer: str = "human",
    reasoning: str = "",
) -> UUID:
    """Create a new edge with polarity and confidence. Write operator, modifies graph."""
    if polarity not in ("+", "-"):
        raise ValueError(f"Invalid polarity: {polarity}. Must be '+' or '-'.")

    edge_id = uuid4()
    action_id = str(uuid4())

    # Determine edge type based on polarity
    edge_type = EdgeType.CAUSES if polarity == "+" else EdgeType.CONTRADICTS

    edge = Edge(
        id=edge_id,
        source_id=source,
        target_id=target,
        type=edge_type,
        polarity=1 if polarity == "+" else -1,
        metadata={
            "polarity": polarity,
            "confidence": confidence,
            "trace_history": [{
                "action_id": action_id,
                "timestamp": __import__('time').time(),
                "action_type": "create_edge",
                "proposer": proposer,
                "reasoning": reasoning,
                "confidence": confidence,
                "metadata": {"polarity": polarity},
            }],
        },
    )
    graph.edges[edge_id] = edge

    logger.info("Created edge %s: %s -> %s (%s)", edge_id, source, target, polarity)
    return edge_id


def set_role(
    graph: Graph,
    node: UUID,
    role: str,
    opinion: Opinion,
    proposer: str = "human",
    reasoning: str = "",
) -> None:
    """Set role for a node with SL opinion. Write operator, modifies graph."""
    node_obj = graph.nodes.get(node)
    if node_obj is None:
        raise ValueError(f"Node {node} not found")

    action_id = str(uuid4())

    # Update node metadata
    node_obj.metadata['role'] = role
    node_obj.metadata['role_opinion'] = opinion
    node_obj.metadata['role_action_id'] = action_id

    # Add trace entry
    node_obj.metadata.setdefault('trace_history', []).append({
        "action_id": action_id,
        "timestamp": __import__('time').time(),
        "action_type": "set_role",
        "proposer": proposer,
        "reasoning": reasoning,
        "confidence": opinion.belief,
        "metadata": {"role": role, "opinion": {"b": opinion.belief, "d": opinion.disbelief, "u": opinion.uncertainty}},
    })

    logger.info("Set role for node %s: %s", node, role)


def set_parameter(
    graph: Graph,
    node: UUID,
    value: Optional[float],
    opinion: Opinion,
    proposer: str = "human",
    reasoning: str = "",
) -> None:
    """Set parameter value and opinion for a node. Write operator, modifies graph."""
    node_obj = graph.nodes.get(node)
    if node_obj is None:
        raise ValueError(f"Node {node} not found")

    action_id = str(uuid4())

    # Create or update parameter
    parameter = Parameter(value=value, opinion=opinion)
    node_obj.metadata['parameter'] = parameter

    # Add trace entry
    node_obj.metadata.setdefault('trace_history', []).append({
        "action_id": action_id,
        "timestamp": __import__('time').time(),
        "action_type": "set_parameter",
        "proposer": proposer,
        "reasoning": reasoning,
        "confidence": opinion.belief,
        "metadata": {"value": value, "opinion": {"b": opinion.belief, "d": opinion.disbelief, "u": opinion.uncertainty}},
    })

    logger.info("Set parameter for node %s: value=%s", node, value)


def merge_nodes(
    graph: Graph,
    a: UUID,
    b: UUID,
    reason: str,
    proposer: str = "human",
) -> UUID:
    """Merge two nodes, keeping the one with more spans. Write operator, modifies graph."""
    node_a = graph.nodes.get(a)
    node_b = graph.nodes.get(b)
    if node_a is None or node_b is None:
        raise ValueError(f"One or both nodes not found: {a}, {b}")

    action_id = str(uuid4())

    # Determine which node to keep (the one with more trace history)
    keep_id = a if len(node_a.metadata.get('trace_history', [])) >= len(node_b.metadata.get('trace_history', [])) else b
    remove_id = b if keep_id == a else a

    # Merge trace histories
    keep_node = graph.nodes[keep_id]
    remove_node = graph.nodes[remove_id]

    keep_traces = keep_node.metadata.get('trace_history', [])
    remove_traces = remove_node.metadata.get('trace_history', [])
    keep_traces.extend(remove_traces)
    keep_node.metadata['trace_history'] = keep_traces

    # Add merge trace entry
    keep_node.metadata.setdefault('trace_history', []).append({
        "action_id": action_id,
        "timestamp": __import__('time').time(),
        "action_type": "merge_nodes",
        "proposer": proposer,
        "reasoning": reason,
        "confidence": 1.0,
        "metadata": {"merged_from": [str(a), str(b)]},
    })

    # Update edges pointing to removed node
    for edge in graph.edges.values():
        if edge.source_id == remove_id:
            edge.source_id = keep_id
        if edge.target_id == remove_id:
            edge.target_id = keep_id

    # Remove the merged node
    del graph.nodes[remove_id]

    logger.info("Merged nodes %s and %s into %s", a, b, keep_id)
    return keep_id


def retract(
    graph: Graph,
    action_id: str,
    reason: str,
    proposer: str = "human",
) -> None:
    """Retract a prior action by superseding it with a new trace entry. Never deletes.

    Args:
        graph: The graph to modify
        action_id: The action_id to retract
        reason: Why this action is being retracted
        proposer: Who is retracting
    """
    # Find the node or edge containing this action_id
    for node in graph.nodes.values():
        if hasattr(node, 'metadata') and node.metadata.get('role_action_id') == action_id:
            # Mark as retracted by adding a superseding trace entry
            node.metadata.setdefault('trace_history', []).append({
                "action_id": str(uuid4()),
                "timestamp": __import__('time').time(),
                "action_type": "retract",
                "proposer": proposer,
                "reasoning": reason,
                "confidence": 1.0,
                "metadata": {"retracted_action_id": action_id},
            })

            # Clear the role assignment
            node.metadata['role'] = None
            node.metadata['role_opinion'] = None
            node.metadata['role_action_id'] = None

            logger.info("Retracted action %s for node %s", action_id, node.id)
            return

    # Check edges
    for edge in graph.edges.values():
        if hasattr(edge, 'metadata'):
            trace_history = edge.metadata.get('trace_history', [])
            for trace in trace_history:
                if isinstance(trace, dict) and trace.get('action_id') == action_id:
                    # Mark as retracted
                    trace_history.append({
                        "action_id": str(uuid4()),
                        "timestamp": __import__('time').time(),
                        "action_type": "retract",
                        "proposer": proposer,
                        "reasoning": reason,
                        "confidence": 1.0,
                        "metadata": {"retracted_action_id": action_id},
                    })

                    logger.info("Retracted action %s for edge %s", action_id, edge.id)
                    return

    logger.warning("Action %s not found for retraction", action_id)
