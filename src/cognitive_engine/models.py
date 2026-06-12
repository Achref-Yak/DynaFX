from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional
from uuid import UUID, uuid4


class NodeType(Enum):
    AXIOM = auto()
    EVIDENCE = auto()
    CONDITION = auto()
    CLAIM = auto()
    COUNTERCLAIM = auto()
    FALLACY = auto()
    JUSTIFICATION = auto()


class EdgeType(Enum):
    INFERS = auto()
    SUPPORTS = auto()
    ATTACKS = auto()
    REBUTS = auto()
    QUALIFIES = auto()
    JUSTIFIES = auto()
    CONTRADICTS = auto()


class ReasoningMode(Enum):
    CAUSAL = auto()
    CONDITIONAL = auto()
    ARGUMENT = auto()
    ANALOGY = auto()


class Severity(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


Opinion = tuple[float, float, float, float]

Warrant = tuple[Opinion, Opinion]


@dataclass
class Span:
    start: int
    end: int
    text: str


@dataclass
class Node:
    id: UUID = field(default_factory=uuid4)
    type: NodeType = NodeType.CLAIM
    text: str = ""
    span: Optional[Span] = None
    abstraction_level: int = 1
    salience: float = 0.5
    opinion: Opinion = (0.0, 0.0, 1.0, 0.5)
    category: int = 2
    metadata: Dict = field(default_factory=dict)


@dataclass
class Edge:
    id: UUID = field(default_factory=uuid4)
    source_id: UUID = field(default_factory=uuid4)
    target_id: UUID = field(default_factory=uuid4)
    type: EdgeType = EdgeType.SUPPORTS
    opinion: Opinion = (0.0, 0.0, 1.0, 0.5)
    warrant: Optional[Warrant] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConversationTree:
    """Conversation Tree Architecture — T = (V, E, r, W).

    Isolates context windows to prevent Logical Context Poisoning.
    W is the window function: get_context(node_id) returns the ancestor
    chain from root to the given node.
    """
    root_id: UUID
    node_ids: set[UUID] = field(default_factory=set)
    parent_map: dict[UUID, UUID] = field(default_factory=dict)

    @classmethod
    def from_graph(cls, graph: Graph, root_id: UUID | None = None) -> ConversationTree:
        if root_id is None:
            candidates = [nid for nid in graph.nodes
                          if not any(e.target_id == nid for e in graph.edges)]
            root_id = candidates[0] if candidates else next(iter(graph.nodes))
        parents: dict[UUID, UUID] = {}
        for e in graph.edges:
            if e.type in (EdgeType.INFERS, EdgeType.SUPPORTS, EdgeType.JUSTIFIES):
                parents[e.target_id] = e.source_id
        all_ids = {root_id}
        stack = [root_id]
        while stack:
            nid = stack.pop()
            for child, pid in parents.items():
                if pid == nid and child not in all_ids:
                    all_ids.add(child)
                    stack.append(child)
        return cls(root_id=root_id, node_ids=all_ids, parent_map=parents)

    def get_context(self, node_id: UUID) -> list[UUID]:
        """Window function W — ancestor chain from root to node (inclusive)."""
        chain: list[UUID] = []
        current = node_id
        while current in self.parent_map or current == self.root_id:
            chain.append(current)
            if current == self.root_id:
                break
            current = self.parent_map.get(current)
            if current is None:
                break
        if chain and chain[-1] != self.root_id:
            chain.append(self.root_id)
        return list(reversed(chain))

    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id.hex,
            "node_ids": [n.hex for n in self.node_ids],
            "parent_map": {k.hex: v.hex for k, v in self.parent_map.items()},
        }


@dataclass
class Graph:
    nodes: Dict[UUID, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    mode: ReasoningMode = ReasoningMode.ARGUMENT
    source_text: str = ""
    metadata: Dict = field(default_factory=dict)
    cta: Optional[ConversationTree] = None

    def to_dict(self) -> dict:
        def _convert(obj):
            if isinstance(obj, Enum):
                return obj.name
            if isinstance(obj, UUID):
                return obj.hex
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            if isinstance(obj, tuple):
                return list(obj)
            if hasattr(obj, "__dict__"):
                return {k: _convert(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
            return obj

        result: dict = {
            "nodes": {
                nid.hex: _convert(n)
                for nid, n in self.nodes.items()
            },
            "edges": [_convert(e) for e in self.edges],
            "mode": self.mode.name,
            "source_text": self.source_text,
            "metadata": self.metadata,
        }
        if self.cta is not None:
            result["cta"] = self.cta.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_compact_str(self) -> str:
        lines: list[str] = []
        for nid, node in self.nodes.items():
            text = node.text[:60].replace("\n", " ")
            lines.append(f"NODE {nid.hex[:8]} {node.type.name} [{node.category}] \"{text}\"")
        for edge in self.edges:
            lines.append(
                f"EDGE {edge.source_id.hex[:8]} --{edge.type.name}--> "
                f"{edge.target_id.hex[:8]}"
            )
        return "\n".join(lines)

    @staticmethod
    def from_dict(data: dict) -> Graph:
        nodes: dict[UUID, Node] = {}
        for nid_hex, nd in data.get("nodes", {}).items():
            node_id = UUID(nid_hex)
            span_data = nd.get("span")
            span = Span(**span_data) if span_data else None
            nodes[node_id] = Node(
                id=node_id,
                type=NodeType[nd.get("type", "CLAIM")],
                text=nd.get("text", ""),
                span=span,
                category=nd.get("category", 2),
                opinion=tuple(nd.get("opinion", (0, 0, 1, 0.5))),
            )

        edges: list[Edge] = []
        for ed in data.get("edges", []):
            warrant = None
            w_data = ed.get("warrant")
            if w_data and len(w_data) == 2:
                warrant = (tuple(w_data[0]), tuple(w_data[1]))
            edges.append(
                Edge(
                    id=UUID(ed["id"]),
                    source_id=UUID(ed["source_id"]),
                    target_id=UUID(ed["target_id"]),
                    type=EdgeType[ed.get("type", "SUPPORTS")],
                    opinion=tuple(ed.get("opinion", (0, 0, 1, 0.5))),
                    warrant=warrant,
                )
            )

        return Graph(
            nodes=nodes,
            edges=edges,
            mode=ReasoningMode[data.get("mode", "ARGUMENT")],
            source_text=data.get("source_text", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Violation:
    type: str
    severity: Severity
    description: str
    node_id: Optional[UUID] = None
    edge_id: Optional[UUID] = None


@dataclass
class ReviewResult:
    status: str
    violations: List[Violation] = field(default_factory=list)
    feedback: str = ""
