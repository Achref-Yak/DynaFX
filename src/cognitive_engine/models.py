from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional
from uuid import UUID, uuid4


class NodeType(Enum):
    CLAIM = auto()
    EVIDENCE = auto()
    CONDITION = auto()


class EdgeType(Enum):
    SUPPORTS = auto()
    CONTRADICTS = auto()
    QUALIFIES = auto()
    INFERS = auto()
    JUSTIFIES = auto()


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
class Graph:
    nodes: Dict[UUID, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    mode: ReasoningMode = ReasoningMode.ARGUMENT
    source_text: str = ""
    metadata: Dict = field(default_factory=dict)

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

        return {
            "nodes": {
                nid.hex: _convert(n)
                for nid, n in self.nodes.items()
            },
            "edges": [_convert(e) for e in self.edges],
            "mode": self.mode.name,
            "source_text": self.source_text,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

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
