from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
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
class Entity:
    """A thing that exists in the world model."""
    id: UUID = field(default_factory=uuid4)
    kind: str = ""
    name: str = ""
    superordinate: Optional[str] = None
    subordinate: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    spans: List[Span] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class WorldRelation:
    """A domain-agnostic relation between two entities."""
    id: UUID = field(default_factory=uuid4)
    source_id: UUID = field(default_factory=uuid4)
    target_id: UUID = field(default_factory=uuid4)
    kind: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class TypedEdge:
    """An edge within an interpretation (string-typed for any domain)."""
    id: UUID = field(default_factory=uuid4)
    source_id: UUID = field(default_factory=uuid4)
    target_id: UUID = field(default_factory=uuid4)
    type: str = ""
    opinion: Opinion = (0.0, 0.0, 1.0, 0.5)
    warrant: Optional[Warrant] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Interpretation:
    """A named view over entities + relations produced by one module."""
    name: str = ""
    roles: Dict[UUID, str] = field(default_factory=dict)
    edges: List[TypedEdge] = field(default_factory=list)


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
    entities: Dict[UUID, Entity] = field(default_factory=dict)
    world_relations: List[WorldRelation] = field(default_factory=list)
    interpretations: Dict[str, Interpretation] = field(default_factory=dict)
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
                return {_convert(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            if isinstance(obj, tuple):
                return list(obj)
            if hasattr(obj, "__dict__"):
                return {k: _convert(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
            return obj

        roles: dict[UUID, str] = {}
        for interp in self.interpretations.values():
            for eid, role in interp.roles.items():
                roles[eid] = role

        outgoing: dict[UUID, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            outgoing[edge.source_id].append(edge)

        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: (x[1].span.start if x[1].span else 0, x[1].text),
        )
        propositions: list[dict] = []
        for nid, node in sorted_nodes:
            nd = _convert(node)
            nd["argumentation_role"] = roles.get(nid, node.type.name)
            nd["outgoing_edges"] = [_convert(e) for e in outgoing.get(nid, [])]
            propositions.append(nd)

        sorted_entities = sorted(
            self.entities.items(),
            key=lambda x: (x[1].spans[0].start if x[1].spans else 0, x[1].name),
        )
        entities_list: list[dict] = [_convert(e) for _, e in sorted_entities]

        sorted_wr = sorted(
            self.world_relations,
            key=lambda r: (r.kind, r.source_id.hex),
        )

        result: dict = {
            "propositions": propositions,
            "entities": entities_list,
            "world_relations": [_convert(r) for r in sorted_wr],
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
        for eid, entity in self.entities.items():
            lines.append(f"ENTITY {eid.hex[:8]} kind={entity.kind} \"{entity.name[:60]}\"")
        for wr in self.world_relations:
            lines.append(f"REL {wr.source_id.hex[:8]} --{wr.kind}--> {wr.target_id.hex[:8]}")
        for name, interp in self.interpretations.items():
            lines.append(f"INTERPRETATION {name}: {len(interp.roles)} roles, {len(interp.edges)} edges")
            for te in interp.edges:
                lines.append(f"  TE {te.source_id.hex[:8]} --{te.type}--> {te.target_id.hex[:8]}")
        return "\n".join(lines)

    @staticmethod
    def from_dict(data: dict) -> Graph:
        nodes: dict[UUID, Node] = {}
        roles: dict[UUID, str] = {}

        old_nodes = data.get("nodes")
        new_propositions = data.get("propositions")

        if old_nodes is not None:
            for nid_hex, nd in old_nodes.items():
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
        elif new_propositions is not None:
            for pd in new_propositions:
                node_id = UUID(pd["id"])
                span_data = pd.get("span")
                span = Span(**span_data) if span_data else None
                nodes[node_id] = Node(
                    id=node_id,
                    type=NodeType[pd.get("type", "CLAIM")],
                    text=pd.get("text", ""),
                    span=span,
                    category=pd.get("category", 2),
                    opinion=tuple(pd.get("opinion", (0, 0, 1, 0.5))),
                )
                role = pd.get("argumentation_role")
                if role:
                    roles[node_id] = role

        edges: list[Edge] = []
        old_edges = data.get("edges")
        if old_edges is not None:
            for ed in old_edges:
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
        elif new_propositions is not None:
            for pd in new_propositions:
                src_id = UUID(pd["id"])
                for e in pd.get("outgoing_edges", []):
                    warrant = None
                    w_data = e.get("warrant")
                    if w_data and len(w_data) == 2:
                        warrant = (tuple(w_data[0]), tuple(w_data[1]))
                    edges.append(
                        Edge(
                            id=UUID(e["id"]),
                            source_id=src_id,
                            target_id=UUID(e["target_id"]),
                            type=EdgeType[e.get("type", "SUPPORTS")],
                            opinion=tuple(e.get("opinion", (0, 0, 1, 0.5))),
                            warrant=warrant,
                        )
                    )

        entities: dict[UUID, Entity] = {}
        old_entities = data.get("entities")
        if isinstance(old_entities, dict):
            for eid_hex, ed in old_entities.items():
                entity_id = UUID(eid_hex)
                entities[entity_id] = Entity(
                    id=entity_id,
                    kind=ed.get("kind", ""),
                    name=ed.get("name", ""),
                    superordinate=ed.get("superordinate"),
                    subordinate=ed.get("subordinate"),
                    attributes=ed.get("attributes", {}),
                    spans=[Span(**s) for s in ed.get("spans", [])],
                    metadata=ed.get("metadata", {}),
                )
        elif isinstance(old_entities, list):
            for ed in old_entities:
                entity_id = UUID(ed["id"])
                entities[entity_id] = Entity(
                    id=entity_id,
                    kind=ed.get("kind", ""),
                    name=ed.get("name", ""),
                    superordinate=ed.get("superordinate"),
                    subordinate=ed.get("subordinate"),
                    attributes=ed.get("attributes", {}),
                    spans=[Span(**s) for s in ed.get("spans", [])],
                    metadata=ed.get("metadata", {}),
                )

        world_relations: list[WorldRelation] = []
        for rd in data.get("world_relations", []):
            world_relations.append(WorldRelation(
                id=UUID(rd["id"]),
                source_id=UUID(rd["source_id"]),
                target_id=UUID(rd["target_id"]),
                kind=rd.get("kind", ""),
                metadata=rd.get("metadata", {}),
            ))

        interpretations: dict[str, Interpretation] = {}
        old_interps = data.get("interpretations")
        if old_interps:
            for name, idata in old_interps.items():
                interp_roles = {UUID(k): v for k, v in idata.get("roles", {}).items()}
                interp_edges = []
                for ted in idata.get("edges", []):
                    warrant = None
                    w_data = ted.get("warrant")
                    if w_data and len(w_data) == 2:
                        warrant = (tuple(w_data[0]), tuple(w_data[1]))
                    interp_edges.append(TypedEdge(
                        id=UUID(ted["id"]),
                        source_id=UUID(ted["source_id"]),
                        target_id=UUID(ted["target_id"]),
                        type=ted.get("type", ""),
                        opinion=tuple(ted.get("opinion", (0, 0, 1, 0.5))),
                        warrant=warrant,
                        metadata=ted.get("metadata", {}),
                    ))
                interpretations[name] = Interpretation(name=name, roles=interp_roles, edges=interp_edges)
        elif roles and data.get("propositions") is not None:
            arg_edges: list[TypedEdge] = []
            for e in edges:
                arg_edges.append(TypedEdge(
                    id=e.id,
                    source_id=e.source_id,
                    target_id=e.target_id,
                    type=e.type.name,
                ))
            interpretations["argumentation"] = Interpretation(
                name="argumentation",
                roles=roles,
                edges=arg_edges,
            )

        cta_data = data.get("cta")
        cta = None
        if cta_data:
            cta = ConversationTree(
                root_id=UUID(cta_data["root_id"]),
                node_ids={UUID(n) for n in cta_data.get("node_ids", [])},
                parent_map={UUID(k): UUID(v) for k, v in cta_data.get("parent_map", {}).items()},
            )

        return Graph(
            nodes=nodes,
            edges=edges,
            entities=entities,
            world_relations=world_relations,
            interpretations=interpretations,
            mode=ReasoningMode[data.get("mode", "ARGUMENT")],
            source_text=data.get("source_text", ""),
            metadata=data.get("metadata", {}),
            cta=cta,
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
