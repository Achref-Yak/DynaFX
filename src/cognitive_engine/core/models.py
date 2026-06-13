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


class FusionSituation(Enum):
    INDEPENDENT_SOURCES = auto()
    CONFLICTING_VIEWS = auto()
    DEPENDENT_SOURCES = auto()
    SAME_SOURCE = auto()


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
        visited: set[UUID] = set()
        current = node_id
        while current in self.parent_map or current == self.root_id:
            if current in visited:
                break
            visited.add(current)
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

    @staticmethod
    def _convert_value(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, UUID):
            return obj.hex
        if isinstance(obj, dict):
            return {Graph._convert_value(k): Graph._convert_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Graph._convert_value(i) for i in obj]
        if isinstance(obj, tuple):
            return list(obj)
        if hasattr(obj, "__dict__"):
            return {k: Graph._convert_value(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        return obj

    @staticmethod
    def _collect_roles(interpretations: Dict[str, Interpretation]) -> Dict[UUID, str]:
        roles: dict[UUID, str] = {}
        for interp in interpretations.values():
            for eid, role in interp.roles.items():
                roles[eid] = role
        return roles

    @staticmethod
    def _build_outgoing_map(edges: List[Edge]) -> Dict[UUID, List[Edge]]:
        outgoing: dict[UUID, list[Edge]] = defaultdict(list)
        for edge in edges:
            outgoing[edge.source_id].append(edge)
        return outgoing

    @staticmethod
    def _serialize_nodes(
        nodes: Dict[UUID, Node],
        roles: Dict[UUID, str],
        outgoing: Dict[UUID, List[Edge]],
    ) -> List[dict]:
        sorted_nodes = sorted(
            nodes.items(),
            key=lambda x: (x[1].span.start if x[1].span else 0, x[1].text),
        )
        propositions: list[dict] = []
        for nid, node in sorted_nodes:
            nd = Graph._convert_value(node)
            nd["argumentation_role"] = roles.get(nid, node.type.name)
            nd["outgoing_edges"] = [Graph._convert_value(e) for e in outgoing.get(nid, [])]
            propositions.append(nd)
        return propositions

    @staticmethod
    def _serialize_entities(entities: Dict[UUID, Entity]) -> List[dict]:
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: (x[1].spans[0].start if x[1].spans else 0, x[1].name),
        )
        return [Graph._convert_value(e) for _, e in sorted_entities]

    @staticmethod
    def _serialize_world_relations(world_relations: List[WorldRelation]) -> List[dict]:
        sorted_wr = sorted(
            world_relations,
            key=lambda r: (r.kind, r.source_id.hex),
        )
        return [Graph._convert_value(r) for r in sorted_wr]

    def to_dict(self) -> dict:
        roles = Graph._collect_roles(self.interpretations)
        outgoing = Graph._build_outgoing_map(self.edges)
        propositions = Graph._serialize_nodes(self.nodes, roles, outgoing)
        entities_list = Graph._serialize_entities(self.entities)
        wr_list = Graph._serialize_world_relations(self.world_relations)

        result: dict = {
            "propositions": propositions,
            "entities": entities_list,
            "world_relations": wr_list,
            "mode": self.mode.name,
            "source_text": self.source_text,
            "metadata": self.metadata,
        }
        if self.cta is not None:
            result["cta"] = self.cta.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @staticmethod
    def _compact_node(nid: UUID, node: Node) -> str:
        text = node.text[:60].replace("\n", " ")
        return f"NODE {nid.hex[:8]} {node.type.name} [{node.category}] \"{text}\""

    @staticmethod
    def _compact_edge(edge: Edge) -> str:
        return f"EDGE {edge.source_id.hex[:8]} --{edge.type.name}--> {edge.target_id.hex[:8]}"

    @staticmethod
    def _compact_entity(eid: UUID, entity: Entity) -> str:
        return f"ENTITY {eid.hex[:8]} kind={entity.kind} \"{entity.name[:60]}\""

    @staticmethod
    def _compact_world_relation(wr: WorldRelation) -> str:
        return f"REL {wr.source_id.hex[:8]} --{wr.kind}--> {wr.target_id.hex[:8]}"

    @staticmethod
    def _compact_interpretation(name: str, interp: Interpretation) -> list[str]:
        lines = [f"INTERPRETATION {name}: {len(interp.roles)} roles, {len(interp.edges)} edges"]
        for te in interp.edges:
            lines.append(f"  TE {te.source_id.hex[:8]} --{te.type}--> {te.target_id.hex[:8]}")
        return lines

    def to_compact_str(self) -> str:
        lines: list[str] = []
        for nid, node in self.nodes.items():
            lines.append(Graph._compact_node(nid, node))
        for edge in self.edges:
            lines.append(Graph._compact_edge(edge))
        for eid, entity in self.entities.items():
            lines.append(Graph._compact_entity(eid, entity))
        for wr in self.world_relations:
            lines.append(Graph._compact_world_relation(wr))
        for name, interp in self.interpretations.items():
            lines.extend(Graph._compact_interpretation(name, interp))
        return "\n".join(lines)

    @staticmethod
    def _parse_warrant(ed: dict) -> Optional[Warrant]:
        w_data = ed.get("warrant")
        if w_data and len(w_data) == 2:
            return (tuple(w_data[0]), tuple(w_data[1]))
        return None

    @staticmethod
    def _parse_node(node_id: UUID, nd: dict, with_role: bool = False) -> tuple[Node, Optional[str]]:
        span_data = nd.get("span")
        span = Span(**span_data) if span_data else None
        node = Node(
            id=node_id,
            type=NodeType[nd.get("type", "CLAIM")],
            text=nd.get("text", ""),
            span=span,
            category=nd.get("category", 2),
            opinion=tuple(nd.get("opinion", (0, 0, 1, 0.5))),
        )
        role = nd.get("argumentation_role") if with_role else None
        return node, role

    @staticmethod
    def _parse_entity(entity_id: UUID, ed: dict) -> Entity:
        return Entity(
            id=entity_id,
            kind=ed.get("kind", ""),
            name=ed.get("name", ""),
            superordinate=ed.get("superordinate"),
            subordinate=ed.get("subordinate"),
            attributes=ed.get("attributes", {}),
            spans=[Span(**s) for s in ed.get("spans", [])],
            metadata=ed.get("metadata", {}),
        )

    @staticmethod
    def _parse_nodes(data: dict) -> tuple[dict[UUID, Node], dict[UUID, str]]:
        nodes: dict[UUID, Node] = {}
        roles: dict[UUID, str] = {}

        old_nodes = data.get("nodes")
        new_propositions = data.get("propositions")

        if old_nodes is not None:
            for nid_hex, nd in old_nodes.items():
                node_id = UUID(nid_hex)
                node, _ = Graph._parse_node(node_id, nd)
                nodes[node_id] = node
        elif new_propositions is not None:
            for pd in new_propositions:
                node_id = UUID(pd["id"])
                node, role = Graph._parse_node(node_id, pd, with_role=True)
                nodes[node_id] = node
                if role:
                    roles[node_id] = role

        return nodes, roles

    @staticmethod
    def _parse_edges(data: dict, nodes: dict[UUID, Node]) -> list[Edge]:
        edges: list[Edge] = []
        old_edges = data.get("edges")
        new_propositions = data.get("propositions")

        if old_edges is not None:
            for ed in old_edges:
                edges.append(
                    Edge(
                        id=UUID(ed["id"]),
                        source_id=UUID(ed["source_id"]),
                        target_id=UUID(ed["target_id"]),
                        type=EdgeType[ed.get("type", "SUPPORTS")],
                        opinion=tuple(ed.get("opinion", (0, 0, 1, 0.5))),
                        warrant=Graph._parse_warrant(ed),
                    )
                )
        elif new_propositions is not None:
            for pd in new_propositions:
                src_id = UUID(pd["id"])
                for e in pd.get("outgoing_edges", []):
                    edges.append(
                        Edge(
                            id=UUID(e["id"]),
                            source_id=src_id,
                            target_id=UUID(e["target_id"]),
                            type=EdgeType[e.get("type", "SUPPORTS")],
                            opinion=tuple(e.get("opinion", (0, 0, 1, 0.5))),
                            warrant=Graph._parse_warrant(e),
                        )
                    )

        return edges

    @staticmethod
    def _parse_entities(data: dict) -> dict[UUID, Entity]:
        entities: dict[UUID, Entity] = {}
        old_entities = data.get("entities")

        if isinstance(old_entities, dict):
            for eid_hex, ed in old_entities.items():
                entity_id = UUID(eid_hex)
                entities[entity_id] = Graph._parse_entity(entity_id, ed)
        elif isinstance(old_entities, list):
            for ed in old_entities:
                entity_id = UUID(ed["id"])
                entities[entity_id] = Graph._parse_entity(entity_id, ed)

        return entities

    @staticmethod
    def _parse_world_relations(data: dict) -> list[WorldRelation]:
        result: list[WorldRelation] = []
        for rd in data.get("world_relations", []):
            result.append(WorldRelation(
                id=UUID(rd["id"]),
                source_id=UUID(rd["source_id"]),
                target_id=UUID(rd["target_id"]),
                kind=rd.get("kind", ""),
                metadata=rd.get("metadata", {}),
            ))
        return result

    @staticmethod
    def _parse_interpretations(
        data: dict, edges: list[Edge], roles: dict[UUID, str],
    ) -> dict[str, Interpretation]:
        old_interps = data.get("interpretations")
        if old_interps:
            return Graph._parse_interpretations_v1(old_interps)
        if roles and data.get("propositions") is not None:
            return Graph._parse_interpretations_v2(edges, roles)
        return {}

    @staticmethod
    def _parse_interpretations_v1(
        old_interps: dict,
    ) -> dict[str, Interpretation]:
        result: dict[str, Interpretation] = {}
        for name, idata in old_interps.items():
            interp_roles = {UUID(k): v for k, v in idata.get("roles", {}).items()}
            interp_edges = []
            for ted in idata.get("edges", []):
                interp_edges.append(TypedEdge(
                    id=UUID(ted["id"]),
                    source_id=UUID(ted["source_id"]),
                    target_id=UUID(ted["target_id"]),
                    type=ted.get("type", ""),
                    opinion=tuple(ted.get("opinion", (0, 0, 1, 0.5))),
                    warrant=Graph._parse_warrant(ted),
                    metadata=ted.get("metadata", {}),
                ))
            result[name] = Interpretation(name=name, roles=interp_roles, edges=interp_edges)
        return result

    @staticmethod
    def _parse_interpretations_v2(
        edges: list[Edge], roles: dict[UUID, str],
    ) -> dict[str, Interpretation]:
        arg_edges: list[TypedEdge] = []
        for e in edges:
            arg_edges.append(TypedEdge(
                id=e.id,
                source_id=e.source_id,
                target_id=e.target_id,
                type=e.type.name,
            ))
        return {
            "argumentation": Interpretation(
                name="argumentation",
                roles=roles,
                edges=arg_edges,
            ),
        }

    @staticmethod
    def _parse_cta(data: dict) -> Optional[ConversationTree]:
        cta_data = data.get("cta")
        if not cta_data:
            return None
        return ConversationTree(
            root_id=UUID(cta_data["root_id"]),
            node_ids={UUID(n) for n in cta_data.get("node_ids", [])},
            parent_map={UUID(k): UUID(v) for k, v in cta_data.get("parent_map", {}).items()},
        )

    @staticmethod
    def from_dict(data: dict) -> Graph:
        nodes, roles = Graph._parse_nodes(data)
        edges = Graph._parse_edges(data, nodes)
        entities = Graph._parse_entities(data)
        world_relations = Graph._parse_world_relations(data)
        interpretations = Graph._parse_interpretations(data, edges, roles)
        cta = Graph._parse_cta(data)

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
class EvidenceCounts:
    positive: int = 0
    negative: int = 0
    uncertainty_pseudocount: float = 2.0


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
