from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from uuid import UUID, uuid4

NodeId = UUID
EdgeId = UUID


class BfoCategory(Enum):
    MATERIAL_ENTITY = auto()
    IMMATERIAL_ENTITY = auto()
    QUALITY = auto()
    REALIZABLE_ENTITY = auto()
    PROCESS = auto()
    TEMPORAL_REGION = auto()
    INFORMATION_CONTENT_ENTITY = auto()


class NodeType(Enum):
    AXIOM = auto()
    EVIDENCE = auto()
    CONDITION = auto()
    CLAIM = auto()
    COUNTERCLAIM = auto()
    FALLACY = auto()
    JUSTIFICATION = auto()
    ENTITY = auto()
    EVENT = auto()
    CONCEPT = auto()
    RULE = auto()
    HYPOTHESIS = auto()
    OBSERVATION = auto()
    DECISION = auto()
    ACTION = auto()
    AGENT = auto()
    PROCESS = auto()
    STATE = auto()
    PROPERTY = auto()
    RESOURCE = auto()
    CONSTRAINT = auto()
    GOAL = auto()
    BELIEF = auto()
    KNOWLEDGE = auto()
    INFORMATION = auto()
    DOCUMENT = auto()
    STOCK = auto()
    FLOW = auto()
    VARIABLE = auto()


class EdgeType(Enum):
    INFERS = auto()
    SUPPORTS = auto()
    ATTACKS = auto()
    REBUTS = auto()
    QUALIFIES = auto()
    JUSTIFIES = auto()
    CONTRADICTS = auto()
    DIRECT = auto()
    CIRCUMSTANTIAL = auto()
    HEARSAY = auto()
    CAUSES = auto()
    SUPPORT = auto()
    ENABLES = auto()
    DEPENDS = auto()
    TEMPORAL = auto()
    SIMILAR = auto()
    EVIDENCE = auto()
    PART_OF = auto()
    CITES = auto()
    FLOWS_TO = auto()
    HAS_ATTRIBUTE = auto()
    LOCATED_AT = auto()
    EMPLOYED_BY = auto()
    ASSOCIATED_WITH = auto()
    CONTACT_OF = auto()
    HAS_GOAL = auto()
    INTENDS = auto()
    KNOWS = auto()
    COMMUNICATED = auto()
    PREFERS = auto()
    USES = auto()
    PRODUCES = auto()
    CONSUMES = auto()
    TRANSFORMS = auto()


_ICE = frozenset({BfoCategory.INFORMATION_CONTENT_ENTITY})
_PROC_AND_ICE = frozenset({BfoCategory.PROCESS, BfoCategory.INFORMATION_CONTENT_ENTITY})
_MAT_AND_IMMAT = frozenset({BfoCategory.MATERIAL_ENTITY, BfoCategory.IMMATERIAL_ENTITY})
_ALL_BFO = frozenset(BfoCategory)

EDGE_BFO_CONSTRAINTS: dict[EdgeType, tuple[frozenset[BfoCategory], frozenset[BfoCategory]]] = {
    EdgeType.INFERS: (_ICE, _ICE),
    EdgeType.SUPPORTS: (_ICE, _ICE),
    EdgeType.REBUTS: (_ICE, _ICE),
    EdgeType.ATTACKS: (_ICE, _ICE),
    EdgeType.CONTRADICTS: (_ICE, _ICE),
    EdgeType.JUSTIFIES: (_ICE, _ICE),
    EdgeType.EVIDENCE: (_ICE, _ICE),
    EdgeType.CITES: (_ICE, _ICE),
    EdgeType.CAUSES: (_PROC_AND_ICE, _PROC_AND_ICE),
    EdgeType.TEMPORAL: (_PROC_AND_ICE, _PROC_AND_ICE),
    EdgeType.FLOWS_TO: (_PROC_AND_ICE, _PROC_AND_ICE),
    EdgeType.PART_OF: (_MAT_AND_IMMAT, _MAT_AND_IMMAT),
    EdgeType.QUALIFIES: (frozenset({BfoCategory.REALIZABLE_ENTITY, BfoCategory.INFORMATION_CONTENT_ENTITY}), _ICE),
    EdgeType.ENABLES: (_ALL_BFO, _ALL_BFO),
    EdgeType.DEPENDS: (_ALL_BFO, _ALL_BFO),
    EdgeType.SIMILAR: (_ALL_BFO, _ALL_BFO),
    EdgeType.DIRECT: (_ICE, _ICE),
    EdgeType.CIRCUMSTANTIAL: (_ICE, _ICE),
    EdgeType.HEARSAY: (_ICE, _ICE),
    EdgeType.SUPPORT: (_ALL_BFO, _ALL_BFO),
    EdgeType.HAS_ATTRIBUTE: (_ALL_BFO, _ICE),
    EdgeType.LOCATED_AT: (_MAT_AND_IMMAT, _MAT_AND_IMMAT),
    EdgeType.EMPLOYED_BY: (_ALL_BFO, _ALL_BFO),
    EdgeType.ASSOCIATED_WITH: (_ALL_BFO, _ALL_BFO),
    EdgeType.CONTACT_OF: (_ALL_BFO, _ALL_BFO),
    EdgeType.HAS_GOAL: (_ALL_BFO, _ICE),
    EdgeType.INTENDS: (_ALL_BFO, _PROC_AND_ICE),
    EdgeType.KNOWS: (_ALL_BFO, _ICE),
    EdgeType.COMMUNICATED: (_ALL_BFO, _ICE),
    EdgeType.PREFERS: (_ALL_BFO, _ICE),
    EdgeType.USES: (_PROC_AND_ICE, _MAT_AND_IMMAT | _ICE),
    EdgeType.PRODUCES: (_PROC_AND_ICE, _MAT_AND_IMMAT | _ICE),
    EdgeType.CONSUMES: (_PROC_AND_ICE, _MAT_AND_IMMAT | _ICE),
    EdgeType.TRANSFORMS: (_PROC_AND_ICE, _PROC_AND_ICE),
}


class ReasoningMode(Enum):
    CAUSAL = auto()
    CONDITIONAL = auto()
    ARGUMENT = auto()
    ANALOGY = auto()


class Severity(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass
class Parameter:
    """Parameter value with optional metadata."""
    value: float | None = None

    def to_dict(self) -> dict:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, d: dict) -> Parameter:
        return cls(value=d.get("value"))


@dataclass
class Payload:
    """Raw content of a node."""
    text: str = ""
    structured: dict = field(default_factory=dict)


@dataclass
class TimeInfo:
    """Temporal metadata for a node."""
    created: float = 0.0
    modified: float = 0.0
    temporal_anchor: str | None = None


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
    payload: Payload = field(default_factory=lambda: Payload(text=""))
    span: Span | None = None
    abstraction_level: int = 1
    salience: float = 0.5
    category: int = 2
    embedding: list[float] | None = None
    timestamps: TimeInfo = field(default_factory=TimeInfo)
    attrs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    bfo_category: BfoCategory | None = None
    container_id: UUID | None = None
    orthogonal_partition: str | None = None


@dataclass
class Edge:
    id: UUID = field(default_factory=uuid4)
    source_id: UUID = field(default_factory=uuid4)
    target_id: UUID = field(default_factory=uuid4)
    type: EdgeType = EdgeType.SUPPORTS
    weight: float = 0.5
    confidence: float = 0.5
    polarity: int = 1
    attrs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class Entity:
    """A thing that exists in the world model."""
    id: UUID = field(default_factory=uuid4)
    kind: str = ""
    name: str = ""
    superordinate: str | None = None
    subordinate: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    bfo_category: BfoCategory | None = None


@dataclass
class WorldRelation:
    """A domain-agnostic relation between two entities."""
    id: UUID = field(default_factory=uuid4)
    source_id: UUID = field(default_factory=uuid4)
    target_id: UUID = field(default_factory=uuid4)
    kind: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class TypedEdge:
    """An edge within an interpretation (string-typed for any domain)."""
    id: UUID = field(default_factory=uuid4)
    source_id: UUID = field(default_factory=uuid4)
    target_id: UUID = field(default_factory=uuid4)
    type: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Interpretation:
    """A named view over entities + relations produced by one module."""
    name: str = ""
    roles: dict[UUID, str] = field(default_factory=dict)
    edges: list[TypedEdge] = field(default_factory=list)


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
                          if not any(e.target_id == nid for e in graph.edges.values())]
            root_id = candidates[0] if candidates else next(iter(graph.nodes))
        parents: dict[UUID, UUID] = {}
        for e in graph.edges.values():
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
class EmergentProperty:
    """A system-level behavior that cannot be reduced to any single component.

    Detected by structural signature matching as a post-pass on the graph.
    condition references bound Parameter ids with threshold expressions.
    """
    name: str = ""
    condition: str = ""
    involved_ids: list[UUID] = field(default_factory=list)
    detected_by: str = ""
    trace_ref: str = ""


@dataclass
class FeedbackLoop:
    """A feedback loop with polarity classification.

    Discovered by MDM-based cycle detection (within-DSM DFS).
    """
    nodes: list[UUID] = field(default_factory=list)
    loop_type: str = ""
    gain_sign: str = ""
    edge_count: int = 0
    negative_count: int = 0

    def to_dict(self) -> dict[str, str | int | list]:
        return {
            "loop_type": self.loop_type,
            "gain_sign": self.gain_sign,
            "edge_count": self.edge_count,
            "negative_count": self.negative_count,
        }


def _strip_defaults(obj: Any, _is_top_level: bool = False) -> Any:
    """Remove empty/default fields from serialized output.

    Strips:
        - None values
        - Empty dicts {} (only from nested structures, not top-level)
        - Empty lists [] (only from nested structures, not top-level)
        - Empty strings ""
        - Default timestamps {"created": 0.0, "modified": 0.0, "temporal_anchor": None}
        - Default numeric fields (abstraction_level=1, salience=0.5, category=2)
        - Embedding vectors (belong in vector DB, not JSON)
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "embedding":
                continue
            if k == "timestamps" and isinstance(v, dict) and (
                v.get("created", 0) == 0.0 and
                v.get("modified", 0) == 0.0 and
                v.get("temporal_anchor") is None
            ):
                continue
            if k == "abstraction_level" and v == 1:
                continue
            if k == "salience" and v == 0.5:
                continue
            if k == "category" and v == 2:
                continue
            stripped = _strip_defaults(v, _is_top_level=False)
            if _is_top_level or (stripped is not None and stripped != {} and stripped != [] and stripped != ""):
                result[k] = stripped
        return result
    if isinstance(obj, list):
        result = [_strip_defaults(i, _is_top_level=False) for i in obj]
        return [i for i in result if i is not None and i != {} and i != [] and i != ""]
    return obj


@dataclass
class Graph:
    nodes: dict[UUID, Node] = field(default_factory=dict)
    edges: dict[UUID, Edge] = field(default_factory=dict)
    entities: dict[UUID, Entity] = field(default_factory=dict)
    world_relations: list[WorldRelation] = field(default_factory=list)
    interpretations: dict[str, Interpretation] = field(default_factory=dict)
    mode: ReasoningMode = ReasoningMode.ARGUMENT
    source_text: str = ""
    metadata: dict = field(default_factory=dict)
    cta: ConversationTree | None = None
    emergent_properties: list[EmergentProperty] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.edges, list):
            self.edges = {e.id: e for e in self.edges}

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
    def _collect_roles(interpretations: dict[str, Interpretation]) -> dict[UUID, str]:
        roles: dict[UUID, str] = {}
        for interp in interpretations.values():
            for eid, role in interp.roles.items():
                roles[eid] = role
        return roles

    @staticmethod
    def _build_outgoing_map(edges: dict[UUID, Edge]) -> dict[UUID, list[Edge]]:
        outgoing: dict[UUID, list[Edge]] = defaultdict(list)
        for edge in edges.values():
            outgoing[edge.source_id].append(edge)
        return outgoing

    @staticmethod
    def _serialize_nodes(
        nodes: dict[UUID, Node],
        roles: dict[UUID, str],
        outgoing: dict[UUID, list[Edge]],
    ) -> list[dict]:
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
    def _serialize_entities(entities: dict[UUID, Entity]) -> list[dict]:
        sorted_entities = sorted(
            entities.items(),
            key=lambda x: (x[1].spans[0].start if x[1].spans else 0, x[1].name),
        )
        return [Graph._convert_value(e) for _, e in sorted_entities]

    @staticmethod
    def _serialize_world_relations(world_relations: list[WorldRelation]) -> list[dict]:
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
        return _strip_defaults(result, _is_top_level=True)

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
        for edge in self.edges.values():
            lines.append(Graph._compact_edge(edge))
        for eid, entity in self.entities.items():
            lines.append(Graph._compact_entity(eid, entity))
        for wr in self.world_relations:
            lines.append(Graph._compact_world_relation(wr))
        for name, interp in self.interpretations.items():
            lines.extend(Graph._compact_interpretation(name, interp))
        return "\n".join(lines)

    @staticmethod
    def _parse_node(node_id: UUID, nd: dict, with_role: bool = False) -> tuple[Node, str | None]:
        span_data = nd.get("span")
        span = Span(**span_data) if span_data else None
        node = Node(
            id=node_id,
            type=NodeType[nd.get("type", "CLAIM")],
            text=nd.get("text", ""),
            payload=Payload(text=nd.get("text", "")),
            span=span,
            category=nd.get("category", 2),
            embedding=nd.get("embedding"),
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
    def _parse_edges(data: dict, nodes: dict[UUID, Node]) -> dict[UUID, Edge]:
        edges: dict[UUID, Edge] = {}
        old_edges = data.get("edges")
        new_propositions = data.get("propositions")

        if old_edges is not None:
            for ed in old_edges:
                e = Edge(
                    id=UUID(ed["id"]),
                    source_id=UUID(ed["source_id"]),
                    target_id=UUID(ed["target_id"]),
                    type=EdgeType[ed.get("type", "SUPPORTS")],
                )
                edges[e.id] = e
        elif new_propositions is not None:
            for pd in new_propositions:
                src_id = UUID(pd["id"])
                for ed in pd.get("outgoing_edges", []):
                    e = Edge(
                        id=UUID(ed["id"]),
                        source_id=src_id,
                        target_id=UUID(ed["target_id"]),
                        type=EdgeType[ed.get("type", "SUPPORTS")],
                    )
                    edges[e.id] = e

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
        data: dict, edges: dict[UUID, Edge], roles: dict[UUID, str],
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
                    metadata=ted.get("metadata", {}),
                ))
            result[name] = Interpretation(name=name, roles=interp_roles, edges=interp_edges)
        return result

    @staticmethod
    def _parse_interpretations_v2(
        edges: dict[UUID, Edge], roles: dict[UUID, str],
    ) -> dict[str, Interpretation]:
        arg_edges: list[TypedEdge] = []
        for e in edges.values():
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
    def _parse_cta(data: dict) -> ConversationTree | None:
        cta_data = data.get("cta")
        if not cta_data:
            return None
        return ConversationTree(
            root_id=UUID(cta_data["root_id"]),
            node_ids={UUID(n) for n in cta_data.get("node_ids", [])},
            parent_map={UUID(k): UUID(v) for k, v in cta_data.get("parent_map", {}).items()},
        )

    @staticmethod
    def _parse_emergent(data: dict) -> list[EmergentProperty]:
        ep_data = data.get("emergent_properties", [])
        result: list[EmergentProperty] = []
        for epd in ep_data:
            result.append(EmergentProperty(
                name=epd.get("name", ""),
                condition=epd.get("condition", ""),
                involved_ids=[UUID(nid) for nid in epd.get("involved_ids", [])],
                detected_by=epd.get("detected_by", ""),
                trace_ref=epd.get("trace_ref", ""),
            ))
        return result

    @staticmethod
    def from_dict(data: dict) -> Graph:
        nodes, roles = Graph._parse_nodes(data)
        edges = Graph._parse_edges(data, nodes)
        entities = Graph._parse_entities(data)
        world_relations = Graph._parse_world_relations(data)
        interpretations = Graph._parse_interpretations(data, edges, roles)
        cta = Graph._parse_cta(data)
        emergent_properties = Graph._parse_emergent(data)

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
            emergent_properties=emergent_properties,
        )


@dataclass
class Context:
    id: UUID
    source_id: str
    text: str
    span: Span | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id.hex,
            "source_id": self.source_id,
            "text": self.text,
        }
        if self.span:
            d["span"] = {"start": self.span.start, "end": self.span.end}
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @staticmethod
    def from_dict(data: dict) -> Context:
        span_data = data.get("span")
        return Context(
            id=UUID(data["id"]),
            source_id=data.get("source_id", ""),
            text=data.get("text", ""),
            span=Span(**span_data) if span_data else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Annotation:
    id: UUID
    target_id: UUID
    annotator: str
    label: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id.hex,
            "target_id": self.target_id.hex,
            "annotator": self.annotator,
            "label": self.label,
            "confidence": self.confidence,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @staticmethod
    def from_dict(data: dict) -> Annotation:
        return Annotation(
            id=UUID(data["id"]),
            target_id=UUID(data["target_id"]),
            annotator=data.get("annotator", ""),
            label=data.get("label", ""),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Trace:
    id: UUID
    trace_type: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id.hex,
            "trace_type": self.trace_type,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict) -> Trace:
        return Trace(
            id=UUID(data["id"]),
            trace_type=data.get("trace_type", ""),
            timestamp=data.get("timestamp", 0.0),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
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
    node_id: UUID | None = None
    edge_id: UUID | None = None


@dataclass
class ReviewResult:
    status: str
    violations: list[Violation] = field(default_factory=list)
    feedback: str = ""
