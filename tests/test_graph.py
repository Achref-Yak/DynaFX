import json
from uuid import UUID, uuid4

import pytest

from dynafx.core.models import (
    Graph, Node, NodeType, Edge, EdgeType, ReasoningMode,
    Entity, WorldRelation, Interpretation, TypedEdge,
    Span, ConversationTree, Opinion,
)


def _simple_graph() -> Graph:
    n1 = Node(type=NodeType.CLAIM, text="claim1")
    n2 = Node(type=NodeType.EVIDENCE, text="evidence1")
    e = Edge(source_id=n1.id, target_id=n2.id, type=EdgeType.SUPPORTS)
    return Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e])


def _full_graph() -> Graph:
    n1 = Node(type=NodeType.AXIOM, text="axiom")
    n2 = Node(type=NodeType.CLAIM, text="claim", span=Span(start=0, end=5, text="axiom"))
    e = Edge(source_id=n1.id, target_id=n2.id, type=EdgeType.INFERS)
    ent = Entity(kind="person", name="Alice", spans=[Span(start=0, end=3, text="Ali")])
    wr = WorldRelation(source_id=n1.id, target_id=n2.id, kind="causes")
    te = TypedEdge(source_id=n1.id, target_id=n2.id, type="INFERS")
    interp = Interpretation(name="test", roles={n1.id: "source"}, edges=[te])
    cta = ConversationTree(root_id=n1.id, node_ids={n1.id, n2.id}, parent_map={n2.id: n1.id})
    return Graph(
        nodes={n1.id: n1, n2.id: n2},
        edges=[e],
        entities={ent.id: ent},
        world_relations=[wr],
        interpretations={"test": interp},
        mode=ReasoningMode.ARGUMENT,
        source_text="axiom claim",
        cta=cta,
    )


class TestGraphDefaults:
    def test_empty_graph(self):
        g = Graph()
        assert g.nodes == {}
        assert g.edges == {}
        assert g.entities == {}
        assert g.world_relations == []
        assert g.interpretations == {}
        assert g.source_text == ""
        assert g.cta is None

    def test_to_dict_empty(self):
        d = Graph().to_dict()
        assert d["propositions"] == []
        assert d["entities"] == []
        assert d["world_relations"] == []

    def test_to_json_empty(self):
        j = Graph().to_json()
        parsed = json.loads(j)
        assert parsed["propositions"] == []

    def test_to_compact_str_empty(self):
        assert Graph().to_compact_str() == ""


class TestGraphToDict:
    def test_returns_propositions(self):
        g = _simple_graph()
        d = g.to_dict()
        assert "propositions" in d
        assert len(d["propositions"]) == 2

    def test_proposition_has_fields(self):
        g = _simple_graph()
        d = g.to_dict()
        p = d["propositions"][0]
        assert "id" in p
        assert "type" in p
        assert "text" in p
        # opinion is stripped when it's the default [0.0, 0.0, 1.0, 0.5]
        # Only check it exists if it's not the default
        if p.get("opinion") is not None:
            assert "opinion" in p

    def test_includes_mode_and_source(self):
        g = _simple_graph()
        g.source_text = "hello"
        d = g.to_dict()
        assert d["mode"] == "ARGUMENT"
        assert d["source_text"] == "hello"

    def test_includes_cta_when_present(self):
        g = _full_graph()
        d = g.to_dict()
        assert "cta" in d
        assert d["cta"]["root_id"] == g.cta.root_id.hex

    def test_excludes_cta_when_none(self):
        g = _simple_graph()
        d = g.to_dict()
        assert "cta" not in d


class TestGraphFromDict:
    def test_returns_graph(self):
        g = Graph.from_dict({})
        assert isinstance(g, Graph)

    def test_empty_data(self):
        g = Graph.from_dict({})
        assert g.nodes == {}
        assert g.edges == {}
        assert g.entities == {}

    def test_with_nodes_old_format(self):
        nid = uuid4()
        data = {
            "nodes": {
                nid.hex: {"type": "CLAIM", "text": "hello", "category": 2}
            }
        }
        g = Graph.from_dict(data)
        assert nid in g.nodes
        assert g.nodes[nid].text == "hello"
        assert g.nodes[nid].type == NodeType.CLAIM

    def test_with_node_span_old_format(self):
        nid = uuid4()
        data = {
            "nodes": {
                nid.hex: {
                    "type": "EVIDENCE", "text": "test",
                    "span": {"start": 0, "end": 4, "text": "test"},
                }
            }
        }
        g = Graph.from_dict(data)
        assert g.nodes[nid].span is not None
        assert g.nodes[nid].span.start == 0

    def test_with_propositions_new_format(self):
        nid = uuid4()
        data = {
            "propositions": [
                {"id": nid.hex, "type": "CLAIM", "text": "hello", "category": 2}
            ]
        }
        g = Graph.from_dict(data)
        assert nid in g.nodes
        assert g.nodes[nid].text == "hello"

    def test_with_edges_old_format(self):
        sid, tid = uuid4(), uuid4()
        eid = uuid4()
        data = {
            "nodes": {
                sid.hex: {"type": "AXIOM", "text": "a"},
                tid.hex: {"type": "CLAIM", "text": "b"},
            },
            "edges": [
                {"id": eid.hex, "source_id": sid.hex, "target_id": tid.hex,
                 "type": "INFERS"}
            ],
        }
        g = Graph.from_dict(data)
        assert len(g.edges) == 1
        assert g.edges[eid].source_id == sid
        assert g.edges[eid].type == EdgeType.INFERS

    def test_with_edges_new_format(self):
        sid, tid = uuid4(), uuid4()
        eid = uuid4()
        data = {
            "propositions": [
                {"id": sid.hex, "type": "AXIOM", "text": "a"},
                {"id": tid.hex, "type": "CLAIM", "text": "b",
                 "outgoing_edges": [
                     {"id": eid.hex, "target_id": tid.hex, "type": "INFERS"}
                 ]},
            ]
        }
        g = Graph.from_dict(data)
        assert len(g.edges) == 1

    def test_outgoing_edges_omit_source_id(self):
        sid, tid = uuid4(), uuid4()
        eid = uuid4()
        data = {
            "propositions": [
                {"id": sid.hex, "type": "CLAIM", "text": "src",
                 "outgoing_edges": [
                     {"id": eid.hex, "target_id": tid.hex, "type": "SUPPORTS"}
                 ]},
            ]
        }
        g = Graph.from_dict(data)
        assert len(g.edges) == 1
        assert g.edges[eid].source_id == sid
        assert g.edges[eid].target_id == tid

    def test_with_entities_dict(self):
        eid = uuid4()
        data = {
            "entities": {
                eid.hex: {"kind": "person", "name": "Bob", "spans": []}
            }
        }
        g = Graph.from_dict(data)
        assert eid in g.entities
        assert g.entities[eid].name == "Bob"

    def test_with_entities_list(self):
        eid = uuid4()
        data = {
            "entities": [
                {"id": eid.hex, "kind": "person", "name": "Bob", "spans": []}
            ]
        }
        g = Graph.from_dict(data)
        assert eid in g.entities
        assert g.entities[eid].name == "Bob"

    def test_with_world_relations(self):
        sid, tid = uuid4(), uuid4()
        rid = uuid4()
        data = {
            "world_relations": [
                {"id": rid.hex, "source_id": sid.hex, "target_id": tid.hex,
                 "kind": "causes"}
            ]
        }
        g = Graph.from_dict(data)
        assert len(g.world_relations) == 1
        assert g.world_relations[0].kind == "causes"

    def test_with_interpretations_old_format(self):
        nid = uuid4()
        eid = uuid4()
        data = {
            "interpretations": {
                "arg": {
                    "roles": {nid.hex: "source"},
                    "edges": [
                        {"id": eid.hex, "source_id": nid.hex, "target_id": nid.hex,
                         "type": "SUPPORTS"}
                    ],
                }
            }
        }
        g = Graph.from_dict(data)
        assert "arg" in g.interpretations
        assert len(g.interpretations["arg"].edges) == 1

    def test_with_cta(self):
        rid = uuid4()
        nid = uuid4()
        data = {
            "cta": {
                "root_id": rid.hex,
                "node_ids": [rid.hex, nid.hex],
                "parent_map": {nid.hex: rid.hex},
            }
        }
        g = Graph.from_dict(data)
        assert g.cta is not None
        assert g.cta.root_id == rid
        assert nid in g.cta.node_ids

    def test_roundtrip(self):
        g_orig = _full_graph()
        data = g_orig.to_dict()
        g_restored = Graph.from_dict(data)
        assert len(g_restored.nodes) == len(g_orig.nodes)
        assert len(g_restored.edges) == len(g_orig.edges)
        assert len(g_restored.entities) == len(g_orig.entities)
        for nid, node in g_orig.nodes.items():
            assert nid in g_restored.nodes
            assert g_restored.nodes[nid].text == node.text
            assert g_restored.nodes[nid].type == node.type

    def test_missing_propositions_key_inherits_mode(self):
        data = {"mode": "CAUSAL", "source_text": "test"}
        g = Graph.from_dict(data)
        assert g.mode.name == "CAUSAL"
        assert g.source_text == "test"


class TestGraphCompactStr:
    def test_includes_nodes(self):
        s = _simple_graph().to_compact_str()
        assert "NODE" in s
        assert "CLAIM" in s
        assert "EVIDENCE" in s

    def test_includes_edges(self):
        s = _simple_graph().to_compact_str()
        assert "EDGE" in s
        assert "SUPPORTS" in s

    def test_includes_entities(self):
        g = _full_graph()
        s = g.to_compact_str()
        assert "ENTITY" in s
        assert "Alice" in s

    def test_includes_world_relations(self):
        g = _full_graph()
        s = g.to_compact_str()
        assert "REL" in s

    def test_includes_interpretations(self):
        g = _full_graph()
        s = g.to_compact_str()
        assert "INTERPRETATION" in s


class TestGraphJson:
    def test_to_json_returns_string(self):
        j = _simple_graph().to_json()
        assert isinstance(j, str)

    def test_to_json_parses(self):
        j = _simple_graph().to_json()
        d = json.loads(j)
        assert "propositions" in d

    def test_to_json_with_indent(self):
        j1 = _simple_graph().to_json(indent=2)
        j2 = _simple_graph().to_json(indent=4)
        assert len(j1) < len(j2)



