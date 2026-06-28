from uuid import UUID

from dynafx.core.models import (
    ConversationTree,
    Graph, Node, NodeType,
    Edge, EdgeType,
)


def _make_graph() -> Graph:
    root = Node(type=NodeType.AXIOM, text="root claim")
    child = Node(type=NodeType.EVIDENCE, text="supporting data")
    leaf = Node(type=NodeType.CLAIM, text="derived conclusion")
    g = Graph(nodes={root.id: root, child.id: child, leaf.id: leaf})
    e1 = Edge(source_id=root.id, target_id=child.id, type=EdgeType.SUPPORTS)
    e2 = Edge(source_id=child.id, target_id=leaf.id, type=EdgeType.INFERS)
    g.edges = {e1.id: e1, e2.id: e2}
    return g


def test_from_graph_detects_root():
    g = _make_graph()
    root_id = list(g.nodes.keys())[0]
    cta = ConversationTree.from_graph(g)
    assert cta.root_id == root_id


def test_from_graph_infers_parents():
    g = _make_graph()
    cta = ConversationTree.from_graph(g)
    node_ids = list(g.nodes.keys())
    assert node_ids[1] in cta.parent_map   # child has parent
    assert node_ids[2] in cta.parent_map   # leaf has parent


def test_get_context_from_root():
    g = _make_graph()
    cta = ConversationTree.from_graph(g)
    root_id = list(g.nodes.keys())[0]
    ctx = cta.get_context(root_id)
    assert ctx == [root_id]


def test_get_context_from_leaf():
    g = _make_graph()
    cta = ConversationTree.from_graph(g)
    ids = list(g.nodes.keys())
    ctx = cta.get_context(ids[2])
    assert ctx[0] == ids[0]  # root first
    assert ctx[-1] == ids[2]  # leaf last
    assert len(ctx) == 3


def test_get_context_unknown_node():
    g = _make_graph()
    cta = ConversationTree.from_graph(g)
    assert cta.get_context(UUID(int=0)) == []


def test_to_dict_roundtrip():
    g = _make_graph()
    cta = ConversationTree.from_graph(g)
    d = cta.to_dict()
    assert "root_id" in d
    assert "node_ids" in d
    assert "parent_map" in d
    assert len(d["node_ids"]) == 3
