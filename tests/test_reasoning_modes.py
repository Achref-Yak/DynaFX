from cognitive_engine.core.models import Graph, Node, Edge, NodeType, EdgeType, ReasoningMode
from cognitive_engine.reason.modes import apply_mode, compute_mode_views, MODE_ACTIVE_EDGES


def _make_graph() -> Graph:
    a = Node(type=NodeType.EVIDENCE, text="data")
    b = Node(type=NodeType.CLAIM, text="conclusion")
    c = Node(type=NodeType.CONDITION, text="if x then y")
    g = Graph(
        nodes={a.id: a, b.id: b, c.id: c},
        edges=[
            Edge(source_id=a.id, target_id=b.id, type=EdgeType.SUPPORTS),
            Edge(source_id=c.id, target_id=b.id, type=EdgeType.QUALIFIES),
            Edge(source_id=a.id, target_id=c.id, type=EdgeType.INFERS),
        ],
    )
    return g


class TestApplyMode:
    def test_causal_mode_filters_edges(self):
        g = _make_graph()
        result = apply_mode(g, ReasoningMode.CAUSAL)
        assert result.mode == ReasoningMode.CAUSAL
        for e in result.edges.values():
            assert e.type in MODE_ACTIVE_EDGES[ReasoningMode.CAUSAL]

    def test_conditional_mode_filters_edges(self):
        g = _make_graph()
        result = apply_mode(g, ReasoningMode.CONDITIONAL)
        assert result.mode == ReasoningMode.CONDITIONAL
        for e in result.edges.values():
            assert e.type in MODE_ACTIVE_EDGES[ReasoningMode.CONDITIONAL]

    def test_argument_mode_filters_edges(self):
        g = _make_graph()
        result = apply_mode(g, ReasoningMode.ARGUMENT)
        assert result.mode == ReasoningMode.ARGUMENT
        for e in result.edges.values():
            assert e.type in MODE_ACTIVE_EDGES[ReasoningMode.ARGUMENT]

    def test_analogy_mode_filters_edges(self):
        g = _make_graph()
        result = apply_mode(g, ReasoningMode.ANALOGY)
        assert result.mode == ReasoningMode.ANALOGY
        for e in result.edges.values():
            assert e.type in MODE_ACTIVE_EDGES[ReasoningMode.ANALOGY]

    def test_original_graph_unchanged(self):
        g = _make_graph()
        original_edge_count = len(g.edges)
        apply_mode(g, ReasoningMode.CAUSAL)
        assert len(g.edges) == original_edge_count


class TestComputeModeViews:
    def test_views_stored_in_metadata(self):
        g = _make_graph()
        result = compute_mode_views(g)
        modes = result.metadata.get("modes", {})
        assert len(modes) == 4
        for mode in ReasoningMode:
            assert mode.name in modes
            assert "active_edge_count" in modes[mode.name]

    def test_view_edge_counts(self):
        g = _make_graph()
        result = compute_mode_views(g)
        modes = result.metadata["modes"]
        assert modes["CAUSAL"]["active_edge_count"] == 2
        assert modes["CONDITIONAL"]["active_edge_count"] == 2
        assert modes["ARGUMENT"]["active_edge_count"] == 1
        assert modes["ANALOGY"]["active_edge_count"] == 1


class TestModeActiveEdges:
    def test_each_mode_has_edges(self):
        for mode in ReasoningMode:
            assert len(MODE_ACTIVE_EDGES[mode]) >= 1

    def test_modes_cover_all_edge_types(self):
        all_active = set()
        for edges in MODE_ACTIVE_EDGES.values():
            all_active.update(edges)
        for et in EdgeType:
            assert et in all_active, f"{et.name} not active in any mode"
