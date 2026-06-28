"""Tests for Abduction, Induction, and Analogy operators."""

import pytest
from uuid import uuid4

from dynafx.core.models import (
    Edge, EdgeType, Graph, Node, NodeType, Span,
)
from dynafx.core.state import State
from dynafx.core.embeddings import EmbeddingModel
from dynafx.operators.abduce import AbductionOperator, Hypothesis
from dynafx.operators.induce import InductionOperator, InductionRule
from dynafx.operators.analogy import AnalogyOperator, AnalogyMapping, InferredEdge


# ── Helper functions ──────────────────────────────────────────────

def _make_causal_graph() -> Graph:
    """Create a graph with causal edges for abduction testing."""
    model = EmbeddingModel.get_instance()

    rain = Node(
        text="It rained last night",
        type=NodeType.CLAIM,
        embedding=model.encode("It rained last night"),
    )
    sprinkler = Node(
        text="The sprinkler was on",
        type=NodeType.CLAIM,
        embedding=model.encode("The sprinkler was on"),
    )
    wet_grass = Node(
        text="The grass is wet",
        type=NodeType.EVIDENCE,
        embedding=model.encode("The grass is wet"),
    )
    wet_path = Node(
        text="The sidewalk is wet",
        type=NodeType.EVIDENCE,
        embedding=model.encode("The sidewalk is wet"),
    )

    graph = Graph(nodes={
        rain.id: rain,
        sprinkler.id: sprinkler,
        wet_grass.id: wet_grass,
        wet_path.id: wet_path,
    })

    edges = [
        Edge(source_id=rain.id, target_id=wet_grass.id, type=EdgeType.CAUSES),
        Edge(source_id=sprinkler.id, target_id=wet_grass.id, type=EdgeType.CAUSES),
        Edge(source_id=rain.id, target_id=wet_path.id, type=EdgeType.CAUSES),
    ]
    graph.edges = {e.id: e for e in edges}

    return graph


def _make_observation_graph() -> Graph:
    """Create a graph with repeated patterns for induction testing."""
    model = EmbeddingModel.get_instance()

    nodes = {}
    for i, color in enumerate(["white", "white", "white", "white", "black"]):
        node = Node(
            text=f"Swan {i+1} is {color}",
            type=NodeType.EVIDENCE,
            embedding=model.encode(f"Swan {i+1} is {color}"),
        )
        nodes[node.id] = node

    return Graph(nodes=nodes)


def _make_source_target_graphs() -> tuple[Graph, Graph]:
    """Create source and target graphs for analogy testing."""
    model = EmbeddingModel.get_instance()

    # Source: Heart → Body
    heart = Node(text="Heart pumps blood", embedding=model.encode("Heart pumps blood"))
    body = Node(text="Body needs blood", embedding=model.encode("Body needs blood"))
    source = Graph(nodes={heart.id: heart, body.id: body})
    e = Edge(source_id=heart.id, target_id=body.id, type=EdgeType.CAUSES)
    source.edges = {e.id: e}

    # Target: Brain, Mind (no edges yet)
    brain = Node(text="Brain processes info", embedding=model.encode("Brain processes info"))
    mind = Node(text="Mind needs info", embedding=model.encode("Mind needs info"))
    target = Graph(nodes={brain.id: brain, mind.id: mind})

    return source, target


# ── AbductionOperator tests ──────────────────────────────────────

class TestAbductionOperator:
    def test_basic_abduction(self):
        graph = _make_causal_graph()
        wet_grass_id = [nid for nid, n in graph.nodes.items()
                        if "wet" in n.text.lower() and "grass" in n.text.lower()][0]

        op = AbductionOperator()
        state = State(graph=graph)
        result = op(state, observation_node_id=str(wet_grass_id))

        abd = result.metadata["abduction_result"]
        assert abd["total_candidates"] == 2
        texts = [h["text"] for h in abd["hypotheses"]]
        assert any("rain" in t.lower() for t in texts)
        assert any("sprinkler" in t.lower() for t in texts)

    def test_abduction_by_text(self):
        graph = _make_causal_graph()
        op = AbductionOperator()
        state = State(graph=graph)
        result = op(state, observation="The grass is wet")

        abd = result.metadata["abduction_result"]
        assert abd["total_candidates"] == 2

    def test_abduction_by_embedding(self):
        graph = _make_causal_graph()
        op = AbductionOperator()
        state = State(graph=graph)
        result = op(state, observation="wet lawn", similarity_threshold=0.3)

        abd = result.metadata["abduction_result"]
        assert abd["total_candidates"] >= 1

    def test_abduction_empty_graph(self):
        op = AbductionOperator()
        state = State(graph=Graph())
        result = op(state, observation="anything")
        assert result.metadata["abduction_result"]["total_candidates"] == 0

    def test_abduction_no_match(self):
        graph = _make_causal_graph()
        op = AbductionOperator()
        state = State(graph=graph)
        result = op(state, observation="quantum physics", similarity_threshold=0.99)
        assert result.metadata["abduction_result"]["total_candidates"] == 0

    def test_abduction_max_hypotheses(self):
        graph = _make_causal_graph()
        wet_grass_id = [nid for nid, n in graph.nodes.items()
                        if "wet" in n.text.lower() and "grass" in n.text.lower()][0]

        op = AbductionOperator()
        state = State(graph=graph)
        result = op(state, observation_node_id=str(wet_grass_id), max_hypotheses=1)

        abd = result.metadata["abduction_result"]
        assert abd["total_candidates"] == 1

    def test_abduction_scoring(self):
        graph = _make_causal_graph()
        wet_grass_id = [nid for nid, n in graph.nodes.items()
                        if "wet" in n.text.lower() and "grass" in n.text.lower()][0]

        op = AbductionOperator()
        state = State(graph=graph)
        result = op(state, observation_node_id=str(wet_grass_id))

        abd = result.metadata["abduction_result"]
        scores = [h["score"] for h in abd["hypotheses"]]
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_abduction_custom_edge_types(self):
        graph = _make_causal_graph()
        wet_grass_id = [nid for nid, n in graph.nodes.items()
                        if "wet" in n.text.lower() and "grass" in n.text.lower()][0]

        op = AbductionOperator()
        state = State(graph=graph)
        result = op(state, observation_node_id=str(wet_grass_id), edge_types=["CAUSES"])

        abd = result.metadata["abduction_result"]
        assert abd["total_candidates"] == 2


# ── InductionOperator tests ──────────────────────────────────────

class TestInductionOperator:
    def test_basic_induction(self):
        graph = _make_observation_graph()
        op = InductionOperator()
        state = State(graph=graph)
        result = op(state, pattern="swan color", min_instances=3)

        ind = result.metadata["induction_result"]
        assert ind["total_observations"] >= 3
        assert len(ind["rules"]) >= 1

    def test_induction_by_type(self):
        graph = _make_observation_graph()
        op = InductionOperator()
        state = State(graph=graph)
        result = op(state, node_type="EVIDENCE", min_instances=3)

        ind = result.metadata["induction_result"]
        assert ind["total_observations"] >= 3

    def test_induction_insufficient_observations(self):
        graph = _make_observation_graph()
        op = InductionOperator()
        state = State(graph=graph)
        result = op(state, pattern="nonexistent concept", min_instances=3)

        ind = result.metadata["induction_result"]
        assert len(ind["rules"]) == 0

    def test_induction_empty_graph(self):
        op = InductionOperator()
        state = State(graph=Graph())
        result = op(state, pattern="anything")
        assert result.metadata["induction_result"]["total_observations"] == 0

    def test_induction_rule_scoring(self):
        graph = _make_observation_graph()
        op = InductionOperator()
        state = State(graph=graph)
        result = op(state, pattern="swan", min_instances=3)

        ind = result.metadata["induction_result"]
        for rule in ind["rules"]:
            assert 0.0 <= rule["coverage"] <= 1.0
            assert 0.0 <= rule["specificity"] <= 1.0

    def test_induction_rule_has_observations(self):
        graph = _make_observation_graph()
        op = InductionOperator()
        state = State(graph=graph)
        result = op(state, pattern="swan", min_instances=3)

        ind = result.metadata["induction_result"]
        for rule in ind["rules"]:
            assert len(rule["observations"]) > 0
            assert len(rule["node_ids"]) > 0


# ── AnalogyOperator tests ────────────────────────────────────────

class TestAnalogyOperator:
    def test_basic_analogy(self):
        source, target = _make_source_target_graphs()
        op = AnalogyOperator()
        state = State(graph=target)
        result = op(state, source_graph=source, target_graph=target, similarity_threshold=0.1)

        ana = result.metadata["analogy_result"]
        assert ana["total_mappings"] >= 1
        assert ana["total_inferred"] >= 1

    def test_analogy_mappings(self):
        source, target = _make_source_target_graphs()
        op = AnalogyOperator()
        state = State(graph=target)
        result = op(state, source_graph=source, target_graph=target, similarity_threshold=0.1)

        ana = result.metadata["analogy_result"]
        for m in ana["mappings"]:
            assert 0.0 <= m["similarity"] <= 1.0
            assert "source_text" in m
            assert "target_text" in m

    def test_analogy_inferred_edges(self):
        source, target = _make_source_target_graphs()
        op = AnalogyOperator()
        state = State(graph=target)
        result = op(state, source_graph=source, target_graph=target, similarity_threshold=0.1)

        ana = result.metadata["analogy_result"]
        for e in ana["inferred_edges"]:
            assert "analogy_from" in e
            assert "edge_type" in e

    def test_analogy_empty_graphs(self):
        op = AnalogyOperator()
        state = State(graph=Graph())
        result = op(state, source_graph=Graph(), target_graph=Graph())
        ana = result.metadata["analogy_result"]
        assert ana["total_mappings"] == 0

    def test_analogy_no_alignment(self):
        source, target = _make_source_target_graphs()
        op = AnalogyOperator()
        state = State(graph=target)
        result = op(state, source_graph=source, target_graph=target, similarity_threshold=0.99)

        ana = result.metadata["analogy_result"]
        assert ana["total_mappings"] == 0

    def test_analogy_via_metadata(self):
        source, target = _make_source_target_graphs()
        op = AnalogyOperator()
        state = State(graph=source)
        state.metadata["other_states"] = [State(graph=target)]
        result = op(state, similarity_threshold=0.1)

        ana = result.metadata["analogy_result"]
        assert ana["total_mappings"] >= 1


# ── CAUSES edge type test ────────────────────────────────────────

class TestCausesEdgeType:
    def test_causes_exists(self):
        assert hasattr(EdgeType, "CAUSES")

    def test_causes_value(self):
        assert EdgeType.CAUSES is not None

    def test_causes_in_graph(self):
        n1 = Node(text="cause")
        n2 = Node(text="effect")
        e = Edge(source_id=n1.id, target_id=n2.id, type=EdgeType.CAUSES)
        graph = Graph(
            nodes={n1.id: n1, n2.id: n2},
            edges={e.id: e},
        )
        assert any(e.type == EdgeType.CAUSES for e in graph.edges.values())
