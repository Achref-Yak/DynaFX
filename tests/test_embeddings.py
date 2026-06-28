"""Tests for embeddings, Compare, Align, and enhanced Attention operators."""

import pytest
from uuid import uuid4

from dynafx.core.models import Graph, Node, NodeType
from dynafx.core.state import State
from dynafx.core.embeddings import EmbeddingModel
from dynafx.operators.compare import CompareOperator
from dynafx.operators.align import AlignOperator
from dynafx.operators.attention import AttentionOperator



# ── EmbeddingModel tests ──────────────────────────────────────────

class TestEmbeddingModel:
    def test_singleton(self):
        m1 = EmbeddingModel.get_instance()
        m2 = EmbeddingModel.get_instance()
        assert m1 is m2

    def test_dimension(self):
        model = EmbeddingModel.get_instance()
        assert model.dimension == 384

    def test_encode(self):
        model = EmbeddingModel.get_instance()
        vec = model.encode("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)

    def test_encode_batch(self):
        model = EmbeddingModel.get_instance()
        vecs = model.encode_batch(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert EmbeddingModel.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert EmbeddingModel.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_distance(self):
        v = [1.0, 0.0, 0.0]
        assert EmbeddingModel.cosine_distance(v, v) == pytest.approx(0.0)

    def test_similarity(self):
        model = EmbeddingModel.get_instance()
        sim = model.similarity("the cat sat on the mat", "a cat sat on a rug")
        assert 0.5 < sim < 1.0

    def test_reset(self):
        EmbeddingModel.reset()
        m1 = EmbeddingModel.get_instance()
        EmbeddingModel.reset()
        m2 = EmbeddingModel.get_instance()
        assert m1 is not m2
        EmbeddingModel.reset()


# ── Node embedding field tests ────────────────────────────────────

class TestNodeEmbedding:
    def test_node_has_embedding_field(self):
        node = Node(text="test")
        assert node.embedding is None

    def test_node_with_embedding(self):
        emb = [0.1] * 384
        node = Node(text="test", embedding=emb)
        assert node.embedding == emb

    def test_graph_serialization_with_embedding(self):
        node = Node(text="test", embedding=[0.1] * 384)
        graph = Graph(nodes={node.id: node})
        data = graph.to_dict()
        # Embeddings are stripped from JSON serialization (belong in vector DB)
        assert "embedding" not in data.get("propositions", [{}])[0] if data.get("propositions") else True
        # The node itself still has the embedding in memory
        assert node.embedding == [0.1] * 384


# ── CompareOperator tests ─────────────────────────────────────────

def _make_graph_with_emb(texts: list[str]) -> Graph:
    model = EmbeddingModel.get_instance()
    nodes = {}
    for text in texts:
        node = Node(text=text, embedding=model.encode(text))
        nodes[node.id] = node
    return Graph(nodes=nodes)


class TestCompareOperator:
    def test_identical_graphs(self):
        g1 = _make_graph_with_emb(["cats are great", "dogs are great"])
        g2 = _make_graph_with_emb(["cats are great", "dogs are great"])
        op = CompareOperator()
        state = State(graph=Graph())
        state.graph = g1
        state.metadata["other_states"] = [State(graph=g2)]
        result = op(state, similarity_threshold=0.5)
        diff = result.metadata["compare_result"]
        assert len(diff["shared_concepts"]) == 2
        assert len(diff["unique_to_a"]) == 0
        assert len(diff["unique_to_b"]) == 0
        assert diff["score"] > 0.8

    def test_completely_different_graphs(self):
        g1 = _make_graph_with_emb(["quantum physics theory"])
        g2 = _make_graph_with_emb(["cooking recipe ingredients"])
        op = CompareOperator()
        state = State(graph=Graph())
        state.graph = g1
        state.metadata["other_states"] = [State(graph=g2)]
        result = op(state, similarity_threshold=0.9)
        diff = result.metadata["compare_result"]
        assert len(diff["shared_concepts"]) == 0
        assert len(diff["unique_to_a"]) == 1
        assert len(diff["unique_to_b"]) == 1

    def test_belief_deltas(self):
        g1 = Node(text="the sky is blue", embedding=EmbeddingModel.get_instance().encode("the sky is blue"))
        g1_node = g1
        g1_node.opinion = (0.8, 0.1, 0.1, 0.5)
        graph_a = Graph(nodes={g1_node.id: g1_node})

        g2_node = Node(text="the sky is blue", embedding=EmbeddingModel.get_instance().encode("the sky is blue"))
        g2_node.opinion = (0.3, 0.5, 0.2, 0.5)
        graph_b = Graph(nodes={g2_node.id: g2_node})

        op = CompareOperator()
        state = State(graph=Graph())
        state.graph = graph_a
        state.metadata["other_states"] = [State(graph=graph_b)]
        result = op(state, similarity_threshold=0.5)
        diff = result.metadata["compare_result"]
        assert len(diff["belief_deltas"]) == 1
        assert diff["belief_deltas"][0]["delta"] == pytest.approx(-0.5, abs=0.01)

    def test_explicit_graphs(self):
        g1 = _make_graph_with_emb(["cats are great"])
        g2 = _make_graph_with_emb(["cats are great"])
        op = CompareOperator()
        state = State(graph=Graph())
        result = op(state, graph_a=g1, graph_b=g2, similarity_threshold=0.5)
        diff = result.metadata["compare_result"]
        assert len(diff["shared_concepts"]) == 1

    def test_no_second_graph(self):
        op = CompareOperator()
        state = State(graph=_make_graph_with_emb(["test"]))
        result = op(state)
        diff = result.metadata["compare_result"]
        assert isinstance(diff, dict)


# ── AlignOperator tests ───────────────────────────────────────────

class TestAlignOperator:
    def test_align_identical_graphs(self):
        g1 = _make_graph_with_emb(["cats are great", "dogs are great"])
        g2 = _make_graph_with_emb(["cats are great", "dogs are great"])
        op = AlignOperator()
        state = State(graph=Graph())
        state.graph = g1
        result = op(state, graphs={"a": g1, "b": g2}, similarity_threshold=0.5)
        alignment = result.metadata["alignment_result"]
        assert len(alignment["alignments"]) == 2
        assert alignment["coverage"] > 0.8

    def test_align_different_graphs(self):
        g1 = _make_graph_with_emb(["cats are great"])
        g2 = _make_graph_with_emb(["dogs are great"])
        op = AlignOperator()
        state = State(graph=Graph())
        state.graph = g1
        result = op(state, graphs={"a": g1, "b": g2}, similarity_threshold=0.9)
        alignment = result.metadata["alignment_result"]
        assert len(alignment["alignments"]) == 2

    def test_align_partial_match(self):
        g1 = _make_graph_with_emb(["cats are great", "quantum physics"])
        g2 = _make_graph_with_emb(["cats are great", "cooking recipe"])
        op = AlignOperator()
        state = State(graph=Graph())
        state.graph = g1
        result = op(state, graphs={"a": g1, "b": g2}, similarity_threshold=0.5)
        alignment = result.metadata["alignment_result"]
        assert len(alignment["alignments"]) >= 2

    def test_no_graphs(self):
        op = AlignOperator()
        state = State(graph=Graph())
        result = op(state)
        alignment = result.metadata["alignment_result"]
        assert len(alignment["alignments"]) == 0

    def test_centroid_computation(self):
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        centroid = AlignOperator._compute_centroid(vecs)
        assert centroid == [0.5, 0.5]

    def test_centroid_empty(self):
        assert AlignOperator._compute_centroid([]) == []


# ── Enhanced AttentionOperator tests ──────────────────────────────

class TestAttentionConceptFilter:
    def test_concept_filter(self):
        model = EmbeddingModel.get_instance()
        g = Graph(nodes={
            uuid4(): Node(text="cats are fluffy", embedding=model.encode("cats are fluffy")),
            uuid4(): Node(text="dogs are loyal", embedding=model.encode("dogs are loyal")),
            uuid4(): Node(text="physics is complex", embedding=model.encode("physics is complex")),
        })
        op = AttentionOperator()
        state = State(graph=g)
        result = op(state, concept="feline pets", concept_threshold=0.3)
        filtered = result.graph
        assert len(filtered.nodes) < 3
        assert len(filtered.nodes) >= 1

    def test_concept_filter_exact(self):
        model = EmbeddingModel.get_instance()
        g = Graph(nodes={
            uuid4(): Node(text="quantum entanglement", embedding=model.encode("quantum entanglement")),
            uuid4(): Node(text="cooking recipes", embedding=model.encode("cooking recipes")),
        })
        op = AttentionOperator()
        state = State(graph=g)
        result = op(state, concept="quantum physics", concept_threshold=0.5)
        filtered = result.graph
        texts = [n.text for n in filtered.nodes.values()]
        assert "quantum entanglement" in texts
        assert "cooking recipes" not in texts

    def test_combined_filters(self):
        model = EmbeddingModel.get_instance()
        n1 = Node(text="cats are fluffy", type=NodeType.CLAIM, embedding=model.encode("cats are fluffy"))
        n1.opinion = (0.9, 0.1, 0.1, 0.5)
        n2 = Node(text="dogs are loyal", type=NodeType.EVIDENCE, embedding=model.encode("dogs are loyal"))
        n2.opinion = (0.3, 0.5, 0.2, 0.5)
        g = Graph(nodes={n1.id: n1, n2.id: n2})
        op = AttentionOperator()
        state = State(graph=g)
        result = op(state, node_type="claim", threshold=0.5, concept="pets")
        filtered = result.graph
        assert all(n.type == NodeType.CLAIM for n in filtered.nodes.values())
        assert all(n.opinion[0] >= 0.5 for n in filtered.nodes.values())



