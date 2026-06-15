"""Tests for the operator framework — core, operators, pipelines, schemas."""

import pytest
from uuid import uuid4

from cognitive_engine.core.models import (
    Edge,
    EdgeType,
    Graph,
    Node,
    NodeType,
    Opinion,
)
from cognitive_engine.core.operator import Operator
from cognitive_engine.core.pipeline import Pipeline
from cognitive_engine.core.schema import Schema, merge_schemas
from cognitive_engine.core.state import State, StateDelta


# ============================================================
# Phase 1: Core Framework Tests
# ============================================================

class TestState:
    """Test State dataclass."""

    def test_create_state(self):
        graph = Graph()
        state = State(graph=graph)
        assert state.graph is graph
        assert state.metadata == {}
        assert state.history == []

    def test_record_delta(self):
        graph = Graph()
        state = State(graph=graph)
        state.record("test_op", "test description")
        assert len(state.history) == 1
        assert state.history[0].operator == "test_op"
        assert state.history[0].description == "test description"

    def test_max_history(self):
        graph = Graph()
        state = State(graph=graph, max_history=3)
        for i in range(5):
            state.record(f"op_{i}", f"desc_{i}")
        assert len(state.history) == 3
        assert state.history[0].operator == "op_2"

    def test_snapshot(self):
        graph = Graph()
        graph.nodes[uuid4()] = Node(text="test")
        state = State(graph=graph, metadata={"key": "value"})
        snap = state.snapshot()
        assert snap["node_count"] == 1
        assert snap["edge_count"] == 0
        assert "key" in snap["metadata_keys"]

    def test_fork(self):
        graph = Graph()
        state = State(graph=graph, metadata={"key": "value"})
        forked = state.fork()
        assert forked.graph is graph
        assert forked.metadata == {"key": "value"}
        assert forked.history == state.history

    def test_repr(self):
        state = State(graph=Graph())
        assert "nodes=0" in repr(state)
        assert "edges=0" in repr(state)


class TestStateDelta:
    """Test StateDelta dataclass."""

    def test_create_delta(self):
        delta = StateDelta(
            timestamp=1.0,
            operator="test",
            description="test",
            node_count=5,
            edge_count=3,
        )
        assert delta.operator == "test"
        assert delta.node_count == 5


class TestPipeline:
    """Test Pipeline composition."""

    def test_create_pipeline(self):
        pipeline = Pipeline(name="test")
        assert pipeline.name == "test"
        assert len(pipeline) == 0

    def test_add_operator(self):
        class MockOp:
            name = "mock"
            def __call__(self, state, **kwargs):
                return state

        pipeline = Pipeline(name="test")
        pipeline.add(MockOp())
        assert len(pipeline) == 1

    def test_run_pipeline(self):
        class CountOp:
            name = "count"
            def __call__(self, state, **kwargs):
                state.metadata["count"] = state.metadata.get("count", 0) + 1
                return state

        pipeline = Pipeline(name="test", operators=[CountOp(), CountOp(), CountOp()])
        state = State(graph=Graph())
        result = pipeline.run(state)
        assert result.metadata["count"] == 3

    def test_fluent_api(self):
        class MockOp:
            name = "mock"
            def __call__(self, state, **kwargs):
                return state

        pipeline = Pipeline(name="test")
        result = pipeline.add(MockOp()).add(MockOp())
        assert result is pipeline
        assert len(pipeline) == 2

    def test_insert_operator(self):
        class OpA:
            name = "a"
            def __call__(self, state, **kwargs):
                state.metadata["ops"] = state.metadata.get("ops", []) + ["a"]
                return state

        class OpB:
            name = "b"
            def __call__(self, state, **kwargs):
                state.metadata["ops"] = state.metadata.get("ops", []) + ["b"]
                return state

        pipeline = Pipeline(name="test", operators=[OpA()])
        pipeline.insert(0, OpB())
        state = State(graph=Graph())
        result = pipeline.run(state)
        assert result.metadata["ops"] == ["b", "a"]

    def test_before_after(self):
        class OpA:
            name = "a"
            def __call__(self, state, **kwargs):
                state.metadata["ops"] = state.metadata.get("ops", []) + ["a"]
                return state

        class OpB:
            name = "b"
            def __call__(self, state, **kwargs):
                state.metadata["ops"] = state.metadata.get("ops", []) + ["b"]
                return state

        pipeline = Pipeline(name="test", operators=[OpA()])
        pipeline.after("a", OpB())
        state = State(graph=Graph())
        result = pipeline.run(state)
        assert result.metadata["ops"] == ["a", "b"]

    def test_replace_operator(self):
        class OpA:
            name = "a"
            def __call__(self, state, **kwargs):
                state.metadata["result"] = "a"
                return state

        class OpB:
            name = "b"
            def __call__(self, state, **kwargs):
                state.metadata["result"] = "b"
                return state

        pipeline = Pipeline(name="test", operators=[OpA()])
        pipeline.replace("a", OpB())
        state = State(graph=Graph())
        result = pipeline.run(state)
        assert result.metadata["result"] == "b"

    def test_copy_pipeline(self):
        class MockOp:
            name = "mock"
            def __call__(self, state, **kwargs):
                return state

        pipeline = Pipeline(name="test", operators=[MockOp()])
        copy = pipeline.copy("test_copy")
        assert copy.name == "test_copy"
        assert len(copy) == 1

    def test_repr(self):
        class OpA:
            name = "a"
            def __call__(self, state, **kwargs):
                return state

        pipeline = Pipeline(name="test", operators=[OpA()])
        assert "test" in repr(pipeline)
        assert "a" in repr(pipeline)


class TestSchema:
    """Test Schema dataclass."""

    def test_create_schema(self):
        schema = Schema(name="test")
        assert schema.name == "test"
        assert schema.merge_strategy == "average"

    def test_get_node_type(self):
        schema = Schema(
            name="test",
            node_types={"claim": NodeType.CLAIM},
        )
        assert schema.get_node_type("claim") == NodeType.CLAIM
        assert schema.get_node_type("unknown") is None

    def test_get_edge_type(self):
        schema = Schema(
            name="test",
            edge_types={
                (NodeType.EVIDENCE, NodeType.CLAIM, "Support"): EdgeType.SUPPORTS,
            },
        )
        result = schema.get_edge_type(NodeType.EVIDENCE, NodeType.CLAIM, "Support")
        assert result == EdgeType.SUPPORTS

    def test_merge_schemas(self):
        s1 = Schema(name="s1", node_types={"claim": NodeType.CLAIM})
        s2 = Schema(name="s2", node_types={"evidence": NodeType.EVIDENCE})
        merged = merge_schemas(s1, s2, name="merged")
        assert merged.name == "merged"
        assert merged.node_types["claim"] == NodeType.CLAIM
        assert merged.node_types["evidence"] == NodeType.EVIDENCE


# ============================================================
# Phase 2: Operator Tests
# ============================================================

class TestOperators:
    """Test all operators."""

    def _make_state(self, text="test text"):
        graph = Graph(source_text=text)
        return State(graph=graph, metadata={"text": text})

    def test_extract_operator(self):
        from cognitive_engine.operators.extract import ExtractOperator
        op = ExtractOperator()
        state = self._make_state("Honor is tied to family reputation.")
        result = op(state, text="Honor is tied to family reputation.")
        assert result.metadata["extracted"] is True
        assert len(result.graph.nodes) > 0

    def test_schema_operator(self):
        from cognitive_engine.operators.schema import SchemaOperator
        from cognitive_engine.schemas.legal import LEGAL_SCHEMA
        op = SchemaOperator()
        state = self._make_state()
        state.graph.nodes[uuid4()] = Node(text="test", type=NodeType.EVIDENCE)
        result = op(state, schema=LEGAL_SCHEMA)
        assert result.metadata["schema_applied"] == "legal"

    def test_graph_operator(self):
        from cognitive_engine.operators.graph import GraphOperator
        op = GraphOperator()
        state = self._make_state()
        result = op(state)
        assert len(result.history) == 1

    def test_propagate_operator(self):
        from cognitive_engine.operators.propagate import PropagateOperator
        op = PropagateOperator()
        state = self._make_state()
        state.graph.nodes[uuid4()] = Node(text="test", type=NodeType.CLAIM)
        result = op(state)
        assert "beliefs" in result.metadata

    def test_constraint_operator(self):
        from cognitive_engine.operators.constraint import ConstraintOperator
        op = ConstraintOperator()
        state = self._make_state()
        state.graph.nodes[uuid4()] = Node(text="test", type=NodeType.CLAIM)
        result = op(state)
        assert "constraint_beliefs" in result.metadata

    def test_attention_operator(self):
        from cognitive_engine.operators.attention import AttentionOperator
        op = AttentionOperator()
        state = self._make_state()
        n1 = Node(text="claim", type=NodeType.CLAIM, opinion=(0.8, 0.1, 0.1, 0.5))
        n2 = Node(text="evidence", type=NodeType.EVIDENCE, opinion=(0.2, 0.3, 0.5, 0.5))
        state.graph.nodes[n1.id] = n1
        state.graph.nodes[n2.id] = n2
        result = op(state, node_type="CLAIM")
        assert len(result.graph.nodes) == 1

    def test_attention_threshold(self):
        from cognitive_engine.operators.attention import AttentionOperator
        op = AttentionOperator()
        state = self._make_state()
        n1 = Node(text="strong", type=NodeType.CLAIM, opinion=(0.9, 0.05, 0.05, 0.5))
        n2 = Node(text="weak", type=NodeType.CLAIM, opinion=(0.2, 0.3, 0.5, 0.5))
        state.graph.nodes[n1.id] = n1
        state.graph.nodes[n2.id] = n2
        result = op(state, threshold=0.5)
        assert len(result.graph.nodes) == 1
        assert list(result.graph.nodes.values())[0].text == "strong"

    def test_compress_operator(self):
        from cognitive_engine.operators.compress import CompressOperator
        op = CompressOperator()
        state = self._make_state()
        n1 = Node(text="root", type=NodeType.CLAIM, opinion=(0.9, 0.05, 0.05, 0.5))
        n2 = Node(text="child", type=NodeType.EVIDENCE, opinion=(0.7, 0.1, 0.2, 0.5))
        state.graph.nodes[n1.id] = n1
        state.graph.nodes[n2.id] = n2
        e = Edge(source_id=n1.id, target_id=n2.id, type=EdgeType.SUPPORTS)
        state.graph.edges[e.id] = e
        result = op(state)
        assert "compressed_chain" in result.metadata
        assert "compression_summary" in result.metadata

    def test_update_operator(self):
        from cognitive_engine.operators.update import UpdateOperator
        op = UpdateOperator()
        state = self._make_state()
        result = op(state, description="test update")
        assert len(result.history) == 1

    def test_merge_operator(self):
        from cognitive_engine.operators.merge import MergeOperator
        op = MergeOperator()
        state = self._make_state()
        state.graph.nodes[uuid4()] = Node(text="node1", type=NodeType.CLAIM)

        g2 = Graph()
        g2.nodes[uuid4()] = Node(text="node2", type=NodeType.EVIDENCE)
        g2.nodes[uuid4()] = Node(text="node1", type=NodeType.CLAIM, opinion=(0.8, 0.1, 0.1, 0.5))

        result = op(state, graphs=[g2])
        assert len(result.graph.nodes) == 2  # dedup

    def test_temporal_operator(self):
        from cognitive_engine.operators.temporal import TemporalOperator
        op = TemporalOperator()
        state = self._make_state()
        state.record("prev", "previous state")
        result = op(state)
        assert "temporal" in result.metadata

    def test_simulate_operator(self):
        from cognitive_engine.operators.simulate import SimulateOperator
        op = SimulateOperator()
        state = self._make_state()
        nid = uuid4()
        state.graph.nodes[nid] = Node(text="test", type=NodeType.CLAIM, opinion=(0.5, 0.3, 0.2, 0.5))
        result = op(state, modifications={str(nid): {"belief": 0.9}})
        assert "simulation" in result.metadata


# ============================================================
# ============================================================
# Phase 3: Schema Tests
# ============================================================

class TestSchemas:
    """Test domain schemas."""

    def test_legal_schema(self):
        from cognitive_engine.schemas.legal import LEGAL_SCHEMA
        assert LEGAL_SCHEMA.name == "legal"
        assert LEGAL_SCHEMA.merge_strategy == "keep_both"

    def test_research_schema(self):
        from cognitive_engine.schemas.research import RESEARCH_SCHEMA
        assert RESEARCH_SCHEMA.name == "research"
        assert RESEARCH_SCHEMA.merge_strategy == "average"

    def test_debate_schema(self):
        from cognitive_engine.schemas.debate import DEBATE_SCHEMA
        assert DEBATE_SCHEMA.name == "debate"
        assert DEBATE_SCHEMA.merge_strategy == "keep_both"

    def test_get_schema(self):
        from cognitive_engine.schemas import get_schema
        schema = get_schema("legal")
        assert schema.name == "legal"

    def test_get_unknown_schema(self):
        from cognitive_engine.schemas import get_schema
        with pytest.raises(ValueError):
            get_schema("unknown")



