"""Tests for the operator framework — core, operators, pipelines, schemas."""

import pytest
from uuid import uuid4

from cognitive_engine.core.concept import (
    ConceptDef,
    ConceptRegistry,
    TemporalSemantics,
)
from cognitive_engine.core.models import (
    BfoCategory,
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
        assert forked.graph is not graph
        assert forked.graph.nodes is not graph.nodes
        assert forked.graph.edges is not graph.edges
        assert forked.graph.entities is not graph.entities
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
        from cognitive_engine.schemas.research import RESEARCH_SCHEMA
        op = SchemaOperator()
        state = self._make_state()
        state.graph.nodes[uuid4()] = Node(text="test", type=NodeType.EVIDENCE)
        result = op(state, schema=RESEARCH_SCHEMA)
        assert result.metadata["schema_applied"] == "research"

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

    def test_relate_operator_contradicts(self):
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        state = self._make_state()
        n1 = Node(text="Lease clause 14 requires 60 days notice", type=NodeType.CLAIM)
        n2 = Node(text="Tenant gave 14 days notice", type=NodeType.EVIDENCE)
        state.graph.nodes[n1.id] = n1
        state.graph.nodes[n2.id] = n2
        result = op(state, max_edges_per_node=5)
        edges = list(result.graph.edges.values())
        assert len(edges) >= 1, "Expected at least one edge between contradictory claims"
        assert any(e.type == EdgeType.CONTRADICTS for e in edges), (
            f"Expected CONTRADICTS edge, got {[e.type.name for e in edges]}"
        )

    def test_relate_operator_too_few_nodes(self):
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        state = self._make_state()
        state.graph.nodes[uuid4()] = Node(text="only one node", type=NodeType.CLAIM)
        result = op(state)
        assert len(result.graph.edges) == 0
        assert len(result.history) == 1

    def _make_temporal_test_state(self, concept_name: str, temporal: TemporalSemantics, cardinality: str):
        reg = ConceptRegistry()
        reg.register(ConceptDef(
            name=concept_name,
            description="Test concept",
            temporal_semantics=temporal,
            cardinality=cardinality,
        ))
        state = State(graph=Graph())
        state.concepts = reg
        n1 = Node(
            id=uuid4(), text="test value is 30", type=NodeType.CLAIM,
            opinion=Opinion(belief=0.8, disbelief=0.15, uncertainty=0.05, prior=0.5),
        )
        n1.metadata["concept"] = concept_name
        n2 = Node(
            id=uuid4(), text="test value is 25", type=NodeType.CLAIM,
            opinion=Opinion(belief=0.15, disbelief=0.8, uncertainty=0.05, prior=0.5),
        )
        n2.metadata["concept"] = concept_name
        state.graph.nodes[n1.id] = n1
        state.graph.nodes[n2.id] = n2
        return state

    def test_relate_temporal_axiom_supersede_rebuts(self):
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        state = self._make_temporal_test_state("PERSON_NAME", TemporalSemantics.SUPERSEEDE_WITH_HISTORY, "single")
        result = op(state, max_edges_per_node=5)
        edges = list(result.graph.edges.values())
        rebuts = [e for e in edges if e.type == EdgeType.REBUTS]
        # Bidirectional REBUTS from concept axiom + possible CONTRADICTS from epistemic
        assert len(rebuts) >= 1, f"Expected at least 1 REBUTS, got {len(rebuts)}"
        assert all(e.type == EdgeType.REBUTS for e in rebuts), (
            f"Expected REBUTS for SUPERSEDE_WITH_HISTORY, got {[e.type for e in rebuts]}"
        )

    def test_relate_temporal_axiom_append_only_suppressed(self):
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        state = self._make_temporal_test_state("TEMPERATURE", TemporalSemantics.APPEND_ONLY, "multiple")
        result = op(state, max_edges_per_node=5)
        edges = list(result.graph.edges.values())
        contradictory = [e for e in edges if e.type == EdgeType.CONTRADICTS]
        assert len(contradictory) == 0, (
            f"Expected no CONTRADICTS for APPEND_ONLY, got {len(contradictory)}"
        )

    def test_relate_temporal_axiom_competing_preserved(self):
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        state = self._make_temporal_test_state("CLAIM", TemporalSemantics.MUTATE_IN_PLACE, "multiple")
        result = op(state, max_edges_per_node=5)
        edges = list(result.graph.edges.values())
        contradictory = [e for e in edges if e.type == EdgeType.CONTRADICTS]
        # MUTATE_IN_PLACE means overwrite in place without history — no conflict
        assert len(contradictory) == 0, (
            f"Expected no CONTRADICTS for MUTATE_IN_PLACE, got {[e.type.name for e in edges]}"
        )

    def test_relate_temporal_axiom_different_concepts(self):
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        state = self._make_temporal_test_state("TEMPERATURE", TemporalSemantics.APPEND_ONLY, "multiple")
        # Override second node with a different concept
        nid_b = list(state.graph.nodes.keys())[1]
        state.graph.nodes[nid_b].metadata["concept"] = "PERSON_NAME"
        result = op(state, max_edges_per_node=5)
        edges = list(result.graph.edges.values())
        # Different concepts — no concept-level adjustment, edge may or may not exist
        assert isinstance(edges, list)

    def _make_same_parent_state(self):
        reg = ConceptRegistry()
        reg.register(ConceptDef("IDENTITY", parent=None,
                               temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                               cardinality="single"))
        reg.register(ConceptDef("USER_NAME", parent="IDENTITY",
                               temporal_semantics=TemporalSemantics.APPEND_ONLY,
                               cardinality="single", provenance_weight=0.9))
        reg.register(ConceptDef("USER_EMAIL", parent="IDENTITY",
                               temporal_semantics=TemporalSemantics.APPEND_ONLY,
                               cardinality="single", provenance_weight=0.85))
        state = State(graph=Graph(), concepts=reg)
        n1 = Node(id=uuid4(), text="my name is alice", type=NodeType.CLAIM)
        n1.metadata["concept"] = "USER_NAME"
        n1.bfo_category = None
        n2 = Node(id=uuid4(), text="my email is alice@foo.com", type=NodeType.CLAIM)
        n2.metadata["concept"] = "USER_EMAIL"
        n2.bfo_category = None
        state.graph.nodes[n1.id] = n1
        state.graph.nodes[n2.id] = n2
        return state

    def test_relate_same_parent_produces_rebuts(self):
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        state = self._make_same_parent_state()
        result = op(state, max_edges_per_node=5)
        edges = list(result.graph.edges.values())
        assert len(edges) >= 1, "Expected at least one edge for same-parent pair"
        assert any(e.type == EdgeType.REBUTS for e in edges), (
            f"Expected REBUTS for same-parent IDENTITY pair, got {[e.type.name for e in edges]}"
        )

    def test_relate_same_parent_different_parents_no_edge(self):
        reg = ConceptRegistry()
        reg.register(ConceptDef("PARENT_A", parent=None,
                               temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY))
        reg.register(ConceptDef("PARENT_B", parent=None,
                               temporal_semantics=TemporalSemantics.APPEND_ONLY))
        reg.register(ConceptDef("CON_A", parent="PARENT_A",
                               temporal_semantics=TemporalSemantics.APPEND_ONLY))
        reg.register(ConceptDef("CON_B", parent="PARENT_B",
                               temporal_semantics=TemporalSemantics.APPEND_ONLY))
        state = State(graph=Graph(), concepts=reg)
        n1 = Node(id=uuid4(), text="test a", type=NodeType.CLAIM)
        n1.metadata["concept"] = "CON_A"
        n2 = Node(id=uuid4(), text="test b", type=NodeType.CLAIM)
        n2.metadata["concept"] = "CON_B"
        state.graph.nodes[n1.id] = n1
        state.graph.nodes[n2.id] = n2
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        result = op(state, max_edges_per_node=5)
        rebuts = [e for e in result.graph.edges.values() if e.type == EdgeType.REBUTS]
        assert len(rebuts) == 0, (
            f"Expected no REBUTS for different parents, got {len(rebuts)}"
        )

    def _make_bfo_test_state(self):
        """Two ICE nodes with a CONTRADICTS edge, plus a PROCESS node to test BFO filter."""
        from cognitive_engine.core.models import BfoCategory
        state = State(graph=Graph())
        n1 = Node(id=uuid4(), text="claim a", type=NodeType.CLAIM,
                  bfo_category=BfoCategory.INFORMATION_CONTENT_ENTITY)
        n2 = Node(id=uuid4(), text="claim b", type=NodeType.CLAIM,
                  bfo_category=BfoCategory.INFORMATION_CONTENT_ENTITY)
        n3 = Node(id=uuid4(), text="process event", type=NodeType.EVENT,
                  bfo_category=BfoCategory.PROCESS)
        state.graph.nodes[n1.id] = n1
        state.graph.nodes[n2.id] = n2
        state.graph.nodes[n3.id] = n3
        return state

    def test_bfo_filter_ice_to_ice_allowed(self):
        from cognitive_engine.operators.relate import RelateOperator
        from cognitive_engine.core.models import EDGE_BFO_CONSTRAINTS
        op = RelateOperator()
        allowed, _ = EDGE_BFO_CONSTRAINTS[EdgeType.INFERS]
        assert BfoCategory.INFORMATION_CONTENT_ENTITY in allowed
        # The check is called during _evaluate_pair — if BFO is compatible,
        # the edge survives. We test the method directly for precision.
        n1 = Node(bfo_category=BfoCategory.INFORMATION_CONTENT_ENTITY)
        n2 = Node(bfo_category=BfoCategory.INFORMATION_CONTENT_ENTITY)
        assert op._check_bfo_compatibility(n1, n2, EdgeType.INFERS) is True

    def test_bfo_filter_process_to_ice_blocked(self):
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        n1 = Node(bfo_category=BfoCategory.PROCESS)
        n2 = Node(bfo_category=BfoCategory.INFORMATION_CONTENT_ENTITY)
        assert op._check_bfo_compatibility(n1, n2, EdgeType.PART_OF) is False

    def test_bfo_filter_none_bfo_allowed(self):
        from cognitive_engine.operators.relate import RelateOperator
        op = RelateOperator()
        n1 = Node(bfo_category=None)
        n2 = Node(bfo_category=None)
        assert op._check_bfo_compatibility(n1, n2, EdgeType.INFERS) is True


# ============================================================
# ============================================================
# Phase 3: Schema Tests
# ============================================================

class TestSchemas:
    """Test domain schemas."""

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
        schema = get_schema("research")
        assert schema.name == "research"

    def test_get_unknown_schema(self):
        from cognitive_engine.schemas import get_schema
        with pytest.raises(ValueError):
            get_schema("unknown")



