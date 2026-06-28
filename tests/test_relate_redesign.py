"""Test the Relate operator redesign: conflict resolution via provenance-weighted Dung."""

from dynafx.core.concept import ConceptDef, TemporalSemantics, ConflictType, ConceptRegistry
from dynafx.core.models import EdgeType, Graph, Node, NodeType, Opinion, Edge
from dynafx.core.state import State
from dynafx.operators.relate import RelateOperator


def _make_state_with_nodes(nodes: list[tuple[str, str, dict]]) -> tuple[State, dict]:
    """Helper: create a State with given nodes.

    Each node is (node_type, text, metadata_override).
    Returns (state, node_id_map).
    """
    state = State(graph=Graph())
    node_ids = {}
    for node_type_str, text, meta in nodes:
        nt = NodeType[node_type_str]
        node = Node(type=nt, text=text, metadata=dict(meta))
        if meta.get("belief"):
            node.opinion = Opinion(belief=meta["belief"])
        state.graph.nodes[node.id] = node
        node_ids[text] = node.id
    return state, node_ids


class TestNumericConflictResolution:
    """Alice and Bob report conflicting metrics: 40% vs 60% efficiency."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_conflicting_values_create_contradicts_edge(self):
        """Two nodes with conflicting numeric values should create CONTRADICTS edge."""
        state, ids = _make_state_with_nodes([
            ("OBSERVATION", "efficiency is 20%", {}),
            ("OBSERVATION", "efficiency is 80%", {}),
        ])
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        contradict_edges = [e for e in edges if e.type == EdgeType.CONTRADICTS]
        assert len(contradict_edges) >= 1, f"Expected CONTRADICTS edge, got {edges}"
        assert contradict_edges[0].weight > 0.5, f"Weight should be >0.5, got {contradict_edges[0].weight}"

    def test_non_conflicting_values_no_contradicts(self):
        """Two nodes with same-direction values should not create CONTRADICTS."""
        state, ids = _make_state_with_nodes([
            ("OBSERVATION", "System efficiency is 80%", {}),
            ("OBSERVATION", "System efficiency is 85%", {}),
        ])
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        contradict_edges = [e for e in edges if e.type == EdgeType.CONTRADICTS]
        # These should not be contradictory (85/80 ratio > 0.5)
        assert len(contradict_edges) == 0, f"Unexpected CONTRADICTS edge: {contradict_edges}"

    def test_different_unit_no_false_contradicts(self):
        """Different units should NOT create CONTRADICTS edge."""
        state, ids = _make_state_with_nodes([
            ("OBSERVATION", "System efficiency is 40%", {}),
            ("OBSERVATION", "System throughput is 80%", {}),
        ])
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        contradict_edges = [e for e in edges if e.type == EdgeType.CONTRADICTS]
        assert len(contradict_edges) == 0, f"Unexpected CONTRADICTS with different units: {contradict_edges}"


class TestSameConceptConflict:
    """Two nodes asserting different values for the same concept (e.g., PERSON_NAME)."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_different_values_for_same_concept_creates_rebuts(self):
        """Two nodes with different values for same concept → REBUTS."""
        state = State(graph=Graph())
        state.concepts = ConceptRegistry({
            c.name: c for c in [
                ConceptDef(
                    name="PERSON_NAME",
                    temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                    cardinality="single",
                    provenance_weight=0.8,
                )
            ]
        })
        node_a = Node(type=NodeType.AGENT, text="My name Alice", metadata={"concept": "PERSON_NAME"})
        node_b = Node(type=NodeType.AGENT, text="My name is Bob", metadata={"concept": "PERSON_NAME"})
        state.graph.nodes[node_a.id] = node_a
        state.graph.nodes[node_b.id] = node_b

        result = self.operator(state)
        edges = list(result.graph.edges.values())
        rebuts_edges = [e for e in edges if e.type == EdgeType.REBUTS]
        assert len(rebuts_edges) >= 1, f"Expected REBUTS edge, got {edges}"
        # Weight should reflect provenance_weight of PERSON_NAME (0.8)
        assert rebuts_edges[0].weight == 0.8, f"Weight should be 0.8, got {rebuts_edges[0].weight}"

    def test_same_value_for_same_concept_creates_supports(self):
        """Two nodes with same value for same concept → SUPPORTS."""
        state = State(graph=Graph())
        state.concepts = ConceptRegistry({
            c.name: c for c in [
                ConceptDef(
                    name="PERSON_NAME",
                    temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                    cardinality="single",
                    provenance_weight=0.8,
                )
            ]
        })
        node_a = Node(type=NodeType.AGENT, text="My name Alice", metadata={"concept": "PERSON_NAME"})
        node_b = Node(type=NodeType.AGENT, text="Her name is Alice", metadata={"concept": "PERSON_NAME"})
        state.graph.nodes[node_a.id] = node_a
        state.graph.nodes[node_b.id] = node_b

        result = self.operator(state)
        edges = list(result.graph.edges.values())
        supports_edges = [e for e in edges if e.type == EdgeType.SUPPORTS]
        assert len(supports_edges) >= 1, f"Expected SUPPORTS edge, got {edges}"


class TestThreeWayConflict:
    """Three nodes with conflicting values: A=10%, B=80%, C=50%."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_three_way_contradicts(self):
        """All three nodes contradict each other."""
        state, ids = _make_state_with_nodes([
            ("OBSERVATION", "success rate is 10%", {}),
            ("OBSERVATION", "success rate is 80%", {}),
            ("OBSERVATION", "success rate is 50%", {}),
        ])
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        contradict_edges = [e for e in edges if e.type == EdgeType.CONTRADICTS]
        # At least 2 pairs should be contradictory (10 vs 80, 10 vs 50)
        assert len(contradict_edges) >= 2, f"Expected >=2 CONTRADICTS edges, got {len(contradict_edges)}"


class TestExtractionLayerEdgesPreserved:
    """Extraction-layer edges should be preserved, not duplicated."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_existing_edges_not_duplicated(self):
        """Edges already in graph should be preserved with fused opinions."""
        state, ids = _make_state_with_nodes([
            ("AGENT", "Alice manages the project", {}),
            ("PROCESS", "The project runs smoothly", {}),
        ])
        # Manually add an extraction-layer edge
        node_ids = list(state.graph.nodes.keys())
        edge = Edge(
            source_id=node_ids[0],
            target_id=node_ids[1],
            type=EdgeType.USES,
            weight=0.9,
            opinion=Opinion(belief=0.45, disbelief=0.05, uncertainty=0.5, prior=0.5),
        )
        state.graph.edges[edge.id] = edge
        edge_key = (edge.source_id, edge.target_id, edge.type)

        result = self.operator(state)
        # Should still have exactly one edge of this type
        uses_edges = [
            e for e in result.graph.edges.values()
            if (e.source_id, e.target_id, e.type) == edge_key
        ]
        assert len(uses_edges) == 1, f"Expected 1 USES edge (no duplicate), got {len(uses_edges)}"
        # Opinion should have been fused
        fused_op = uses_edges[0].opinion
        assert fused_op.belief > 0.0, "Fused opinion should have non-zero belief"

    def test_extraction_layer_edge_becomes_source_for_dung(self):
        """An extraction-layer edge is preserved and fed into Dung's post-processing."""
        state = State(graph=Graph())
        state.concepts = ConceptRegistry({
            c.name: c for c in [
                ConceptDef(
                    name="PERSON_NAME",
                    temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                    cardinality="single",
                    provenance_weight=0.8,
                )
            ]
        })
        node_a = Node(type=NodeType.AGENT, text="My name Alice", metadata={"concept": "PERSON_NAME"})
        node_b = Node(type=NodeType.AGENT, text="My name is Bob", metadata={"concept": "PERSON_NAME"})
        state.graph.nodes[node_a.id] = node_a
        state.graph.nodes[node_b.id] = node_b

        # Simulate extraction-layer CITES edge
        edge = Edge(
            source_id=node_a.id,
            target_id=node_b.id,
            type=EdgeType.CITES,
            weight=0.8,
            opinion=Opinion(belief=0.4, disbelief=0.1, uncertainty=0.5, prior=0.5),
        )
        state.graph.edges[edge.id] = edge

        # Run operator — should not duplicate CITES, but should add REBUTS
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        cites_edges = [e for e in edges if e.type == EdgeType.CITES]
        rebuts_edges = [e for e in edges if e.type == EdgeType.REBUTS]

        assert len(cites_edges) == 1, f"Expected 1 CITES (preserved), got {len(cites_edges)}"
        assert len(rebuts_edges) >= 1, f"Expected REBUTS from concept axiom, got {len(rebuts_edges)}"


class TestDungPostProcessing:
    """Dung's Argumentation should filter non-acceptable arguments."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_rebuts_edge_has_provenance_weight(self):
        """REBUTS edges from concept axiom should have provenance_weight in opinion.
        Bidirectional: both A→B and B→A are created.
        Dung's post-processing: in a symmetric attack, one node is accepted,
        the other rejected. The rejected node's outgoing REBUTS is down-weighted
        by ×0.3."""
        state = State(graph=Graph())
        state.concepts = ConceptRegistry({
            c.name: c for c in [
                ConceptDef(
                    name="EMAIL",
                    temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                    cardinality="single",
                    provenance_weight=0.9,
                )
            ]
        })
        node_a = Node(type=NodeType.AGENT, text="Email is alice@co.com", metadata={"concept": "EMAIL"})
        node_b = Node(type=NodeType.AGENT, text="Email is bob@co.com", metadata={"concept": "EMAIL"})
        state.graph.nodes[node_a.id] = node_a
        state.graph.nodes[node_b.id] = node_b

        result = self.operator(state)
        edges = list(result.graph.edges.values())
        rebuts_edges = [e for e in edges if e.type == EdgeType.REBUTS]
        # Bidirectional: both A→B and B→A
        assert len(rebuts_edges) == 2, f"Expected 2 REBUTS (bidirectional), got {len(rebuts_edges)}"
        weights = sorted([e.weight for e in rebuts_edges])
        # One edge at raw provenance weight (0.9), one down-weighted by Dung (0.9×0.3=0.27)
        assert weights[0] == 0.27, f"Down-weighted edge should be 0.27, got {weights[0]}"
        assert weights[1] == 0.9, f"Accepted edge should be 0.9, got {weights[1]}"
        # Opinion belief should be >0 for both
        for e in rebuts_edges:
            assert e.opinion.belief > 0.0, f"Opinion belief should be >0, got {e.opinion.belief}"


class TestAppendOnlyConceptNoConflict:
    """APPEND_ONLY concepts should not trigger conflict."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_append_only_no_rebuts(self):
        """Two nodes with same concept + APPEND_ONLY → NO_CONFLICT, no REBUTS."""
        state = State(graph=Graph())
        state.concepts = ConceptRegistry({
            c.name: c for c in [
                ConceptDef(
                    name="VERSION",
                    temporal_semantics=TemporalSemantics.APPEND_ONLY,
                    cardinality="single",
                )
            ]
        })
        node_a = Node(type=NodeType.ENTITY, text="Version 1.0 released", metadata={"concept": "VERSION"})
        node_b = Node(type=NodeType.ENTITY, text="Version 1.1 released", metadata={"concept": "VERSION"})
        state.graph.nodes[node_a.id] = node_a
        state.graph.nodes[node_b.id] = node_b

        result = self.operator(state)
        edges = list(result.graph.edges.values())
        rebuts_edges = [e for e in edges if e.type == EdgeType.REBUTS]
        contradict_edges = [e for e in edges if e.type == EdgeType.CONTRADICTS]
        assert len(rebuts_edges) == 0, f"Unexpected REBUTS for APPEND_ONLY: {rebuts_edges}"
        assert len(contradict_edges) == 0, f"Unexpected CONTRADICTS for APPEND_ONLY: {contradict_edges}"


class TestParentConceptConflict:
    """Different concepts with same parent should create REBUTS edge."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_sibling_concepts_same_parent_rebuts(self):
        """Two nodes with concepts sharing a parent → bidirectional REBUTS edges.
        Dung's post-processing: one edge at raw weight, one down-weighted by ×0.3."""
        state = State(graph=Graph())
        state.concepts = ConceptRegistry({
            c.name: c for c in [
                ConceptDef(
                    name="TEMPERATURE",
                    temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                    cardinality="single",
                    provenance_weight=0.6,
                    parent="OPERATING_CONDITION",
                ),
                ConceptDef(
                    name="PRESSURE",
                    temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                    cardinality="single",
                    provenance_weight=0.7,
                    parent="OPERATING_CONDITION",
                ),
                ConceptDef(
                    name="OPERATING_CONDITION",
                    temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                    cardinality="single",
                ),
            ]
        })

        node_a = Node(type=NodeType.ENTITY, text="Temperature is 100°C", metadata={"concept": "TEMPERATURE"})
        node_b = Node(type=NodeType.ENTITY, text="Pressure is 200 kPa", metadata={"concept": "PRESSURE"})
        state.graph.nodes[node_a.id] = node_a
        state.graph.nodes[node_b.id] = node_b

        result = self.operator(state)
        edges = list(result.graph.edges.values())
        rebuts_edges = [e for e in edges if e.type == EdgeType.REBUTS]
        # Bidirectional
        assert len(rebuts_edges) == 2, f"Expected 2 REBUTS (bidirectional), got {len(rebuts_edges)}"
        weights = sorted([e.weight for e in rebuts_edges])
        # Weight = max(0.6, 0.7) = 0.7; one edge raw (0.7), one down-weighted (0.7×0.3=0.21)
        assert weights[0] == 0.21, f"Down-weighted edge should be 0.21, got {weights[0]}"
        assert weights[1] == 0.7, f"Accepted edge should be 0.7, got {weights[1]}"


class TestEdgePriority:
    """World-model edges should have higher priority than argumentation edges."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_world_model_edge_wins_over_argumentation(self):
        """When both world-model and argumentation signals fire, world-model wins."""
        state, ids = _make_state_with_nodes([
            ("AGENT", "Alice causes the system to run efficiently", {}),
            ("PROCESS", "System runs efficiently", {}),
        ])
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        # CAUSES (world-model, priority=100) should win over SUPPORTS (arg, priority=26)
        types = [e.type for e in edges]
        assert EdgeType.CAUSES in types, f"Expected CAUSES edge, got {types}"
        if EdgeType.SUPPORTS in types:
            causes_edge = next(e for e in edges if e.type == EdgeType.CAUSES)
            supports_edge = next(e for e in edges if e.type == EdgeType.SUPPORTS)
            assert causes_edge.weight >= supports_edge.weight, "CAUSES should have higher weight"


class TestSLInvariant:
    """SL invariant: belief + disbelief + uncertainty == 1.0 for all opinions."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_evaluate_pair_opinions_satisfy_sl_invariant(self):
        """Every Opinion created by _evaluate_pair must satisfy b+d+u=1.0."""
        state, ids = _make_state_with_nodes([
            ("OBSERVATION", "efficiency is 20%", {}),
            ("OBSERVATION", "efficiency is 80%", {}),
        ])
        result = self.operator(state)
        for edge in result.graph.edges.values():
            if edge.opinion:
                total = edge.opinion.belief + edge.opinion.disbelief + edge.opinion.uncertainty
                assert abs(total - 1.0) < 1e-9, (
                    f"SL invariant violated for {edge.type.name}: "
                    f"b={edge.opinion.belief} + d={edge.opinion.disbelief} + "
                    f"u={edge.opinion.uncertainty} = {total} != 1.0"
                )

    def test_concept_driven_opinions_satisfy_sl_invariant(self):
        """Opinions from concept-driven edges must also satisfy b+d+u=1.0."""
        state = State(graph=Graph())
        state.concepts = ConceptRegistry({
            c.name: c for c in [
                ConceptDef(
                    name="PERSON_NAME",
                    temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                    cardinality="single",
                    provenance_weight=0.8,
                )
            ]
        })
        node_a = Node(type=NodeType.AGENT, text="My name Alice", metadata={"concept": "PERSON_NAME"})
        node_b = Node(type=NodeType.AGENT, text="My name is Bob", metadata={"concept": "PERSON_NAME"})
        state.graph.nodes[node_a.id] = node_a
        state.graph.nodes[node_b.id] = node_b

        result = self.operator(state)
        for edge in result.graph.edges.values():
            if edge.opinion:
                total = edge.opinion.belief + edge.opinion.disbelief + edge.opinion.uncertainty
                assert abs(total - 1.0) < 1e-9, (
                    f"SL invariant violated for {edge.type.name}: "
                    f"b={edge.opinion.belief} + d={edge.opinion.disbelief} + "
                    f"u={edge.opinion.uncertainty} = {total} != 1.0"
                )


class TestCrossAxiomSuppression:
    """Multiple axioms should produce multiple edges for the same pair."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_lexical_and_numeric_both_fire(self):
        """A pair with both causal signal AND numeric opposition should produce
        both CAUSES and CONTRADICTS edges (not just the winner)."""
        state, ids = _make_state_with_nodes([
            ("OBSERVATION", "the causes efficiency dropped to 20%", {}),
            ("OBSERVATION", "the causes efficiency is 80%", {}),
        ])
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        types = [e.type for e in edges]
        # Both CAUSES (lexical) and CONTRADICTS (numeric) should be present
        assert EdgeType.CAUSES in types, f"Expected CAUSES edge, got {types}"
        assert EdgeType.CONTRADICTS in types, f"Expected CONTRADICTS edge, got {types}"

    def test_lexical_and_type_both_fire(self):
        """A pair with both a causal lexical signal AND a type-compatible pattern
        should produce edges from both axioms."""
        state, ids = _make_state_with_nodes([
            ("AGENT", "Alice causes the system to run", {}),
            ("PROCESS", "system runs smoothly", {}),
        ])
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        types = [e.type for e in edges]
        # CAUSES from lexical + ENABLES from type pattern should coexist
        assert EdgeType.CAUSES in types, f"Expected CAUSES edge, got {types}"
        assert EdgeType.ENABLES in types, f"Expected ENABLES edge, got {types}"


class TestDirectionality:
    """Symmetric contradictions should create edges in both directions."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_numeric_contradiction_is_bidirectional(self):
        """Numeric opposition should create CONTRADICTS in both directions."""
        state, ids = _make_state_with_nodes([
            ("OBSERVATION", "success rate is 10%", {}),
            ("OBSERVATION", "success rate is 80%", {}),
        ])
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        contradict_edges = [e for e in edges if e.type == EdgeType.CONTRADICTS]
        assert len(contradict_edges) == 2, f"Expected 2 CONTRADICTS (bidirectional), got {len(contradict_edges)}"
        src_ids = {e.source_id for e in contradict_edges}
        tgt_ids = {e.target_id for e in contradict_edges}
        # Both directions present: A→B and B→A
        node_ids = list(ids.values())
        assert node_ids[0] in src_ids and node_ids[1] in tgt_ids, "Missing A→B direction"
        assert node_ids[1] in src_ids and node_ids[0] in tgt_ids, "Missing B→A direction"

    def test_same_concept_rebuts_is_bidirectional(self):
        """Same-concept value conflict should create REBUTS in both directions."""
        state = State(graph=Graph())
        state.concepts = ConceptRegistry({
            c.name: c for c in [
                ConceptDef(
                    name="PERSON_NAME",
                    temporal_semantics=TemporalSemantics.SUPERSEEDE_WITH_HISTORY,
                    cardinality="single",
                    provenance_weight=0.8,
                )
            ]
        })
        node_a = Node(type=NodeType.AGENT, text="My name Alice", metadata={"concept": "PERSON_NAME"})
        node_b = Node(type=NodeType.AGENT, text="My name is Bob", metadata={"concept": "PERSON_NAME"})
        state.graph.nodes[node_a.id] = node_a
        state.graph.nodes[node_b.id] = node_b

        result = self.operator(state)
        edges = list(result.graph.edges.values())
        rebuts_edges = [e for e in edges if e.type == EdgeType.REBUTS]
        assert len(rebuts_edges) == 2, f"Expected 2 REBUTS (bidirectional), got {len(rebuts_edges)}"
        src_ids = {e.source_id for e in rebuts_edges}
        assert node_a.id in src_ids and node_b.id in src_ids, "REBUTS should be bidirectional"


class TestThreeWayConflictDung:
    """Three-way conflict: Dung's should handle the cycle correctly."""

    def setup_method(self):
        self.operator = RelateOperator()

    def test_three_way_contradicts_bidirectional(self):
        """Three nodes with conflicting values should have bidirectional CONTRADICTS.
        Only pairs with ratio < 0.5 trigger: 10/50=0.2, 10/80=0.125 → 2 pairs.
        50/80=0.625 > 0.5 → no CONTRADICTS. So 2 pairs × 2 directions = 4 edges."""
        state, ids = _make_state_with_nodes([
            ("OBSERVATION", "success rate is 10%", {}),
            ("OBSERVATION", "success rate is 80%", {}),
            ("OBSERVATION", "success rate is 50%", {}),
        ])
        result = self.operator(state)
        edges = list(result.graph.edges.values())
        contradict_edges = [e for e in edges if e.type == EdgeType.CONTRADICTS]
        assert len(contradict_edges) == 4, f"Expected 4 CONTRADICTS (2 pairs × 2 dirs), got {len(contradict_edges)}"


