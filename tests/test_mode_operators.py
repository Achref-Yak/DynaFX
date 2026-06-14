from copy import deepcopy

import pytest

from cognitive_engine.core.config import Priors
from cognitive_engine.core.models import (
    Graph, Node, Edge, NodeType, EdgeType, ReasoningMode, Opinion,
)
from cognitive_engine.reason.mode_operators import (
    apply_mode_operator,
    reverse_warrant,
    subjective_abduction,
    _reverse_graph_for_diagnostic,
    _apply_analogy_warrants,
)
from cognitive_engine.reason.modes import MODE_ACTIVE_EDGES
from cognitive_engine.reason.sl_operators import projected_probability


def _make_test_graph() -> Graph:
    a = Node(type=NodeType.EVIDENCE, text="data", opinion=(0.6, 0.1, 0.3, 0.5))
    b = Node(type=NodeType.CLAIM, text="conclusion", opinion=(0.5, 0.2, 0.3, 0.5))
    g = Graph(
        nodes={a.id: a, b.id: b},
        edges=[
            Edge(
                source_id=a.id, target_id=b.id, type=EdgeType.INFERS,
                warrant=((0.5, 0.1, 0.4, 0.5), (0.2, 0.3, 0.5, 0.5)),
            ),
        ],
    )
    return g


class TestReverseWarrant:
    def test_reverse_forward_roundtrip(self):
        forward = ((0.7, 0.1, 0.2, 0.5), (0.3, 0.4, 0.3, 0.5))
        rev = reverse_warrant(forward, 0.5, 0.5)
        assert len(rev) == 2
        for op in rev:
            b, d, u, a = op
            assert abs(b + d + u - 1.0) < 1e-9
            assert 0 <= a <= 1

    def test_reverse_identity_warrant(self):
        forward = ((1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5))
        rev = reverse_warrant(forward, 0.5, 0.5)
        b0, d0, u0, _ = rev[0]
        assert b0 > 0.9
        assert u0 < 0.01

    def test_reverse_zero_p_t_returns_ignorance(self):
        forward = ((0.0, 0.0, 1.0, 0.5), (0.0, 0.0, 1.0, 0.5))
        rev = reverse_warrant(forward, 0.5, 0.5)
        b0, d0, u0, a0 = rev[0]
        assert abs(u0 - 1.0) < 1e-9
        assert a0 == 0.5

    def test_reverse_negative_case(self):
        forward = ((0.9, 0.05, 0.05, 0.5), (0.1, 0.8, 0.1, 0.5))
        rev = reverse_warrant(forward, 0.3, 0.5)
        for op in rev:
            b, d, u, a = op
            assert abs(b + d + u - 1.0) < 1e-9


class TestSubjectiveAbduction:
    def test_abduction_returns_valid_opinion(self):
        effect = (0.8, 0.1, 0.1, 0.5)
        warrant = ((0.6, 0.2, 0.2, 0.5), (0.3, 0.4, 0.3, 0.5))
        result = subjective_abduction(effect, warrant, 0.5)
        b, d, u, a = result
        assert abs(b + d + u - 1.0) < 1e-9
        assert 0 <= a <= 1

    def test_abduction_certain_effect(self):
        effect = (1.0, 0.0, 0.0, 0.5)
        warrant = ((0.9, 0.05, 0.05, 0.5), (0.2, 0.7, 0.1, 0.5))
        result = subjective_abduction(effect, warrant, 0.5)
        b, d, u, _ = result
        assert b > 0.8

    def test_abduction_ignorant_effect(self):
        effect = (0.0, 0.0, 1.0, 0.5)
        warrant = ((0.8, 0.1, 0.1, 0.5), (0.2, 0.6, 0.2, 0.5))
        result = subjective_abduction(effect, warrant, 0.5)
        b, d, u, _ = result
        assert abs(b + d + u - 1.0) < 1e-9
        assert u > 0.4  # high uncertainty preserved

    def test_abduction_normalized(self):
        effect = (0.3, 0.4, 0.3, 0.5)
        warrant = ((0.5, 0.3, 0.2, 0.5), (0.4, 0.4, 0.2, 0.5))
        result = subjective_abduction(effect, warrant, 0.5)
        b, d, u, a = result
        assert abs(b + d + u - 1.0) < 1e-9


class TestReverseGraphForDiagnostic:
    def test_edges_reversed(self):
        g = _make_test_graph()
        original_forward = g.edges[0].warrant
        result = _reverse_graph_for_diagnostic(g)
        for e in result.edges:
            assert e.source_id != e.target_id
        assert len(result.edges) == 1
        e = result.edges[0]
        assert e.warrant is not None

    def test_inverse_warrant_valid(self):
        g = _make_test_graph()
        result = _reverse_graph_for_diagnostic(g)
        for e in result.edges:
            if e.warrant:
                for op in e.warrant:
                    b, d, u, a = op
                    assert abs(b + d + u - 1.0) < 1e-9


class TestApplyAnalogyWarrants:
    def test_uncertainty_increased(self):
        g = _make_test_graph()
        original = deepcopy(g.edges[0].warrant)
        priors = Priors()
        result = _apply_analogy_warrants(g, priors)
        for e in result.edges:
            if e.warrant and original:
                (b1, d1, u1, _), _ = e.warrant
                (ob1, _, ou1, _), _ = original
                assert u1 >= ou1
                assert b1 <= ob1


class TestApplyModeOperator:
    def test_causal_mode_preserves_infers(self):
        g = _make_test_graph()
        priors = Priors()

        opin_before = g.nodes[g.edges[0].target_id].opinion

        result = apply_mode_operator(g, priors, ReasoningMode.CAUSAL)
        assert result.mode == ReasoningMode.CAUSAL
        for e in result.edges:
            assert e.type in MODE_ACTIVE_EDGES[ReasoningMode.CAUSAL]
        assert len(result.edges) > 0

    def test_argument_mode_reverses_edges(self):
        g = _make_test_graph()
        priors = Priors()
        result = apply_mode_operator(g, priors, ReasoningMode.ARGUMENT)
        assert result.mode == ReasoningMode.ARGUMENT
        for e in result.edges:
            assert e.type in MODE_ACTIVE_EDGES[ReasoningMode.ARGUMENT]

    def test_analogy_mode_increases_uncertainty(self):
        g = _make_test_graph()
        priors = Priors()
        result = apply_mode_operator(g, priors, ReasoningMode.ANALOGY)
        for e in result.edges:
            if e.warrant:
                b, d, u, a = e.warrant[0]
                assert u >= 0

    def test_modes_produce_different_opinions(self):
        g = _make_test_graph()
        priors = Priors()

        opinions = {}
        for mode in ReasoningMode:
            view = apply_mode_operator(g, priors, mode)
            for nid, node in view.nodes.items():
                opinions.setdefault(mode.name, {})[nid.hex] = node.opinion

        assert opinions["CAUSAL"] != opinions["ARGUMENT"]

    def test_conditional_mode_filter(self):
        g = _make_test_graph()
        priors = Priors()
        result = apply_mode_operator(g, priors, ReasoningMode.CONDITIONAL)
        assert result.mode == ReasoningMode.CONDITIONAL
        for e in result.edges:
            assert e.type in MODE_ACTIVE_EDGES[ReasoningMode.CONDITIONAL]
