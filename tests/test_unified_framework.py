"""Tests for unified framework — coefficients, master equation, reasoner."""
import json
from pathlib import Path
from uuid import uuid4
from cognitive_engine.unified.coefficients import Coefficients
from cognitive_engine.unified.master_equation import master_equation, compute_support_sum, compute_attack_sum
from cognitive_engine.unified.objective import count_violations, compute_objective
from cognitive_engine.unified.reasoner import UnifiedReasoner, ReasoningResult
from cognitive_engine.core.models import Graph, Node, Edge, EdgeType, NodeType
from cognitive_engine.levels.base import ReasoningContext


# ── Coefficients ──────────────────────────────────────────────────
class TestCoefficients:
    def test_defaults(self):
        c = Coefficients()
        assert c.alpha == 0.3
        assert c.beta == 0.3
        assert c.gamma == 0.2
        assert c.delta == 0.2
        assert abs(c.alpha + c.beta + c.gamma + c.delta - 1.0) < 1e-6

    def test_save_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "coeffs.json"
        c = Coefficients(alpha=0.25, beta=0.35, gamma=0.25, delta=0.15)
        c.save(path)
        c2 = Coefficients.load(path)
        assert c2.alpha == 0.25
        assert c2.beta == 0.35
        assert c2.gamma == 0.25
        assert c2.delta == 0.15
        assert c2.level7_max_iterations == 100


# ── Master Equation ───────────────────────────────────────────────
def _make_graph():
    n1 = Node(id=uuid4(), text="A", type=NodeType.CLAIM, opinion=(0.7, 0.2, 0.1, 0.5))
    n2 = Node(id=uuid4(), text="B", type=NodeType.EVIDENCE, opinion=(0.5, 0.3, 0.2, 0.5))
    e1 = Edge(id=uuid4(), source_id=n2.id, target_id=n1.id, type=EdgeType.SUPPORTS, warrant=((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5)))
    g = Graph(nodes={n1.id: n1, n2.id: n2}, edges=[e1])
    return g


class TestMasterEquation:
    def test_basic(self):
        g = _make_graph()
        coeffs = Coefficients()
        beliefs = {nid: 0.8 for nid in g.nodes}
        probabilities = {nid: 0.7 for nid in g.nodes}
        logic_consistency = {nid: 0.9 for nid in g.nodes}
        attack_strengths = {nid: 0.1 for nid in g.nodes}
        violations = {nid: 0 for nid in g.nodes}
        result = master_equation(g, beliefs, probabilities, logic_consistency, attack_strengths, violations, coeffs)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result.values())


# ── Support/Attack Sums ───────────────────────────────────────────
class TestSupportAttackSums:
    def test_support_sum(self):
        g = _make_graph()
        nid = list(g.nodes.keys())[0]
        result = compute_support_sum(nid, g, {list(g.nodes.keys())[1]: 0.8})
        assert result > 0.0


# ── Violations ────────────────────────────────────────────────────
class TestViolations:
    def test_no_violations(self):
        g = _make_graph()
        violations = count_violations(g)
        assert isinstance(violations, dict)


# ── Objective ─────────────────────────────────────────────────────
class TestObjective:
    def test_basic(self):
        g = _make_graph()
        truth_values = {nid: 0.8 for nid in g.nodes}
        violations = {nid: 0 for nid in g.nodes}
        coeffs = Coefficients()
        objective = compute_objective(g, truth_values, violations, coeffs)
        assert isinstance(objective, float)


# ── Reasoner ──────────────────────────────────────────────────────
class TestUnifiedReasoner:
    def test_reason_returns_result(self):
        g = _make_graph()
        reasoner = UnifiedReasoner()
        result = reasoner.reason(g)
        assert isinstance(result, ReasoningResult)
        assert len(result.beliefs) == 2
        assert result.objective != 0.0 or len(result.beliefs) == 0  # objective computed

    def test_beliefs_are_float(self):
        g = _make_graph()
        reasoner = UnifiedReasoner()
        result = reasoner.reason(g)
        for node_id, belief in result.beliefs.items():
            assert isinstance(belief, float)
