import pytest

from cognitive_engine.sl_operators import (
    conjunction,
    disjunction,
    cumulative_fusion,
    conditional_deduction,
    projected_probability,
    dirichlet_strength,
    _clamp,
)


def test_projected_probability():
    assert projected_probability((1.0, 0.0, 0.0, 0.5)) == pytest.approx(1.0)
    assert projected_probability((0.0, 1.0, 0.0, 0.5)) == pytest.approx(0.0)
    assert projected_probability((0.0, 0.0, 1.0, 0.5)) == pytest.approx(0.5)
    assert projected_probability((0.8, 0.1, 0.1, 0.5)) == pytest.approx(0.85)


def test_dirichlet_strength():
    assert dirichlet_strength((0.8, 0.1, 0.1, 0.5)) == pytest.approx(9.0)
    assert dirichlet_strength((0.0, 0.0, 1.0, 0.5)) == pytest.approx(0.0)


def test_conjunction_sums_to_one():
    result = conjunction((0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5))
    assert sum(result[:3]) == pytest.approx(1.0)


def test_disjunction_sums_to_one():
    result = disjunction((0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5))
    assert sum(result[:3]) == pytest.approx(1.0)


def test_cumulative_fusion_reduces_uncertainty():
    result = cumulative_fusion((0.8, 0.1, 0.1, 0.5), (0.7, 0.2, 0.1, 0.5))
    assert sum(result[:3]) == pytest.approx(1.0)
    assert result[2] < 0.1


def test_cumulative_fusion_certain():
    result = cumulative_fusion((1.0, 0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.5))
    assert result[0] == pytest.approx(1.0)
    assert result[2] == pytest.approx(0.0)
    assert sum(result[:3]) == pytest.approx(1.0)


def test_conditional_deduction_properties():
    omega_p = (0.8, 0.1, 0.1, 0.5)
    warrant = ((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5))
    result = conditional_deduction(omega_p, warrant)
    assert sum(result[:3]) == pytest.approx(1.0)
    assert result[0] > result[1]


def test_conditional_deduction_preserves_uncertainty():
    omega_p = (0.8, 0.1, 0.1, 0.5)
    warrant = ((0.9, 0.05, 0.05, 0.5), (0.0, 1.0, 0.0, 0.5))
    result = conditional_deduction(omega_p, warrant)
    assert result[2] >= 0.1


def test_clamp_normalizes():
    result = _clamp((0.9, 0.3, 0.1, 0.5))
    assert sum(result[:3]) == pytest.approx(1.0)


def test_clamp_zero_total():
    result = _clamp((0.0, 0.0, 0.0, 0.5))
    assert result[2] == 1.0


def test_conjunction_lowers_belief():
    result = conjunction((0.8, 0.1, 0.1, 0.5), (0.5, 0.3, 0.2, 0.5))
    assert result[0] < 0.8


def test_disjunction_raises_belief():
    result = disjunction((0.8, 0.1, 0.1, 0.5), (0.5, 0.3, 0.2, 0.5))
    assert result[0] > 0.8


def test_compute_opinions_with_new_node_types():
    from cognitive_engine.sl_operators import compute_opinions
    from cognitive_engine.config import Priors

    priors = Priors()
    assert "COUNTERCLAIM" in priors.source_type_map
    assert "AXIOM" in priors.source_type_map
    assert "FALLACY" in priors.source_type_map
    assert "JUSTIFICATION" in priors.source_type_map
    assert "ATTACKS" in priors.edge_warrants
    assert "REBUTS" in priors.edge_warrants
