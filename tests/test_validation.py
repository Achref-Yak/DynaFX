"""Tests for model validation."""

from dynafx.system.dsl import parse_sysd


def test_valid_simple_model():
    m = parse_sysd("""
model 'Good'
  dt 1
  from 0 to 10
  stock 'S': 100
    - 'Out': S * 0.1
""")
    vr = m.validate()
    assert vr.is_valid


def test_unknown_identifier():
    m = parse_sysd("""
model 'Bad'
  dt 1
  from 0 to 10
  stock 'S': 100
    - 'Out': FooBar
""")
    vr = m.validate()
    assert not vr.is_valid
    assert any("FooBar" in e.message for e in vr.errors)


def test_unknown_identifier_in_aux():
    m = parse_sysd("""
model 'BadAux'
  dt 1
  from 0 to 10
  stock 'S': 100
    - 'Out': rate
  aux 'rate': S * Foo
""")
    vr = m.validate()
    assert not vr.is_valid
    assert any("Foo" in e.message for e in vr.errors)


def test_unknown_identifier_with_params():
    """Parameter names should be valid when passed to validate()."""
    m = parse_sysd("""
model 'Param'
  dt 1
  from 0 to 10
  stock 'S': 100
    - 'Out': beta * S
""")
    vr = m.validate(params={"beta"})
    assert vr.is_valid


def test_flow_conservation_warning():
    m = parse_sysd("""
model 'Unbalanced'
  dt 1
  from 0 to 10
  stock 'A': 100
    - 'Flow1': 10
""")
    vr = m.validate()
    assert len(vr.warnings) >= 1
    assert any("only one side" in w.message for w in vr.warnings)


def test_flow_conservation_error():
    m = parse_sysd("""
model 'Overbalanced'
  dt 1
  from 0 to 10
  stock 'A': 100
    - 'F': 10
  stock 'B': 100
    + 'F': 10
  stock 'C': 100
    - 'F': 10
""")
    vr = m.validate()
    assert not vr.is_valid
    assert any("3 times" in e.message for e in vr.errors)


def test_non_negativity_warning():
    m = parse_sysd("""
model 'NegRisk'
  dt 1
  from 0 to 10
  stock 'S': 100
    - 'Out': S * 0.2
""")
    vr = m.validate()
    assert any("negative" in w.message for w in vr.warnings)


def test_zero_stock_divisor_warning():
    m = parse_sysd("""
model 'DivZero'
  dt 1
  from 0 to 10
  stock 'S': 0
    + 'In': 10
  stock 'T': 100
    - 'Out': T / S
""")
    vr = m.validate()
    assert any("divisor" in w.message for w in vr.warnings)


def test_table_t_as_builtin():
    m = parse_sysd("""
model 'Tbl'
  dt 1
  from 0 to 10
  table 'demand'
    x: [0, 5]
    y: [10, 20]
  stock 'S': 100
    - 'Out': demand(t)
""")
    vr = m.validate()
    assert vr.is_valid


def test_validation_result_merge():
    from dynafx.system.dsl import ValidationResult, ValidationIssue
    r1 = ValidationResult()
    r1.errors.append(ValidationIssue("error", "e1"))
    r2 = ValidationResult()
    r2.warnings.append(ValidationIssue("warning", "w1"))
    r1.merge(r2)
    assert len(r1.errors) == 1
    assert len(r1.warnings) == 1
