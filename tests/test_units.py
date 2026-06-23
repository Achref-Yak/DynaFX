"""Tests for units checking — Phase 5."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TMPDIR", "/tmp")

from cognitive_engine.system.units import (
    Unit,
    UnitRegistry,
    UnitChecker,
    UnitCheckResult,
    UnitViolation,
)
from cognitive_engine.system.dsl import parse_sysd, SysdModel


# ═══════════════════════════════════════════════════════════════
# Unit parsing
# ═══════════════════════════════════════════════════════════════

class TestUnitParsing:
    def test_dimensionless(self):
        u = Unit()
        assert u.is_dimensionless()
        assert str(u) == "dimensionless"

    def test_single_factor(self):
        u = Unit.from_str("people")
        assert u.factors == (("people", 1.0),)
        assert not u.is_dimensionless()
        assert str(u) == "people"

    def test_division(self):
        u = Unit.from_str("people/year")
        assert u.factors == (("people", 1.0), ("year", -1.0))
        assert str(u) == "people*1/year"

    def test_multiplication(self):
        u = Unit.from_str("kg*m/s^2")
        # This should parse as kg*m/s^2
        assert ("kg", 1.0) in u.factors
        assert ("m", 1.0) in u.factors
        assert ("s", -2.0) in u.factors

    def test_power(self):
        u = Unit.from_str("m^2")
        assert u.factors == (("m", 2.0),)

    def test_complex(self):
        u = Unit.from_str("kg*m/s")
        assert ("kg", 1.0) in u.factors
        assert ("m", 1.0) in u.factors
        assert ("s", -1.0) in u.factors

    def test_dimensionless_string(self):
        u = Unit.from_str("dimensionless")
        assert u.is_dimensionless()

    def test_empty_string(self):
        u = Unit.from_str("")
        assert u.is_dimensionless()


# ═══════════════════════════════════════════════════════════════
# Unit arithmetic
# ═══════════════════════════════════════════════════════════════

class TestUnitArithmetic:
    def test_multiply(self):
        a = Unit.from_str("kg")
        b = Unit.from_str("m/s^2")
        c = a * b
        assert ("kg", 1.0) in c.factors
        assert ("m", 1.0) in c.factors
        assert ("s", -2.0) in c.factors

    def test_divide(self):
        a = Unit.from_str("people")
        b = Unit.from_str("year")
        c = a / b
        assert c.factors == (("people", 1.0), ("year", -1.0))

    def test_power(self):
        a = Unit.from_str("m")
        b = a ** 2
        assert b.factors == (("m", 2.0),)

    def test_power_zero(self):
        a = Unit.from_str("m")
        b = a ** 0
        assert b.is_dimensionless()

    def test_multiply_cancels(self):
        a = Unit.from_str("m/s")
        b = Unit.from_str("s")
        c = a * b
        assert c.factors == (("m", 1.0),)

    def test_divide_cancels(self):
        a = Unit.from_str("people")
        b = Unit.from_str("people")
        c = a / b
        assert c.is_dimensionless()

    def test_equality(self):
        a = Unit.from_str("people/year")
        b = Unit.from_str("people/year")
        assert a == b

    def test_inequality(self):
        a = Unit.from_str("people/year")
        b = Unit.from_str("people/month")
        assert a != b


# ═══════════════════════════════════════════════════════════════
# Unit Registry
# ═══════════════════════════════════════════════════════════════

class TestUnitRegistry:
    def test_resolve_builtin(self):
        reg = UnitRegistry()
        u = reg.resolve("people")
        assert u is not None
        # "people" is an alias for "person"
        assert u.factors == (("person", 1.0),)

    def test_resolve_alias(self):
        reg = UnitRegistry()
        u = reg.resolve("people")
        assert u is not None

    def test_resolve_unknown(self):
        reg = UnitRegistry()
        u = reg.resolve("frobnitz")
        assert u is None

    def test_register_custom(self):
        reg = UnitRegistry()
        reg.register("widget", Unit.from_str("widget"))
        u = reg.resolve("widget")
        assert u is not None

    def test_is_time_unit(self):
        reg = UnitRegistry()
        assert reg.is_time_unit("year")
        assert reg.is_time_unit("day")
        assert not reg.is_time_unit("people")


# ═══════════════════════════════════════════════════════════════
# DSL parsing with units
# ═══════════════════════════════════════════════════════════════

class TestDSLUnitsParsing:
    def test_stock_units(self):
        src = '''
model 'Test'
  stock "Population": 1000 ~people~
  aux "births": 50 ~people/year~
  + births
  dt 1
'''
        m = parse_sysd(src)
        assert m.stocks[0].units == "people"

    def test_aux_units(self):
        src = '''
model 'Test'
  stock "Pop": 1000
  aux "rate": 0.05 ~1/year~
  dt 1
'''
        m = parse_sysd(src)
        assert m.aux_vars[0].units == "1/year"

    def test_flow_units(self):
        src = '''
model 'Test'
  stock "Pop": 1000
    + births: 50 ~people/year~
    - deaths: 30 ~people/year~
  dt 1
'''
        m = parse_sysd(src)
        assert m.stocks[0].flows[0].units == "people/year"
        assert m.stocks[0].flows[1].units == "people/year"

    def test_no_units(self):
        src = '''
model 'Test'
  stock "Pop": 1000
  aux "rate": 0.05
  dt 1
'''
        m = parse_sysd(src)
        assert m.stocks[0].units == ""
        assert m.aux_vars[0].units == ""

    def test_complex_units(self):
        src = '''
model 'Test'
  stock "Energy": 100 ~kg*m^2/s^2~
  dt 1
'''
        m = parse_sysd(src)
        assert "kg" in m.stocks[0].units
        assert "m^2" in m.stocks[0].units
        assert "s^2" in m.stocks[0].units


# ═══════════════════════════════════════════════════════════════
# Unit checking
# ═══════════════════════════════════════════════════════════════

class TestUnitChecking:
    def test_consistent_model(self):
        src = '''
model 'Population'
  stock "Population": 1000 ~people~
  aux "births": 50 ~people/year~
  aux "deaths": 30 ~people/year~
  + births
  - deaths
  dt 1
  from 0 to 100
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        assert result.passed

    def test_mismatched_flow_units(self):
        src = '''
model 'Test'
  stock "Population": 1000 ~people~
  aux "births": 50 ~people/month~
  aux "time_unit": 1 ~year~
  + births
  dt 1
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        # Should report error: flow is people/month but stock is people
        # (stock has no time dimension, so checker can't verify — test with time unit)
        # With time_unit declared, checker infers stock needs flow in people/year
        # but flow is people/month → mismatch
        assert not result.passed
        assert len(result.errors) > 0

    def test_inconsistent_inflow_outflow(self):
        src = '''
model 'Test'
  stock "Population": 1000 ~people~
  aux "births": 50 ~people/year~
  aux "deaths": 30 ~dollars/year~
  + births
  - deaths
  dt 1
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        # Should report error: deaths has wrong units
        assert not result.passed

    def test_dimensionless_stock(self):
        src = '''
model 'Test'
  stock "Fraction": 0.5
  aux "change": 0.01 ~1/year~
  + change
  dt 1
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        # Should pass (dimensionless stock with dimensionless flow)
        assert result.passed

    def test_no_units_skips_check(self):
        src = '''
model 'Test'
  stock "Pop": 1000
  aux "rate": 0.05
  + rate
  dt 1
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        # No units declared → nothing to check → passes
        assert result.passed


# ═══════════════════════════════════════════════════════════════
# Unit propagation
# ═══════════════════════════════════════════════════════════════

class TestUnitPropagation:
    def test_infer_from_variable(self):
        src = '''
model 'Test'
  stock "Pop": 1000 ~people~
  aux "growth": Pop * 0.05
  + growth
  dt 1
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        # growth should inherit units from Pop
        assert result.passed

    def test_infer_from_multiplication(self):
        src = '''
model 'Test'
  stock "Money": 1000 ~dollar~
  aux "rate": 0.05 ~1/year~
  aux "interest": Money * rate ~dollar/year~
  + interest
  dt 1
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        # interest is dollar/year = stock/time → consistent
        assert result.passed

    def test_inconsistent_flow_units(self):
        src = '''
model 'Test'
  stock "Money": 1000 ~dollar~
  aux "rate": 0.05 ~1/year~
  aux "interest": Money * rate ~dollar/month~
  + interest
  dt 1
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        # interest is dollar/month but expected dollar/year → inconsistent
        assert not result.passed

    def test_consistent_multiplication(self):
        src = '''
model 'Test'
  stock "Money": 1000 ~dollar~
  aux "rate": 0.05
  aux "interest": Money * rate
  + interest
  dt 1
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        # Both dimensionless — consistent
        assert result.passed

    def test_infer_from_function(self):
        src = '''
model 'Test'
  stock "Pop": 1000 ~people~
  aux "smooth_pop": SMOOTH(Pop, 5)
  dt 1
'''
        m = parse_sysd(src)
        checker = UnitChecker()
        result = checker.check(m)
        # smooth_pop should have same units as Pop
        assert result.passed


# ═══════════════════════════════════════════════════════════════
# UnitCheckResult
# ═══════════════════════════════════════════════════════════════

class TestUnitCheckResult:
    def test_passed_no_violations(self):
        r = UnitCheckResult()
        assert r.passed
        assert r.errors == []
        assert r.warnings == []

    def test_failed_on_errors(self):
        v = UnitViolation(
            name="x",
            expected=Unit.from_str("people"),
            actual=Unit.from_str("dollar"),
            message="mismatch",
        )
        r = UnitCheckResult(violations=[v])
        assert not r.passed

    def test_warnings_dont_fail(self):
        v = UnitViolation(
            name="x",
            expected=Unit.from_str("people"),
            actual=Unit.from_str("people/year"),
            message="mismatch",
            severity="warning",
        )
        r = UnitCheckResult(violations=[v])
        assert r.passed
        assert len(r.warnings) == 1

    def test_to_dict(self):
        v = UnitViolation(
            name="x",
            expected=Unit.from_str("people"),
            actual=Unit.from_str("dollar"),
            message="mismatch",
        )
        r = UnitCheckResult(violations=[v], checked_names=["x"])
        d = r.to_dict()
        assert d["passed"] is False
        assert d["num_errors"] == 1
        assert len(d["violations"]) == 1
