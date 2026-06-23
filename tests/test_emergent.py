"""Tests for emergent properties and stock-flow consistency checker."""

import pytest
from cognitive_engine.system.dsl import parse_sysd, SysdModel, StockDef, FlowDef, AuxDef
from cognitive_engine.system.emergent import (
    EmergentProperty,
    Condition,
    Effect,
    ComparisonOp,
    EffectType,
    ConsistencyResult,
    run_consistency_checks,
    check_outflow_partitions,
    check_flow_sides,
    check_zero_divisors,
)


# ── Condition tests ──────────────────────────────────────────────

class TestCondition:
    def test_gt_true(self):
        c = Condition("x", ComparisonOp.GT, 1.0)
        assert c.evaluate({"x": 1.5}) is True

    def test_gt_false(self):
        c = Condition("x", ComparisonOp.GT, 1.0)
        assert c.evaluate({"x": 0.5}) is False

    def test_ge_boundary(self):
        c = Condition("x", ComparisonOp.GE, 1.0)
        assert c.evaluate({"x": 1.0}) is True

    def test_lt(self):
        c = Condition("x", ComparisonOp.LT, 5.0)
        assert c.evaluate({"x": 3.0}) is True

    def test_le_boundary(self):
        c = Condition("x", ComparisonOp.LE, 5.0)
        assert c.evaluate({"x": 5.0}) is True

    def test_eq(self):
        c = Condition("x", ComparisonOp.EQ, 1.0)
        assert c.evaluate({"x": 1.0}) is True
        assert c.evaluate({"x": 1.1}) is False

    def test_ne(self):
        c = Condition("x", ComparisonOp.NE, 1.0)
        assert c.evaluate({"x": 0.5}) is True
        assert c.evaluate({"x": 1.0}) is False

    def test_missing_variable_defaults_zero(self):
        c = Condition("x", ComparisonOp.GT, 0.5)
        assert c.evaluate({}) is False  # 0.0 > 0.5 is False

    def test_str(self):
        c = Condition("healthcare_stress_avg", ComparisonOp.GT, 1.2)
        assert str(c) == "healthcare_stress_avg > 1.2"


# ── Effect tests ─────────────────────────────────────────────────

class TestEffect:
    def test_multiply(self):
        e = Effect("mortality", EffectType.MULTIPLY, value=2.0)
        assert e.apply(0.03, {}) == 0.06

    def test_add(self):
        e = Effect("rate", EffectType.ADD, value=0.1)
        assert e.apply(0.5, {}) == 0.6

    def test_set(self):
        e = Effect("capacity", EffectType.SET, value=100.0)
        assert e.apply(50.0, {}) == 100.0

    def test_replace_expr(self):
        e = Effect("x", EffectType.REPLACE_EXPR, expr="y + 1")
        # REPLACE_EXPR falls through to returning base
        assert e.apply(5.0, {}) == 5.0


# ── EmergentProperty tests ──────────────────────────────────────

class TestEmergentProperty:
    def test_check_transition_off_to_on(self):
        ep = EmergentProperty(
            name="overload",
            description="Healthcare overload",
            condition=Condition("stress", ComparisonOp.GT, 1.2),
        )
        assert ep.active is False
        transitioned = ep.check({"stress": 1.5}, t=10.0)
        assert transitioned is True
        assert ep.active is True
        assert ep.activation_times == [10.0]

    def test_check_no_transition(self):
        ep = EmergentProperty(
            name="overload",
            description="Healthcare overload",
            condition=Condition("stress", ComparisonOp.GT, 1.2),
        )
        ep.check({"stress": 0.5}, t=10.0)
        assert ep.active is False
        transitioned = ep.check({"stress": 0.8}, t=20.0)
        assert transitioned is False
        assert ep.activation_times == []

    def test_check_transition_on_to_off(self):
        ep = EmergentProperty(
            name="overload",
            description="Healthcare overload",
            condition=Condition("stress", ComparisonOp.GT, 1.2),
        )
        ep.check({"stress": 1.5}, t=10.0)
        assert ep.active is True
        transitioned = ep.check({"stress": 0.5}, t=20.0)
        assert transitioned is True
        assert ep.active is False
        assert ep.activation_times == [10.0, 20.0]

    def test_apply_effects_when_active(self):
        ep = EmergentProperty(
            name="overload",
            description="Healthcare overload",
            condition=Condition("stress", ComparisonOp.GT, 1.2),
            effects=[Effect("mortality", EffectType.MULTIPLY, value=2.0)],
        )
        ep.active = True
        state = {"mortality": 0.03, "stress": 1.5}
        modified = ep.apply_effects(state)
        assert modified["mortality"] == 0.06
        assert modified["stress"] == 1.5  # unchanged

    def test_apply_effects_when_inactive(self):
        ep = EmergentProperty(
            name="overload",
            description="Healthcare overload",
            condition=Condition("stress", ComparisonOp.GT, 1.2),
            effects=[Effect("mortality", EffectType.MULTIPLY, value=2.0)],
        )
        ep.active = False
        state = {"mortality": 0.03}
        modified = ep.apply_effects(state)
        assert modified == {"mortality": 0.03}

    def test_multiple_effects(self):
        ep = EmergentProperty(
            name="overload",
            description="Healthcare overload",
            condition=Condition("stress", ComparisonOp.GT, 1.2),
            effects=[
                Effect("mortality", EffectType.MULTIPLY, value=2.0),
                Effect("capacity_target", EffectType.ADD, value=100),
            ],
        )
        ep.active = True
        state = {"mortality": 0.03, "capacity_target": 50}
        modified = ep.apply_effects(state)
        assert modified["mortality"] == 0.06
        assert modified["capacity_target"] == 150

    def test_summary(self):
        ep = EmergentProperty(
            name="overload",
            description="Healthcare overload",
            condition=Condition("stress", ComparisonOp.GT, 1.2),
            effects=[Effect("mortality", EffectType.MULTIPLY, value=2.0)],
            severity=0.8,
        )
        s = ep.summary()
        assert s["name"] == "overload"
        assert s["condition"] == "stress > 1.2"
        assert s["severity"] == 0.8
        assert len(s["effects"]) == 1
        assert "belief" not in s  # no SL types in SD

    def test_provenance(self):
        ep = EmergentProperty(
            name="overload",
            description="Healthcare overload",
            condition=Condition("stress", ComparisonOp.GT, 1.2),
            provenance=["Hospitalized", "Healthcare_Capacity", "effective_mortality"],
        )
        assert len(ep.provenance) == 3
        assert "Hospitalized" in ep.provenance


# ── Consistency checker tests ───────────────────────────────────

class TestConsistencyChecker:
    def test_valid_model_no_violations(self):
        model = SysdModel(
            name="test",
            stocks=[
                StockDef(name="A", initial=100, flows=[
                    FlowDef(name="in", direction="+", expr="10"),
                    FlowDef(name="out", direction="-", expr="A * 1.0 / 10"),
                ]),
            ],
        )
        result = run_consistency_checks(model)
        assert result.is_valid
        assert result.checks_run == 4

    def test_one_sided_flow_warning(self):
        model = SysdModel(
            name="test",
            stocks=[
                StockDef(name="A", initial=100, flows=[
                    FlowDef(name="in", direction="+", expr="10"),
                ]),
            ],
        )
        result = run_consistency_checks(model)
        warnings = [v for v in result.violations if v.rule == "one_sided_flow"]
        assert len(warnings) == 1
        assert "in" in warnings[0].message

    def test_zero_divisor_warning(self):
        model = SysdModel(
            name="test",
            stocks=[
                StockDef(name="A", initial=0, flows=[
                    FlowDef(name="in", direction="+", expr="10"),
                ]),
                StockDef(name="B", initial=100, flows=[
                    FlowDef(name="flow", direction="-", expr="B / A"),
                    FlowDef(name="flow", direction="+", expr="B / A"),
                ]),
            ],
        )
        result = run_consistency_checks(model)
        div_warnings = [v for v in result.violations if v.rule == "zero_divisor"]
        assert len(div_warnings) >= 1

    def test_partition_sum_error(self):
        """Outflow fractions summing to >1 should be flagged.

        In the real pandemic model, the pattern is:
            Infected * 0.4 / infectious_period
        where 0.4 is the fraction and /infectious_period is the rate.
        The checker extracts the fraction parts (0.4, 0.5, 0.2) and sums them.
        """
        model = SysdModel(
            name="test",
            stocks=[
                StockDef(name="A", initial=100, flows=[
                    FlowDef(name="f1", direction="+", expr="10"),
                    FlowDef(name="f1", direction="-", expr="A * 0.4"),
                    FlowDef(name="f2", direction="-", expr="A * 0.5"),
                    FlowDef(name="f3", direction="-", expr="A * 0.2"),
                ]),
            ],
        )
        result = run_consistency_checks(model)
        partition_errors = [v for v in result.violations if v.rule == "partition_sum"]
        assert len(partition_errors) == 1
        assert "1.1" in partition_errors[0].message

    def test_to_validation(self):
        model = SysdModel(
            name="test",
            stocks=[
                StockDef(name="A", initial=100, flows=[
                    FlowDef(name="in", direction="+", expr="10"),
                ]),
            ],
        )
        result = run_consistency_checks(model)
        val = result.to_validation()
        assert len(val.warnings) > 0

    def test_print_report(self, capsys):
        model = SysdModel(
            name="test",
            stocks=[
                StockDef(name="A", initial=100, flows=[
                    FlowDef(name="in", direction="+", expr="10"),
                ]),
            ],
        )
        result = run_consistency_checks(model)
        result.print_report()
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_valid_two_stock_flow(self):
        """A valid two-stock flow with balanced fractions."""
        model = SysdModel(
            name="test",
            stocks=[
                StockDef(name="A", initial=100, flows=[
                    FlowDef(name="transfer", direction="-", expr="A * 0.1 / 5"),
                ]),
                StockDef(name="B", initial=0, flows=[
                    FlowDef(name="transfer", direction="+", expr="A * 0.1 / 5"),
                ]),
            ],
        )
        result = run_consistency_checks(model)
        assert result.is_valid

    def test_cross_type_flow_warning(self):
        """Flow connecting MATERIAL to FINANCIAL stock should warn."""
        model = SysdModel(
            name="test",
            stocks=[
                StockDef(name="Inventory", initial=100, flows=[
                    FlowDef(name="purchase", direction="-", expr="Inventory * 0.1"),
                ]),
                StockDef(name="Budget", initial=1000, flows=[
                    FlowDef(name="purchase", direction="+", expr="Inventory * 0.1"),
                ]),
            ],
        )
        result = run_consistency_checks(model)
        cross = [v for v in result.violations if v.rule == "cross_type_flow"]
        assert len(cross) == 1


# ── Integration: parse + consistency ─────────────────────────────

class TestConsistencyIntegration:
    def test_sir_model_valid(self):
        source = '''
model "SIR"
  dt 1.0
  from 0 to 100

  stock "Susceptible": 990
    - "Infection": beta * Susceptible * Infected / N

  stock "Infected": 10
    + "Infection": beta * Susceptible * Infected / N
    - "Recovery": gamma * Infected

  stock "Recovered": 0
    + "Recovery": gamma * Infected
'''
        model = parse_sysd(source)
        result = run_consistency_checks(model)
        # SIR has a one-sided Infection flow in Susceptible (only -)
        # and one-sided in Infected (only +) — that's fine, they connect
        # Two-sided: Recovery connects Infected(-) and Recovered(+)
        one_sided = [v for v in result.violations if v.rule == "one_sided_flow"]
        # Infection appears in 2 stocks (Susceptible -, Infected +) — OK
        # Recovery appears in 2 stocks (Infected -, Recovered +) — OK
        assert len(one_sided) == 0

    def test_partition_bug_detected(self):
        """Simulate the pandemic model bug: recovery + severe + mortality != 1."""
        source = '''
model "Buggy"
  dt 0.25
  from 0 to 100

  stock "Infected": 100
    - "Recovery": Infected * 0.4
    - "Hospitalization": Infected * 0.5
    - "Fatality": Infected * 0.2
'''
        model = parse_sysd(source)
        result = run_consistency_checks(model)
        partition_errors = [v for v in result.violations if v.rule == "partition_sum"]
        assert len(partition_errors) == 1
        # 0.4 + 0.5 + 0.2 = 1.1, excess 0.1
        assert "1.1" in partition_errors[0].message
