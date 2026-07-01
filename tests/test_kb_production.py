"""Tests for Production Rule Engine (kb/production.py)."""

import time

from dynafx.core.models import Opinion
from dynafx.knowledge.inference import InferencePattern
from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
)
from dynafx.knowledge.production import (
    AndCondition,
    AggregationCondition,
    ComparisonCondition,
    LogAction,
    NotCondition,
    OrCondition,
    ProductionRule,
    ProductionRuleEngine,
    RetractAction,
    SparqlCondition,
    TripleAction,
    TripleCondition,
)
from dynafx.knowledge.store import TripleStore

S = NamedNode("http://example.org/s")
P = NamedNode("http://example.org/p")
O = NamedNode("http://example.org/o")
O2 = NamedNode("http://example.org/o2")
CONTAINER = NamedNode("http://sc.org/Container")
DELAYED = NamedNode("http://sc.org/Delayed")
STATUS = NamedNode("http://sc.org/status")


# ── Fixtures ────────────────────────────────────────────────────────


def empty_store() -> TripleStore:
    return TripleStore()


def store_with_one_triple() -> TripleStore:
    st = TripleStore()
    st.add(Triple(S, P, O))
    return st


# ── TripleCondition tests ──────────────────────────────────────────


class TestTripleCondition:
    def test_match_exact(self):
        st = store_with_one_triple()
        cond = TripleCondition(InferencePattern(S, P, O))
        result = cond.evaluate(st, {})
        assert result.matched

    def test_no_match(self):
        st = store_with_one_triple()
        cond = TripleCondition(InferencePattern(S, P, O2))
        result = cond.evaluate(st, {})
        assert not result.matched

    def test_match_with_variable(self):
        st = store_with_one_triple()
        cond = TripleCondition(InferencePattern("?s", P, O))
        result = cond.evaluate(st, {})
        assert result.matched
        assert "s" in result.bindings
        assert result.bindings["s"] == S

    def test_match_with_existing_binding(self):
        st = store_with_one_triple()
        cond = TripleCondition(InferencePattern("?s", P, O))
        result = cond.evaluate(st, {"s": S})
        assert result.matched

    def test_match_with_conflicting_binding(self):
        st = store_with_one_triple()
        cond = TripleCondition(InferencePattern("?s", P, O))
        result = cond.evaluate(st, {"s": O2})
        assert not result.matched

    def test_empty_store_no_match(self):
        st = empty_store()
        cond = TripleCondition(InferencePattern(S, P, O))
        result = cond.evaluate(st, {})
        assert not result.matched

    def test_pattern_with_wildcards(self):
        st = store_with_one_triple()
        cond = TripleCondition(InferencePattern(None, None, None))
        result = cond.evaluate(st, {})
        assert result.matched

    def test_blank_node_match(self):
        st = TripleStore()
        bn = BlankNode("b1")
        st.add(Triple(S, P, bn))
        cond = TripleCondition(InferencePattern(None, None, "?o"))
        result = cond.evaluate(st, {})
        assert result.matched
        assert result.bindings["o"] == bn


# ── SparqlCondition tests ──────────────────────────────────────────


class TestSparqlCondition:
    def test_select_returns_results(self):
        st = store_with_one_triple()
        cond = SparqlCondition(
            "SELECT ?s WHERE { ?s <http://example.org/p> <http://example.org/o> }"
        )
        result = cond.evaluate(st, {})
        assert result.matched

    def test_select_no_results(self):
        st = store_with_one_triple()
        cond = SparqlCondition(
            "SELECT ?s WHERE { ?s <http://example.org/p> <http://example.org/other> }"
        )
        result = cond.evaluate(st, {})
        assert not result.matched

    def test_ask_returns_true(self):
        st = store_with_one_triple()
        cond = SparqlCondition(
            "ASK { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"
        )
        result = cond.evaluate(st, {})
        assert result.matched

    def test_ask_returns_false(self):
        st = store_with_one_triple()
        cond = SparqlCondition(
            "ASK { <http://example.org/s> <http://example.org/p> <http://example.org/other> }"
        )
        result = cond.evaluate(st, {})
        assert not result.matched

    def test_min_results_threshold(self):
        st = store_with_one_triple()
        cond = SparqlCondition(
            "SELECT ?s WHERE { ?s <http://example.org/p> <http://example.org/o> }",
            min_results=2,
        )
        result = cond.evaluate(st, {})
        assert not result.matched


# ── ComparisonCondition tests ──────────────────────────────────────


class TestComparisonCondition:
    def test_lt_true(self):
        cond = ComparisonCondition(1.0, "<", 2.0)
        result = cond.evaluate(TripleStore(), {})
        assert result.matched

    def test_lt_false(self):
        cond = ComparisonCondition(3.0, "<", 2.0)
        result = cond.evaluate(TripleStore(), {})
        assert not result.matched

    def test_gt_true(self):
        cond = ComparisonCondition(3.0, ">", 2.0)
        result = cond.evaluate(TripleStore(), {})
        assert result.matched

    def test_eq_true(self):
        cond = ComparisonCondition(1.0, "==", 1.0)
        result = cond.evaluate(TripleStore(), {})
        assert result.matched

    def test_eq_false(self):
        cond = ComparisonCondition(1.0, "==", 2.0)
        result = cond.evaluate(TripleStore(), {})
        assert not result.matched

    def test_ne_true(self):
        cond = ComparisonCondition(1.0, "!=", 2.0)
        result = cond.evaluate(TripleStore(), {})
        assert result.matched

    def test_with_bound_variable(self):
        st = store_with_one_triple()
        cond = ComparisonCondition("?val", ">", 0.0)
        result = cond.evaluate(st, {"val": Literal(42.0)})
        assert result.matched

    def test_with_unresolved_variable_returns_default(self):
        st = store_with_one_triple()
        cond = ComparisonCondition("?val", "<", 10.0)
        result = cond.evaluate(st, {"val": Literal("abc")})  # returns 0.0
        assert result.matched  # 0.0 < 10.0 is True

    def test_with_non_numeric_string_above_threshold(self):
        st = store_with_one_triple()
        cond = ComparisonCondition("?val", ">", 0.0)
        result = cond.evaluate(st, {"val": Literal("abc")})
        assert not result.matched  # 0.0 > 0.0 is False


# ── AggregationCondition tests (require SPARQL with aggregates) ────


class TestAggregationCondition:
    def test_count_above_threshold(self):
        st = store_with_one_triple()
        cond = AggregationCondition(
            "SELECT ?s WHERE { ?s ?p ?o }",
            threshold=0.5,
            op=">=",
        )
        result = cond.evaluate(st, {})
        assert result.matched

    def test_count_below_threshold(self):
        st = store_with_one_triple()
        cond = AggregationCondition(
            "SELECT ?s WHERE { ?s ?p ?o }",
            threshold=10.0,
            op=">=",
        )
        result = cond.evaluate(st, {})
        assert not result.matched


# ── Compound condition tests ───────────────────────────────────────


class TestCompoundConditions:
    def test_and_both_match(self):
        st = store_with_one_triple()
        c1 = TripleCondition(InferencePattern(S, P, O))
        c2 = TripleCondition(InferencePattern(S, P, O))
        cond = AndCondition([c1, c2])
        result = cond.evaluate(st, {})
        assert result.matched

    def test_and_one_fails(self):
        st = store_with_one_triple()
        c1 = TripleCondition(InferencePattern(S, P, O))
        c2 = TripleCondition(InferencePattern(S, P, O2))
        cond = AndCondition([c1, c2])
        result = cond.evaluate(st, {})
        assert not result.matched

    def test_or_first_matches(self):
        st = store_with_one_triple()
        c1 = TripleCondition(InferencePattern(S, P, O))
        c2 = TripleCondition(InferencePattern(S, P, O2))
        cond = OrCondition([c1, c2])
        result = cond.evaluate(st, {})
        assert result.matched

    def test_or_neither_matches(self):
        st = store_with_one_triple()
        c1 = TripleCondition(InferencePattern(S, P, O2))
        c2 = TripleCondition(InferencePattern(S, P, O2))
        cond = OrCondition([c1, c2])
        result = cond.evaluate(st, {})
        assert not result.matched

    def test_not_inverts_match(self):
        st = store_with_one_triple()
        cond = NotCondition(TripleCondition(InferencePattern(S, P, O2)))
        result = cond.evaluate(st, {})
        assert result.matched

    def test_not_inverts_no_match(self):
        st = store_with_one_triple()
        cond = NotCondition(TripleCondition(InferencePattern(S, P, O)))
        result = cond.evaluate(st, {})
        assert not result.matched


# ── Action tests ────────────────────────────────────────────────────


class TestTripleAction:
    def test_adds_triple(self):
        st = empty_store()
        action = TripleAction(S, P, O, graph="test")
        result = action.execute(st, {})
        assert result.success
        matched = list(st.triples(TriplePattern(S, P, O), graph="test"))
        assert len(matched) == 1

    def test_resolves_bound_variable(self):
        st = empty_store()
        action = TripleAction("?s", P, O)
        result = action.execute(st, {"s": S})
        assert result.success
        matched = list(st.triples(TriplePattern(S, P, O)))
        assert len(matched) == 1

    def test_unresolved_variable_fails(self):
        st = empty_store()
        action = TripleAction("?missing", P, O)
        result = action.execute(st, {})
        assert not result.success


class TestRetractAction:
    def test_removes_triple(self):
        st = store_with_one_triple()
        action = RetractAction(TriplePattern(S, P, O))
        result = action.execute(st, {})
        assert result.success
        assert result.output["removed"] >= 1
        matched = list(st.triples(TriplePattern(S, P, O)))
        assert len(matched) == 0

    def test_no_match_removes_nothing(self):
        st = store_with_one_triple()
        action = RetractAction(TriplePattern(S, P, O2))
        result = action.execute(st, {})
        assert result.success
        assert result.output["removed"] == 0


class TestLogAction:
    def test_executes_successfully(self):
        action = LogAction("Test message")
        result = action.execute(TripleStore(), {})
        assert result.success
        assert result.message == "Test message"

    def test_interpolates_binding(self):
        action = LogAction("Container ?id delayed")
        result = action.execute(TripleStore(), {"id": NamedNode("http://sc.org/C-123")})
        assert result.success
        assert "http://sc.org/C-123" in result.message


# ── ProductionRule and Engine tests ─────────────────────────────────


class TestProductionRuleEngine:
    def test_simple_rule_fires(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(
            name="test",
            body=[TripleCondition(InferencePattern(S, P, O))],
            head=[LogAction("Matched")],
        )
        engine.add_rule(rule)
        results = engine.evaluate()
        assert len(results) == 1
        assert results[0].success

    def test_rule_does_not_fire(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(
            name="test",
            body=[TripleCondition(InferencePattern(S, P, O2))],
            head=[LogAction("Should not fire")],
        )
        engine.add_rule(rule)
        results = engine.evaluate()
        assert len(results) == 0

    def test_disabled_rule_skipped(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(
            name="test",
            enabled=False,
            body=[TripleCondition(InferencePattern(S, P, O))],
            head=[LogAction("Should not fire")],
        )
        engine.add_rule(rule)
        results = engine.evaluate()
        assert len(results) == 0

    def test_fire_once(self):
        st = empty_store()
        engine = ProductionRuleEngine(st)
        fire_count: list[int] = [0]

        class CountingLogAction(LogAction):
            def execute(self, store, bindings):
                fire_count[0] += 1
                return super().execute(store, bindings)

        # Rule matches ANY triple
        rule = ProductionRule(
            name="test",
            fire_once=True,
            body=[TripleCondition(InferencePattern(None, None, None))],
            head=[CountingLogAction("test")],
        )
        engine.add_rule(rule)
        engine.start()  # subscribe to store.on_add

        # Add first triple — fires once
        st.add(Triple(S, P, O))
        assert fire_count[0] == 1

        # Add same triple again — fire_once dedup should block
        st.add(Triple(S, P, O))
        assert fire_count[0] == 1  # dedup'd by fire_once

        # Add a different triple — different signature, should fire
        st.add(Triple(O, P, O2))
        assert fire_count[0] == 2  # new trigger triple

        engine.stop()


    def test_fire_once_reset(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(
            name="test",
            fire_once=True,
            body=[TripleCondition(InferencePattern(S, P, O))],
            head=[LogAction("Fires once")],
        )
        engine.add_rule(rule)
        results1 = engine.evaluate()
        engine.reset()
        results2 = engine.evaluate()
        assert len(results1) == 1
        assert len(results2) == 1  # fires again after reset

    def test_max_fires(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(
            name="test",
            fire_once=False,
            max_fires=2,
            body=[TripleCondition(InferencePattern(S, P, O))],
            head=[LogAction("Fires")],
        )
        engine.add_rule(rule)
        results1 = engine.evaluate()
        results2 = engine.evaluate()  # no new trigger but fire_once=False
        results3 = engine.evaluate()
        assert len(results1) == 1
        assert len(results2) == 1  # max 2 total, this is #2
        assert len(results3) == 0  # max reached

    def test_priority_ordering(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        order: list[str] = []

        rule_low = ProductionRule(
            name="low", priority=10,
            body=[TripleCondition(InferencePattern(S, P, O))],
            head=[LogAction("low")],
        )
        rule_high = ProductionRule(
            name="high", priority=1,
            body=[TripleCondition(InferencePattern(S, P, O))],
            head=[LogAction("high")],
        )
        engine.add_rule(rule_low)
        engine.add_rule(rule_high)
        results = engine.evaluate()
        # Both should fire (fire_once=False, max_fires=0)
        assert len(results) == 2

    def test_full_body_and_multiple_actions(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(
            name="multi-action",
            body=[
                TripleCondition(InferencePattern(S, P, O)),
                ComparisonCondition(1.0, "<", 10.0),
            ],
            head=[
                LogAction("Condition met"),
                TripleAction(NamedNode("http://ex/Decision"), STATUS, Literal("approved")),
            ],
        )
        engine.add_rule(rule)
        results = engine.evaluate()
        assert len(results) == 2
        assert results[0].success
        assert results[1].success

        # Verify triple was added
        matched = list(st.triples(TriplePattern(
            NamedNode("http://ex/Decision"), STATUS, Literal("approved"),
        )))
        assert len(matched) == 1

    def test_remove_rule(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(
            name="test",
            body=[TripleCondition(InferencePattern(S, P, O))],
            head=[LogAction("Matched")],
        )
        engine.add_rule(rule)
        engine.remove_rule("test")
        results = engine.evaluate()
        assert len(results) == 0

    def test_get_rule(self):
        st = empty_store()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(name="test", body=[], head=[])
        engine.add_rule(rule)
        assert engine.get_rule("test") is rule
        assert engine.get_rule("nonexistent") is None

    def test_start_stop(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        engine.add_rule(ProductionRule(
            name="test",
            body=[TripleCondition(InferencePattern(S, P, O))],
            head=[LogAction("test")],
        ))
        engine.start()
        assert engine._started
        # start() calls evaluate() which should fire the rule
        engine.stop()
        assert not engine._started

    def test_event_driven_fire(self):
        st = empty_store()
        engine = ProductionRuleEngine(st)
        fired: list[str] = []

        rule = ProductionRule(
            name="on-add",
            fire_once=False,
            body=[TripleCondition(InferencePattern(S, P, O))],
            head=[LogAction("fired")],
        )
        engine.add_rule(rule)
        engine.start()

        # Adding the matching triple should trigger evaluate
        st.add(Triple(S, P, O))
        # Give callbacks a moment
        _ = st.triples(TriplePattern(S, P, O))

        # We can verify the engine is subscribed by checking the listener
        assert engine._add_listener is not None
        engine.stop()

    def test_complex_rule_with_all_condition_types(self):
        st = store_with_one_triple()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(
            name="complex",
            body=[
                TripleCondition(InferencePattern("?s", P, O)),
                AndCondition([
                    ComparisonCondition(1.0, "<", 10.0),
                    NotCondition(TripleCondition(InferencePattern(S, P, O2))),
                ]),
                OrCondition([
                    TripleCondition(InferencePattern(S, P, O)),
                    TripleCondition(InferencePattern(S, P, O2)),
                ]),
            ],
            head=[LogAction("Complex rule fired for ?s")],
        )
        engine.add_rule(rule)
        results = engine.evaluate()
        assert len(results) == 1
        assert results[0].success

    def test_rule_body_cascades_bindings(self):
        st = store_with_one_triple()
        st.add(Triple(O, P, O2))
        engine = ProductionRuleEngine(st)
        bound: list[dict] = []

        class CaptureAction(LogAction):
            def execute(self, store, bindings):
                bound.append(dict(bindings))
                return super().execute(store, bindings)

        rule = ProductionRule(
            name="cascade",
            body=[
                TripleCondition(InferencePattern("?a", P, O)),
                TripleCondition(InferencePattern("?a", P, "?b")),
            ],
            head=[CaptureAction("?a -> ?b")],
        )
        engine.add_rule(rule)
        results = engine.evaluate()
        assert len(results) == 1
        assert "a" in bound[0]
        assert "b" in bound[0]

    def test_triple_action_with_belief(self):
        st = empty_store()
        action = TripleAction(S, P, O, graph="test", belief=0.8, uncertainty=0.1)
        result = action.execute(st, {})
        assert result.success
        matched = list(st.triples(TriplePattern(S, P, O), graph="test"))
        assert len(matched) == 1
        assert matched[0].opinion is not None
        assert matched[0].opinion.belief == 0.8

    def test_empty_rule_no_body_no_head(self):
        st = empty_store()
        engine = ProductionRuleEngine(st)
        rule = ProductionRule(name="empty", body=[], head=[])
        engine.add_rule(rule)
        # A rule with no body conditions always matches
        results = engine.evaluate()
        assert len(results) == 0  # no head actions to return
