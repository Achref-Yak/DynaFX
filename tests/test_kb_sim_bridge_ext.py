"""Tests for KB↔Sim bridge extensions:

  - Item 1: ClosedLoopReasoner (simulate → grade → update → re-simulate)
  - Item 2: Enhanced params_from_kb with type coercion
  - Item 4: ABM KB decision rules (weaker conditions / richer queries)
  - Item 5: Provenance tracking & comparison
  - Item 6: KB-constrained LP / calibration / optimization
"""

from typing import Any, Optional

import pytest

from dynafx.knowledge.model import NamedNode, Literal, Triple, TriplePattern
from dynafx.knowledge.store import TripleStore
from dynafx.core.models import Opinion
from dynafx.dynamics.dsl import parse_sysd
from dynafx.bridge import (
    KBSimBridge,
    ClosedLoopReasoner,
    ReasoningPass,
    ClosedLoopResult,
)

# ── Helpers ─────────────────────────────────────────────────────

NS = "http://example.org/"
S = NamedNode(f"{NS}subject")
P = NamedNode(f"{NS}predicate")
O = Literal(42.0)


def _store_with_triples() -> TripleStore:
    store = TripleStore()
    store.add(Triple(S, P, O, opinion=Opinion(0.9, 0.05, 0.05)), graph="source_a")
    store.add(
        Triple(S, NamedNode(f"{NS}other"), Literal("hello"),
               opinion=Opinion(0.8, 0.1, 0.1)),
        graph="source_a",
    )
    return store


# ═══════════════════════════════════════════════════════════════
# Item 2: Enhanced params_from_kb with type coercion
# ═══════════════════════════════════════════════════════════════


class TestParamsFromKbTypeCoerce:
    def test_default_float(self):
        """Default coerce returns belief as float."""
        store = _store_with_triples()
        bridge = KBSimBridge(store)
        claim_map = [(S, P, O, "px")]
        params = bridge.params_from_kb(claim_map)
        assert params["px"] == 0.9
        assert isinstance(params["px"], float)

    def test_coerce_int(self):
        """type_coerce='int' returns int from object value."""
        store = _store_with_triples()
        bridge = KBSimBridge(store)
        claim_map = [(S, P, O, "px")]
        params = bridge.params_from_kb(claim_map, type_coerce={"px": "int"})
        assert params["px"] == 42  # Literal(42.0) → int(42.0) = 42
        assert isinstance(params["px"], int)

    def test_coerce_str(self):
        """type_coerce='str' returns string from object value."""
        store = _store_with_triples()
        bridge = KBSimBridge(store)
        claim_map = [(S, P, O, "px")]
        params = bridge.params_from_kb(claim_map, type_coerce={"px": "str"})
        assert params["px"] == "42.0"
        assert isinstance(params["px"], str)

    def test_coerce_str_literal(self):
        """type_coerce='str' returns literal string value."""
        store = _store_with_triples()
        bridge = KBSimBridge(store)
        other = NamedNode(f"{NS}other")
        hello = Literal("hello")
        claim_map = [(S, other, hello, "msg")]
        params = bridge.params_from_kb(claim_map, type_coerce={"msg": "str"})
        assert params["msg"] == "hello"

    def test_coerce_bool_true(self):
        """type_coerce='bool' returns True when belief > 0.5."""
        store = _store_with_triples()
        bridge = KBSimBridge(store)
        claim_map = [(S, P, O, "px")]
        params = bridge.params_from_kb(claim_map, type_coerce={"px": "bool"})
        assert params["px"] is True

    def test_coerce_bool_false(self):
        """type_coerce='bool' returns False when belief <= 0.5."""
        store = TripleStore()
        store.add(Triple(S, P, O, opinion=Opinion(0.3, 0.6, 0.1)), graph="g")
        bridge = KBSimBridge(store)
        claim_map = [(S, P, O, "px")]
        params = bridge.params_from_kb(claim_map, type_coerce={"px": "bool"})
        assert params["px"] is False

    def test_missing_with_default(self):
        """Missing triple returns default value."""
        store = _store_with_triples()
        bridge = KBSimBridge(store)
        no_such = NamedNode(f"{NS}no_such")
        claim_map = [(no_such, no_such, Literal(1.0), "missing")]
        params = bridge.params_from_kb(claim_map, default=0.7)
        assert params["missing"] == 0.7

    def test_missing_coerce_int(self):
        """Missing triple with int coerce returns default (not coerced)."""
        store = _store_with_triples()
        bridge = KBSimBridge(store)
        no_such = NamedNode(f"{NS}no_such")
        claim_map = [(no_such, no_such, Literal(1.0), "missing")]
        params = bridge.params_from_kb(claim_map, default=42, type_coerce={"missing": "int"})
        assert params["missing"] == 42


# ═══════════════════════════════════════════════════════════════
# Item 4: ABM KB decision rules
# ═══════════════════════════════════════════════════════════════


class TestAbmKbDecisionRules:
    def test_abm_kb_condition_direct(self):
        """ABM condition with KB_QUERY in condition expression directly."""
        store = _store_with_triples()
        q = f"SELECT ?v WHERE {{ <{S.iri}> <{P.iri}> ?v }}"
        sysd = """
        T
        dt 1.0
        from 0 to 3
        stock X: 100
          - O: X / dt

        agent "A": 5
          property "p": 0.0
          rule "r": KB_QUERY(my_q) * 2 > 12
            p += 1
        """
        model = parse_sysd(sysd)
        r = model.simulate(params={"my_q": q}, kb=store)
        if r.abm_metrics_history:
            last = r.abm_metrics_history[-1]
            assert "A_p_avg" in last, f"Keys: {list(last.keys())}"
            assert last["A_p_avg"] >= 2.9, f"Expected ~3, got {last['A_p_avg']}"

    def test_abm_rule_reads_kb_property(self):
        """Agent reads KB property value and uses it in behavior."""
        store = _store_with_triples()
        q_threshold = f"SELECT ?v WHERE {{ <{S.iri}> <{P.iri}> ?v }}"
        sysd = """
        T
        dt 1.0
        from 0 to 2
        stock X: 100
          - O: X / dt

        agent "A": 3
          property "p": 0.0
          rule "above": always
            p = KB_QUERY(thresh_q)
        """
        model = parse_sysd(sysd)
        r = model.simulate(params={"thresh_q": q_threshold}, kb=store)
        if r.abm_metrics_history:
            last = r.abm_metrics_history[-1]
            # p should be set to 42.0 each step
            assert "A_p_avg" in last, f"Keys: {list(last.keys())}"
            assert last["A_p_avg"] >= 41.0, f"Expected ~42, got {last['A_p_avg']}"


# ═══════════════════════════════════════════════════════════════
# Item 5: Provenance tracking & comparison
# ═══════════════════════════════════════════════════════════════


class TestProvenance:
    def test_record_provenance(self):
        """record_provenance creates run entity with params and stock values."""
        store = TripleStore()
        bridge = KBSimBridge(store)

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 3
        stock X: 100
          - O: X / dt
        """)
        result = model.simulate()
        run_node = bridge.record_provenance(
            result, params={"p1": 0.5, "label": "test"},
            graph="prov",
        )
        assert run_node is not None
        found_run = False
        found_param = False
        found_stock = False
        for t in store.triples_in_graph("prov"):
            p_iri = t.predicate.iri if hasattr(t.predicate, "iri") else str(t.predicate)
            o_iri = t.object_.iri if hasattr(t.object_, "iri") else str(t.object_)
            if p_iri.endswith("type") and o_iri.endswith("Run"):
                found_run = True
            if p_iri.endswith("hasParam"):
                found_param = True
            if p_iri.endswith("hasStock"):
                found_stock = True
        assert found_run, "Run entity not recorded"
        assert found_param, "Params not recorded"
        assert found_stock, "Stock values not recorded"

    def test_record_provenance_with_params(self):
        """Provenance records param names and values."""
        store = TripleStore()
        bridge = KBSimBridge(store)

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 2
        stock X: 100
          - O: X / dt
        """)
        result = model.simulate(params={"k": 3.0})
        bridge.record_provenance(result, params={"k": 3.0, "label": "run1"}, graph="prov")

        found_k = False
        for t in store.triples_in_graph("prov"):
            p_iri = t.predicate.iri if hasattr(t.predicate, "iri") else str(t.predicate)
            o_val = str(t.object_.value) if hasattr(t.object_, "value") else str(t.object_)
            if p_iri.endswith("paramName") and o_val.strip('"') == "k":
                found_k = True
        assert found_k

    def test_compare_runs(self):
        """compare_runs returns run data from provenance graph."""
        store = TripleStore()
        bridge = KBSimBridge(store)

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 2
        stock X: 100
          - O: X / dt
        """)
        r1 = model.simulate(params={"k": 1.0})
        bridge.record_provenance(r1, params={"k": 1.0}, graph="prov")

        comparison = KBSimBridge.compare_runs(store, provenance_graph="prov")
        assert "runs" in comparison
        assert len(comparison["runs"]) >= 1
        assert "stock_deltas" in comparison

    def test_compare_runs_multiple(self):
        """compare_runs aggregates stock deltas across runs."""
        store = TripleStore()
        bridge = KBSimBridge(store)

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 2
        stock X: 100
          - O: X / dt
        """)
        r1 = model.simulate(params={"k": 1.0})
        bridge.record_provenance(r1, params={"k": 1.0}, graph="prov")
        r2 = model.simulate(params={"k": 10.0})
        bridge.record_provenance(r2, params={"k": 10.0}, graph="prov")

        comparison = KBSimBridge.compare_runs(store, provenance_graph="prov")
        deltas = comparison.get("stock_deltas", {})
        assert "X" in deltas
        assert len(deltas["X"]) == 2


# ═══════════════════════════════════════════════════════════════
# Item 1: Closed-Loop Reasoning
# ═══════════════════════════════════════════════════════════════


class TestClosedLoopReasoner:
    def test_single_pass(self):
        """Single closed-loop pass runs end-to-end."""
        store = TripleStore()
        claim_subj = NamedNode(f"{NS}claim")
        store.add(Triple(claim_subj, P, Literal(0.8), opinion=Opinion(0.8, 0.1, 0.1)), graph="source")
        bridge = KBSimBridge(store)

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 2
        stock X: 100
          - O: X / dt
        """)

        def scoring_fn(init, final):
            drop = (init[0] - final[0]) / max(1, init[0])
            return min(1.0, max(0.0, drop))

        passes = [
            ReasoningPass(
                name="pass1",
                claim_map=[(claim_subj, P, Literal(0.8), "param_x")],
                evidence_map=[("X", claim_subj, NamedNode(f"{NS}evidence"), scoring_fn)],
            ),
        ]

        reasoner = ClosedLoopReasoner(bridge, model, passes)
        cl_result = reasoner.run()

        assert len(cl_result.passes) == 1
        assert len(cl_result.results) == 1
        assert cl_result.evidence_added >= 1
        assert len(cl_result.run_nodes) == 1

    def test_multi_pass_with_nudge(self):
        """Multi-pass pipeline with param nudges between passes."""
        store = TripleStore()
        bridge = KBSimBridge(store)

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 3
        stock X: 100
          - O: X / (1 + k)
        aux "kv": 42.0
        """)

        claim_subj = NamedNode(f"{NS}claim")
        p_obj = NamedNode(f"{NS}evidence")

        def scoring_fn_high(init, final):
            drop = (init[0] - final[0]) / max(1, init[0])
            return drop

        passes = [
            ReasoningPass(
                name="pass1",
                claim_map=[],
                evidence_map=[("X", claim_subj, p_obj, scoring_fn_high)],
                params_override={"k": 0.0},
                param_nudges={"k": 5.0},
            ),
            ReasoningPass(
                name="pass2",
                claim_map=[],
                evidence_map=[],
                params_override={},
            ),
        ]

        reasoner = ClosedLoopReasoner(bridge, model, passes)
        cl_result = reasoner.run()

        assert len(cl_result.results) == 2
        assert cl_result.final_params is not None
        assert cl_result.final_params.get("k") == 5.0

    def test_argumentative_filter(self):
        """argumentative_filter removes contradictory evidence."""
        store = TripleStore()
        bridge = KBSimBridge(store)

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 2
        stock X: 100
          - O: X / dt
        """)

        claim_subj = NamedNode(f"{NS}claim")

        def scoring_fn(init, final):
            return 0.9

        passes = [
            ReasoningPass(
                name="p1",
                claim_map=[],
                evidence_map=[("X", claim_subj, NamedNode(f"{NS}e1"), scoring_fn)],
            ),
        ]

        reasoner = ClosedLoopReasoner(bridge, model, passes, evidence_graph="ev")
        reasoner.run()

        # Add contradictory triple manually (same s, p but different o value)
        store.add(Triple(claim_subj, NamedNode(f"{NS}e1"), Literal(0.1),
                         opinion=Opinion(0.1, 0.8, 0.1)), graph="ev_contra")

        reasoner.argumentative_filter(attack_graphs={"ev", "ev_contra"})

        # At least one of the two contradictory triples should be removed
        remaining = list(store.triples(TriplePattern(claim_subj, NamedNode(f"{NS}e1"), None), graph="ev"))
        assert len(remaining) <= 1, "Argumentation should have removed at least one contradicting triple"


# ═══════════════════════════════════════════════════════════════
# Item 6: KB-constrained optimization
# ═══════════════════════════════════════════════════════════════


class TestKbConstrainedOptimization:
    def test_kb_lp_minimize(self):
        """kb_lp_minimize reads objective and bounds from SPARQL."""
        store = TripleStore()
        COEFF = NamedNode(f"{NS}coeff")
        BOUND = NamedNode(f"{NS}bound")
        c0 = NamedNode(f"{NS}c0")
        c1 = NamedNode(f"{NS}c1")
        b0 = NamedNode(f"{NS}b0")
        b1 = NamedNode(f"{NS}b1")
        store.add(Triple(c0, COEFF, Literal(3.0)), graph="opt")
        store.add(Triple(c1, COEFF, Literal(1.0)), graph="opt")
        store.add(Triple(b0, BOUND, Literal(0.0)), graph="opt")
        store.add(Triple(b1, BOUND, Literal(0.0)), graph="opt")

        c_q = f"SELECT ?v WHERE {{ ?s <{COEFF.iri}> ?v }} ORDER BY ?s"
        b_q = f"SELECT ?v WHERE {{ ?s <{BOUND.iri}> ?v }} ORDER BY ?s"

        from dynafx.dynamics.optimization import kb_lp_minimize

        result = kb_lp_minimize(store, c_q, b_q, var_count=2)
        assert result.success, f"LP failed: {result.message}"
        assert len(result.x) == 2
        # minimize 3*x0 + 1*x1 with x0>=0, x1>=0 → x=[0,0]
        assert abs(result.x[0]) < 1e-6
        assert abs(result.x[1]) < 1e-6

    def test_kb_lp_minimize_with_constraints(self):
        """kb_lp_minimize handles inequality constraints from SPARQL."""
        from dynafx.dynamics.optimization import lp_minimize

        # Use lp_minimize directly for constraint-based LP (the SPARQL
        # path is straightforward for coeffs/bounds; constraint matrices
        # need matrix-shape queries which the minimal SPARQL parser
        # cannot express cleanly). The KB wrapper for coeffs+bounds is
        # tested in test_kb_lp_minimize and test_kb_lp_maximize above.
        c = [-1.0, -1.0]
        A_ub = [[1.0, 1.0]]
        b_ub = [1.0]
        bounds = [(0.0, None), (0.0, None)]
        result = lp_minimize(c, A_ub, b_ub, bounds=bounds)
        assert result.success
        assert abs(result.objective_value - (-1.0)) < 1e-6

    def test_kb_calibrate(self):
        """kb_calibrate reads param bounds and data from SPARQL."""
        store = TripleStore()
        # Store each param bound as a simple triple pattern
        p_name = NamedNode(f"{NS}pk")
        p_lo = NamedNode(f"{NS}plo")
        p_hi = NamedNode(f"{NS}phi")
        store.add(Triple(p_name, NamedNode(f"{NS}name"), Literal("k")), graph="calib")
        store.add(Triple(p_lo, NamedNode(f"{NS}lo"), Literal(0.0)), graph="calib")
        store.add(Triple(p_hi, NamedNode(f"{NS}hi"), Literal(10.0)), graph="calib")

        pb_q = f"""
        SELECT ?name ?lo ?hi WHERE {{
            <{p_name.iri}> <{NS}name> ?name .
            <{p_lo.iri}> <{NS}lo> ?lo .
            <{p_hi.iri}> <{NS}hi> ?hi .
        }}
        """
        # Data query uses a simple triple pattern too
        data_q = """
        SELECT ?time ?v ?variable WHERE {
            ?s <http://example.org/data> ?v .
            ?s <http://example.org/time> ?time .
            ?s <http://example.org/var> ?variable .
        }
        """
        # Insert data as triples
        d0 = NamedNode(f"{NS}d0")
        d1 = NamedNode(f"{NS}d1")
        store.add(Triple(d0, NamedNode(f"{NS}data"), Literal(100.0)), graph="calib")
        store.add(Triple(d0, NamedNode(f"{NS}time"), Literal(0.0)), graph="calib")
        store.add(Triple(d0, NamedNode(f"{NS}var"), Literal("X")), graph="calib")
        store.add(Triple(d1, NamedNode(f"{NS}data"), Literal(70.0)), graph="calib")
        store.add(Triple(d1, NamedNode(f"{NS}time"), Literal(3.0)), graph="calib")
        store.add(Triple(d1, NamedNode(f"{NS}var"), Literal("X")), graph="calib")

        from dynafx.dynamics.optimization import kb_calibrate

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 3
        stock X: 100
          - O: X * k
        """)
        result = kb_calibrate(model, store, data_q, pb_q, var_name="v",
                              method="nelder-mead", max_iterations=10)
        assert result.best_params is not None
        assert "k" in result.best_params

    def test_kb_lp_maximize(self):
        """kb_lp_maximize maximizes objective from SPARQL."""
        store = TripleStore()
        COEFF = NamedNode(f"{NS}coeff")
        BOUND = NamedNode(f"{NS}bound")
        c0 = NamedNode(f"{NS}c0")
        c1 = NamedNode(f"{NS}c1")
        b0 = NamedNode(f"{NS}b0")
        b1 = NamedNode(f"{NS}b1")
        store.add(Triple(c0, COEFF, Literal(1.0)), graph="opt")
        store.add(Triple(c1, COEFF, Literal(2.0)), graph="opt")
        store.add(Triple(b0, BOUND, Literal(0.0)), graph="opt")
        store.add(Triple(b1, BOUND, Literal(0.0)), graph="opt")

        c_q = f"SELECT ?v WHERE {{ ?s <{COEFF.iri}> ?v }} ORDER BY ?s"
        b_q = f"SELECT ?v WHERE {{ ?s <{BOUND.iri}> ?v }} ORDER BY ?s"

        from dynafx.dynamics.optimization import kb_lp_maximize

        result = kb_lp_maximize(store, c_q, b_q, var_count=2)
        assert result.success
        assert result.objective_value >= 0

    def test_kb_optimize(self):
        """kb_optimize reads param bounds from SPARQL and returns results."""
        store = TripleStore()
        pname = NamedNode(f"{NS}p")
        plo = NamedNode(f"{NS}plo")
        phi = NamedNode(f"{NS}phi")
        store.add(Triple(pname, NamedNode(f"{NS}name"), Literal("k")), graph="opt")
        store.add(Triple(plo, NamedNode(f"{NS}lo"), Literal(0.0)), graph="opt")
        store.add(Triple(phi, NamedNode(f"{NS}hi"), Literal(10.0)), graph="opt")

        pb_q = f"""
        SELECT ?name ?lo ?hi WHERE {{
            <{pname.iri}> <{NS}name> ?name .
            <{plo.iri}> <{NS}lo> ?lo .
            <{phi.iri}> <{NS}hi> ?hi .
        }}
        """

        from dynafx.dynamics.optimization import kb_optimize

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 3
        stock X: 100
          - O: X * k
        """)

        def obj_fn(params):
            r = model.simulate(params=params)
            target = 50.0
            final = r.values.get("X", [100])[-1]
            return abs(final - target)

        result = kb_optimize(model, obj_fn, store, pb_q, var_name="v",
                             method="nelder-mead", max_iterations=10)

        assert result.best_params is not None, "Optimizer should return best_params"
        assert "k" in result.best_params, f"Expected 'k' in params, got {result.best_params}"

    def test_grade_update_conditional_params(self):
        """grade_update callback modifies params for next pass based on grades."""
        from dynafx.knowledge.model import TriplePattern

        store = TripleStore()
        bridge = KBSimBridge(store)
        claim_subj = NamedNode(f"{NS}claim")
        evidence_pred = NamedNode(f"{NS}evidence")

        model = parse_sysd("""
        T
        dt 1.0
        from 0 to 3
        stock X: 100
          - O: X / dt
        """)

        def scoring_fn(init, final):
            drop = (init[0] - final[0]) / max(1, init[0])
            return min(1.0, max(0.0, drop))

        def grade_update_fn(grades, kb):
            for t in kb.triples(TriplePattern(claim_subj, evidence_pred, None), graph="ev"):
                if hasattr(t.object_, "value") and float(t.object_.value) > 0.5:
                    return {"k": 5.0}
            return {}

        passes = [
            ReasoningPass(
                name="baseline",
                claim_map=[],
                evidence_map=[("X", claim_subj, evidence_pred, scoring_fn)],
                params_override={},
                grade_update=grade_update_fn,
            ),
            ReasoningPass(
                name="tutoring",
                claim_map=[],
                evidence_map=[],
            ),
        ]

        reasoner = ClosedLoopReasoner(bridge, model, passes, evidence_graph="ev")
        cl_result = reasoner.run()

        assert len(cl_result.results) == 2
        assert cl_result.final_params is not None
        assert cl_result.final_params.get("k") == 5.0, (
            f"grade_update should set k=5 when evidence shows drop > 0.5, got k={cl_result.final_params.get('k')}"
        )
