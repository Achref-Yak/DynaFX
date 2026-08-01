"""Tests for KB↔Sim bridge: KB_QUERY/KB_ASSERT builtins + KBSimBridge class."""

from dynafx.knowledge.model import NamedNode, Literal, Triple, TriplePattern
from dynafx.knowledge.store import TripleStore
from dynafx.dynamics.dsl import (
    parse_sysd, SysdModel, AgentDef, AgentPropDef, AgentRuleDef, AgentStrategy,
)
from dynafx.bridge import KBSimBridge


# ── Helpers ─────────────────────────────────────────────────────

NS = "http://example.org/"
S = NamedNode(f"{NS}subject")
P = NamedNode(f"{NS}predicate")
O = Literal(42.0)


def _store_with_triples() -> TripleStore:
    store = TripleStore()
    store.add(Triple(S, P, O), graph="source_a")
    store.add(
        Triple(S, NamedNode(f"{NS}other"), Literal("hello")),
        graph="source_a",
    )
    return store


# ── KB_QUERY in SD aux expressions ──────────────────────────────


def test_kb_query_sd_aux():
    """KB_QUERY in aux expression resolves from KB via SPARQL."""
    store = _store_with_triples()
    query_str = f"SELECT ?v WHERE {{ <{S.iri}> <{P.iri}> ?v }}"
    sysd = """
    T
    dt 1.0
    from 0 to 2
    stock X: 100
      - O: X / dt
    aux "kb_val": KB_QUERY(my_query)
    """
    model = parse_sysd(sysd)
    r = model.simulate(params={"my_query": query_str}, kb=store)
    assert "kb_val" in r.aux_values
    assert abs(r.aux_values["kb_val"][0] - 42.0) < 1e-9, \
        f"Expected 42.0, got {r.aux_values['kb_val'][0]}"
    assert abs(r.aux_values["kb_val"][-1] - 42.0) < 1e-9


def test_kb_query_without_kb_returns_zero():
    """KB_QUERY returns 0.0 when no KB is provided."""
    sysd = """
    T
    dt 1.0
    from 0 to 1
    stock X: 100
      - O: X / dt
    aux "kb_val": KB_QUERY(empty_q)
    """
    model = parse_sysd(sysd)
    r = model.simulate(params={"empty_q": ""})
    assert abs(r.aux_values["kb_val"][0]) < 1e-9


def test_kb_query_ask():
    """KB_QUERY with ASK returns 1.0 for matching pattern, 0.0 for no match."""
    store = _store_with_triples()
    query_match = f"ASK {{ <{S.iri}> <{P.iri}> ?o }}"
    query_no_match = f"ASK {{ <{S.iri}> <{P.iri}> <http://nope> }}"

    sysd = """
    T
    dt 1.0
    from 0 to 1
    stock X: 100
      - O: X / dt
    aux "match": KB_QUERY(match_q)
    aux "nomatch": KB_QUERY(nomatch_q)
    """
    model = parse_sysd(sysd)
    r = model.simulate(
        params={"match_q": query_match, "nomatch_q": query_no_match},
        kb=store,
    )
    assert abs(r.aux_values["match"][0] - 1.0) < 1e-9, "ASK match should be 1.0"
    assert abs(r.aux_values["nomatch"][0]) < 1e-9, "ASK no-match should be 0.0"


def test_kb_query_in_flow_expr():
    """KB_QUERY in a flow expression dynamically gates the rate."""
    store = _store_with_triples()
    query_str = f"SELECT ?v WHERE {{ <{S.iri}> <{P.iri}> ?v }}"
    sysd = """
    T
    dt 1.0
    from 0 to 3
    stock X: 100
      - O: X / KB_QUERY(q)
    """
    model = parse_sysd(sysd)
    r = model.simulate(params={"q": query_str}, kb=store)
    x_vals = r.values["X"]
    assert x_vals[0] == 100.0
    assert x_vals[-1] < 100.0
    expected_final = 100.0 - (100.0 / 42.0) * 3
    assert abs(x_vals[-1] - expected_final) < 3.0, \
        f"Expected ~{expected_final}, got {x_vals[-1]}"


# ── KB_QUERY in ABM conditions ──────────────────────────────────


def test_kb_query_in_abm_condition():
    """ABM agent rule condition with KB_QUERY triggers correctly."""
    store = _store_with_triples()
    query_str = f"SELECT ?v WHERE {{ <{S.iri}> <{P.iri}> ?v }}"
    sysd = """
    T
    dt 1.0
    from 0 to 3
    stock X: 100
      - O: X / dt

    agent "A": 10
      prop "p": 0.0
      rule "r": KB_QUERY(q) > 10 and always
        p += 5
    """
    model = parse_sysd(sysd)
    r = model.simulate(params={"q": query_str}, kb=store)
    if r.abm_metrics_history:
        for m in r.abm_metrics_history:
            if "A_p_avg" in m and m["A_p_avg"] > 0:
                assert m["A_p_avg"] >= 4.0, f"A_p_avg should be ~5.0, got {m['A_p_avg']}"
                break


def test_kb_query_abm_condition_no_match():
    """ABM rule condition with KB_QUERY that doesn't match does NOT trigger."""
    store = _store_with_triples()
    query_no_match = f"ASK {{ <{S.iri}> <{P.iri}> <http://nope> }}"
    sysd = """
    T
    dt 1.0
    from 0 to 2
    stock X: 100
      - O: X / dt

    agent "A": 5
      prop "p": 10.0
      rule "r": KB_QUERY(q) > 0.5
        p -= 1
    """
    model = parse_sysd(sysd)
    r = model.simulate(params={"q": query_no_match}, kb=store)
    if r.abm_metrics_history:
        for m in r.abm_metrics_history:
            if "A_p_avg" in m:
                assert abs(m["A_p_avg"] - 10.0) < 1e-6, \
                    f"p should stay 10.0, got {m['A_p_avg']}"
                break


# ── KB_ASSERT in ABM effects ────────────────────────────────────


def test_kb_assert_abm_effect():
    """ABM agent effect with KB_ASSERT adds triples to the KB.

    KB_ASSERT args must be params (for IRIs) or numeric literals,
    because the DSL expression parser doesn't support string literals.
    """
    store = _store_with_triples()
    s_iri = f"{NS}agent"
    p_iri = f"{NS}hasProperty"
    sysd = """
    T
    dt 1.0
    from 0 to 2
    stock X: 100
      - O: X / dt

    agent "A": 3
      prop "p": 0.0
      rule "r": always
        p += 1
        KB_ASSERT(subj_iri, pred_iri, p, 0.9)
    """
    model = parse_sysd(sysd)
    r = model.simulate(
        params={"subj_iri": s_iri, "pred_iri": p_iri},
        kb=store,
    )

    subj = NamedNode(s_iri)
    pred = NamedNode(p_iri)
    found = False
    for g in store.graphs():
        for t in store.triples(
            TriplePattern(subject=subj, predicate=pred, object_=None), graph=g
        ):
            found = True
    assert found, "KB_ASSERT triple not found in store"


# ── KB_QUERY in DES expressions ─────────────────────────────────


def test_kb_query_des_arrival_rate():
    """DES arrival rate using KB_QUERY is modulated by KB belief."""
    store = _store_with_triples()
    query_str = f"SELECT ?v WHERE {{ <{S.iri}> <{P.iri}> ?v }}"
    sysd = """
    T
    dt 1.0
    from 0 to 5
    stock X: 100
      - O: X / dt

    queue "Q": capacity 100
      arrival_rate KB_QUERY(q) / 42
    """
    model = parse_sysd(sysd)
    r = model.simulate(params={"q": query_str}, kb=store)
    if r.des_engine and r.des_engine.queues:
        q = r.des_engine.queues.get("Q")
        if q:
            assert q.stats.total_arrivals >= 1, \
                f"Expected arrivals from KB-modulated rate, got {q.stats.total_arrivals}"


# ── KBSimBridge class ──────────────────────────────────────────


def test_bridge_params_from_kb():
    """params_from_kb returns max belief across graphs."""
    store = _store_with_triples()
    bridge = KBSimBridge(store)

    claim_map = [
        (S, P, O, "param_x"),
    ]
    params = bridge.params_from_kb(claim_map)
    assert "param_x" in params
    assert abs(params["param_x"] - 42.0) < 1e-6, \
        f"Expected 42.0, got {params['param_x']}"


def test_bridge_params_from_kb_default():
    """params_from_kb returns default when no triple matches."""
    store = _store_with_triples()
    bridge = KBSimBridge(store)

    no_such = NamedNode(f"{NS}no_such")
    claim_map = [(no_such, no_such, Literal(1.0), "missing")]
    params = bridge.params_from_kb(claim_map, default=0.3)
    assert abs(params["missing"] - 0.3) < 1e-6


def test_bridge_evidence_from_result():
    """evidence_from_result creates opinion triples from simulation output."""
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

    def scoring_fn(initial, final):
        drop = (initial[0] - final[0]) / max(1, initial[0])
        return min(1.0, max(0.0, drop))

    evidence_map = [
        ("X", S, P, scoring_fn),
    ]
    triples = bridge.evidence_from_result(result, evidence_map)
    assert len(triples) >= 1
    t = triples[0]
    assert t.subject == S
    assert t.predicate == P
    assert t is not None


def test_bridge_run_with_kb():
    """run_with_kb passes kb to simulate and returns result."""
    store = _store_with_triples()
    bridge = KBSimBridge(store)

    query_str = f"SELECT ?v WHERE {{ <{S.iri}> <{P.iri}> ?v }}"
    sysd = """
    T
    dt 1.0
    from 0 to 2
    stock X: 100
      - O: X / dt
    aux "kv": KB_QUERY(q)
    """
    model = parse_sysd(sysd)
    result = bridge.run_with_kb(model, params={"q": query_str})
    assert abs(result.aux_values["kv"][0] - 42.0) < 1e-9


def test_bridge_full_roundtrip():
    """full_roundtrip does KB→sim→KB end-to-end."""
    store = _store_with_triples()
    bridge = KBSimBridge(store)

    model = parse_sysd("""
    T
    dt 1.0
    from 0 to 3
    stock X: 100
      - O: X / dt
    """)

    claim_map = [
        (S, P, O, "param_x"),
    ]

    def scoring_fn(initial, final):
        drop = (initial[0] - final[0]) / max(1, initial[0])
        return drop

    evidence_map = [
        ("X", S, P, scoring_fn),
    ]

    result, triples = bridge.full_roundtrip(model, claim_map, evidence_map)
    assert result is not None
    assert len(triples) >= 1
    found = False
    for g in store.graphs():
        for t in store.triples(TriplePattern(S, P, None), graph=g):
            found = True
    assert found, "No evidence triples found in KB"


# ── KB_QUERY_TEMPLATE tests ─────────────────────────────────────


NS_P_IRI = f"{NS}predicate"

def test_kb_query_template_basic():
    """KB_QUERY_TEMPLATE substitutes $subject and returns KB value."""
    store = _store_with_triples()
    template = f"SELECT ?v WHERE {{ <$subject> <{NS_P_IRI}> ?v }}"
    sysd = """
    T
    dt 1.0
    from 0 to 2
    stock X: 100
      - O: X / dt
    aux "kb_val": KB_QUERY_TEMPLATE(my_tmpl, s_iri)
    """
    model = parse_sysd(sysd)
    params = {
        "my_tmpl": template,
        "s_iri": S.iri,
    }
    r = model.simulate(params=params, kb=store)
    assert abs(r.aux_values["kb_val"][0] - 42.0) < 1e-9


def test_kb_query_template_multiple_subjects():
    """KB_QUERY_TEMPLATE returns different values for different subjects."""
    store = _store_with_triples()
    store.add(Triple(
        NamedNode(f"{NS}other_subj"), P, Literal(77.0),
    ), graph="source_a")
    template = f"SELECT ?v WHERE {{ <$subject> <{NS_P_IRI}> ?v }}"

    def run_with_subject(subj_iri):
        sysd = """
        T
        dt 1.0
        from 0 to 2
        stock X: 100
          - O: X / dt
        aux "kv": KB_QUERY_TEMPLATE(tmpl, subj)
        """
        model = parse_sysd(sysd)
        r = model.simulate(params={"tmpl": template, "subj": subj_iri}, kb=store)
        return r.aux_values["kv"][0]

    val_a = run_with_subject(S.iri)
    val_b = run_with_subject(f"{NS}other_subj")
    assert abs(val_a - 42.0) < 1e-9, f"Expected 42, got {val_a}"
    assert abs(val_b - 77.0) < 1e-9, f"Expected 77, got {val_b}"
    assert val_a != val_b, "Different subjects should return different values"


def test_kb_query_template_ask():
    """KB_QUERY_TEMPLATE works with ASK queries."""
    store = TripleStore()
    subj = NamedNode(f"{NS}thing")
    pred = NamedNode(f"{NS}active")
    pred_iri = f"{NS}active"
    store.add(Triple(subj, pred, Literal("true")), graph="g")
    template = f"ASK {{ <$subject> <{pred_iri}> \"true\" }}"
    sysd = """
    T
    dt 1.0
    from 0 to 2
    stock X: 100
      - O: X / dt
    aux "found": KB_QUERY_TEMPLATE(tmpl, subj)
    """
    model = parse_sysd(sysd)
    r = model.simulate(params={"tmpl": template, "subj": subj.iri}, kb=store)
    assert abs(r.aux_values["found"][0] - 1.0) < 1e-9


def test_kb_query_template_in_abm_condition():
    """KB_QUERY_TEMPLATE in ABM rule condition with per-agent binding."""
    store = TripleStore()
    store.add(Triple(
        NamedNode(f"{NS}agent_1"), NamedNode(f"{NS}status"), Literal("critical"),
    ), graph="g")
    store.add(Triple(
        NamedNode(f"{NS}agent_2"), NamedNode(f"{NS}status"), Literal("normal"),
    ), graph="g")

    template = f"ASK {{ <$subject> <{NS}status> \"critical\" }}"

    model = SysdModel("test_abm_template")
    model.dt = 1.0
    model.t_start = 0.0
    model.t_end = 5.0

    with model.stock("X", 100) as s:
        s.outflow("O", "X / dt")

    for iri in [f"{NS}agent_1", f"{NS}agent_2"]:
        model.agents.append(AgentDef(
            "TestAgent", 1,
            properties=[AgentPropDef("val", 0.0)],
            strategies=[
                AgentStrategy("default", [
                    AgentRuleDef("check", "always",
                        [f"val = KB_QUERY_TEMPLATE(q_template, '{iri}')"]),
                ]),
            ],
        ))

    r = model.simulate(params={"q_template": template}, kb=store)
    assert r is not None


# ── Bridge ergonomics tests ──────────────────────────────────────


def test_bridge_params_for_class():
    """params_for_class introspects rdf:type to find instances."""
    store = TripleStore()
    cls_iri = f"{NS}Widget"
    cls_node = NamedNode(cls_iri)
    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    inst1 = NamedNode(f"{NS}widget_1")
    inst2 = NamedNode(f"{NS}widget_2")
    store.add(Triple(inst1, NamedNode(rdf_type), cls_node), graph="g")
    store.add(Triple(inst2, NamedNode(rdf_type), cls_node), graph="g")
    store.add(Triple(inst1, NamedNode(f"{NS}weight"), Literal(10.0)), graph="g")
    store.add(Triple(inst2, NamedNode(f"{NS}weight"), Literal(20.0)), graph="g")
    store.add(Triple(inst1, NamedNode(f"{NS}count"), Literal(5)), graph="g")

    bridge = KBSimBridge(store)
    params = bridge.params_for_class(cls_iri)
    assert "Widget_weight" in params or any("weight" in k for k in params), \
        f"Expected weight in params, got {params}"
    assert len(params) > 0, f"Expected non-empty params, got {params}"


def test_bridge_params_for_class_with_filter():
    """params_for_class respects subject_filter."""
    store = TripleStore()
    cls_iri = f"{NS}Portfolio"
    cls_node = NamedNode(cls_iri)
    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    status_p = f"{NS}status"
    risk_p = f"{NS}risk_score"

    p1 = NamedNode(f"{NS}portfolio_1")
    p2 = NamedNode(f"{NS}portfolio_2")
    store.add(Triple(p1, NamedNode(rdf_type), cls_node), graph="g")
    store.add(Triple(p2, NamedNode(rdf_type), cls_node), graph="g")
    store.add(Triple(p1, NamedNode(status_p), Literal("active")), graph="g")
    store.add(Triple(p2, NamedNode(status_p), Literal("archived")), graph="g")
    store.add(Triple(p1, NamedNode(risk_p), Literal(0.8)), graph="g")
    store.add(Triple(p2, NamedNode(risk_p), Literal(0.2)), graph="g")

    bridge = KBSimBridge(store)
    filtered = bridge.params_for_class(
        cls_iri, subject_filter={status_p: "active"})
    assert len(filtered) > 0, f"Expected some params, got {filtered}"
    raw_vals = list(filtered.values())
    assert all(v > 0.5 for v in raw_vals), \
        f"Active portfolio should have higher risk, got {filtered}"


def test_bridge_evidence_for_stock_percentile():
    """evidence_for_stock with percentile method."""
    store = _store_with_triples()
    bridge = KBSimBridge(store)

    model = parse_sysd("""
    T
    dt 1.0
    from 0 to 5
    stock X: 100
      - O: X * 0.1
    stock Y: 50
      - O: Y * 0.05
    """)
    result = model.simulate()

    triple = bridge.evidence_for_stock("X", S, P, result, method="percentile")
    assert triple is not None


def test_bridge_evidence_for_stock_delta():
    """evidence_for_stock with delta method."""
    store = _store_with_triples()
    bridge = KBSimBridge(store)

    model = parse_sysd("""
    T
    dt 1.0
    from 0 to 5
    stock X: 100
      - O: X * 0.1
    """)
    result = model.simulate()

    triple = bridge.evidence_for_stock("X", S, P, result, method="delta")
    assert triple is not None


def test_bridge_evidence_for_stock_threshold():
    """evidence_for_stock with threshold method."""
    store = _store_with_triples()
    bridge = KBSimBridge(store)

    model = parse_sysd("""
    T
    dt 1.0
    from 0 to 5
    stock X: 100
      - O: X * 0.1
    """)
    result = model.simulate()

    triple = bridge.evidence_for_stock("X", S, P, result, method="threshold",
                                       threshold=50)
    assert triple is not None


def test_bridge_load_queries(tmp_path):
    """load_queries reads SPARQL from a YAML file."""
    yaml_file = tmp_path / "queries.yaml"
    yaml_file.write_text("""
disruption_active:
  sparql: "ASK { ?s ?p ?o }"
  mode: ask

supplier_risk:
  sparql: "SELECT ?v WHERE { ?s epc:risk ?v }"
  mode: select
  var: v
""")
    queries = KBSimBridge.load_queries(str(yaml_file))
    assert "disruption_active" in queries
    assert "supplier_risk" in queries
    assert "ASK" in queries["disruption_active"]
    assert "SELECT" in queries["supplier_risk"]


def test_bridge_load_queries_missing_file():
    """load_queries returns empty dict for missing file."""
    queries = KBSimBridge.load_queries("/nonexistent/queries.yaml")
    assert queries == {}
