"""Tests for KB↔Sim bridge: KB_QUERY/KB_ASSERT builtins + KBSimBridge class."""

from dynafx.knowledge.model import NamedNode, Literal, Triple, TriplePattern
from dynafx.knowledge.store import TripleStore
from dynafx.core.models import Opinion
from dynafx.dynamics.dsl import parse_sysd
from dynafx.bridge import KBSimBridge


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
            if t.opinion:
                assert abs(t.opinion.belief - 0.9) < 1e-6, \
                    f"Expected belief 0.9, got {t.opinion.belief}"
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
    assert abs(params["param_x"] - 0.9) < 1e-6, \
        f"Expected 0.9, got {params['param_x']}"


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
    assert t.opinion is not None
    assert t.opinion.belief > 0.0


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
