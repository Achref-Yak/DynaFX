#!/usr/bin/env python3
"""Knowledge Base Showcase — RDF/SPARQL/Inference/Confidence with SL grading.

Narrative: Three intelligence sources (Alpha, Beta, Gamma) report on
Acme Corp and its relationships. Each source has different confidence.
We fuse their reports and grade the results.
"""

from dynafx.kb import (
    # Model
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
    # Store
    TripleStore,
    # Turtle
    parse_turtle,
    serialize_turtle,
    # SPARQL
    QueryResult,
    evaluate,
    parse_sparql,
    # Inference
    RuleEngine,
    rdfs_rules,
    # Confidence
    QueryGrade,
    fuse_graphs,
    grade_query,
    # SL
    Opinion,
    cumulative_fusion,
)

# ── 1. Schema (RDFS ontology) ─────────────────────────────────────

SCHEMA = """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .

ex:Organization rdf:type rdfs:Class .
ex:Person rdf:type rdfs:Class .
ex:Product rdf:type rdfs:Class .

ex:hasCEO rdfs:domain ex:Organization .
ex:hasCEO rdfs:range ex:Person .

ex:revenue rdfs:domain ex:Organization .
ex:revenue rdfs:range rdfs:Literal .

ex:subsidiary rdfs:domain ex:Organization .
ex:subsidiary rdfs:range ex:Organization .
"""

# ── 2. Source graph data (Turtle strings) ─────────────────────────

ALPHA = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:AcmeCorp ex:hasCEO ex:Alice .
ex:AcmeCorp ex:revenue "1000000"^^xsd:integer .
ex:AcmeCorp rdf:type ex:Organization .
"""

BETA = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:AcmeCorp ex:hasCEO ex:Alice .
ex:AcmeCorp ex:subsidiary ex:BobCorp .
ex:BobCorp ex:hasCEO ex:Bob .
ex:BobCorp rdf:type ex:Organization .
"""

GAMMA = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:AcmeCorp ex:revenue "1000000"^^xsd:integer .
ex:AcmeCorp ex:subsidiary ex:CarolCorp .
ex:CarolCorp rdf:type ex:Organization .
"""


def main():
    print("=" * 70)
    print("KB SHOWCASE — Multi-source Intelligence Fusion with SL Grading")
    print("=" * 70)

    # ── 3. Build store ────────────────────────────────────────────

    store = TripleStore()

    # Parse schema into default graph
    schema_store = parse_turtle(SCHEMA, default_graph="schema")
    for t in list(schema_store.triples(TriplePattern())):
        store.add(t, graph="schema")
    print(f"\nSchema parsed: {len(list(store.triples_in_graph('schema')))} triples")

    # Parse each source into its named graph, injecting opinions
    src_a = _parse_with_opinion(ALPHA, Opinion(0.85, 0.05, 0.10))
    src_b = _parse_with_opinion(BETA, Opinion(0.60, 0.20, 0.20))
    src_c = _parse_with_opinion(GAMMA, Opinion(0.40, 0.30, 0.30))

    for t in src_a:
        store.add(t, graph="alpha")
    for t in src_b:
        store.add(t, graph="beta")
    for t in src_c:
        store.add(t, graph="gamma")

    print(f"Alpha: {len(list(store.triples_in_graph('alpha')))} triples (b=0.85)")
    print(f"Beta:  {len(list(store.triples_in_graph('beta')))} triples (b=0.60)")
    print(f"Gamma: {len(list(store.triples_in_graph('gamma')))} triples (b=0.40)")
    print(f"Total store: {len(store)} triples")

    # ── 4. SPARQL queries ─────────────────────────────────────────

    print("\n" + "-" * 50)
    print("SPARQL QUERIES")
    print("-" * 50)

    acme = NamedNode("http://example.org/AcmeCorp")
    ex_ns = "http://example.org/"

    # Query 1: Who is Acme's CEO?
    q1 = f"""\
PREFIX ex: <{ex_ns}>
SELECT ?ceo WHERE {{
  ex:AcmeCorp ex:hasCEO ?ceo .
}}
"""
    r1 = evaluate(parse_sparql(q1), store)
    _show_query("Acme's CEO", r1)

    # Query 2: What is Acme's revenue? (conflicting sources)
    q2 = f"""\
PREFIX ex: <{ex_ns}>
SELECT ?rev WHERE {{
  ex:AcmeCorp ex:revenue ?rev .
}}
"""
    r2 = evaluate(parse_sparql(q2), store)
    _show_query("Acme's revenue (all sources)", r2)

    # Query 3: All subsidiaries
    q3 = f"""\
PREFIX ex: <{ex_ns}>
SELECT ?sub WHERE {{
  ex:AcmeCorp ex:subsidiary ?sub .
}}
"""
    r3 = evaluate(parse_sparql(q3), store)
    _show_query("Acme's subsidiaries", r3)

    # Query 4: Organizations (ASK)
    q4 = f"""\
PREFIX ex: <{ex_ns}>
ASK WHERE {{
  ex:AcmeCorp rdf:type ex:Organization .
}}
"""
    r4 = evaluate(parse_sparql(q4), store)
    print(f"\n  Is Acme an Organization? (ASK): {bool(r4.cardinality)}")

    # ── 5. RDFS Inference ─────────────────────────────────────────

    print("\n" + "-" * 50)
    print("RDFS INFERENCE")
    print("-" * 50)

    before = len(store)
    engine = RuleEngine(rdfs_rules(), max_iterations=10)
    new_count = engine.apply(store)
    print(f"  Before: {before} triples | +{new_count} inferred | After: {len(store)}")

    # Check inferred types
    # hasCEO domain=Organization → AcmeCorp rdf:type Organization (already asserted)
    # hasCEO range=Person → Alice rdf:type Person
    # hasCEO domain=Organization → BobCorp rdf:type Organization (from beta)
    # revenue domain=Organization → AcmeCorp rdf:type Organization
    # subsidiary domain=Organization → AcmeCorp, BobCorp, CarolCorp rdf:type Organization
    alice = NamedNode("http://example.org/Alice")
    bob = NamedNode("http://example.org/Bob")
    pat_type = lambda s: TriplePattern(s, NamedNode(f"{ex_ns}type"), NamedNode(f"{ex_ns}Person"))
    print(f"  Alice rdf:type Person: {TriplePattern(alice, NamedNode('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), NamedNode(f'{ex_ns}Person')) in store}")

    # ── 6. Graph fusion ──────────────────────────────────────────

    print("\n" + "-" * 50)
    print("GRAPH FUSION (cumulative)")
    print("-" * 50)

    result = fuse_graphs(store, ["alpha", "beta", "gamma"],
                         target_graph="fused", method="cumulative")
    print(f"  Fused {result.fused_count} overlapping triples "
          f"(agreement ratio: {result.agreement_ratio:.2f})")

    # Show the fused revenue (Alpha 1M vs Gamma 1M, different opinions)
    print("\n  Fused Acme revenue opinions per source:")
    for g in ["alpha", "beta", "gamma", "fused"]:
        for t in store.triples(TriplePattern(acme, NamedNode(f"{ex_ns}revenue"), None), graph=g):
            o = t.opinion
            print(f"    [{g:6s}]  {t.object_}  b={o.belief:.2f} d={o.disbelief:.2f} u={o.uncertainty:.2f}")

    print("\n  Fused Acme CEO opinions per source:")
    for g in ["alpha", "beta", "fused"]:
        for t in store.triples(TriplePattern(acme, NamedNode(f"{ex_ns}hasCEO"), None), graph=g):
            o = t.opinion
            print(f"    [{g:6s}]  {t.object_}  b={o.belief:.2f} d={o.disbelief:.2f} u={o.uncertainty:.2f}")

    # ── 7. Query grading ──────────────────────────────────────────

    print("\n" + "-" * 50)
    print("QUERY GRADING")
    print("-" * 50)

    grade = grade_query(r2)
    print(f"\n  'Acme revenue' query grade:")
    print(f"    Avg belief:      {grade.avg_belief:.3f}")
    print(f"    Avg disbelief:   {grade.avg_disbelief:.3f}")
    print(f"    Avg uncertainty: {grade.avg_uncertainty:.3f}")
    print(f"    Consensus:       {grade.consensus.upper()}")
    print(f"    Cardinality:     {grade.cardinality}")
    print(f"    Label:           {grade.label()}")

    # Grade a higher-confidence query
    grade_ceo = grade_query(r1)
    print(f"\n  'Acme CEO' query grade:")
    print(f"    Avg belief:      {grade_ceo.avg_belief:.3f}")
    print(f"    Consensus:       {grade_ceo.consensus.upper()}")
    print(f"    Label:           {grade_ceo.label()}")

    # ── 8. Turtle roundtrip ───────────────────────────────────────

    print("\n" + "-" * 50)
    print("TURTLE SERIALIZATION (fused graph snippet)")
    print("-" * 50)

    fused_triples = list(store.triples_in_graph("fused"))
    ttl = serialize_turtle(fused_triples[:6],
                           prefixes={"ex": ex_ns})
    for line in ttl.strip().split("\n")[:12]:
        print(f"  {line}")

    # ── 9. Summary ────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total triples:  {len(store)}")
    print(f"  Named graphs:   {len(store.graphs())}")
    print(f"  Inference pass: +{new_count} triples")
    print(f"  Fused triples:  {result.fused_count}")
    print(f"  Graphs:         {store.graphs()}")
    print("=" * 70)


# ── Helpers ──────────────────────────────────────────────────────


def _parse_with_opinion(text: str, opinion: Opinion) -> list[Triple]:
    """Parse Turtle and override all triples with a given opinion."""
    temp_store = parse_turtle(text)
    result = []
    for t in list(temp_store.triples(TriplePattern())):
        result.append(Triple(t.subject, t.predicate, t.object_, opinion=opinion))
    return result


def _show_query(label: str, result: QueryResult) -> None:
    print(f"\n  {label}:")
    if not result.bindings:
        print("    (no results)")
        return
    for i, (binding, opin_map) in enumerate(
        zip(result.bindings, result.opinions)
    ):
        desc = ", ".join(f"{k}={v.n3() if hasattr(v, 'n3') else v}"
                         for k, v in binding.items())
        opin_strs = []
        for var, opin in opin_map.items():
            if opin is not None:
                opin_strs.append(f"{var}=(b={opin.belief:.2f})")
        opin_part = f" [{', '.join(opin_strs)}]" if opin_strs else ""
        print(f"    [{i+1}]  {desc}{opin_part}")


if __name__ == "__main__":
    main()
