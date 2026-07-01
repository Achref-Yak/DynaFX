#!/usr/bin/env python3
"""Argumentation Showcase — claim filtering via Dung grounded semantics.

Pipeline:
  Turtle sources → Named graphs → RDFS inference
  → ARGUMENTATION (filter defeated claims)
  → SL fusion → Query grading

Three sources (Alpha=reliable, Bravo=medium, Charlie=unreliable) make
agreeing and contradictory claims about Atlas Corp. Argumentation
removes Charlie's contradicted claims before fusion, producing cleaner
results.
"""

from dynafx.core.models import Opinion
from dynafx.knowledge import (
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
    TripleStore,
    FusionResult,
    QueryGrade,
    RuleEngine,
    argumentative_filter,
    evaluate,
    fuse_graphs,
    grade_query,
    parse_sparql,
    parse_turtle,
    rdfs_rules,
    serialize_turtle,
)
from dynafx.epistemics.argumentation import (
    PROV_NS,
    SOURCE_RELIABILITY,
    AttackType,
    build_framework,
)

EX = "http://example.org/"
atlas = NamedNode(f"{EX}AtlasCorp")
alice = NamedNode(f"{EX}Alice")
bob = NamedNode(f"{EX}Bob")
carol = NamedNode(f"{EX}Carol")
omega = NamedNode(f"{EX}OmegaCorp")
theta = NamedNode(f"{EX}ThetaCorp")
hasCEO = NamedNode(f"{EX}hasCEO")
revenue = NamedNode(f"{EX}revenue")
subsidiary = NamedNode(f"{EX}subsidiary")
publicRating = NamedNode(f"{EX}publicRating")

# ── 1. Schema ─────────────────────────────────────────────────────

SCHEMA = """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .
@prefix prov: <http://cognitive.engine/provenance#> .

ex:Organization rdf:type rdfs:Class .
ex:Person rdf:type rdfs:Class .

ex:hasCEO rdfs:domain ex:Organization .
ex:hasCEO rdfs:range ex:Person .
ex:revenue rdfs:domain ex:Organization .
ex:revenue rdfs:range rdfs:Literal .
ex:subsidiary rdfs:domain ex:Organization .
ex:subsidiary rdfs:range ex:Organization .
ex:publicRating rdfs:domain ex:Organization .
ex:publicRating rdfs:range rdfs:Literal .
"""

# ── 2. Source data ────────────────────────────────────────────────

ALPHA = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:AtlasCorp ex:hasCEO ex:Alice .
ex:AtlasCorp ex:revenue "5000000"^^xsd:integer .
ex:AtlasCorp ex:subsidiary ex:OmegaCorp .
ex:AtlasCorp rdf:type ex:Organization .
"""

BRAVO = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:AtlasCorp ex:hasCEO ex:Alice .
ex:AtlasCorp ex:revenue "5000000"^^xsd:integer .
ex:AtlasCorp ex:subsidiary ex:OmegaCorp .
ex:AtlasCorp rdf:type ex:Organization .
"""

CHARLIE = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:AtlasCorp ex:revenue "2000000"^^xsd:integer .
ex:AtlasCorp ex:subsidiary ex:ThetaCorp .
ex:AtlasCorp ex:publicRating "Good" .
ex:AtlasCorp rdf:type ex:Organization .
"""


def _parse_with_opinion(text: str, opinion: Opinion) -> list[Triple]:
    temp = parse_turtle(text)
    return [
        Triple(t.subject, t.predicate, t.object_, opinion=opinion)
        for t in temp.triples(TriplePattern())
    ]


def _show_result(label: str, r):
    print(f"\n  {label}:")
    if not r.bindings:
        print("    (no results)")
        return
    for i, (b, om) in enumerate(zip(r.bindings, r.opinions)):
        desc = ", ".join(f"{k}={v.n3() if hasattr(v, 'n3') else v}" for k, v in b.items())
        opin = ", ".join(f"{k}=(b={o.belief:.2f})" for k, o in om.items() if o)
        print(f"    [{i+1}]  {desc}  [{opin}]")


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("ARGUMENTATION SHOWCASE — Dung grounded semantics + SL fusion")
    print("=" * 72)

    store = TripleStore()

    # Schema
    schema_store = parse_turtle(SCHEMA)
    for t in schema_store.triples(TriplePattern()):
        store.add(t, graph="schema")
    print(f"\nSchema: {len(list(store.triples_in_graph('schema')))} triples")

    # Sources with opinions
    src_a = _parse_with_opinion(ALPHA, Opinion(0.85, 0.05, 0.10))
    src_b = _parse_with_opinion(BRAVO, Opinion(0.55, 0.15, 0.30))
    src_c = _parse_with_opinion(CHARLIE, Opinion(0.20, 0.60, 0.20))

    for t in src_a: store.add(t, graph="alpha")
    for t in src_b: store.add(t, graph="bravo")
    for t in src_c: store.add(t, graph="charlie")

    # Source reliability meta-triples (inline — could come from ReputationTracker)
    store.add(Triple(NamedNode("alpha"), SOURCE_RELIABILITY, Literal(0.9)), graph="meta")
    store.add(Triple(NamedNode("bravo"), SOURCE_RELIABILITY, Literal(0.6)), graph="meta")
    store.add(Triple(NamedNode("charlie"), SOURCE_RELIABILITY, Literal(0.15)), graph="meta")

    print(f"Alpha: 5 triples (b=0.85, reliability=0.90)")
    print(f"Bravo: 5 triples (b=0.55, reliability=0.60)")
    print(f"Charlie: 4 triples (b=0.20, reliability=0.15)")
    print(f"Total: {len(store)} triples in {len(store.graphs())} graphs")

    # ── 3. RDFS inference ──────────────────────────────────────────

    print("\n" + "-" * 72)
    print("RDFS INFERENCE")
    print("-" * 72)
    before = len(store)
    engine = RuleEngine(rdfs_rules(), max_iterations=10)
    new = engine.apply(store)
    print(f"  {before} → +{new} → {len(store)} triples")

    # ── 4. BEFORE argumentation: raw fusion ────────────────────────

    print("\n" + "-" * 72)
    print("BEFORE ARGUMENTATION — fusion includes all contradictions")
    print("-" * 72)
    result_raw = fuse_graphs(store, ["alpha", "bravo", "charlie"],
                             target_graph="raw_fused", method="cumulative")
    print(f"  Fused {result_raw.fused_count} overlapping triples")

    r1 = evaluate(parse_sparql(f"PREFIX ex: <{EX}> SELECT ?rev WHERE {{ ex:AtlasCorp ex:revenue ?rev }}"),
                  store)
    _show_result("Atlas revenue (BEFORE argumentation)", r1)

    # ── 5. Build argumentation framework ───────────────────────────

    print("\n" + "-" * 72)
    print("ARGUMENTATION FRAMEWORK")
    print("-" * 72)

    af = build_framework(
        store, ["alpha", "bravo", "charlie", "meta"],
        min_belief=0.2,
        auto_rebut=True,
        auto_undermine_low_belief=True,
    )
    ext = af.compute_grounded()
    print(f"  Arguments:    {len(af.arguments)}")
    print(f"  Attacks:      {len(af.attacks)}")
    print(f"  Survivors:    {len(ext)} / {len(af.arguments)}")

    # Show attack breakdown
    rebut = [a for a in af.attacks if a.attack_type == AttackType.REBUT]
    undermine = [a for a in af.attacks if a.attack_type == AttackType.UNDERMINE]
    print(f"  Rebut:        {len(rebut)}")
    print(f"  Undermine:    {len(undermine)}")

    print("\n  Attack map:")
    for a in af.attacks[:8]:
        src = af.arguments.get(a.source_id)
        tgt = af.arguments.get(a.target_id)
        s_str = f"{a.source_id} [{_shorten(str(src.triple.subject.iri))}]" if src else a.source_id
        t_str = f"{a.target_id} [{_shorten(str(tgt.triple.predicate.iri))}]" if tgt else a.target_id
        check = "✓" if a.target_id in ext else "✗"
        print(f"    {s_str} ─{a.attack_type.value}→ {t_str}  {check}")

    # ── 6. AFTER argumentation: filtered fusion ────────────────────

    print("\n" + "-" * 72)
    print("AFTER ARGUMENTATION — only surviving claims fused")
    print("-" * 72)

    filtered = argumentative_filter(store, ["alpha", "bravo", "charlie", "meta"],
                                    min_belief=0.2)

    # Re-attach schema and meta graphs (they have no contradictions, but
    # argumentative_filter removes triples not in the source graphs).
    # Actually, argumentative_filter returns a store with only acceptable triples
    # across all graphs. Let me re-parse schema into it for clean state.
    result_filtered = fuse_graphs(
        filtered,
        [g for g in filtered.graphs() if g != "fused"],
        target_graph="fused",
        method="cumulative",
    )
    print(f"  Triples after filter: {len(filtered)}")
    print(f"  Fused {result_filtered.fused_count} overlapping triples")

    r2 = evaluate(parse_sparql(f"PREFIX ex: <{EX}> SELECT ?rev WHERE {{ ex:AtlasCorp ex:revenue ?rev }}"),
                  filtered)
    _show_result("Atlas revenue (AFTER argumentation)", r2)

    # ── 7. Compare side-by-side ────────────────────────────────────

    print("\n" + "-" * 72)
    print("BEFORE vs AFTER — revenue claims")
    print("-" * 72)

    print(f"  {'Claim':<25} {'BEFORE':<15} {'AFTER':<15}")
    print(f"  {'─'*25} {'─'*15} {'─'*15}")
    for g in ["alpha", "bravo", "charlie"]:
        for t in store.triples(TriplePattern(atlas, revenue, None), graph=g):
            o = t.opinion
            survives = any(
                ft == t for ft in filtered.triples(TriplePattern(atlas, revenue, None), graph=g)
            )
            print(f"  {str(t.object_):<25} {f'b={o.belief:.2f} ({g})':<15} {'KEPT' if survives else 'DEFEATED':<15}")

    # ── 8. Export fused Turtle ─────────────────────────────────────

    print("\n" + "-" * 72)
    print("FUSED OUTPUT (survivors, Turtle)")
    print("-" * 72)
    fused_triples = list(filtered.triples_in_graph("fused"))
    if fused_triples:
        ttl = serialize_turtle(fused_triples[:6], prefixes={"ex": EX})
        for line in ttl.strip().split("\n")[:10]:
            print(f"  {line}")

    # ── 9. Summary ─────────────────────────────────────────────────

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Source graphs:       alpha (b=0.85, rel=0.90)")
    print(f"                       bravo (b=0.55, rel=0.60)")
    print(f"                       charlie (b=0.20, rel=0.15)")
    print(f"  Total triples:       {len(store)}")
    print(f"  After filter:        {len(filtered)}")
    print(f"  Attacks generated:   {len(af.attacks)}")
    print(f"  Grounded survivors:  {len(ext)} / {len(af.arguments)}")
    print(f"  Fused (before):      {result_raw.fused_count} triples")
    print(f"  Fused (after):       {result_filtered.fused_count} triples")
    print("=" * 72)
    print("  Charlie's contradicted claims ($2M revenue, ThetaCorp")
    print("  subsidiary, 'Good' rating) are rebutted/undermined by")
    print("  Alpha+Bravo corroboration and low source reliability.")
    print("  Only corroborated claims reach SL fusion.")
    print("=" * 72)


def _shorten(iri: str) -> str:
    """Shorten an IRI for display."""
    for prefix, ns in [("ex:", EX), ("prov:", PROV_NS)]:
        if iri.startswith(ns):
            return iri.replace(ns, prefix, 1)
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.split("/")[-1] if "/" in iri else iri


if __name__ == "__main__":
    main()
