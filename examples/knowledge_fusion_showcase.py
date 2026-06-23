#!/usr/bin/env python3
"""Knowledge Fusion Showcase — full pipeline with KBT + argumentation + EvidenceMatrix.

Pipeline:
  Turtle sources → Named graphs → RDFS inference
  → KBT (compute source reliability)
  → EvidenceMatrix (structured consensus analysis)
  → ARGUMENTATION (filter defeated claims using KBT reliability)
  → SL fusion → SPARQL query grading

Three sources (Alpha=reliable, Bravo=medium, Charlie=unreliable) make
agreeing and contradictory claims about AtlasCorp. The pipeline
automatically identifies contradictions, scores source trustworthiness,
and produces a clean fused knowledge base.
"""

from cognitive_engine.core.models import Opinion
from cognitive_engine.kb.model import (
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
)
from cognitive_engine.kb.store import TripleStore
from cognitive_engine.kb.turtle import parse_turtle, serialize_turtle
from cognitive_engine.kb.sparql import evaluate, parse_sparql
from cognitive_engine.kb.inference import RuleEngine, rdfs_rules
from cognitive_engine.kb.confidence import (
    fuse_graphs,
    grade_query,
)
from cognitive_engine.reason.argumentation import (
    AttackType,
    build_framework,
)
from cognitive_engine.reason.evidence import ConsensusLevel, EvidenceMatrix
from cognitive_engine.reason.fusion import cumulative_fusion
from cognitive_engine.reason.kbt import compute_kbt

EX = "http://example.org/"
atlas = NamedNode(f"{EX}AtlasCorp")
hasCEO = NamedNode(f"{EX}hasCEO")
revenue = NamedNode(f"{EX}revenue")
subsidiary = NamedNode(f"{EX}subsidiary")
publicRating = NamedNode(f"{EX}publicRating")

# ── 1. SCHEMA — RDFS domain/range definitions ────────────────────

SCHEMA = """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .

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

# ── 2. SOURCE DATA — three sources with varying reliability ──────

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


def _shorten(iri: str) -> str:
    for prefix, ns in [("ex:", EX)]:
        if iri.startswith(ns):
            return iri.replace(ns, prefix, 1)
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.split("/")[-1] if "/" in iri else iri


def _claim_key(t: Triple) -> str:
    """Map a triple to a human-readable claim name."""
    pred = _shorten(str(t.predicate.iri))
    obj = str(t.object_.value) if hasattr(t.object_, "value") else _shorten(str(t.object_.iri))
    return f"{pred}={obj}"


def _build_evidence_matrix(store: TripleStore, source_graphs: list[str]) -> EvidenceMatrix:
    """Build an EvidenceMatrix from named graphs.

    Gathers opinions for each triple across all sources,
    mapping them to human-readable claim names.
    """
    matrix = EvidenceMatrix()
    for g in source_graphs:
        opinions: dict[str, Opinion] = {}
        for t in store.triples_in_graph(g):
            claim = _claim_key(t)
            if claim not in opinions:
                opinions[claim] = t.opinion or Opinion()
            else:
                opinions[claim] = Opinion.from_tuple(
                    cumulative_fusion(opinions[claim], t.opinion or Opinion())
                )
        matrix.add_source(g, opinions)
    return matrix


def _print_results(label: str, r):
    print(f"\n  {label}:")
    if not r.bindings:
        print("    (no results)")
        return
    for i, (b, om) in enumerate(zip(r.bindings, r.opinions)):
        parts = []
        for k, v in b.items():
            if hasattr(v, "iri"):
                parts.append(f"{k}={_shorten(v.iri)}")
            elif hasattr(v, "value"):
                parts.append(f"{k}={v.value}")
            else:
                parts.append(f"{k}={v}")
        opin = ", ".join(f"{k}=(b={o.belief:.2f})" for k, o in om.items() if o)
        print(f"    [{i+1}]  {', '.join(parts)}  [{opin}]")


def _consensus_label(c: ConsensusLevel) -> str:
    return {
        ConsensusLevel.STRONG_AGREEMENT: "STRONG AGREEMENT",
        ConsensusLevel.MILD_AGREEMENT: "MILD AGREEMENT",
        ConsensusLevel.CONTESTED: "CONTESTED",
        ConsensusLevel.STRONG_DISAGREEMENT: "STRONG DISAGREEMENT",
    }.get(c, str(c))


def main():
    # ── Setup ─────────────────────────────────────────────────────
    store = TripleStore()

    # Schema
    schema_store = parse_turtle(SCHEMA)
    for t in schema_store.triples(TriplePattern()):
        store.add(t, graph="schema")

    # Sources with opinions reflecting their reliability
    sources = [
        ("alpha", ALPHA, Opinion(0.85, 0.05, 0.10)),
        ("bravo", BRAVO, Opinion(0.55, 0.15, 0.30)),
        ("charlie", CHARLIE, Opinion(0.20, 0.60, 0.20)),
    ]
    for g, text, opinion in sources:
        temp = parse_turtle(text)
        for t in temp.triples(TriplePattern()):
            store.add(Triple(t.subject, t.predicate, t.object_, opinion=opinion), graph=g)

    source_graphs = ["alpha", "bravo", "charlie"]

    print("=" * 78)
    print("KNOWLEDGE FUSION SHOWCASE — KBT + Argumentation + EvidenceMatrix")
    print("=" * 78)
    print(f"\n  Schema:     {len(list(store.triples_in_graph('schema')))} triples")
    for g, _, op in sources:
        cnt = len(list(store.triples_in_graph(g)))
        print(f"  {g:12s} {cnt} triples  (b={op.belief:.2f}, d={op.disbelief:.2f}, u={op.uncertainty:.2f})")
    print(f"  Total:      {len(store)} triples in {len(store.graphs())} graphs")

    # ── 3. RDFS Inference ─────────────────────────────────────────
    print("\n" + "-" * 78)
    print("RDFS INFERENCE")
    print("-" * 78)
    before = len(store)
    engine = RuleEngine(rdfs_rules(), max_iterations=10)
    new = engine.apply(store)
    print(f"  {before} → +{new} → {len(store)} triples")

    # ── 4. KBT — Knowledge-Based Trust ────────────────────────────
    print("\n" + "-" * 78)
    print("KBT — AUTOMATIC SOURCE RELIABILITY SCORING")
    print("-" * 78)
    kbt_result = compute_kbt(store, source_graphs)
    print(f"  Converged:    {kbt_result.converged} ({kbt_result.iterations} iterations)")
    print(f"\n  {'Source':<12} {'KBT Trust':<12} {'Iterations':<12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12}")
    for g in source_graphs:
        trust = kbt_result.source_trust[g]
        hist = kbt_result.trust_history[g]
        print(f"  {g:<12} {trust:<12.4f} {len(hist):<12}")
    print(f"\n  Trust trajectory:")
    for g in source_graphs:
        vals = kbt_result.trust_history[g]
        bar = "█" * int(vals[-1] * 20) + "░" * (20 - int(vals[-1] * 20))
        print(f"    {g:<10} {bar} {vals[-1]:.4f}")

    # ── 5. EvidenceMatrix — structured consensus ──────────────────
    print("\n" + "-" * 78)
    print("EVIDENCE MATRIX — STRUCTURED CONSENSUS ANALYSIS")
    print("-" * 78)
    matrix = _build_evidence_matrix(store, source_graphs)
    em_result = matrix.compute()
    print(f"  Sources: {em_result.source_count}, Claims: {em_result.claim_count}")
    print(f"\n  {'Claim':<30} {'Consensus':<20} {'Sources':<8} {'Mean b':<8}")
    print(f"  {'─'*30} {'─'*20} {'─'*8} {'─'*8}")
    for claim in sorted(em_result.claim_names):
        ca = em_result.claims[claim]
        n = ca.source_count
        label = _consensus_label(ca.consensus)
        print(f"  {claim:<30} {label:<20} {n:<8} {ca.belief_mean:<8.3f}")
    print(f"\n  Contested claims: {em_result.contested_claims()}")

    # ── 6. Build argumentation framework (one build, two uses) ────
    print("\n" + "-" * 78)
    print("ARGUMENTATION FRAMEWORK — DUNG GROUNDED SEMANTICS")
    print("-" * 78)
    af = build_framework(
        store, source_graphs + ["meta"],
        min_belief=0.2,
        min_attack_strength=0.5,
        auto_rebut=True,
        auto_undermine_low_belief=True,
    )
    ext = af.compute_grounded()

    rebut = [a for a in af.attacks if a.attack_type == AttackType.REBUT]
    undermine = [a for a in af.attacks if a.attack_type == AttackType.UNDERMINE]
    undercut = [a for a in af.attacks if a.attack_type == AttackType.UNDERCUT]
    print(f"  Arguments:    {len(af.arguments)}")
    print(f"  Attacks:      {len(af.attacks)}  (rebut={len(rebut)}, "
          f"undermine={len(undermine)}, undercut={len(undercut)})")
    print(f"  Survivors:    {len(ext)} / {len(af.arguments)}")

    # Build reverse map: (spo, graph) → argument ID
    spo_to_argid: dict[tuple, str] = {}
    for aid, arg in af.arguments.items():
        if arg.source_graph:
            spo_to_argid[(arg.triple.spo, arg.source_graph)] = aid

    print(f"\n  Attack map (first 10):")
    for a in af.attacks[:10]:
        src = af.arguments.get(a.source_id)
        tgt = af.arguments.get(a.target_id)
        if src and tgt:
            sp = _shorten(str(src.triple.predicate.iri))
            tp = _shorten(str(tgt.triple.predicate.iri))
            so = str(src.triple.object_.value) if hasattr(src.triple.object_, "value") else _shorten(str(src.triple.object_.iri))
            to = str(tgt.triple.object_.value) if hasattr(tgt.triple.object_, "value") else _shorten(str(tgt.triple.object_.iri))
            check = "✓" if a.target_id in ext else "✗"
            print(f"    {a.attack_type.value:<10} {sp}={so}  ─→  {tp}={to}  {check}")

    # ── 7. BEFORE vs AFTER argumentation ──────────────────────────
    print("\n" + "-" * 78)
    print("BEFORE ARGUMENTATION — RAW FUSION (includes contradictions)")
    print("-" * 78)
    result_raw = fuse_graphs(store, source_graphs, target_graph="raw_fused", method="cumulative")
    print(f"  Fused {result_raw.fused_count} overlapping triples")

    r_rev_before = evaluate(
        parse_sparql(f"PREFIX ex: <{EX}> SELECT ?rev WHERE {{ ex:AtlasCorp ex:revenue ?rev }}"),
        store,
    )
    _print_results("AtlasCorp revenue (BEFORE)", r_rev_before)

    print("\n" + "-" * 78)
    print("AFTER ARGUMENTATION — FILTERED FUSION (survivors only)")
    print("-" * 78)

    # Build filtered store from grounded extension (same framework)
    filtered = TripleStore()
    for t in store.triples_in_graph("schema"):
        filtered.add(t, graph="schema")
    for t in store.triples_in_graph("meta"):
        filtered.add(t, graph="meta")
    kept = 0
    for g in source_graphs:
        for t in store.triples_in_graph(g):
            aid = spo_to_argid.get((t.spo, g))
            if aid is not None and aid in ext:
                filtered.add(t, graph=g)
                kept += 1
    result_filtered = fuse_graphs(
        filtered,
        [g for g in filtered.graphs() if g not in ("raw_fused",)],
        target_graph="fused",
        method="cumulative",
    )
    print(f"  Survivors:    {kept} / {len(store)} data triples kept")
    print(f"  Total triples in filtered store: {len(filtered)}")
    print(f"  Fused {result_filtered.fused_count} overlapping triples")

    r_rev_after = evaluate(
        parse_sparql(f"PREFIX ex: <{EX}> SELECT ?rev WHERE {{ ex:AtlasCorp ex:revenue ?rev }}"),
        filtered,
    )
    _print_results("AtlasCorp revenue (AFTER)", r_rev_after)

    # ── 8. Side-by-side claim comparison ─────────────────────────
    print("\n" + "-" * 78)
    print("SIDE-BY-SIDE — EVERY CLAIM BEFORE vs AFTER")
    print("-" * 78)
    print(f"  {'Claim':<35} {'Source':<10} {'BEFORE b':<10} {'AFTER b':<10} {'Status':<10}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for g in source_graphs:
        for t in store.triples_in_graph(g):
            claim = _claim_key(t)
            ob = t.opinion.belief if t.opinion else 0.5
            survives = any(
                ft == t for ft in filtered.triples(TriplePattern(
                    t.subject, t.predicate, t.object_
                ), graph=g)
            )
            after_b = 0.0
            if survives:
                for ft in filtered.triples(TriplePattern(
                    t.subject, t.predicate, t.object_
                ), graph=g):
                    after_b = ft.opinion.belief if ft.opinion else 0.5
            status = "KEPT" if survives else "DEFEATED"
            print(f"  {claim:<35} {g:<10} {ob:<10.3f} {after_b:<10.3f} {status:<10}")

    # ── 9. SPARQL query grading ──────────────────────────────────
    print("\n" + "-" * 78)
    print("SPARQL QUERY GRADING — CONFIDENCE ON FUSED KNOWLEDGE")
    print("-" * 78)

    queries = [
        ("CEO of AtlasCorp", f"PREFIX ex: <{EX}> SELECT ?ceo WHERE {{ ex:AtlasCorp ex:hasCEO ?ceo }}"),
        ("Revenue of AtlasCorp", f"PREFIX ex: <{EX}> SELECT ?rev WHERE {{ ex:AtlasCorp ex:revenue ?rev }}"),
        ("Subsidiary of AtlasCorp", f"PREFIX ex: <{EX}> SELECT ?sub WHERE {{ ex:AtlasCorp ex:subsidiary ?sub }}"),
        ("Public rating", f"PREFIX ex: <{EX}> SELECT ?r WHERE {{ ex:AtlasCorp ex:publicRating ?r }}"),
        ("All known facts", f"PREFIX ex: <{EX}> SELECT ?p ?o WHERE {{ ex:AtlasCorp ?p ?o }}"),
    ]

    for label, sparql in queries:
        algebra = parse_sparql(sparql)
        result = evaluate(algebra, filtered)
        grade = grade_query(result)
        print(f"\n  {label}:")
        _print_results("  Result", result)
        print(f"    Grade: avg b={grade.avg_belief:.3f}, "
              f"consensus={grade.consensus} ({grade.label()})")

    # ── 10. Export fused Turtle ──────────────────────────────────
    print("\n" + "-" * 78)
    print("FUSED KNOWLEDGE — TURTLE OUTPUT")
    print("-" * 78)
    fused_triples = list(filtered.triples_in_graph("fused"))
    if fused_triples:
        ttl = serialize_turtle(fused_triples, prefixes={"ex": EX})
        for line in ttl.strip().split("\n")[:15]:
            print(f"  {line}")
        if len(ttl.strip().split("\n")) > 15:
            print(f"  ... ({len(ttl.strip().split('\n'))} total lines)")

    # ── 11. Summary ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Pipeline:   Schema → 3 sources → RDFS inference → KBT")
    print(f"              → EvidenceMatrix → Argumentation → Fusion → SPARQL")
    print(f"  Sources:    alpha (b=0.85), bravo (b=0.55), charlie (b=0.20)")
    print(f"  KBT trust:  ", end="")
    for g in source_graphs:
        print(f"{g}={kbt_result.source_trust[g]:.3f} ", end="")
    print()
    print(f"  RDFS inf:   +{new} triples derived")
    print(f"  Attacks:    {len(af.attacks)} ({len(rebut)} rebut, "
          f"{len(undermine)} undermine, {len(undercut)} undercut)")
    print(f"  Survivors:  {len(ext)}/{len(af.arguments)} arguments in grounded extension")
    print(f"  Fused raw:  {result_raw.fused_count} triples")
    print(f"  Fused aft:  {result_filtered.fused_count} triples")
    print(f"  Contested:  {em_result.contested_claims()}")
    print(f"  Consensus:  {sum(1 for c in em_result.claims.values() if c.consensus == ConsensusLevel.STRONG_AGREEMENT)} strong, "
          f"{sum(1 for c in em_result.claims.values() if c.consensus == ConsensusLevel.CONTESTED)} contested")
    print("=" * 78)
    print("  Charlie's contradicted claims ($2M revenue, ThetaCorp")
    print("  subsidiary) are rebutted by Alpha+Bravo corroboration")
    print("  and undermined by low KBT trust scores. Only")
    print("  corroborated claims survive argumentation and reach SL")
    print("  fusion. SPARQL queries against the filtered KB show")
    print("  higher confidence consensus.")
    print("=" * 78)


if __name__ == "__main__":
    main()
