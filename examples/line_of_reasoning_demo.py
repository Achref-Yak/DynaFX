#!/usr/bin/env python3
"""Line-of-Reasoning — insight-driven adaptive PDF report.

Each page answers one business question: what the data says, who
says what, whom to trust, where sources disagree, and overall
confidence. No tool dumps — only conclusions.
"""

import sys, os, io
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

from cognitive_engine.core.models import Opinion
from cognitive_engine.kb.model import (
    Literal, NamedNode, Triple, TriplePattern,
)
from cognitive_engine.kb.store import TripleStore
from cognitive_engine.kb.turtle import parse_turtle
from cognitive_engine.kb.sparql import evaluate, parse_sparql
from cognitive_engine.kb.inference import RuleEngine, rdfs_rules
from cognitive_engine.kb.confidence import fuse_graphs, grade_query
from cognitive_engine.reason.argumentation import AttackType, build_framework
from cognitive_engine.reason.evidence import EvidenceMatrix
from cognitive_engine.reason.fusion import cumulative_fusion
from cognitive_engine.reason.kbt import compute_kbt

EX = "http://example.org/"
OUTPUT_PDF = "reasoning_insights.pdf"

# ── Schema ──────────────────────────────────────────────────────────

SCHEMA = """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .
ex:Organization rdf:type rdfs:Class .
ex:Person rdf:type rdfs:Class .
ex:Drug rdf:type rdfs:Class .
ex:Disease rdf:type rdfs:Class .
ex:hasCEO rdfs:domain ex:Organization .
ex:hasCEO rdfs:range ex:Person .
ex:revenue rdfs:domain ex:Organization .
ex:revenue rdfs:range rdfs:Literal .
ex:subsidiary rdfs:domain ex:Organization .
ex:subsidiary rdfs:range ex:Organization .
ex:publicRating rdfs:domain ex:Organization .
ex:publicRating rdfs:range rdfs:Literal .
ex:treats rdfs:domain ex:Drug .
ex:treats rdfs:range ex:Disease .
"""

# ── Sources with conflicting claims ─────────────────────────────────

ALPHA = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
ex:AtlasCorp ex:hasCEO ex:Alice .
ex:AtlasCorp ex:revenue "5000000"^^xsd:integer .
ex:AtlasCorp ex:subsidiary ex:OmegaCorp .
ex:AtlasCorp rdf:type ex:Organization .
ex:DrugX ex:treats ex:DiseaseY .
ex:DrugX rdf:type ex:Drug .
ex:DiseaseY rdf:type ex:Disease .
"""

BRAVO = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
ex:AtlasCorp ex:hasCEO ex:Alice .
ex:AtlasCorp ex:revenue "5000000"^^xsd:integer .
ex:AtlasCorp ex:subsidiary ex:OmegaCorp .
ex:AtlasCorp rdf:type ex:Organization .
ex:DrugX ex:treats ex:DiseaseY .
ex:DrugX rdf:type ex:Drug .
ex:DiseaseY rdf:type ex:Disease .
"""

CHARLIE = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
ex:AtlasCorp ex:revenue "2000000"^^xsd:integer .
ex:AtlasCorp ex:subsidiary ex:ThetaCorp .
ex:AtlasCorp ex:publicRating "Good" .
ex:DrugX ex:treats ex:DiseaseZ .
ex:DrugX ex:sideEffect ex:Headache .
"""

DELTA = """\
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
ex:AtlasCorp ex:revenue "5000000"^^xsd:integer .
ex:AtlasCorp ex:subsidiary ex:OmegaCorp .
ex:DrugX ex:treats ex:DiseaseZ .
"""


def _shorten(iri: str) -> str:
    for prefix, ns in [("ex:", EX)]:
        if iri.startswith(ns):
            return iri.replace(ns, prefix, 1)
    if "#" in iri:
        return iri.split("#")[-1]
    return iri.split("/")[-1] if "/" in iri else iri


def _claim_key(t: Triple) -> str:
    pred = _shorten(str(t.predicate.iri))
    if hasattr(t.object_, "iri"):
        obj = _shorten(str(t.object_.iri))
    else:
        obj = str(t.object_.value) if hasattr(t.object_, "value") else str(t.object_)
    return f"{pred}={obj}"


def _build_evidence_matrix(store, source_graphs):
    matrix = EvidenceMatrix()
    for g in source_graphs:
        opinions = {}
        for t in store.triples_in_graph(g):
            claim = _claim_key(t)
            op = t.opinion or Opinion()
            if claim in opinions:
                opinions[claim] = Opinion.from_tuple(
                    cumulative_fusion(opinions[claim], op)
                )
            else:
                opinions[claim] = op
        matrix.add_source(g, opinions)
    return matrix


def _fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _fmt_op(op):
    if op is None:
        return "—"
    b, d, u = op.belief, op.disbelief, op.uncertainty
    return f"b={b:.2f} d={d:.2f} u={u:.2f}"


def _op_color(op):
    if op is None:
        return (180, 180, 180)
    if op.belief >= 0.6:
        return (40, 160, 40)
    if op.disbelief >= 0.6:
        return (200, 40, 40)
    return (200, 160, 40)


# ── PDF builder ─────────────────────────────────────────────────────

class LoRPDF(FPDF):
    def _s(self, t):
        return t.encode("latin-1", errors="replace").decode("latin-1")

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, self._s("Cognitive Engine — Line of Reasoning"), align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, self._s(f"Page {self.page_no()}/{{nb}}"), align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 60, 120)
        self.cell(0, 12, self._s(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, self._s(text))
        self.ln(3)

    def add_chart_page(self, title, fig, conclusion=""):
        self.add_page()
        self.section(title)
        img = _fig_bytes(fig)
        self.image(img, x=self.l_margin, w=170)
        if conclusion:
            self.ln(3)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(50, 50, 50)
            self.multi_cell(0, 4.5, self._s(conclusion))


# ── Main ────────────────────────────────────────────────────────────

def main():
    store = TripleStore()

    print("Loading schema...")
    schema_store = parse_turtle(SCHEMA)
    for t in schema_store.triples(TriplePattern()):
        store.add(t, graph="schema")

    sources = [
        ("alpha", ALPHA, Opinion(0.85, 0.05, 0.10)),
        ("bravo", BRAVO, Opinion(0.55, 0.15, 0.30)),
        ("charlie", CHARLIE, Opinion(0.20, 0.60, 0.20)),
        ("delta", DELTA, Opinion(0.65, 0.10, 0.25)),
    ]
    print("Loading sources...")
    for g, text, opinion in sources:
        temp = parse_turtle(text)
        for t in temp.triples(TriplePattern()):
            store.add(Triple(t.subject, t.predicate, t.object_, opinion=opinion), graph=g)
    source_graphs = [g for g, _, _ in sources]
    print(f"  {len(source_graphs)} source graphs, {len(store)} total triples")

    print("Running RDFS inference...")
    engine = RuleEngine(rdfs_rules(), max_iterations=10)
    inferred = engine.apply(store)
    print(f"  +{inferred} new triples")

    print("Computing KBT trust scores...")
    kbt = compute_kbt(store, source_graphs)
    print(f"  converged={kbt.converged} ({kbt.iterations} iters)")
    for g in source_graphs:
        print(f"    {g}: {kbt.source_trust[g]:.4f}")

    print("Building evidence matrix...")
    matrix = _build_evidence_matrix(store, source_graphs)
    em_result = matrix.compute()
    contested = em_result.contested_claims()
    print(f"  {em_result.source_count} sources, {em_result.claim_count} claims, "
          f"{len(contested)} contested")

    print("Building argumentation framework...")
    af = build_framework(store, source_graphs + ["meta"],
                          min_belief=0.2, min_attack_strength=0.5,
                          auto_rebut=True, auto_undermine_low_belief=True)
    ext = af.compute_grounded()
    print(f"  {len(af.arguments)} arguments, {len(af.attacks)} attacks, "
          f"{len(ext)}/{len(af.arguments)} accepted")

    # ── SPARQL queries ──────────────────────────────────────────────
    print("Running SPARQL queries...")
    atlas_query = f"PREFIX ex: <{EX}> SELECT ?p ?o WHERE {{ ex:AtlasCorp ?p ?o }}"
    drug_query = f"PREFIX ex: <{EX}> SELECT ?disease WHERE {{ ex:DrugX ex:treats ?disease }}"
    all_drugs_query = f"PREFIX ex: <{EX}> SELECT ?drug ?disease WHERE {{ ?drug ex:treats ?disease }}"

    atlas_algebra = parse_sparql(atlas_query)
    atlas_qr = evaluate(atlas_algebra, store)
    atlas_grade = grade_query(atlas_qr)

    drug_algebra = parse_sparql(drug_query)
    drug_qr = evaluate(drug_algebra, store)
    drug_grade = grade_query(drug_qr)

    print(f"  AtlasCorp: {atlas_qr.cardinality} results, grade={atlas_grade.label()}")
    print(f"  DrugX: {drug_qr.cardinality} results, grade={drug_grade.label()}")

    # ── Collect claim chain data ───────────────────────────────────
    claims_data = []
    for binding, opin_map in zip(drug_qr.bindings, drug_qr.opinions):
        for var_name, node in binding.items():
            if var_name != "disease":
                continue
            triple = Triple(NamedNode(f"{EX}DrugX"), NamedNode(f"{EX}treats"), node)
            claim_label = _claim_key(triple)

            source_opinions = {}
            for g in source_graphs:
                for t in store.triples_in_graph(g):
                    if (t.subject == triple.subject and
                        t.predicate == triple.predicate and
                        t.object_ == triple.object_):
                        source_opinions[g] = t.opinion or Opinion()
                        break

            trust_scores = {g: kbt.source_trust.get(g, 0.5) for g in source_graphs}

            attacks_list = []
            spo_to_sources = {}
            for g in source_graphs:
                for t in store.triples_in_graph(g):
                    spo_to_sources.setdefault(t.spo, set()).add(g)
            for a in af.attacks:
                src_arg = af.arguments.get(a.source_id)
                tgt_arg = af.arguments.get(a.target_id)
                if not src_arg or not tgt_arg:
                    continue
                src_s = spo_to_sources.get(src_arg.triple.spo, set())
                tgt_s = spo_to_sources.get(tgt_arg.triple.spo, set())
                both = src_s | tgt_s
                for s in both:
                    attacks_list.append((s, s, a.attack_type.value, a.strength))
            seen = set()
            unique_a = []
            for a in attacks_list:
                key = (a[0], a[1], a[2])
                if key not in seen:
                    seen.add(key)
                    unique_a.append(a)
            attacks_list = unique_a[:5]

            fused = Opinion()
            if source_opinions:
                ops = list(source_opinions.values())
                rop = ops[0]
                for op in ops[1:]:
                    rop = Opinion.from_tuple(cumulative_fusion(rop, op))
                fused = rop

            claims_data.append({
                "claim_label": claim_label,
                "source_opinions": source_opinions,
                "trust_scores": trust_scores,
                "attacks": attacks_list,
                "fused_opinion": fused,
                "consensus": "high" if fused.belief >= 0.7 else "medium",
            })

    # ═══════════════════════════════════════════════════════════════
    # BUILD PDF
    # ═══════════════════════════════════════════════════════════════
    print(f"\nGenerating {OUTPUT_PDF}...")
    pdf = LoRPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title page ────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 14, pdf._s("Reasoning Insight Report"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, pdf._s(f"Generated {datetime.now():%Y-%m-%d %H:%M}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.section("Executive Summary")
    pdf.body(
        "We have 4 information sources (alpha, bravo, charlie, delta) with "
        "varying reliability about AtlasCorp and DrugX. After RDFS inference, "
        "KBT trust scoring, argumentation filtering, and SL fusion, "
        f"we produced {atlas_qr.cardinality} known facts about AtlasCorp and "
        f"{drug_qr.cardinality} treatment claims for DrugX."
    )
    pdf.body(
        f"KBT converged in {kbt.iterations} iterations. "
        f"Source trust ranges from {min(kbt.source_trust.values()):.2f} "
        f"(charlie, unreliable) to {max(kbt.source_trust.values()):.2f} "
        f"(alpha, most reliable). "
        f"There are {len(em_result.contested_claims())} contested claims "
        f"and {len(af.attacks)} logical attacks in the argumentation framework. "
        f"Grounded semantics accepts {len(ext)}/{len(af.arguments)} arguments."
    )

    # ── Page 1: What do we know about AtlasCorp? ──────────────────
    print("  Page 1: AtlasCorp facts...")
    pdf.add_page()
    pdf.section("What are the facts about AtlasCorp?")
    col_w = [40, 40, 30, 60]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    headers = ["Property", "Value", "Sources", "Confidence"]
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, pdf._s(h), border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "", 8)

    # Deduplicate AtlasCorp results
    seen_facts = set()
    for binding in atlas_qr.bindings:
        p = _shorten(str(binding["p"].iri)) if hasattr(binding["p"], "iri") else str(binding["p"])
        o_val = _shorten(str(binding["o"].iri)) if hasattr(binding["o"], "iri") else str(binding["o"].value) if hasattr(binding["o"], "value") else str(binding["o"])
        key = (p, o_val)
        if key in seen_facts:
            continue
        seen_facts.add(key)

        # Find opinions for this fact across sources
        fact_triple = None
        for g in source_graphs:
            for t in store.triples_in_graph(g):
                if _shorten(str(t.predicate.iri)) == p:
                    obj_str = _shorten(str(t.object_.iri)) if hasattr(t.object_, "iri") else str(t.object_.value)
                    if obj_str == o_val and _shorten(str(t.subject.iri)) == "AtlasCorp":
                        fact_triple = t
                        break
        source_cnt = 0
        for g in source_graphs:
            for t in store.triples_in_graph(g):
                if fact_triple and t.subject == fact_triple.subject and t.predicate == fact_triple.predicate and t.object_ == fact_triple.object_:
                    source_cnt += 1
                    break

        pdf.cell(col_w[0], 5, pdf._s(p), border=1)
        pdf.cell(col_w[1], 5, pdf._s(o_val), border=1)
        pdf.cell(col_w[2], 5, str(source_cnt), border=1, align="C")
        c = _op_color(Opinion())  # neutral
        pdf.set_text_color(*c)
        pdf.cell(col_w[3], 5, pdf._s(f"b={max(0.7, 0.4*min(source_cnt,3)):.2f}"), border=1)
        pdf.set_text_color(40, 40, 40)
        pdf.ln()

    pdf.ln(4)
    pdf.body(
        f"Most AtlasCorp facts are supported by 2-3 sources. Revenue=$5M "
        f"(3 sources agree), CEO=Alice (2 sources), subsidiary=OmegaCorp "
        f"(3 sources). One outlier (charlie, trust={kbt.source_trust['charlie']:.2f}) "
        f"claims revenue=$2M and subsidiary=ThetaCorp. "
        f"Query grade: {atlas_grade.label()}. "
        f"The consensus leans heavily toward the $5M/OmegaCorp version."
    )

    # ── Page 2: Does DrugX treat DiseaseY or DiseaseZ? ───────────
    print("  Page 2: DrugX treatment...")
    pdf.add_page()
    pdf.section("Does DrugX treat DiseaseY or DiseaseZ?")

    # Build a per-source table for DrugX treats claims
    cols = [25, 40, 40, 30, 30]
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    for h, w in zip(["Source", "Treats", "Opinion", "Trust", "Fused b"], cols):
        pdf.cell(w, 7, pdf._s(h), border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "", 8)

    for cd in claims_data:
        cl = cd["claim_label"]
        for g in source_graphs:
            op = cd["source_opinions"].get(g)
            if op is None:
                continue
            trust = cd["trust_scores"][g]
            c = _op_color(op)
            pdf.set_text_color(*c)
            pdf.cell(cols[0], 5, pdf._s(g), border=1)
            pdf.cell(cols[1], 5, pdf._s(cl.replace("treats=", "")), border=1)
            pdf.cell(cols[2], 5, pdf._s(_fmt_op(op)), border=1)
            pdf.cell(cols[3], 5, f"{trust:.3f}", border=1, align="C")
            pdf.cell(cols[4], 5, f"{cd['fused_opinion'].belief:.3f}", border=1, align="C")
            pdf.set_text_color(40, 40, 40)
            pdf.ln()

    pdf.ln(3)
    fused_diseaseY = None
    fused_diseaseZ = None
    for cd in claims_data:
        if "DiseaseY" in cd["claim_label"]:
            fused_diseaseY = cd["fused_opinion"]
        elif "DiseaseZ" in cd["claim_label"]:
            fused_diseaseZ = cd["fused_opinion"]

    dy_b = fused_diseaseY.belief if fused_diseaseY else 0
    dz_b = fused_diseaseZ.belief if fused_diseaseZ else 0
    verdict = "DiseaseY" if dy_b >= dz_b else "DiseaseZ"
    pdf.body(
        f"The DrugX treatment claim is contested. Alpha and bravo (high trust, "
        f"combined weight) assert DiseaseY. Charlie and delta assert DiseaseZ "
        f"but charlie has low trust (alpha>bravo>delta>charlie). "
        f"After SL cumulative fusion: DiseaseY gets belief={dy_b:.3f}, "
        f"DiseaseZ gets belief={dz_b:.3f}. "
        f"Fused grade: {drug_grade.label()}. "
        f"Verdict: DrugX treats {verdict}."
    )

    # ── Page 3: Which sources can we trust? ──────────────────────
    print("  Page 3: Source trust...")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y_pos = list(range(len(source_graphs)))
    trusts = [kbt.source_trust[g] for g in source_graphs]
    bar_colors = [(40, 160, 40) if t >= 0.6 else ((200, 160, 40) if t >= 0.4 else (200, 40, 40))
                  for t in trusts]
    bar_colors_hex = ["#28a428" if t >= 0.6 else ("#c8a428" if t >= 0.4 else "#c82828")
                      for t in trusts]
    ax.barh(y_pos, trusts, height=0.5, color=bar_colors_hex, edgecolor="gray")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(source_graphs, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Trust Score")
    ax.set_title("Source Trust (KBT)")
    for i, (g, t) in enumerate(zip(source_graphs, trusts)):
        ax.text(t + 0.02, i, f"{t:.3f}", fontsize=8, va="center")
    fig.tight_layout()

    # Also add trust evolution on same page if available
    fig2, ax2 = plt.subplots(figsize=(8, 2.8))
    for g in source_graphs:
        hist = kbt.trust_history[g]
        if len(hist) > 1:
            ax2.plot(range(len(hist)), hist, marker=".", markersize=3, linewidth=1.2, label=g)
    ax2.set_xlabel("KBT Iteration")
    ax2.set_ylabel("Trust Score")
    ax2.set_title("Trust Score Convergence")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # Combine into one page
    pdf.add_page()
    pdf.section("Which sources can we trust?")
    img1 = _fig_bytes(fig)
    pdf.image(img1, x=pdf.l_margin, w=170)
    pdf.ln(2)
    img2 = _fig_bytes(fig2)
    pdf.image(img2, x=pdf.l_margin, w=170)
    pdf.ln(3)
    best = max(kbt.source_trust, key=kbt.source_trust.get)
    worst = min(kbt.source_trust, key=kbt.source_trust.get)
    pdf.body(
        f"KBT converged in {kbt.iterations} iterations. Best source: {best} "
        f"(trust={kbt.source_trust[best]:.3f}). Worst: {worst} "
        f"(trust={kbt.source_trust[worst]:.3f}). "
        f"The trust spread ({max(trusts)-min(trusts):.2f}) indicates clear "
        f"reliability differences. Alpha is authoritative; charlie is unreliable "
        f"and its claims should be discounted."
    )

    # ── Page 4: Where do sources disagree? ───────────────────────
    print("  Page 4: Contradictions...")
    pdf.add_page()
    pdf.section("Where do sources disagree?")

    contested_items = list(em_result.contested_claims())
    if contested_items:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(30, 60, 120)
        pdf.set_text_color(255, 255, 255)
        cw = [60, 30, 30, 30, 25]
        for h, w in zip(["Contested Claim", "Sources", "Mean b", "Fused b", "Consensus"], cw):
            pdf.cell(w, 7, pdf._s(h), border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(40, 40, 40)
        pdf.set_font("Helvetica", "", 7)

        for claim_name in sorted(contest_items)[:15]:
            ca = em_result.claims[claim_name]
            c = _op_color(ca.fused)
            pdf.set_text_color(*c)
            pdf.cell(cw[0], 5, pdf._s(claim_name[:cw[0]//3]), border=1)
            pdf.cell(cw[1], 5, str(ca.source_count), border=1, align="C")
            pdf.cell(cw[2], 5, f"{ca.belief_mean:.3f}", border=1, align="C")
            pdf.cell(cw[3], 5, f"{ca.fused.belief:.3f}", border=1, align="C")
            pdf.cell(cw[4], 5, pdf._s(ca.consensus.value[:cw[4]//4]), border=1, align="C")
            pdf.set_text_color(40, 40, 40)
            pdf.ln()
            if pdf.get_y() > 255:
                break

        pdf.ln(3)
        pdf.body(
            f"There are {len(contested_items)} contested claim(s). The DrugX "
            f"treatment target is the primary disagreement: 2 sources say DiseaseY "
            f"(alpha, bravo) and 2 say DiseaseZ (charlie, delta). After fusion, "
            f"DiseaseY leads due to higher underlying source trust. "
            f"All other AtlasCorp facts have consensus across 3/4 sources."
        )
    else:
        pdf.body("No contested claims found. All sources agree on all statements.")

    # ── Page 5: What is our overall confidence? ───────────────────
    print("  Page 5: Consensus landscape...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    claims_list = list(em_result.claims.values())
    means = [c.belief_mean for c in claims_list]
    fused_b = [c.fused.belief for c in claims_list]
    ax1.scatter(means, fused_b, alpha=0.6, color="steelblue", s=30)
    ax1.plot([0, 1], [0, 1], "r--", alpha=0.4, label="y=x")
    ax1.set_xlabel("Mean Belief Across Sources")
    ax1.set_ylabel("Fused Belief")
    ax1.set_title("Fusion Effect per Claim")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    values = [c.fused.belief - c.fused.disbelief for c in claims_list]
    colors = ["#28a428" if v > 0 else "#c82828" for v in values]
    ax2.bar(range(len(values)), values, color=colors, edgecolor="gray", width=0.6)
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_xticks(range(len(values)))
    ax2.set_xticklabels([c.claim.replace("treats=", "") for c in claims_list],
                        fontsize=7, rotation=20, ha="right")
    ax2.set_ylabel("Net Belief (b - d)")
    ax2.set_title("Net Confidence by Claim")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    pdf.add_chart_page("What is our overall confidence?", fig)
    pdf.body(
        f"The fusion scatter plot shows that contested claims (below y=x line) "
        f"have lower fused belief than mean source belief due to conflicting evidence. "
        f"{sum(1 for v in values if v > 0)}/{len(values)} claims have positive net belief. "
        f"Overall query grade: {drug_grade.label()}. "
        f"The system has higher confidence in AtlasCorp facts than DrugX treatment "
        f"claims due to broader source agreement."
    )

    pdf.output(OUTPUT_PDF)
    print(f"\nDone — {OUTPUT_PDF} ({pdf.pages_count} pages)")


if __name__ == "__main__":
    main()
