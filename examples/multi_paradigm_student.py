#!/usr/bin/env python3
"""
Multi-Paradigm Student Math Performance — KG + KBT + Argumentation + SD + ABM + DES.

Pipeline (3 passes):
  KG sources → KBT → Argumentation → SL fusion
  → Bridge: fused opinions → simulation params
  → Simulate (SD + ABM + DES) → extract new evidence
  → KG updated → re-run fusion → re-simulate

Each pass reveals a different layer of the student's situation:
  Pass 1 (pre-diagnosis):  sources conflict → reinforcing loop dominates
  Pass 2 (intervention):    practice tests + therapy activated → balancing loop engages
  Pass 3 (follow-up):       "scores up, anxiety stable" → refined belief structure
"""

from datetime import datetime
from pathlib import Path

from dynafx.core.models import Opinion
from dynafx.kb.model import (
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
)
from dynafx.kb.store import TripleStore
from dynafx.kb.turtle import parse_turtle
from dynafx.kb.inference import RuleEngine, rdfs_rules
from dynafx.kb.confidence import (
    fuse_graphs,
    grade_query,
)
from dynafx.kb.sparql import evaluate, parse_sparql
from dynafx.kb.sparql import Variable as SPARQLVar
from dynafx.reason.argumentation import (
    AttackType,
    build_framework,
)
from dynafx.reason.evidence import ConsensusLevel, EvidenceMatrix
from dynafx.reason.fusion import cumulative_fusion
from dynafx.reason.kbt import compute_kbt
from dynafx.system.dsl import (
    AuxDef,
    AgentDef,
    AgentPropDef,
    AgentRuleDef,
    QueueDef,
    ResourceDef,
    parse_sysd_file,
)

# ── Namespaces ────────────────────────────────────────────────────

EX = "http://example.org/student/"
NS = EX
s = NamedNode(f"{NS}Student")
t = NamedNode(f"{NS}Teacher")
p = NamedNode(f"{NS}Parent")
psy = NamedNode(f"{NS}Psychologist")
stu = NamedNode(f"{NS}StudentSelf")
lit = NamedNode(f"{NS}Literature")
obs = NamedNode(f"{NS}Observer")

hasIssue = NamedNode(f"{NS}hasIssue")
observes = NamedNode(f"{NS}observes")
effectiveFor = NamedNode(f"{NS}effectiveFor")
activeFor = NamedNode(f"{NS}activeFor")
causes = NamedNode(f"{NS}causes")

ANXIETY = "anxiety"
ATTENTION = "attention"
PACING = "pacing"
SCORES_DOWN = "scores_dropped"
SCORES_UP_ANXIETY_STABLE = "scores_up_anxiety_stable"
PRACTICE_TESTS = "practice_tests"
THERAPY = "therapy"

SD_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "student_math.sysd"

# ── Turtle source data ───────────────────────────────────────────

PREFIXES = f"""\
@prefix : <{NS}> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
"""

TEACHER = PREFIXES + """\
:Student :hasIssue "attention" .
:Student :hasIssue "scores_dropped" .
:Teacher :observes :Student .
"""

PARENT = PREFIXES + """\
:Student :hasIssue "pacing" .
:Student :hasIssue "scores_dropped" .
"""

PSYCHOLOGIST = PREFIXES + """\
:Student :hasIssue "anxiety" .
:Student :hasIssue "scores_dropped" .
:LowStakesTesting :effectiveFor "anxiety" .
"""

STUDENT_SELF = PREFIXES + """\
:Student :hasIssue "anxiety" .
:Student :hasIssue "pacing" .
"""

LITERATURE = PREFIXES + """\
:Math :causes "test_anxiety" .
:TestAnxiety :reduces "performance" .
:LowStakesTesting :effectiveFor "anxiety" .
"""

# ── Schema (RDFS) ────────────────────────────────────────────────

SCHEMA = """\
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix : <http://example.org/student/> .

:Student rdf:type rdfs:Resource .
:Teacher rdf:type rdfs:Resource .
:Parent rdf:type rdfs:Resource .
:Psychologist rdf:type rdfs:Resource .
:Observer rdf:type rdfs:Resource .

:hasIssue rdfs:domain :Student .
:observes rdfs:domain :Observer .
:effectiveFor rdfs:domain :LowStakesTesting .
"""


# ── Bridge: KG opinions → SD params ──────────────────────────────

CLAIM_MAP: list[tuple[NamedNode, NamedNode, object, str]] = [
    (s, hasIssue, Literal(ANXIETY), "KG_anxiety_belief"),
    (s, hasIssue, Literal(ATTENTION), "KG_attention_belief"),
    (s, hasIssue, Literal(PACING), "KG_pacing_belief"),
]

# Observation triples (synthetic — not in any source, added per-pass)
OBSERVATION_MAP: list[tuple[str, str, float]] = [
    # (observation_graph_name, observation_label, belief_if_true)
]


def _claim_key(t: Triple) -> str:
    pred = str(t.predicate.iri).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    val = (t.object_.value if hasattr(t.object_, "value")
           else str(t.object_.iri).rsplit("/", 1)[-1])
    return f"{pred}={val}"


def bridge_fused_to_params(store: TripleStore) -> dict[str, float]:
    """Read surviving claim opinions from filtered store and map to SD parameters.

    Iterates all non-schema/meta graphs (surviving claims after argumentation)
    and finds the max belief for each claim across all source graphs.
    """
    params: dict[str, float] = {}
    for subj, pred, obj, param_name in CLAIM_MAP:
        max_belief = 0.0
        has_any = False
        for g in store.graphs():
            if g in ("schema", "meta", "fused"):
                continue
            for t in store.triples(TriplePattern(subj, pred, obj), graph=g):
                b = t.opinion.belief if t.opinion else 0.5
                max_belief = max(max_belief, b)
                has_any = True
        params[param_name] = max_belief if has_any else 0.5
    return params


def extract_evidence_from_result(
    result, pass_idx: int,
) -> list[tuple[NamedNode, NamedNode, object, float]]:
    """Analyze simulation result and produce new evidence triples.

    Returns (subject, predicate, object, belief) tuples for the KG.
    """
    new: list[tuple[NamedNode, NamedNode, object, float]] = []

    anxiety = result.values.get("Math_Anxiety", [])
    perf = result.values.get("Math_Performance", [])
    efficacy = result.values.get("Self_Efficacy", [])

    if not anxiety or not perf:
        return new

    initial_a, final_a = anxiety[0], anxiety[-1]
    initial_p, final_p = perf[0], perf[-1]
    initial_e, final_e = efficacy[0], efficacy[-1]

    if pass_idx == 0:
        # After pre-diagnosis: did performance drop significantly?
        pct_drop = (initial_p - final_p) / max(1, initial_p)
        if pct_drop > 0.2:
            new.append((obs, observes, Literal(SCORES_DOWN), min(0.9, 0.5 + pct_drop)))
        # Did anxiety spike?
        anxiety_pct = (final_a - initial_a) / max(1, initial_a)
        if anxiety_pct > 0.3:
            new.append((obs, observes, Literal("anxiety_spiked"), min(0.9, 0.5 + anxiety_pct)))

    elif pass_idx == 1:
        # After intervention: did scores improve but anxiety stay high?
        perf_improved = final_p > initial_p * 1.1
        anxiety_stable = abs(final_a - initial_a) < initial_a * 0.15
        if perf_improved and anxiety_stable:
            new.append((obs, observes, Literal(SCORES_UP_ANXIETY_STABLE), 0.85))
        elif perf_improved:
            new.append((obs, observes, Literal("scores_improved"), 0.80))
        # Did intervention appear to work?
        if perf_improved:
            new.append((NamedNode(f"{NS}Observation"), effectiveFor,
                       Literal(PRACTICE_TESTS), 0.75))

    return new


# ── Build initial KG ─────────────────────────────────────────────

SOURCE_CONFIG = [
    ("teacher", TEACHER, Opinion(0.85, 0.05, 0.10)),
    ("parent", PARENT, Opinion(0.55, 0.20, 0.25)),
    ("psychologist", PSYCHOLOGIST, Opinion(0.80, 0.08, 0.12)),
    ("student_self", STUDENT_SELF, Opinion(0.65, 0.15, 0.20)),
    ("literature", LITERATURE, Opinion(0.95, 0.02, 0.03)),
]


def build_initial_store() -> TripleStore:
    """Create a TripleStore with schema + all source graphs."""
    store = TripleStore()

    for t in parse_turtle(SCHEMA).triples(TriplePattern()):
        store.add(t, graph="schema")

    for g, text, opinion in SOURCE_CONFIG:
        temp = parse_turtle(text)
        for t in temp.triples(TriplePattern()):
            store.add(
                Triple(t.subject, t.predicate, t.object_, opinion=opinion),
                graph=g,
            )
    return store


# ── EvidenceMatrix from store ────────────────────────────────────

def build_evidence_matrix(store: TripleStore, source_graphs: list[str]):
    """Build an EvidenceMatrix from named graph opinions."""
    matrix = EvidenceMatrix()
    for g in source_graphs:
        opinions: dict[str, Opinion] = {}
        for t in store.triples(TriplePattern(), graph=g):
            claim = _claim_key(t)
            op = t.opinion or Opinion()
            opinions[claim] = op
        matrix.add_source(g, opinions)
    return matrix.compute()


# ── Run one pass ─────────────────────────────────────────────────

def run_pass(
    store: TripleStore,
    source_graphs: list[str],
    pass_num: int,
    extra_params: dict[str, float] | None = None,
    phase_name: str = "",
) -> dict:
    """Run one full KG → simulation pass.

    Returns a dict with all results for display.
    """
    header = phase_name.upper() if phase_name else f"PASS {pass_num + 1}"
    print(f"\n{'=' * 78}")
    print(f"  PASS {pass_num + 1} — {header}")
    print(f"{'=' * 78}")

    # ── 1. RDFS inference ──────────────────────────────────────
    engine = RuleEngine(rdfs_rules(), max_iterations=10)
    inferred = engine.apply(store)
    if pass_num == 0:
        print(f"  RDFS inference: +{inferred} triples")

    # ── 2. KBT ─────────────────────────────────────────────────
    kbt = compute_kbt(store, source_graphs)
    print(f"  KBT converged in {kbt.iterations} iterations:")
    for g in sorted(source_graphs):
        if g in ("schema", "meta", "observation", "fused"):
            continue
        trust = kbt.source_trust.get(g, 0.5)
        bar_len = int(trust * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {g:15s} {bar} {trust:.4f}")

    # ── 3. Argumentation ────────────────────────────────────────
    af = build_framework(
        store, source_graphs + ["meta"],
        min_belief=0.2,
        min_attack_strength=0.5,
        auto_rebut=True,
        auto_undermine_low_belief=True,
    )
    ext = af.compute_grounded()
    rebuts = sum(1 for a in af.attacks if a.attack_type == AttackType.REBUT)
    undermines = sum(1 for a in af.attacks if a.attack_type == AttackType.UNDERMINE)
    print(f"  Argumentation: {len(af.arguments)} args, {len(af.attacks)} attacks "
          f"({rebuts} rebut, {undermines} undermine)")
    print(f"    Survivors: {len(ext)} / {len(af.arguments)}")

    # ── 4. Filter + fuse ───────────────────────────────────────
    spo_to_argid = {}
    for aid, arg in af.arguments.items():
        if arg.source_graph:
            spo_to_argid[(arg.triple.spo, arg.source_graph)] = aid

    filtered = TripleStore()
    for g in ("schema", "meta"):
        for t in store.triples_in_graph(g):
            filtered.add(t, graph=g)
    for g in source_graphs:
        if g in ("schema", "meta"):
            continue
        for t in store.triples_in_graph(g):
            aid = spo_to_argid.get((t.spo, g))
            if aid is not None and aid in ext:
                filtered.add(t, graph=g)

    fusion = fuse_graphs(
        filtered,
        [g for g in filtered.graphs() if g not in ("schema", "meta", "fused")],
        target_graph="fused",
        method="cumulative",
    )
    print(f"  Fusion: {fusion.fused_count} overlapping triples fused")

    # ── 5. EvidenceMatrix ──────────────────────────────────────
    em_graphs = [g for g in source_graphs if g not in ("schema", "meta")]
    em_result = build_evidence_matrix(store, em_graphs)
    print(f"  EvidenceMatrix: {em_result.source_count} sources, {em_result.claim_count} claims")
    for claim in sorted(em_result.claim_names):
        ca = em_result.claims[claim]
        print(f"    {claim:35s} {ca.consensus.value:20s} (b={ca.belief_mean:.3f}, {ca.source_count} src)")

    # ── 6. Bridge → params ─────────────────────────────────────
    # Read from original store (not filtered survivors) so max-belief
    # across all sources drives SD params even when argumentation
    # grounded semantics is too skeptical to resolve rebut attacks.
    sd_params = bridge_fused_to_params(store)
    for obs_g in store.graphs():
        if obs_g.startswith("observation"):
            for t in store.triples(TriplePattern(), graph=obs_g):
                claim = _claim_key(t)
                if t.predicate.iri == f"{NS}observes":
                    lit = t.object_.value if hasattr(t.object_, "value") else ""
                    sd_params[f"KG_obs_{lit}"] = t.opinion.belief if t.opinion else 0.5

    if extra_params:
        sd_params.update(extra_params)

    print(f"  SD params: ", end="")
    display_params = {k: f"{v:.3f}" for k, v in sorted(sd_params.items())}
    print(display_params)

    # ── 7. Load SD model + ABM/DES ──────────────────────────────
    model = parse_sysd_file(str(SD_MODEL_PATH))
    model.dt = 0.25
    model.t_span = (0.0, 84.0)  # 12 weeks * 7 days

    # Carry over initial values from previous pass (if provided)
    if extra_params:
        for stock in model.stocks:
            key = f"{stock.name}_initial"
            if key in extra_params:
                stock.initial = extra_params[key]
                # Remove from extra_params so it doesn't go as a regular param
                extra_params = {k: v for k, v in extra_params.items() if k != key}

    # Wire ABM engagement_mod → SD aux
    for aux in model.aux_vars:
        if aux.name == "abm_engagement_mod":
            aux.expr = "Student_engagement_mod_avg"
            break

    model.agents = [
        AgentDef("Student", count=1, properties=[
            AgentPropDef("perceived_threat", initial=0.3, min=0.0, max=1.0),
            AgentPropDef("engagement_mod", initial=0.0, min=-0.3, max=0.3),
        ], rules=[
            AgentRuleDef("threat_avoid",
                condition="Math_Anxiety > 50",
                effects=["engagement_mod = -0.15"]),
            AgentRuleDef("mild_anxiety_boost",
                condition="Math_Anxiety > 25 AND Math_Anxiety <= 50",
                effects=["engagement_mod = 0.05"]),
            AgentRuleDef("calm_engagement",
                condition="Math_Anxiety <= 25",
                effects=["engagement_mod = 0.15"]),
        ]),
    ]

    model.queues = [
        QueueDef("therapy", capacity=3, service_time="7.0"),
    ]
    model.resources = [
        ResourceDef("psychologist", capacity=3),
    ]

    # ── 8. Simulate ─────────────────────────────────────────────
    result = model.simulate(params=sd_params)

    print(f"  Simulation: {result.steps} steps (t=0 to {result.times[-1]:.0f})")
    init_a = result.values['Math_Anxiety'][0]
    final_a = result.values['Math_Anxiety'][-1]
    init_p = result.values['Math_Performance'][0]
    final_p = result.values['Math_Performance'][-1]
    init_e = result.values['Self_Efficacy'][0]
    final_e = result.values['Self_Efficacy'][-1]
    print(f"    Math_Anxiety:    {init_a:.1f} → {final_a:.1f}  "
          f"({'↑' if final_a > init_a else '↓'}{abs(final_a - init_a):.1f})")
    print(f"    Math_Performance: {init_p:.1f} → {final_p:.1f}  "
          f"({'↑' if final_p > init_p else '↓'}{abs(final_p - init_p):.1f})")
    print(f"    Self_Efficacy:   {init_e:.1f} → {final_e:.1f}  "
          f"({'↑' if final_e > init_e else '↓'}{abs(final_e - init_e):.1f})")

    # Detect dominant loop
    if pass_num == 0 and final_p < init_p * 0.8:
        print(f"    → Reinforcing loop dominant (anxiety↑ performance↓)")
    elif pass_num >= 1 and final_p > init_p * 1.1:
        print(f"    → Balancing loop engaged (performance recovering)")

    if result.abm_engine:
        abm_metrics = result.abm_engine.get_metrics()
        print(f"  ABM metrics: {abm_metrics}")

    if result.des_engine:
        des_stats = result.des_engine.get_all_stats()
        print(f"  DES stats: {des_stats}")

    return {
        "store": store,
        "filtered": filtered,
        "kbt": kbt,
        "af": af,
        "ext": ext,
        "fusion": fusion,
        "em_result": em_result,
        "sd_params": sd_params,
        "result": result,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  MULTI-PARADIGM STUDENT MODEL — KG × KBT × Argumentation × SD × ABM × DES")
    print("=" * 78)
    print()
    print("  A student is failing math. Four sources disagree on why.")
    print("  Three simulation passes, each informed by the previous pass's outcomes.")
    print()

    store = build_initial_store()
    source_graphs = ["teacher", "parent", "psychologist", "student_self", "literature"]
    all_results: list[dict] = []
    phase_names = ["pre_diagnosis", "intervention", "follow_up"]

    for pass_num in range(3):
        # Set scenario-driven params for this pass
        if pass_num == 0:
            phase_extra = {"KG_intervention_active": 0.0, "KG_therapy_active": 0.0}
        else:
            phase_extra = {"KG_intervention_active": 1.0, "KG_therapy_active": 1.0}

        # Carry forward final state from previous pass as initial values
        if pass_num > 0:
            prev = all_results[-1]["result"]
            for stock in ["Math_Anxiety", "Math_Performance", "Self_Efficacy"]:
                phase_extra[f"{stock}_initial"] = prev.values[stock][-1]

        # Add observation triples from previous pass (if any)
        obs_graphs = [g for g in store.graphs() if g.startswith("observation")]
        current_sources = source_graphs + obs_graphs + ["meta"]
        current_sources = [g for g in current_sources if g in store.graphs()]

        p_result = run_pass(store, current_sources, pass_num, phase_extra,
                            phase_name=phase_names[pass_num])
        all_results.append(p_result)

        # Extract evidence from result for next pass
        new_evidence = extract_evidence_from_result(p_result["result"], pass_num)
        if new_evidence:
            obs_num = pass_num + 1
            obs_graph = f"observation_{obs_num}"
            for subj, pred, obj, belief in new_evidence:
                store.add(
                    Triple(subj, pred, obj, opinion=Opinion(belief, 1.0 - belief, 0.0)),
                    graph=obs_graph,
                )
            print(f"  → New evidence added to graph '{obs_graph}':")
            for subj, pred, obj, belief in new_evidence:
                print(f"      ({_claim_key(Triple(subj, pred, obj, opinion=Opinion(belief, 0, 0)))}) "
                      f"b={belief:.2f}")

        # Print phase transition
        if pass_num == 0:
            print(f"  → INTERVENTION PHASE: practice tests + therapy activated")
        elif pass_num == 1:
            print(f"  → FOLLOW-UP PHASE: updated KG beliefs from observations")

    # ── Summary comparison ─────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  COMPARISON — 3 PASSES")
    print(f"{'=' * 78}")
    print(f"  {'':20s} {'Pass 1 (Pre)':>18s} {'Pass 2 (Interv)':>18s} {'Pass 3 (F/up)':>18s}")
    print(f"  {'─'*20} {'─'*18} {'─'*18} {'─'*18}")

    for stock in ["Math_Anxiety", "Math_Performance", "Self_Efficacy"]:
        vals = []
        for r in all_results:
            v = r["result"].values[stock]
            vals.append((v[0], v[-1]))
        label = f"{stock}  "
        print(f"  {label:20s} {vals[0][0]:6.1f}→{vals[0][1]:<6.1f}  "
              f"{vals[1][0]:6.1f}→{vals[1][1]:<6.1f}  "
              f"{vals[2][0]:6.1f}→{vals[2][1]:<6.1f}")

    print(f"  {'─'*20} {'─'*18} {'─'*18} {'─'*18}")
    print(f"  {'KBT trust':20s}", end="")
    for r in all_results:
        kbt = r["kbt"]
        avg = sum(kbt.source_trust.values()) / max(1, len(kbt.source_trust))
        print(f" {avg:>8.3f}      ", end="")
    print()

    print(f"\n  {'Source':15s} {'Pass 1 KBT':>12s} {'Pass 2 KBT':>12s} {'Pass 3 KBT':>12s}")
    print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*12}")
    for g in ["teacher", "parent", "psychologist", "student_self", "literature"]:
        trusts = []
        for r in all_results:
            trusts.append(r["kbt"].source_trust.get(g, 0.5))
        if any(t != 0.5 for t in trusts):
            print(f"  {g:15s} {trusts[0]:>12.4f} {trusts[1]:>12.4f} {trusts[2]:>12.4f}")

    # KBT sources per pass (only show those that changed)
    print()
    print(f"  Pass 1 (pre-diagnosis):    sources conflict, reinforcing loop dominates")
    print(f"  Pass 2 (intervention):     practice tests + therapy activated")
    p2_anxiety = all_results[1]["result"].values["Math_Anxiety"]
    p2_perf = all_results[1]["result"].values["Math_Performance"]
    p3_anxiety = all_results[2]["result"].values["Math_Anxiety"]
    a_final = p2_anxiety[-1]
    p_final = p2_perf[-1]
    p3_a_final = p3_anxiety[-1]
    if p_final > 45:
        print(f"  Pass 2 outcome:           Performance recovered to {p_final:.0f}")
    if abs(p3_a_final - a_final) / max(1, a_final) < 0.1 and p_final > 45:
        print(f"  Pass 3 follow-up:         Scores up, anxiety stable "
              f"({a_final:.0f}→{p3_a_final:.0f}) → refined belief")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
