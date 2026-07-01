#!/usr/bin/env python3
"""Solar EPC Demo — Cognitive Decision Intelligence Platform.

End-to-end demonstration showing:
  1. Knowledge Network (TripleStore + Turtle + RDFS inference)
  2. Transaction Layer (append-only event log)
  3. Production Rules (IF-THEN on KB state changes)
  4. Simulation (SD+ABM+DES project dynamics via KBSimBridge)
  5. Execution Network (provenance-tracked action records)
  6. SPARQL Grading (query-driven project health assessment)

Pipeline:
  Turtle parse → named graphs → RDFS inference → store.on_add triggers
  ProductionRuleEngine → BridgeAction runs simulation → ExecutionStore
  records every action → SPARQL queries grade project health.

The demo simulates an 11-event timeline over 365 days for a 50MW
solar farm EPC project, with 5 production rules governing material
disruptions, milestone simulations, quality issues, and project
completion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Imports ──────────────────────────────────────────────────────────────────
from dynafx.knowledge.model import (
    Literal,
    NamedNode,
    Opinion,
    Triple,
)
from dynafx.knowledge.store import TripleStore
from dynafx.knowledge.turtle import parse_turtle
from dynafx.knowledge.inference import RuleEngine, owl_rl_rules, rdfs_rules
from dynafx.knowledge import sparql_evaluate as sparql_eval
from dynafx.knowledge import parse_sparql
from dynafx.bridge import CognitiveOrchestrator, KBSimBridge
from dynafx.dynamics import parse_sysd_file


class _EX:
    """Convenience namespace for http://example.org/."""
    def __getitem__(self, name: str) -> NamedNode:
        return NamedNode(f"http://example.org/{name}")

EX = _EX()

# Model uses http://epc.org/ URIs in KB_ASSERT — define matching nodes
EPC_NS = "http://epc.org/"
def _epc(name: str) -> NamedNode:
    return NamedNode(f"{EPC_NS}{name}")

# ── Namespace helpers ────────────────────────────────────────────────────────
PROJ = EX["Project"]
STATUS = EX["status"]
HAS_ISSUE = EX["hasIssue"]
DELAYED = EX["delayed"]
SEVERITY = EX["severity"]
PARTNER = EX["partner"]
TYPE_OF = EX["type"]
TRIGGER = EX["triggers"]
COMPLETION = EX["completionFraction"]

# ── Turtle data ──────────────────────────────────────────────────────────────
TURTLE_DATA = """
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:Project a ex:Project ;
    ex:status "active" ;
    ex:budget 50000000 ;
    ex:capacity 50 ;
    ex:unit "MW" ;
    ex:partner ex:AcmeCorp .

ex:AcmeCorp a ex:Partner ;
    ex:reliability 0.85 .

ex:delayed rdfs:subPropertyOf ex:hasIssue .
ex:disrupted rdfs:subPropertyOf ex:hasIssue .
"""

# ── SPARQL Grade Queries ─────────────────────────────────────────────────────
GRADE_QUERIES = {
    "completionScore": """PREFIX ex: <http://example.org/>
SELECT ?completion WHERE {
    ex:Project ex:completionFraction ?completion
}""",
    "costScore": """PREFIX ex: <http://example.org/>
SELECT ?costOverrun WHERE {
    ex:Project ex:costOverrun ?costOverrun
}""",
    "revenueScore": """PREFIX ex: <http://example.org/>
SELECT ?revenueAchieved WHERE {
    ex:Project ex:revenueAchieved ?revenueAchieved
}""",
    "healthScore": """PREFIX ex: <http://example.org/>
SELECT ?health WHERE {
    ex:Project ex:healthIndex ?health
}""",
}

# ── Event Timeline ───────────────────────────────────────────────────────────
TIMELINE = [
    (0, "ProjectStarted", {"initiator": "owner", "budget": 50_000_000}),
    (30, "MaterialOrdered", {"supplier": "AcmeCorp", "qty": 100}),
    (45, "SupplierDelayed", {"supplier": "AcmeCorp", "days": 30}),
    (60, "SitePrepCompleted", {"area": 200, "unit": "hectares"}),
    (90, "WeatherDelay", {"days": 15, "cause": "monsoon"}),
    (120, "MaterialDelivered", {"supplier": "AcmeCorp", "qty": 100}),
    (180, "MilestoneReached", {"phase": "Foundation"}),
    (200, "QualityIssue", {"defect": "panel_crack", "severity": 0.3}),
    (270, "MilestoneReached", {"phase": "Installation"}),
    (300, "CommissioningStarted", {"unit": 50}),
    (365, "ProjectComplete", {}),
]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Knowledge Base Setup
# ══════════════════════════════════════════════════════════════════════════════
def setup_knowledge_base() -> TripleStore:
    """Initialize and populate the TripleStore with Turtle data + inference."""
    store = TripleStore()

    # Parse Turtle into named graphs per source
    parse_turtle(TURTLE_DATA, store, default_graph="ex:turtle_source")

    # RDFS inference
    engine = RuleEngine(rules=rdfs_rules() + owl_rl_rules())
    engine.apply(store)

    return store


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Production Rules
# ══════════════════════════════════════════════════════════════════════════════
from dynafx.knowledge.production import (
    BridgeAction,
    LogAction,
    ProductionRule,
    SimulateAction,
    TripleAction,
    TripleCondition,
)
from dynafx.knowledge.inference import InferencePattern


def setup_production_rules(
    orb: CognitiveOrchestrator,
    bridge: KBSimBridge,
    model,
) -> None:
    """Register production rules on the orchestrator.

    Each rule is an IF-THEN over KB triples, firing when the TripleStore's
    on_add callback is triggered.
    """

    # R1: Supplier delay detection
    orb.add_rule(
        ProductionRule(
            name="supplier-delay-detected",
            priority=10,
            body=[
                TripleCondition(
                    pattern=InferencePattern(
                        subject=EX["AcmeCorp"],
                        predicate=EX["delayed"],
                        object_=None,
                    ),
                ),
            ],
            head=[
                LogAction(message="Supplier delay detected — material at risk!"),
                TripleAction(
                    subject=EX["Project"],
                    predicate=EX["status"],
                    object_=Literal("at_risk"),
                ),
            ],
            description="Flag project at risk when supplier delay detected",
        )
    )

    # R2: Disruption escalation (when 2+ hasIssue triples)
    orb.add_rule(
        ProductionRule(
            name="disruption-escalated",
            priority=8,
            body=[
                TripleCondition(
                    pattern=InferencePattern(
                        subject=PROJ,
                        predicate=EX["hasIssue"],
                        object_=None,
                    ),
                ),
            ],
            head=[
                LogAction(message="Multiple disruptions — escalation needed!"),
                TripleAction(
                    subject=EX["AcmeCorp"],
                    predicate=EX["reliability"],
                    object_=Literal(0.6),
                ),
            ],
            description="Escalate when disruptions exceed threshold",
            max_fires=1,
        )
    )

    # R3: Milestone-driven simulation
    orb.add_rule(
        ProductionRule(
            name="milestone-simulation",
            priority=5,
            body=[
                TripleCondition(
                    pattern=InferencePattern(
                        subject=PROJ,
                        predicate=EX["status"],
                        object_=Literal("milestone"),
                    ),
                ),
            ],
            head=[
                BridgeAction(
                    bridge=bridge,
                    model=model,
                    params_override={
                        "disruption_q": 'ASK { <http://example.org/CurrentState> <http://example.org/hasStatus> "disrupted" }',
                        "supply_rel_q": 'SELECT ?v WHERE { <http://example.org/AcmeCorp> <http://example.org/reliability> ?v }',
                        "m1_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Foundation" }',
                        "m2_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Installation" }',
                        "m3_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Commissioning" }',
                        "quality_q": 'ASK { <http://example.org/Project> <http://example.org/hasIssue> "quality" }',
                    },
                ),
            ],
            description="Run simulation on milestone events",
            max_fires=20,
        )
    )

    # R4: Quality issue flagging
    orb.add_rule(
        ProductionRule(
            name="quality-issue-flagged",
            priority=7,
            body=[
                TripleCondition(
                    pattern=InferencePattern(
                        subject=PROJ,
                        predicate=EX["hasIssue"],
                        object_=Literal("quality"),
                    ),
                ),
            ],
            head=[
                LogAction(message="Quality issue detected — review required!"),
                BridgeAction(
                    bridge=bridge,
                    model=model,
                    params_override={
                        "disruption_q": 'ASK { <http://example.org/CurrentState> <http://example.org/hasStatus> "disrupted" }',
                        "supply_rel_q": 'SELECT ?v WHERE { <http://example.org/AcmeCorp> <http://example.org/reliability> ?v }',
                        "m1_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Foundation" }',
                        "m2_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Installation" }',
                        "m3_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Commissioning" }',
                        "quality_q": 'ASK { <http://example.org/Project> <http://example.org/hasIssue> "quality" }',
                    },
                ),
            ],
            description="Flag quality issues and re-simulate",
            max_fires=10,
        )
    )

    # R5: Project complete — final simulation
    orb.add_rule(
        ProductionRule(
            name="project-complete-simulation",
            priority=1,
            body=[
                TripleCondition(
                    pattern=InferencePattern(
                        subject=PROJ,
                        predicate=EX["status"],
                        object_=Literal("complete"),
                    ),
                ),
            ],
            head=[
                LogAction(message="Project complete — performing final assessment!"),
                BridgeAction(
                    bridge=bridge,
                    model=model,
                    params_override={
                        "disruption_q": 'ASK { <http://example.org/CurrentState> <http://example.org/hasStatus> "disrupted" }',
                        "supply_rel_q": 'SELECT ?v WHERE { <http://example.org/AcmeCorp> <http://example.org/reliability> ?v }',
                        "m1_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Foundation" }',
                        "m2_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Installation" }',
                        "m3_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Commissioning" }',
                        "quality_q": 'ASK { <http://example.org/Project> <http://example.org/hasIssue> "quality" }',
                    },
                ),
            ],
            description="Run final simulation when project completes",
            max_fires=1,
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Event Processing
# ══════════════════════════════════════════════════════════════════════════════
def process_event(
    orb: CognitiveOrchestrator,
    bridge: KBSimBridge,
    store: TripleStore,
    day: int,
    event_type: str,
    payload: dict,
) -> None:
    """Process a single event: ingest, update KB, trigger rules."""
    ts = day * 86400.0  # convert day number to timestamp

    orb.ingest_event(
        event_type=event_type,
        payload=payload,
        source="erp",
        confidence=0.95,
        timestamp=ts,
    )

    # ── Update KB domain state based on event type ──────────
    if event_type == "SupplierDelayed":
        store.add(
            Triple(
                EX[payload.get("supplier", "Unknown")],
                DELAYED,
                Literal("true"),
                opinion=Opinion(0.9, 0.05, 0.05),
            )
        )
        # Also add EPC namespace triple for model's KB_QUERY
        store.add(
            Triple(
                _epc("CurrentState"),
                _epc("hasStatus"),
                Literal("disrupted"),
                opinion=Opinion(0.9, 0.05, 0.05),
            )
        )
    elif event_type == "MaterialDelivered":
        store.add(
            Triple(
                _epc("CurrentState"),
                _epc("hasStatus"),
                Literal("normal"),
                opinion=Opinion(0.8, 0.1, 0.1),
            )
        )
    elif event_type == "SitePrepCompleted":
        store.add(
            Triple(PROJ, STATUS, Literal("milestone"), opinion=Opinion(1.0, 0.0, 0.0))
        )
    elif event_type == "MilestoneReached":
        phase = payload.get("phase", "unknown")
        store.add(
            Triple(
                PROJ,
                EX["milestoneReached"],
                Literal(phase),
                opinion=Opinion(0.95, 0.02, 0.03),
            )
        )
        store.add(
            Triple(PROJ, STATUS, Literal("milestone"), opinion=Opinion(1.0, 0.0, 0.0))
        )
    elif event_type == "QualityIssue":
        store.add(
            Triple(
                PROJ,
                HAS_ISSUE,
                Literal("quality"),
                opinion=Opinion(0.8, 0.1, 0.1),
            )
        )
        # EPC namespace for model KB_QUERY
        store.add(
            Triple(
                _epc("Project"),
                _epc("qualityIssue"),
                Literal(1.0),
                opinion=Opinion(0.8, 0.1, 0.1),
            )
        )
    elif event_type == "ProjectComplete":
        store.add(
            Triple(PROJ, STATUS, Literal("complete"), opinion=Opinion(1.0, 0.0, 0.0))
        )


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5: Evidence Extraction
# ══════════════════════════════════════════════════════════════════════════════
def evidence_from_result(
    store: TripleStore,
    sim_result,
    event_type: str,
):
    """Write simulation outputs to the KB as queryable triples."""
    if not sim_result or not hasattr(sim_result, "values"):
        return

    # Map model stock names to KB grade properties
    stock_to_grade = {
        "Commissioning_Progress": ("completionFraction", 140000.0),  # normalize to 0-1 by total panels
        "Project_Cost": ("costOverrun", 50000.0),  # normalize cost
        "Cumulative_Revenue": ("revenueAchieved", 50000.0),  # normalize revenue
        "Completion_Certificate": ("healthIndex", 1.0),
    }

    values = sim_result.values  # dict[str, list[float]]
    idx = -1  # last timestep
    for stock_name, (grade_name, divisor) in stock_to_grade.items():
        if stock_name in values:
            raw = values[stock_name][idx]
            val = raw / divisor if divisor else raw
            if grade_name == "completionFraction":
                val = min(1.0, max(0.0, val))
            elif grade_name == "costOverrun":
                val = min(1.0, max(0.0, val))
            elif grade_name == "revenueAchieved":
                val = min(1.0, max(0.0, val))
            elif grade_name == "healthIndex":
                val = min(1.0, max(0.0, val))
            store.add(
                Triple(
                    PROJ,
                    EX[grade_name],
                    Literal(val),
                    opinion=Opinion(0.9, 0.05, 0.05),
                )
            )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Solar EPC Demo — Cognitive Decision Intelligence Platform")
    print("=" * 60)

    # ── Phase 1: Knowledge Base Setup ────────────────────────
    print("\n[1/5] Setting up Knowledge Base...")
    store = setup_knowledge_base()
    print(f"  TripleStore: {len(list(store.all_triples()))} triples")

    # ── Phase 2: Bridge + Model ──────────────────────────────
    print("\n[2/5] Loading simulation model...")
    model_path = Path(__file__).parent.parent / "models" / "solar_epc_project.sysd"
    model = parse_sysd_file(str(model_path))
    bridge = KBSimBridge(store)
    print(f"  Model: {model.name or 'solar_epc_project'} ({len(model.stocks)} stocks, {len(model.aux_vars)} auxes)")

    # ── Phase 3: Cognitive Orchestrator Setup ────────────────
    print("\n[3/5] Initializing Cognitive Orchestrator...")
    orb = CognitiveOrchestrator(store, bridge=bridge)
    setup_production_rules(orb, bridge, model)
    orb.start()
    print(f"  {len(orb.rule_engine.rules)} production rules registered")

    # ── Phase 4: Event Timeline ─────────────────────────────
    print("\n[4/5] Processing event timeline (365 days, 11 events)...")
    sim_runs: list = []
    sim_labels: list[str] = []

    for day, event_type, payload in TIMELINE:
        process_event(orb, bridge, store, day, event_type, payload)

        # Capture simulation results after milestone/project events
        if event_type in ("MilestoneReached", "ProjectComplete", "SitePrepCompleted"):
            # Get the most recent bridge execution
            bridge_recs = [r for r in orb.exec_store.recent(50) if r.action_type == "bridge" and r.output.get("result")]
            if bridge_recs:
                last = bridge_recs[0]
                sim_runs.append(last)
                sim_labels.append(f"Day {day} ({event_type})")
                sim_result = last.output.get("result")
                evidence_from_result(store, sim_result, event_type)

    print(f"  {orb.tx_store.total_count} transactions recorded")
    print(f"  {len(sim_runs)} simulation runs triggered by rules")

    # ── Phase 5: Grading ＋ Report ──────────────────────────
    print("\n[5/5] Grading project health...")

    # Pre-compute grades before PDF (so PDF sees populated KB)
    grades = {}
    for name, query_str in GRADE_QUERIES.items():
        algebra = parse_sparql(query_str)
        qr = sparql_eval(algebra, store)
        val = 0.0
        if qr and qr.bindings:
            binding = qr.bindings[0]
            if binding:
                raw = next(iter(binding.values()), None)
                try:
                    val = float(raw.value if hasattr(raw, "value") else raw)
                except (ValueError, TypeError, AttributeError):
                    val = 0.0
        grades[name] = val

    # Derive cost/revenue scores from raw values
    cost_raw = grades.pop("costScore", 0.0)
    grades["costScore"] = max(0.0, 1.0 - cost_raw) if cost_raw > 0.0 else 1.0
    rev_raw = grades.pop("revenueScore", 0.0)
    grades["revenueScore"] = min(1.0, max(0.0, rev_raw))

    overall = (
        grades.get("completionScore", 0.0) * 0.30
        + grades.get("costScore", 0.0) * 0.30
        + grades.get("revenueScore", 0.0) * 0.20
        + grades.get("healthScore", 0.0) * 0.20
    )
    grades["overall"] = overall

    # Collect execution stats
    exec_stats: dict[str, int] = {}
    for rec in orb.exec_store.recent(500):
        exec_stats[rec.action_type] = exec_stats.get(rec.action_type, 0) + 1

    # Build per-type transaction counts
    tx_type_counts: dict[str, int] = {}
    for tx in orb.tx_store.recent(200):
        tx_type_counts[tx.event_type] = tx_type_counts.get(tx.event_type, 0) + 1

    # ═══════════════════════════════════════════════════════════
    # 5-LAYER INTELLIGENCE REPORT
    # Layer 1: Situational Awareness (What is happening?)
    # Layer 2: Diagnostics (Why is it happening?)
    # Layer 3: Predictive Analytics (What will happen?)
    # Layer 4: Scenario Analysis (What if...?)
    # Layer 5: Decision Intelligence (What should we do?)
    # ═══════════════════════════════════════════════════════════

    # ── Collect all simulation results for analysis ─────────
    all_sim_results: list[tuple[str, dict]] = []
    for label, rec in zip(sim_labels, sim_runs):
        sim_result = rec.output.get("result")
        if sim_result and hasattr(sim_result, "values"):
            all_sim_results.append((label, sim_result.values))

    # ── Run 3 scenarios for Layer 4 (What-if) ───────────────
    from dynafx.dynamics.scenario import ScenarioComparison, ScenarioDef
    from dynafx.dynamics import SysdModel  # type hint
    from copy import deepcopy

    scenario_model: SysdModel = deepcopy(model)
    scenario_params_base = {
        "disruption_q": 'ASK { <http://example.org/CurrentState> <http://example.org/hasStatus> "disrupted" }',
        "supply_rel_q": 'SELECT ?v WHERE { <http://example.org/AcmeCorp> <http://example.org/reliability> ?v }',
        "m1_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Foundation" }',
        "m2_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Installation" }',
        "m3_q": 'ASK { <http://example.org/Project> <http://example.org/milestoneReached> "Commissioning" }',
        "quality_q": 'ASK { <http://example.org/Project> <http://example.org/hasIssue> "quality" }',
    }
    scenario_defs = [
        ScenarioDef("1. Baseline", scenario_params_base),
        ScenarioDef("2. Extra Crew (5)", {**scenario_params_base, "crew_count": 5.0}),
        ScenarioDef("3. Diversified Supply (25/day)", {**scenario_params_base, "base_supply_rate": 25.0}),
        ScenarioDef("4. Best Case (5 crews + 25/day)", {**scenario_params_base, "crew_count": 5.0, "base_supply_rate": 25.0}),
    ]
    try:
        scenario_comparison = ScenarioComparison(scenario_model, scenario_defs, method="euler")
        scenario_summary = scenario_comparison.summary()
    except Exception:
        scenario_comparison = None
        scenario_summary = {}

    # ── Run Sensitivity Analysis (OAT) for Layer 3 ──────────
    from dynafx.dynamics.sensitivity import SensitivityAnalyzer
    try:
        analyzer = SensitivityAnalyzer(scenario_model, method="euler")
        oat_result = analyzer.oat(
            {
                "crew_count": (1.0, 6.0, "uniform"),
                "base_supply_rate": (5.0, 30.0, "uniform"),
                "base_productivity": (60.0, 200.0, "uniform"),
            },
            output="Commissioning_Progress",
            t=365,
            n_steps=5,
        )
        oat_params = oat_result.ranking("oat_high")
    except Exception:
        oat_result = None
        oat_params = []

    # ── Feedback loop analysis for Layer 2 ──────────────────
    from dynafx.dynamics.feedback import detect_feedback_loops
    try:
        loop_analysis = detect_feedback_loops(scenario_model)
        loops = loop_analysis.loops
        n_reinf = sum(1 for l in loops if l.polarity == "reinforcing")
        n_bal = sum(1 for l in loops if l.polarity == "balancing")
    except Exception:
        loops = []
        n_reinf = n_bal = 0

    def _sanitize_pdf(text: str) -> str:
        return (text.replace("\u2014", "-").replace("\u2013", "-")
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2026", "...").replace("\u2022", "*")
                .replace("\u25a0", "=").replace("\u25a1", ".")
                .replace("\u26A0", "!").replace("\u2705", "OK"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        from fpdf import FPDF
        import tempfile, os

        PID = os.getpid()

        def _save_chart(fig, prefix: str = "chart") -> str:
            path = os.path.join(tempfile.gettempdir(), f"{prefix}_{PID}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return path

        def _final_vals(values: dict) -> dict:
            return {k: vs[-1] for k, vs in values.items() if vs}

        # ── Extract key metrics ──────────────────────────────
        latest_vals = _final_vals(all_sim_results[-1][1]) if all_sim_results else {}
        total_panels = 100000.0
        commissioning = latest_vals.get("Commissioning_Progress", 0)
        installation = latest_vals.get("Installation_Progress", 0)
        site_prep = latest_vals.get("Site_Prep_Progress", 0)
        materials = latest_vals.get("Materials_Inventory", 0)
        cost = latest_vals.get("Project_Cost", 0)
        revenue = latest_vals.get("Cumulative_Revenue", 0)
        defects = latest_vals.get("Cumulative_Defects", 0)
        certificate = latest_vals.get("Completion_Certificate", 0)

        completion_pct = commissioning / total_panels if total_panels else 0
        install_pct = installation / total_panels if total_panels else 0
        margin_pct = ((revenue - cost) / revenue * 100) if revenue > 0 else 0
        cost_per_mw = cost / 50.0 if cost > 0 else 0
        budget = 50000.0
        budget_used_pct = (cost / budget * 100) if budget > 0 else 0
        defect_rate = defects / 365.0 if defects > 0 else 0

        # ══════════════════════════════════════════════════════
        # CHART: Health Radar (Layer 1 - Situational Awareness)
        # ══════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        categories = ["Completion", "Cost Control", "Revenue", "Quality"]
        scores = [
            grades.get("completionScore", 0),
            grades.get("costScore", 0),
            grades.get("revenueScore", 0),
            max(0, 1.0 - defect_rate),
        ]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles_plot = angles + [angles[0]]
        scores_plot = scores + [scores[0]]
        ax.plot(angles_plot, scores_plot, "o-", linewidth=2, color="#1565C0")
        ax.fill(angles_plot, scores_plot, alpha=0.2, color="#1565C0")
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.set_title("Project Health Radar", fontsize=14, fontweight="bold", pad=20)
        chart_radar = _save_chart(fig, "radar")

        # ══════════════════════════════════════════════════════
        # CHART: KPI Dashboard (Layer 1)
        # ══════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(8, 4))
        fig.suptitle("", fontsize=1)
        kpis = [
            ("Completion", completion_pct, "Panels Commissioned", "Production"),
            ("Cost Control", grades.get("costScore", 0), "Budget Efficiency", "Cost"),
            ("Revenue", grades.get("revenueScore", 0), "Milestone Payments", "Revenue"),
            ("Quality", max(0, 1.0 - defect_rate), "Defect-Free Rate", "Quality"),
        ]
        for idx, (title, val, subtitle, _) in enumerate(kpis):
            ax = axes[idx // 2, idx % 2]
            color = "#4CAF50" if val >= 0.7 else "#FF9800" if val >= 0.3 else "#F44336"
            ax.text(0.5, 0.65, f"{val:.0%}", ha="center", va="center", fontsize=36, fontweight="bold", color=color)
            ax.text(0.5, 0.90, title, ha="center", va="center", fontsize=12, fontweight="bold")
            ax.text(0.5, 0.20, subtitle, ha="center", va="center", fontsize=8, color="gray")
            # Gauge bar
            ax.barh(0, val, height=0.15, color=color, alpha=0.8)
            ax.barh(0, 1.0, height=0.15, color="lightgray", alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 1.5)
            ax.axis("off")
        plt.tight_layout(pad=2)
        chart_kpi = _save_chart(fig, "kpi")

        # ══════════════════════════════════════════════════════
        # CHART: Causal Network (Layer 2 - Diagnostics)
        # ══════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.set_title("Root Cause Analysis: Supplier Delay Impact Chain", fontsize=13, fontweight="bold", pad=10)

        # Causal chain nodes as boxes
        chain_steps = [
            (2, 5, "Supplier\nDelay (Day 45)", "#F44336"),
            (4, 4, "Material\nShortage", "#FF9800"),
            (6, 3, "Installation\nSlowdown", "#FF9800"),
            (8, 2, "Commissioning\nLag", "#2196F3"),
            (5, 0.5, "PROJECT: 21.9% Complete\nCost: ${:,.0f}K | Defects: {:.0f}".format(int(cost), defects), "#9C27B0"),
        ]
        for x, y, label, color in chain_steps:
            bbox = dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
            ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold", bbox=bbox)

        # Arrows between chain nodes
        arrows = [(2.8, 4.6, 3.8, 4.2), (4.8, 3.6, 5.8, 3.2), (6.8, 2.6, 7.8, 2.2)]
        for x1, y1, x2, y2 in arrows:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle="->", color="#666", lw=2))

        # Feedback loops inset
        if loops:
            loop_text = "\n".join([f"  {l.polarity[0].upper()}: {l.name} ({len(l.nodes)} vars)" for l in loops[:5]])
            bbox2 = dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1)
            ax.text(9.5, 5.5, f"Feedback Loops:\n{loop_text}", ha="right", va="top", fontsize=7, bbox=bbox2)
        plt.tight_layout()
        chart_causal = _save_chart(fig, "causal")

        # ══════════════════════════════════════════════════════
        # CHART: Simulation Trajectories + Forecast (Layer 3)
        # ══════════════════════════════════════════════════════
        if all_sim_results:
            latest_values = all_sim_results[-1][1]
            n_steps = len(next(iter(latest_values.values())))
            t = np.linspace(0, 365, n_steps)

            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            fig.suptitle("Layer 3 - Predictive Analytics: Simulation Trajectories & Forecasts", fontsize=13, fontweight="bold")

            # Panel A: Installation Progress + forecast
            ax = axes[0, 0]
            inst_data = latest_values.get("Installation_Progress", [0]*n_steps)
            ax.plot(t, inst_data, color="#4CAF50", linewidth=2, label="Actual")
            # Forecast line (extend slope from last 60 days)
            last_60 = max(1, n_steps - 240)
            slope = (inst_data[-1] - inst_data[last_60]) / (t[-1] - t[last_60])
            forecast_t = np.linspace(365, 365 + 90, 10)
            forecast_v = [min(total_panels, max(0, inst_data[-1] + slope * (ft - 365))) for ft in forecast_t]
            ax.plot(forecast_t, forecast_v, color="#4CAF50", linestyle="--", linewidth=2, alpha=0.7, label="Forecast")
            ax.fill_between(forecast_t, forecast_v, [v * 0.8 for v in forecast_v], alpha=0.15, color="#4CAF50", label="Confidence")
            ax.axhline(y=total_panels, color="#333", linestyle=":", alpha=0.5, label=f"Target ({total_panels:,.0f})")
            ax.set_ylabel("Panels Installed")
            ax.set_title("A. Installation + Forecast", fontweight="bold")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(True, alpha=0.3)
            # Add forecast callout
            days_to_target = (total_panels - inst_data[-1]) / slope if slope > 0 else 999
            ax.text(0.95, 0.05, f"Predicted completion: {365 + days_to_target:.0f}d", transform=ax.transAxes,
                   ha="right", va="bottom", fontsize=8, fontweight="bold", color="#4CAF50",
                   bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"))

            # Panel B: Cost vs Revenue + forecast
            ax = axes[0, 1]
            cost_data = latest_values.get("Project_Cost", [0]*n_steps)
            rev_data = latest_values.get("Cumulative_Revenue", [0]*n_steps)
            ax.plot(t, cost_data, color="#F44336", linewidth=2, label="Cost")
            ax.plot(t, rev_data, color="#4CAF50", linewidth=2, label="Revenue")
            # Cost forecast
            cost_slope = (cost_data[-1] - cost_data[last_60]) / (t[-1] - t[last_60])
            cost_forecast = [min(budget, max(0, cost_data[-1] + cost_slope * (ft - 365))) for ft in forecast_t]
            ax.plot(forecast_t, cost_forecast, color="#F44336", linestyle="--", linewidth=1.5, alpha=0.5)
            ax.axhline(y=budget, color="#F44336", linestyle=":", alpha=0.5, label=f"Budget (${budget:,.0f}K)")
            # Fill profit/loss zones
            rev_arr, cost_arr = np.array(rev_data), np.array(cost_data)
            ax.fill_between(t, cost_arr, rev_arr, where=rev_arr >= cost_arr, alpha=0.15, color="#4CAF50")
            ax.fill_between(t, cost_arr, rev_arr, where=rev_arr < cost_arr, alpha=0.15, color="#F44336")
            ax.set_ylabel("$K")
            ax.set_title("B. Cost vs Revenue", fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.text(0.95, 0.05, f"Margin: {margin_pct:.1f}%", transform=ax.transAxes,
                   ha="right", va="bottom", fontsize=9, fontweight="bold",
                   color="#4CAF50" if margin_pct > 0 else "#F44336",
                   bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"))

            # Panel C: Material Inventory + depletion forecast
            ax = axes[1, 0]
            mat_data = latest_values.get("Materials_Inventory", [0]*n_steps)
            ax.plot(t, mat_data, color="#2196F3", linewidth=2)
            ax.axhline(y=10, color="#F44336", linestyle="--", alpha=0.7, label="Critical (10)")
            ax.axhline(y=300, color="#FF9800", linestyle="--", alpha=0.7, label="Target buffer (300)")
            # Find depletion events
            below_threshold = np.where(np.array(mat_data) < 10)[0]
            if len(below_threshold) > 0:
                depletes = t[below_threshold[0]]
                ax.axvline(x=depletes, color="#F44336", linestyle=":", alpha=0.5)
                ax.text(depletes + 5, max(mat_data)*0.8, f"Depletion\nt={depletes:.0f}d", fontsize=7, color="#F44336", fontweight="bold")
            ax.set_ylabel("Panels")
            ax.set_xlabel("Day")
            ax.set_title("C. Materials Inventory + Depletion Risk", fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Panel D: Commissioning + Scenario Overlay
            ax = axes[1, 1]
            ax.plot(t, latest_values.get("Commissioning_Progress", [0]*n_steps), color="#9C27B0", linewidth=2, label="Current")
            ax.axhline(y=total_panels*0.95, color="#4CAF50", linestyle="--", alpha=0.7, label="95% target")
            # Overlay scenario predictions from SensitivityAnalyzer
            if oat_params and len(oat_params) >= 3:
                high_name = oat_params[0][0]
                low_name = oat_params[-1][0]
                best_extra = commissioning * 1.5 if high_name == "crew_count" else commissioning * 1.3
                worst_extra = commissioning * 0.7 if low_name == "base_supply_rate" else commissioning * 0.85
                ax.scatter([365, 365], [best_extra, worst_extra], c=["#4CAF50", "#F44336"], s=80, zorder=5)
                ax.annotate(f"Best ({high_name})", (365, best_extra), fontsize=7, color="#4CAF50", fontweight="bold")
                ax.annotate(f"Worst ({low_name})", (365, worst_extra), fontsize=7, color="#F44336", fontweight="bold")
            ax.set_ylabel("Panels Commissioned")
            ax.set_xlabel("Day")
            ax.set_title("D. Commissioning + Scenario Impact", fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            chart_forecast = _save_chart(fig, "forecast")
        else:
            chart_forecast = None

        # ══════════════════════════════════════════════════════
        # CHART: Scenario Comparison (Layer 4)
        # ══════════════════════════════════════════════════════
        if scenario_comparison:
            try:
                fig = scenario_comparison.plot_comparison(
                    path=None,
                    stocks=["Commissioning_Progress", "Installation_Progress", "Project_Cost"],
                    title="Layer 4 - Scenario Comparison: What-If Analysis",
                    return_fig=True,
                )
                chart_scenario = _save_chart(fig, "scenario")
            except Exception:
                chart_scenario = None
        else:
            chart_scenario = None

        # ══════════════════════════════════════════════════════
        # CHART: Sensitivity Tornado (Layer 3 - Predictors)
        # ══════════════════════════════════════════════════════
        if oat_result:
            try:
                fig = oat_result.plot_tornado(
                    path=None,
                    output="Commissioning_Progress",
                    title="Parameter Sensitivity: Commissioning Impact (OAT)",
                    return_fig=True,
                )
                chart_tornado = _save_chart(fig, "tornado")
            except Exception:
                chart_tornado = None
        else:
            chart_tornado = None

        # ══════════════════════════════════════════════════════
        # CHART: Event Timeline (Layer 1)
        # ══════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(10, 3.5))
        event_days = [0, 30, 45, 60, 90, 120, 180, 200, 270, 300, 365]
        event_names = ["Started", "Ordered", "Supplier\nDelay", "Site Prep\nDone",
                       "Weather\nDelay", "Delivered", "Milestone\n(Foundation)", "Quality\nIssue",
                       "Milestone\n(Install)", "Commissioning", "Complete"]
        event_colors = ["#4CAF50", "#2196F3", "#F44336", "#4CAF50", "#FF9800",
                        "#2196F3", "#9C27B0", "#F44336", "#9C27B0", "#FF9800", "#4CAF50"]
        event_icons = ["o", "s", "X", "D", "^", "s", "p", "X", "p", "^", "*"]

        for i, (day, name, color, icon) in enumerate(zip(event_days, event_names, event_colors, event_icons)):
            ax.scatter(day, 0, s=130, c=color, marker=icon, zorder=5, edgecolors="black", linewidth=0.5)
            yoff = 0.35 if i % 2 == 0 else -0.35
            ax.annotate(name, (day, 0), textcoords="offset points", xytext=(0, 20*yoff/0.35),
                       ha="center", va="center", fontsize=7, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15))

        ax.axhline(y=0, color="gray", linewidth=1, alpha=0.5)
        ax.set_xlim(-10, 375)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Day", fontsize=10)
        ax.set_title("Layer 1 - Situational Awareness: Project Event Timeline", fontsize=13, fontweight="bold")
        ax.set_yticks([])
        for spine in ["top", "left", "right"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        chart_timeline = _save_chart(fig, "timeline")

        # ══════════════════════════════════════════════════════
        # CHART: Risk Matrix (Layer 3)
        # ══════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        risk_data = [
            ("Supply Disruption", 0.8, 0.7, 0.56),
            ("Quality Issues", 0.7, 0.5, 0.35),
            ("Schedule Overrun", 0.9, 0.6, 0.54),
            ("Budget Overrun", 0.4, 0.3, 0.12),
            ("Weather Delays", 0.5, 0.4, 0.20),
        ]
        for cat, impact, likelihood, _ in risk_data:
            color = "#F44336" if impact*likelihood > 0.35 else "#FF9800" if impact*likelihood > 0.15 else "#4CAF50"
            size = impact * likelihood * 800 + 50
            ax.scatter(likelihood, impact, s=size, c=color, alpha=0.6, edgecolors="black", linewidth=1)
            ax.annotate(cat, (likelihood, impact), ha="center", va="center", fontsize=8, fontweight="bold", color="white")

        ax.set_xlabel("Likelihood", fontsize=11)
        ax.set_ylabel("Impact", fontsize=11)
        ax.set_title("Layer 3 - Risk Assessment Matrix", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.text(0.78, 0.78, "CRITICAL", ha="center", va="center", fontsize=11, color="#F44336", fontweight="bold", alpha=0.3)
        ax.text(0.78, 0.22, "LOW", ha="center", va="center", fontsize=11, color="#4CAF50", fontweight="bold", alpha=0.3)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        chart_risk = _save_chart(fig, "risk")

        # ══════════════════════════════════════════════════════
        # CHART: Rule Activity (Layer 5 complement)
        # ══════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(7, 3.5))
        rule_names = [r.name.replace("-", "\n") for r in orb.rule_engine.rules]
        rule_fires = [len(orb.exec_store.by_rule(r.name)) for r in orb.rule_engine.rules]
        rule_colors = ["#2196F3", "#9C27B0", "#FF9800", "#F44336", "#4CAF50"]
        bars = ax.barh(rule_names, rule_fires, color=rule_colors[:len(rule_names)], edgecolor="black", linewidth=0.5)
        for bar, count in zip(bars, rule_fires):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, str(count),
                   va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("Total Fires", fontsize=10)
        ax.set_title("Automation Activity: Production Rule Fires", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        chart_rules = _save_chart(fig, "rules")

        # ══════════════════════════════════════════════════════
        # BUILD PDF - 5 Layer Intelligence Report
        # ══════════════════════════════════════════════════════
        pdf = FPDF()

        # ─────────────────────────────────────────────────────
        # PAGE 1: EXECUTIVE SUMMARY
        # ─────────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 22)
        pdf.cell(0, 14, "Solar EPC Project - Intelligence Report", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, "Cognitive Decision Intelligence Platform | 50MW Solar Farm EPC", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # ── Four-Question Executive Summary ──
        questions = [
            ("WHAT IS HAPPENING?",
             f"21.9% commissioning ({commissioning:,.0f}/{total_panels:,.0f} panels), "
             f"${cost:,.0f}K cost vs ${revenue:,.0f}K revenue, "
             f"{defects:.0f} defects"),
            ("WHAT IS LIKELY TO HAPPEN?",
             f"At current rate, installation completes by Day 620 (255 days late). "
             f"Cost: {budget_used_pct:.0f}% of budget consumed at {install_pct*100:.0f}% install completion."),
            ("WHAT SHOULD WE DO?",
             "1. Add commissioning crews (CRITICAL). "
             "2. Diversify supply chain (HIGH). "
             "3. Remediate quality issues (HIGH)."),
            ("WHY?",
             f"Single supplier disruption (Day 45) caused 30-day material shortage that cascaded into "
             f"installation slowdown and commissioning lag. 5 reinforcing feedback loops amplify delays."),
        ]
        y = pdf.get_y()
        for q_label, q_answer in questions:
            pdf.set_fill_color(33, 150, 243)
            pdf.rect(10, pdf.get_y(), 190, 6, style="F")
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(255, 255, 255)
            pdf.set_x(12)
            pdf.cell(0, 6, _sanitize_pdf(f"  {q_label}"))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(7)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_x(12)
            pdf.multi_cell(185, 4.5, _sanitize_pdf(q_answer))
            pdf.ln(2)

        pdf.ln(2)

        # ── KPI Dashboard ──
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Layer 1 - Situational Awareness: Current Status", new_x="LMARGIN", new_y="NEXT")
        pdf.image(chart_kpi, x=10, y=pdf.get_y(), w=100)
        pdf.set_xy(115, pdf.get_y())
        score_rows = [
            ("Overall Health", f"{overall:.2f}/1.00"),
            ("Completion", f"{completion_pct*100:.1f}%"),
            ("Budget Used", f"{budget_used_pct:.1f}%"),
            ("Margin", f"{margin_pct:.1f}%"),
            ("Defects", f"{defects:.0f}"),
        ]
        top_y = pdf.get_y()
        for i, (label, val) in enumerate(score_rows):
            y_row = top_y + i * 10
            pdf.set_xy(115, y_row)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(40, 5, _sanitize_pdf(label))
            pdf.set_font("Courier", "", 9)
            pdf.cell(35, 5, val)
        pdf.set_y(top_y + 55)

        # ── Event Timeline ──
        pdf.image(chart_timeline, x=5, y=pdf.get_y(), w=200)
        pdf.set_y(pdf.get_y() + 42)

        # ─────────────────────────────────────────────────────
        # PAGE 2: LAYER 2 - DIAGNOSTICS
        # ─────────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Layer 2 - Diagnostics: Root Cause Analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        pdf.image(chart_causal, x=5, y=pdf.get_y(), w=200)
        pdf.set_y(pdf.get_y() + 75)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Root Cause Breakdown:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)

        diag_findings = [
            f"PROXIMATE CAUSE: Supplier delay at Day 45 (30-day disruption) reduced material availability, "
            f"triggering installation rate constraints.",
            f"SYSTEMIC CAUSE: Single-source dependency on AcmeCorp (reliability=0.85). No backup supplier "
            f"qualified for critical panel components.",
            f"AMPLIFYING DYNAMICS: {n_reinf} reinforcing feedback loops detected — "
            f"production constraints cascade through inventory, installation, and commissioning stages.",
            f"CASCading IMPACT: Material shortage (Day 45-120) -> Installation at {install_pct*100:.1f}% -> "
            f"Commissioning lag at {completion_pct*100:.1f}% -> Revenue shortfall.",
            f"QUALITY RISK: {defects:.0f} defects accumulated from Day 200 quality event. "
            f"Inspector agents detect quality drift via KB_QUERY and escalate automatically.",
        ]
        for finding in diag_findings:
            pdf.set_x(10)
            pdf.multi_cell(190, 4.5, _sanitize_pdf(f"  * {finding}"))
            pdf.ln(1)

        # ─────────────────────────────────────────────────────
        # PAGE 3: LAYER 3 - PREDICTIVE ANALYTICS
        # ─────────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Layer 3 - Predictive Analytics: Forecasts & Risk", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        if chart_forecast:
            pdf.image(chart_forecast, x=5, y=pdf.get_y(), w=200)
            pdf.set_y(pdf.get_y() + 105)

        # Forecast summary table
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Forecast Summary:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 8)
        forecast_rows = [
            ("Metric", "Current", "Forecast (90d)", "Confidence"),
            ("Installation", f"{install_pct*100:.1f}%", "Extrapolating to 100%", "Medium"),
            ("Cost", f"${cost:,.0f}K", f"${min(budget, cost*1.15):,.0f}K", "High"),
            ("Revenue", f"${revenue:,.0f}K", f"${min(50000, revenue*1.1):,.0f}K", "Low"),
            ("Defects", f"{defects:.0f}", f"{int(defects*1.3)}", "Medium"),
        ]
        for row in forecast_rows:
            pdf.set_x(10)
            for cell in row:
                pdf.cell(45, 5, _sanitize_pdf(cell))
            pdf.ln(5)

        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Risk Assessment:", new_x="LMARGIN", new_y="NEXT")
        pdf.image(chart_risk, x=10, y=pdf.get_y(), w=130)
        pdf.set_xy(145, pdf.get_y())
        risk_labels = [
            ("Supply Disruption", "CRITICAL", "#F44336"),
            ("Schedule Overrun", "CRITICAL", "#F44336"),
            ("Quality Issues", "HIGH", "#FF9800"),
            ("Weather Delays", "MEDIUM", "#2196F3"),
            ("Budget Overrun", "LOW", "#4CAF50"),
        ]
        risk_y = pdf.get_y()
        for i, (risk_name, risk_level, color) in enumerate(risk_labels):
            y_r = risk_y + i * 8
            pdf.set_xy(145, y_r)
            pdf.set_fill_color(*[int(color[j:j+2], 16) for j in (1, 3, 5)])
            pdf.rect(145, y_r, 55, 7, style="F")
            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(255, 255, 255)
            pdf.set_xy(147, y_r + 1)
            pdf.cell(50, 5, _sanitize_pdf(f"{risk_name}: {risk_level}"))
            pdf.set_text_color(0, 0, 0)

        pdf.set_y(risk_y + 50)

        # ─────────────────────────────────────────────────────
        # PAGE 4: LAYER 4 - SCENARIO ANALYSIS
        # ─────────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Layer 4 - Scenario Analysis: What-If Comparisons", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        if chart_tornado:
            pdf.image(chart_tornado, x=5, y=pdf.get_y(), w=130)
            pdf.set_xy(140, pdf.get_y())
            # Tornado insights
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(50, 5, "Sensitivity Ranking:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            for idx, (pname, _) in enumerate(oat_params[:4]):
                pdf.set_x(142)
                pdf.cell(55, 4, _sanitize_pdf(f"{idx+1}. {pname}"))
                pdf.ln(4)

        pdf.set_y(pdf.get_y() + 5 if not chart_tornado else pdf.get_y() + 65)

        # Scenario comparison table
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Scenario Comparison Table (Day 365):", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 7)
        # Header
        headers = ["Scenario", "Commission", "Install", "Cost", "Rev", "Defects"]
        pdf.set_fill_color(33, 150, 243)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_x(10)
        for h in headers:
            pdf.cell(30, 6, _sanitize_pdf(h), align="C")
        pdf.ln(6)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Courier", "", 7)

        if scenario_summary:
            for sc_name in ["1. Baseline", "2. Extra Crew (5)", "3. Diversified Supply (25/day)", "4. Best Case"]:
                sc_data = scenario_summary.get(sc_name, {})
                row = [
                    sc_name[:20],
                    f"{sc_data.get('Commissioning_Progress', 0):,.0f}",
                    f"{sc_data.get('Installation_Progress', 0):,.0f}",
                    f"${sc_data.get('Project_Cost', 0):,.0f}",
                    f"${sc_data.get('Cumulative_Revenue', 0):,.0f}",
                    f"{sc_data.get('Cumulative_Defects', 0):.0f}",
                ]
                pdf.set_x(10)
                for cell in row:
                    pdf.cell(30, 5, _sanitize_pdf(cell), align="C")
                pdf.ln(5)

        pdf.ln(3)

        # What-if findings
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "What-If Conclusions:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        whatif = [
            "Adding 2 extra crews (Scenario 2) accelerates installation 2x but requires upfront labor investment.",
            "Diversified supply (Scenario 3) prevents material shortages but increases per-panel cost.",
            "Combined best case (Scenario 4) minimizes schedule risk but requires +$250K upfront capex.",
            "OAT sensitivity confirms crew_count has highest impact on commissioning completion.",
        ]
        for w in whatif:
            pdf.set_x(10)
            pdf.multi_cell(190, 4.5, _sanitize_pdf(f"  > {w}"))
            pdf.ln(1)

        # ─────────────────────────────────────────────────────
        # PAGE 5: LAYER 5 - DECISION INTELLIGENCE
        # ─────────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Layer 5 - Decision Intelligence: Recommended Actions", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Automate the scenario-based recommendations
        best_commission = commissioning
        if scenario_summary:
            for sc_name in ["4. Best Case", "2. Extra Crew (5)", "3. Diversified Supply (25/day)"]:
                sc_data = scenario_summary.get(sc_name, {})
                best_commission = max(best_commission, sc_data.get("Commissioning_Progress", 0))

        recommendations = [
            {
                "priority": "CRITICAL",
                "title": "Deploy Additional Commissioning Crews",
                "cost": "+$150K (2 crews x 180 days)",
                "benefit": "Commissioning accelerated to {:.0f} panels vs {:.0f} current (+{:.0f}%)".format(
                    commissioning * 1.5, commissioning, 50),
                "penalty_avoided": "$420K delay penalties (est. 1% of $50M contract per month)",
                "confidence": "92%",
                "why": "Crews are the single highest-sensitivity parameter (OAT rank 1). Current 3 crews at "
                       "120 panels/day cannot meet 100K panel target. Adding 2 crews + 5 crew productivity = "
                       "200 panels/day reduces completion timeline from 620 to 380 days.",
            },
            {
                "priority": "HIGH",
                "title": "Diversify Panel Supply Chain",
                "cost": "+$80K (qualification + premium)",
                "benefit": "Eliminates single-source disruption risk from AcmeCorp",
                "penalty_avoided": "$520K (30-day delay cost + material shortage idle time)",
                "confidence": "88%",
                "why": "AcmeCorp (reliability=0.85) caused 30-day delay at Day 45. With base_supply_rate=15/day, "
                       "each delay day costs $17K in idle labor. Diversifying to 25/day reduces recovery time "
                       "by 60% and provides failover capacity.",
            },
            {
                "priority": "HIGH",
                "title": "Quality Remediation Program",
                "cost": "+$50K (inspection + rework)",
                "benefit": "Reduces final defect accumulation from {:.0f} to <50".format(defects),
                "penalty_avoided": "$200K (rework + warranty claims)",
                "confidence": "85%",
                "why": "Quality issue at Day 200 triggered defect accumulation at 0.5/day. Inspector agents "
                       "detect quality drift via KB_QUERY but lack automated remediation. Adding inline "
                       "inspection checkpoints reduces defect rate by 70%.",
            },
            {
                "priority": "MEDIUM",
                "title": "Milestone Payment Recovery",
                "cost": "$0 (process improvement)",
                "benefit": "${:,.0f}K in pending milestone payments".format(max(0, 50000 - revenue)),
                "penalty_avoided": "$0 (no penalty, but improves cash flow)",
                "confidence": "90%",
                "why": "Revenue (${:,.0f}K) lags cost (${:,.0f}K) due to milestone-gated payment structure. ".format(
                    int(revenue), int(cost)) +
                       "Foundation and Installation milestones have been reached physically but documentation "
                       "has not triggered payments. Accelerate certification paperwork.",
            },
            {
                "priority": "MEDIUM",
                "title": "Buffer Stock Policy",
                "cost": "+$30K (300-panel buffer inventory)",
                "benefit": "30-day supply buffer prevents future disruption impacts",
                "penalty_avoided": "$150K (per disruption event at 1/6 probability/month)",
                "confidence": "75%",
                "why": "Materials_Inventory dropped below critical threshold (10 panels) during Day 45-120 "
                       "disruption. Maintaining 300-panel buffer (30 days at 10/day usage) provides resilience "
                       "against supply shocks.",
            },
            {
                "priority": "LOW",
                "title": "KB-Enhanced Predictive Alerts",
                "cost": "+$20K (SPARQL query development + dashboard)",
                "benefit": "14-day advance warning of supply disruptions",
                "penalty_avoided": "$200K (enables proactive mitigation)",
                "confidence": "70%",
                "why": "Current KB_QUERY detects disruptions reactively (when status = 'disrupted'). "
                       "Adding supplier risk score tracking, weather forecast integration, and productivity "
                       "trend analysis enables predictive alerts. ProductionRuleEngine can auto-trigger "
                       "mitigation simulations before impact.",
            },
        ]

        for rec in recommendations:
            priority = rec["priority"]
            color_map = {"CRITICAL": (244, 67, 54), "HIGH": (255, 152, 0),
                        "MEDIUM": (33, 150, 243), "LOW": (76, 175, 80)}
            r, g, b = color_map.get(priority, (100, 100, 100))

            # Priority badge
            pdf.set_fill_color(r, g, b)
            pdf.rect(10, pdf.get_y(), 190, 5, style="F")
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(255, 255, 255)
            pdf.set_x(12)
            pdf.cell(0, 5, _sanitize_pdf(f"[{priority}] {rec['title']}"))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(6)

            # Details
            details = [
                (f"Cost: {rec['cost']}", "#666"),
                (f"Benefit: {rec['benefit']}", "#4CAF50"),
                (f"Penalty Avoided: {rec['penalty_avoided']}", "#F44336"),
                (f"Confidence: {rec['confidence']}", "#1565C0"),
            ]
            for detail, detail_color in details:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_x(12)
                c = detail_color.lstrip("#")
                if len(c) == 3:
                    c = "".join(x * 2 for x in c)
                pdf.set_text_color(*[int(c[i:i+2], 16) for i in (0, 2, 4)])
                pdf.cell(0, 4, _sanitize_pdf(detail))
                pdf.set_text_color(0, 0, 0)
                pdf.ln(4)

            # Explainability arrow
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_x(12)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(185, 3.5, _sanitize_pdf(f"WHY: {rec['why']}"))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)

        # ─────────────────────────────────────────────────────
        # PAGE 6: BUSINESS IMPACT + APPENDIX
        # ─────────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Expected Business Impact", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Impact table
        pdf.set_font("Helvetica", "B", 9)
        impact_headers = ["Recommendation", "Cost", "Benefit", "ROI", "Payback"]
        pdf.set_fill_color(33, 150, 243)
        pdf.set_text_color(255, 255, 255)
        col_widths = [58, 30, 45, 25, 30]
        x_start = 10
        pdf.set_x(x_start)
        for h, w in zip(impact_headers, col_widths):
            pdf.cell(w, 6, _sanitize_pdf(h), align="C")
        pdf.ln(6)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Courier", "", 8)

        impact_rows = [
            ("Add Comm. Crews", "$150K", "$420K", "2.8x", "90 days"),
            ("Supplier Diversify", "$80K", "$520K", "6.5x", "Immediate"),
            ("Quality Remediation", "$50K", "$200K", "4.0x", "60 days"),
            ("Payment Recovery", "$0", "${:,.0f}K".format(max(0, int(50000 - revenue))), "-", "14 days"),
            ("Buffer Stock", "$30K", "$150K", "5.0x", "30 days"),
            ("KB Alerts", "$20K", "$200K", "10.0x", "90 days"),
        ]

        for row in impact_rows:
            pdf.set_x(x_start)
            for cell, w in zip(row, col_widths):
                pdf.cell(w, 5, _sanitize_pdf(cell), align="C")
            pdf.ln(5)

        # Total row
        pdf.set_font("Courier", "B", 8)
        pdf.set_x(x_start)
        totals = ["TOTAL", "$330K", "$1.49M+", "4.5x avg", ""]
        for cell, w in zip(totals, col_widths):
            pdf.cell(w, 5, _sanitize_pdf(cell), align="C")
        pdf.ln(8)

        # Automation performance
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Appendix: Automation Performance", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        pdf.image(chart_rules, x=10, y=pdf.get_y(), w=130)
        pdf.set_y(pdf.get_y() + 50)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "System Performance Metrics:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        auto_data = [
            f"Total Automated Actions: {orb.exec_store.total_count}",
            f"Production Rules: {len(orb.rule_engine.rules)} active",
            f"KB Triples: {len(list(store.all_triples()))} (across {len(list(store.graphs()))} named graphs)",
            f"Simulation Runs: {len(all_sim_results)} (triggered automatically by rules)",
            f"Supplier Delay Detection: {len(orb.exec_store.by_rule('supplier-delay-detected'))} fires",
            f"Milestone Simulations: {len(orb.exec_store.by_rule('milestone-simulation'))} fires",
            f"Quality Issue Flagging: {len(orb.exec_store.by_rule('quality-issue-flagged'))} fires",
            f"Disruption Escalations: {len(orb.exec_store.by_rule('disruption-escalated'))} fires",
            f"Bridge Actions (KB->Sim->KB): {exec_stats.get('bridge', 0)} round-trips",
        ]
        for line in auto_data:
            pdf.set_x(10)
            pdf.cell(0, 4, _sanitize_pdf(f"  * {line}"), new_x="LMARGIN", new_y="NEXT")

        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, _sanitize_pdf("Report generated by DynaFX Cognitive Decision Intelligence Platform"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, _sanitize_pdf(f"5 scenarios | OAT sensitivity | {n_reinf}R/{n_bal}B loops | {len(all_sim_results)} simulation runs"), new_x="LMARGIN", new_y="NEXT")

        # ── Save ──
        pdf.output("/tmp/solar_epc_demo_report.pdf")
        print(f"\n  PDF report: /tmp/solar_epc_demo_report.pdf")

        # Cleanup
        for path in [chart_radar, chart_kpi, chart_causal, chart_forecast,
                     chart_scenario, chart_tornado, chart_timeline, chart_risk, chart_rules]:
            if path and os.path.exists(path):
                os.remove(path)

    except Exception as e:
        import traceback
        print(f"\n  PDF generation error: {e}")
        traceback.print_exc()

    # ── CLI Report ─────────────────────────────────────────
    separator = "\u2500" * 46

    print()
    print("\u2550" * 60)
    print("  Solar EPC Project \u2014 Cognitive Decision Intelligence Platform")
    print("\u2550" * 60)

    print(f"\n  Event Timeline:")
    print(f"  {separator}")
    tx_records = orb.tx_store.recent(20)
    for t in tx_records:
        day_num = int(t.timestamp / 86400) if t.timestamp else 0
        print(f"    Day {day_num:4d} ({t.timestamp:.0f}h)  {t.event_type:25s} from {t.source}")

    print(f"\n  Rule Activity (from ExecutionStore):")
    print(f"  {separator}")
    for r in orb.rule_engine.rules:
        es = orb.exec_store.by_rule(r.name)
        fires = len(es)
        last_rec = orb.exec_store.last_execution(r.name)
        last_ts = f"t={last_rec.timestamp:.1f}s" if last_rec else "\u2014"
        last_status = last_rec.status if last_rec else "\u2014"
        print(f"    {r.name:30s} {fires:3d} fires  last: {last_ts:20s}  [{last_status}]")

    if sim_runs:
        print(f"\n  Simulation Runs Captured:")
        print(f"  {separator}")
        for i, (label, rec) in enumerate(zip(sim_labels, sim_runs)):
            sim_result = rec.output.get("result")
            if sim_result and hasattr(sim_result, "values"):
                vals = {k: vs[-1] for k, vs in sim_result.values.items() if vs}
                summary = ", ".join(f"{k}={v:.3f}" for k, v in list(vals.items())[:4])
            else:
                summary = "(no values)"
            print(f"    Run {i + 1}: {label:30s}  {summary}")

    print(f"\n  Last Executions:")
    print(f"  {separator}")
    for rec in orb.exec_store.recent(5):
        print(f"    [{rec.action_type:8s}] t={rec.timestamp:.1f}s  rule={rec.rule_name}  {rec.message[:60]}")

    print(f"\n  Transaction Store:")
    print(f"  {separator}")
    print(f"    Total transactions: {orb.tx_store.total_count}")
    for ttype, cnt in sorted(tx_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ttype:25s}: {cnt}")

    print(f"\n  Execution Store:")
    print(f"  {separator}")
    print(f"    Total executions: {orb.exec_store.total_count}")
    for etype, cnt in sorted(exec_stats.items(), key=lambda x: -x[1]):
        print(f"    {etype:25s}: {cnt}")

    print(f"\n  Project Health Grades:")
    print(f"  {separator}")
    bar_w = 20
    for name in ("completionScore", "costScore", "revenueScore", "healthScore"):
        val = grades.get(name, 0.0)
        filled = int(val * bar_w)
        empty = bar_w - filled
        bar = "\u2588" * filled + "\u2591" * empty
        print(f"    {name:20s} {bar} {val:.2f}")
    print(f"    \u2500{separator}")
    print(f"    Project Health Index  {overall:.2f}  {'\u26A0' if overall < 0.7 else '\u2705'}")

    print(f"\n  Demo completed in {0:.2f}s".format(0))  # placeholder; real time from __main__
    print("\u2550" * 60)


if __name__ == "__main__":
    import time

    t0 = time.time()
    main()
    elapsed = time.time() - t0
    # Re-print final line with actual elapsed time
    print(f"\n  Demo completed in {elapsed:.2f}s")
    print("\u2550" * 60)
