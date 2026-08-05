"""DynaFX — systems thinking framework:

  Dynamics     System Dynamics, Agent-Based Modeling, Discrete Event Simulation
  Knowledge    RDF/OWL/SPARQL triple store
  Bridge       KB <-> simulation glue (params, KB_QUERY, evidence, rules)
  Patterns     Reusable cross-paradigm model factories
  Core         Foundational data models (Graph, Node, Edge, etc.)

Quick start::

    from dynafx import SysdModel, parse_sysd_file
    model = parse_sysd_file("model.sysd")
    result = model.simulate()
"""

from dynafx.bridge import (
    ClosedLoopReasoner,
    ClosedLoopResult,
    KBSimBridge,
    ReasoningPass,
    grade_queries,
)
from dynafx.dynamics import (
    ScenarioComparison,
    ScenarioDef,
    SysdModel,
    SysdModelResult,
    causal_trace,
    detect_feedback_loops,
    parse_sysd,
    parse_sysd_file,
)
from dynafx.dynamics.optimization import (
    ParetoResult,
    calibrate,
    kb_calibrate,
    kb_lp_maximize,
    kb_lp_minimize,
    kb_optimize,
    lp_maximize,
    lp_minimize,
    optimize,
    pareto_optimize,
)
from dynafx.knowledge import TripleStore, parse_turtle

__all__ = [
    "ClosedLoopReasoner",
    "ClosedLoopResult",
    "KBSimBridge",
    "ParetoResult",
    "ReasoningPass",
    "ScenarioComparison",
    "ScenarioDef",
    "SysdModel",
    "SysdModelResult",
    "TripleStore",
    "calibrate",
    "causal_trace",
    "detect_feedback_loops",
    "grade_queries",
    "kb_calibrate",
    "kb_lp_maximize",
    "kb_lp_minimize",
    "kb_optimize",
    "lp_maximize",
    "lp_minimize",
    "optimize",
    "pareto_optimize",
    "parse_sysd",
    "parse_sysd_file",
    "parse_turtle",
]
