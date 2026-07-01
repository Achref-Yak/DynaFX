"""DynaFX — systems thinking framework with four pillars:

  Dynamics     System Dynamics, Agent-Based Modeling, Discrete Event Simulation
  Knowledge    RDF/OWL/SPARQL triple store with SL confidence grading
  Epistemics   Subjective Logic algebra, evidence matrix, argumentation, KBT
  Core         Foundational data models (Opinion, Graph, etc.)

Quick start::

    from dynafx import SysdModel, parse_sysd_file
    model = parse_sysd_file("model.sysd")
    result = model.simulate()
"""

from dynafx.dynamics import SysdModel, SysdModelResult, parse_sysd, parse_sysd_file
from dynafx.dynamics import causal_trace, detect_feedback_loops
from dynafx.dynamics import ScenarioComparison, ScenarioDef
from dynafx.dynamics.optimization import lp_minimize, lp_maximize, calibrate, optimize
from dynafx.dynamics.optimization import kb_lp_minimize, kb_lp_maximize, kb_calibrate, kb_optimize
from dynafx.knowledge import TripleStore, parse_turtle
from dynafx.epistemics import cumulative_fusion, EvidenceMatrix
from dynafx.bridge import KBSimBridge, ClosedLoopReasoner, ReasoningPass, ClosedLoopResult, grade_queries

__all__ = [
    # SD / multi-paradigm
    "SysdModel",
    "SysdModelResult",
    "parse_sysd",
    "parse_sysd_file",
    "causal_trace",
    "detect_feedback_loops",
    "ScenarioComparison",
    "ScenarioDef",
    # Optimization
    "lp_minimize",
    "lp_maximize",
    "calibrate",
    "optimize",
    # KB-constrained optimization
    "kb_lp_minimize",
    "kb_lp_maximize",
    "kb_calibrate",
    "kb_optimize",
    # Knowledge base
    "TripleStore",
    "parse_turtle",
    # Reasoning
    "cumulative_fusion",
    "EvidenceMatrix",
    # Bridge
    "KBSimBridge",
    "ClosedLoopReasoner",
    "ReasoningPass",
    "ClosedLoopResult",
    "grade_queries",
]
