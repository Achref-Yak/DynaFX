"""DynaFX — System Dynamics, Agent-Based Modeling, Discrete Event Simulation,
Knowledge Base, and Subjective Logic reasoning.

Domain-agnostic multi-paradigm simulation framework.

Quick start::

    from dynafx import SysdModel, parse_sysd_file
    model = parse_sysd_file("model.sysd")
    result = model.simulate()

Subpackages: system, kb, sl, reason, tbox
"""

from dynafx.system import SysdModel, SysdModelResult, parse_sysd, parse_sysd_file
from dynafx.system import causal_trace, detect_feedback_loops
from dynafx.system import ScenarioComparison, ScenarioDef
from dynafx.kb import TripleStore, parse_turtle
from dynafx.reason import cumulative_fusion, EvidenceMatrix

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
    # Knowledge base
    "TripleStore",
    "parse_turtle",
    # Reasoning
    "cumulative_fusion",
    "EvidenceMatrix",
]
