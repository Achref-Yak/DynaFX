"""System Dynamics (SD) — stocks, flows, equations, simulation.

Primary API::

    from dynafx.system import SysdModel, parse_sysd_file
    model = parse_sysd_file("model.sysd")
    result = model.simulate()
    result.plot("output.png")

Subpackages: dsl, causal, feedback, scenario, units, agent, des, vensim
"""

from dynafx.system.dsl import (
    SysdModel,
    SysdModelResult,
    parse_sysd,
    parse_sysd_file,
    ValidationResult,
    ValidationIssue,
    StockDef,
    FlowDef,
    AuxDef,
    TableDef,
    AgentDef,
    QueueDef,
    ResourceDef,
    EventDef,
    SubmodelDef,
    IncludeDef,
)
from dynafx.system.causal import (
    causal_trace,
    causes_tree,
    effects_tree,
    causes_strip,
)
from dynafx.system.feedback import (
    detect_feedback_loops,
    loops_for_variable,
)
from dynafx.system.scenario import (
    ScenarioComparison,
    ScenarioDef,
    ScenarioResult,
)
from dynafx.system.equations import (
    compile_equations,
    get_equation_summary,
    simulate_equations,
    rk4_step,
    euler_step,
)
from dynafx.system.emergent import (
    EmergentProperty,
    Condition,
    Effect,
    ComparisonOp,
    EffectType,
    ConsistencyResult,
    ConsistencyViolation,
    run_consistency_checks,
)
from dynafx.system.agent import (
    AgentInstance,
    ABMEngine,
)
from dynafx.system.des import (
    DESEngine,
    EventQueue,
    Queue,
    Resource,
    QueueStats,
    ResourceStats,
)
from dynafx.system.units import (
    Unit,
    UnitRegistry,
    UnitChecker,
    UnitCheckResult,
    UnitViolation,
)
# Basic Formal Ontology alignment — available on request
from dynafx.system.bfo import (
    get_bfo_alignment as _get_bfo_alignment,
    BfoContinuantCategory as _BfoContinuantCategory,
)
__all__ = [
    # DSL & simulation
    "SysdModel",
    "SysdModelResult",
    "parse_sysd",
    "parse_sysd_file",
    "ValidationResult",
    "ValidationIssue",
    "StockDef",
    "FlowDef",
    "AuxDef",
    "TableDef",
    # ABM
    "AgentDef",
    "AgentInstance",
    "ABMEngine",
    # DES
    "QueueDef",
    "ResourceDef",
    "EventDef",
    "DESEngine",
    "EventQueue",
    "Queue",
    "Resource",
    "QueueStats",
    "ResourceStats",
    # Submodels
    "SubmodelDef",
    "IncludeDef",
    # Structural analysis
    "causal_trace",
    "causes_tree",
    "effects_tree",
    "causes_strip",
    "detect_feedback_loops",
    "loops_for_variable",
    # Scenario comparison
    "ScenarioComparison",
    "ScenarioDef",
    "ScenarioResult",
    # Equation compilation
    "compile_equations",
    "get_equation_summary",
    "simulate_equations",
    "rk4_step",
    "euler_step",
    # Emergent properties
    "EmergentProperty",
    "Condition",
    "Effect",
    "ComparisonOp",
    "EffectType",
    "ConsistencyResult",
    "ConsistencyViolation",
    "run_consistency_checks",
    # Units
    "Unit",
    "UnitRegistry",
    "UnitChecker",
    "UnitCheckResult",
    "UnitViolation",
]
