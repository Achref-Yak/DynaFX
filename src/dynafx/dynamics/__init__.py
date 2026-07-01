"""Dynamics — System Dynamics, Agent-Based Modeling, Discrete Event Simulation.

Primary API::

    from dynafx.dynamics import SysdModel, parse_sysd_file
    model = parse_sysd_file("model.sysd")
    result = model.simulate()
    result.plot("output.png")

Subpackages: dsl, causal, feedback, scenario, units, agent, des, vensim
"""

from dynafx.dynamics.dsl import (
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
from dynafx.dynamics.causal import (
    causal_trace,
    causes_tree,
    effects_tree,
    causes_strip,
)
from dynafx.dynamics.feedback import (
    detect_feedback_loops,
    loops_for_variable,
)
from dynafx.dynamics.scenario import (
    ScenarioComparison,
    ScenarioDef,
    ScenarioResult,
)
from dynafx.dynamics.equations import (
    compile_equations,
    get_equation_summary,
    simulate_equations,
    rk4_step,
    euler_step,
)
from dynafx.dynamics.emergent import (
    EmergentProperty,
    Condition,
    Effect,
    ComparisonOp,
    EffectType,
    ConsistencyResult,
    ConsistencyViolation,
    run_consistency_checks,
)
from dynafx.dynamics.agent import (
    AgentInstance,
    ABMEngine,
)
from dynafx.dynamics.des import (
    DESEngine,
    EventQueue,
    Queue,
    Resource,
    QueueStats,
    ResourceStats,
)
from dynafx.dynamics.units import (
    Unit,
    UnitRegistry,
    UnitChecker,
    UnitCheckResult,
    UnitViolation,
)
from dynafx.dynamics.bfo import (
    get_bfo_alignment as _get_bfo_alignment,
    BfoContinuantCategory as _BfoContinuantCategory,
)
from dynafx.dynamics.signal_chain import SignalChain
from dynafx.dynamics.sensitivity import SensitivityAnalyzer, SensitivityResult
from dynafx.dynamics.optimization import (
    LPResult,
    CalibrationResult,
    OptimizationResult,
    lp_minimize,
    lp_maximize,
    calibrate,
    optimize,
    kb_lp_minimize,
    kb_lp_maximize,
    kb_calibrate,
    kb_optimize,
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
    # Templates
    "SignalChain",
    # Sensitivity analysis
    "SensitivityAnalyzer",
    "SensitivityResult",
    # Optimization
    "LPResult",
    "CalibrationResult",
    "OptimizationResult",
    "lp_minimize",
    "lp_maximize",
    "calibrate",
    "optimize",
    # KB-constrained optimization
    "kb_lp_minimize",
    "kb_lp_maximize",
    "kb_calibrate",
    "kb_optimize",
]
