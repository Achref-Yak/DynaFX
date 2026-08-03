"""Dynamics — System Dynamics, Agent-Based Modeling, Discrete Event Simulation.

Primary API::

    from dynafx.dynamics import SysdModel, parse_sysd_file
    model = parse_sysd_file("model.sysd")
    result = model.simulate()
    result.plot("output.png")

Subpackages: dsl, causal, feedback, scenario, units, agent, des
"""

from dynafx.dynamics.agent import (
    ABMEngine,
    AgentInstance,
    Message,
)
from dynafx.dynamics.causal import (
    causal_trace,
    causes_strip,
    causes_tree,
    effects_tree,
)
from dynafx.dynamics.des import (
    DESEngine,
    EventQueue,
    Queue,
    QueueStats,
    Resource,
    ResourceStats,
)
from dynafx.dynamics.dsl import (
    AgentDef,
    AgentRuleDef,
    AgentStrategy,
    AuxDef,
    EventDef,
    FlowDef,
    IncludeDef,
    QueueDef,
    ResourceDef,
    StockDef,
    SubmodelDef,
    SysdModel,
    SysdModelResult,
    TableDef,
    ValidationIssue,
    ValidationResult,
    parse_sysd,
    parse_sysd_file,
)
from dynafx.dynamics.emergent import (
    ComparisonOp,
    Condition,
    ConsistencyResult,
    ConsistencyViolation,
    Effect,
    EffectType,
    EmergentProperty,
    run_consistency_checks,
)
from dynafx.dynamics.equations import (
    compile_equations,
    euler_step,
    get_equation_summary,
    rk4_step,
    simulate_equations,
)
from dynafx.dynamics.feedback import (
    detect_feedback_loops,
    loops_for_variable,
)
from dynafx.dynamics.optimization import (
    CalibrationResult,
    LPResult,
    OptimizationResult,
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
from dynafx.dynamics.scenario import (
    ScenarioComparison,
    ScenarioDef,
    ScenarioResult,
)
from dynafx.dynamics.sensitivity import SensitivityAnalyzer, SensitivityResult
from dynafx.dynamics.units import (
    Unit,
    UnitChecker,
    UnitCheckResult,
    UnitRegistry,
    UnitViolation,
)
from dynafx.patterns.signal_chain import SignalChain  # re-exported for compat

__all__ = [  # noqa: RUF022  section-grouped, not alphabetical
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
    "AgentRuleDef",
    "AgentStrategy",
    "AgentInstance",
    "ABMEngine",
    "Message",
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
    "ParetoResult",
    "lp_minimize",
    "lp_maximize",
    "calibrate",
    "optimize",
    "pareto_optimize",
    # KB-constrained optimization
    "kb_lp_minimize",
    "kb_lp_maximize",
    "kb_calibrate",
    "kb_optimize",
]
