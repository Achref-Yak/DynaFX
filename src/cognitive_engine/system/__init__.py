"""System Dynamics (SD) — stocks, flows, equations, simulation.

Works on its own system: SysdModel, DSL, RK4/Euler solvers,
emergent properties, consistency checking, Vensim import.
Independent of Subjective Logic.
"""

from cognitive_engine.system.equations import (
    compile_equations,
    get_equation_summary,
    simulate_equations,
    rk4_step,
    euler_step,
    Equation,
    LoopType,
)
from cognitive_engine.system.bfo import (
    get_bfo_alignment,
    get_all_bfo_alignments,
    validate_bfo_alignment,
    get_bfo_description,
    get_continuant_roles,
    get_occurrent_roles,
    get_bfo_summary,
    BfoContinuantCategory,
)
from cognitive_engine.system.emergent import (
    EmergentProperty,
    Condition,
    Effect,
    ComparisonOp,
    EffectType,
    ConsistencyResult,
    ConsistencyViolation,
    run_consistency_checks,
)
from cognitive_engine.system.agent import (
    AgentInstance,
    ABMEngine,
)
from cognitive_engine.system.des import (
    DESEngine,
    EventQueue,
    Queue,
    Resource,
    QueueStats,
    ResourceStats,
)
from cognitive_engine.system.units import (
    Unit,
    UnitRegistry,
    UnitChecker,
    UnitCheckResult,
    UnitViolation,
)
__all__ = [
    # Equation compilation
    "compile_equations",
    "get_equation_summary",
    "simulate_equations",
    "rk4_step",
    "euler_step",
    "Equation",
    "LoopType",
    # BFO alignment
    "get_bfo_alignment",
    "get_all_bfo_alignments",
    "validate_bfo_alignment",
    "get_bfo_description",
    "get_continuant_roles",
    "get_occurrent_roles",
    "get_bfo_summary",
    "BfoContinuantCategory",
    # Emergent properties
    "EmergentProperty",
    "Condition",
    "Effect",
    "ComparisonOp",
    "EffectType",
    "ConsistencyResult",
    "ConsistencyViolation",
    "run_consistency_checks",
    # ABM
    "AgentInstance",
    "ABMEngine",
    # DES
    "DESEngine",
    "EventQueue",
    "Queue",
    "Resource",
    "QueueStats",
    "ResourceStats",
    # Units
    "Unit",
    "UnitRegistry",
    "UnitChecker",
    "UnitCheckResult",
    "UnitViolation",
]
