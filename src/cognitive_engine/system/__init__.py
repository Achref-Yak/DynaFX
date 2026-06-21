"""System modeling module for cognitive_engine.

Provides operators for converting human-defined knowledge graphs
into simulatable system dynamics models.
"""

from cognitive_engine.system.operators import (
    # Read operators
    list_nodes,
    get_node,
    get_edge,
    query_contested,
    query_by_role,
    get_trace_history,
    # Write operators
    create_node,
    create_edge,
    set_role,
    set_parameter,
    merge_nodes,
    retract,
    # Data classes
    Role,
    RoleAssignment,
    TraceEntry,
)
from cognitive_engine.system.parameters import (
    bind_parameters,
    get_parameter_summary,
)
from cognitive_engine.system.equations import (
    compile_equations,
    get_equation_summary,
    simulate_equations,
    rk4_step,
    euler_step,
    Equation,
    LoopClassification,
    LoopType,
)
from cognitive_engine.system.validation import (
    validate_system_internal,
    get_validation_summary,
    ArgumentType,
    ValidationResult,
    ValidationResultDetail,
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

__all__ = [
    # Read operators
    "list_nodes",
    "get_node",
    "get_edge",
    "query_contested",
    "query_by_role",
    "get_trace_history",
    # Write operators
    "create_node",
    "create_edge",
    "set_role",
    "set_parameter",
    "merge_nodes",
    "retract",
    # Parameter binding
    "bind_parameters",
    "get_parameter_summary",
    # Equation compilation
    "compile_equations",
    "get_equation_summary",
    "simulate_equations",
    "rk4_step",
    "euler_step",
    "Equation",
    "LoopClassification",
    "LoopType",
    # Validation
    "validate_system_internal",
    "get_validation_summary",
    "ArgumentType",
    "ValidationResult",
    "ValidationResultDetail",
    # BFO alignment
    "get_bfo_alignment",
    "get_all_bfo_alignments",
    "validate_bfo_alignment",
    "get_bfo_description",
    "get_continuant_roles",
    "get_occurrent_roles",
    "get_bfo_summary",
    "BfoContinuantCategory",
    # Data classes
    "Role",
    "RoleAssignment",
    "TraceEntry",
]
