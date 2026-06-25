"""Subjective Logic (SL) — graph utilities and parameter binding.

Note: SL opinion algebra lives in cognitive_engine.reason.fusion.
This package contains graph-level utilities moved from system/sl/.
"""

from cognitive_engine.sl.operators import (
    list_nodes,
    get_node,
    get_edge,
    query_contested,
    query_by_role,
    get_trace_history,
    create_node,
    create_edge,
    set_role,
    set_parameter,
    merge_nodes,
    retract,
)
from cognitive_engine.sl.validation import (
    ArgumentType,
    ValidationResult,
    Argument,
    Attack,
    ValidationResultDetail,
    validate_system_internal,
    get_validation_summary,
)
from cognitive_engine.sl.parameters import (
    bind_parameters,
    get_parameter_summary,
)

__all__ = [
    "list_nodes",
    "get_node",
    "get_edge",
    "query_contested",
    "query_by_role",
    "get_trace_history",
    "create_node",
    "create_edge",
    "set_role",
    "set_parameter",
    "merge_nodes",
    "retract",
    "ArgumentType",
    "ValidationResult",
    "Argument",
    "Attack",
    "ValidationResultDetail",
    "validate_system_internal",
    "get_validation_summary",
    "bind_parameters",
    "get_parameter_summary",
]
