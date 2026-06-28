"""Subjective Logic (SL) — graph utilities and parameter binding.

Core SL algebra lives in ``dynafx.reason``::

    from dynafx.reason import Opinion, cumulative_fusion, consensus_compromise

This package (``dynafx.sl``) contains graph-level utilities for
manipulating SL-annotated graphs (nodes, edges, roles, parameters).
"""

from dynafx.sl.operators import (
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
from dynafx.sl.validation import (
    ArgumentType,
    ValidationResult,
    ValidationArgument,
    ValidationAttack,
    ValidationResultDetail,
    validate_system_internal,
    get_validation_summary,
)
import warnings as _warnings

def __getattr__(name: str):
    if name == "Argument":
        _warnings.warn(
            "dynafx.sl.Argument is deprecated, use dynafx.sl.ValidationArgument",
            DeprecationWarning, stacklevel=2,
        )
        return ValidationArgument
    if name == "Attack":
        _warnings.warn(
            "dynafx.sl.Attack is deprecated, use dynafx.sl.ValidationAttack",
            DeprecationWarning, stacklevel=2,
        )
        return ValidationAttack
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "ValidationArgument",
    "ValidationAttack",
    "ValidationResultDetail",
    "validate_system_internal",
    "get_validation_summary",
    "bind_parameters",
    "get_parameter_summary",
]
