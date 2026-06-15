from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from cognitive_engine.core.state import State


@runtime_checkable
class Operator(Protocol):
    """Uniform interface for all operators.

    Every operator:
      - Takes a State + optional kwargs
      - Returns a new State

    This is the industry standard pattern (Option C):
      - External: uniform (composable)
      - Internal: typed (safe)

    Example:
        class ExtractOperator:
            name = "extract"

            def __call__(self, state: State, text: str = None, **kwargs) -> State:
                graph = extract_from_text(text)
                return State(graph=graph, ...)
    """
    name: str

    def __call__(self, state: State, **kwargs) -> State: ...
