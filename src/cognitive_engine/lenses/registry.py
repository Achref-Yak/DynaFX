"""Lens registry — register, apply, and list Graph → Graph transformers."""

from __future__ import annotations

from typing import Callable

from cognitive_engine.core.models import Graph

LensFn = Callable[[Graph], Graph]
_registry: dict[str, LensFn] = {}


def register_lens(name: str, fn: LensFn) -> None:
    """Register a lens function under a name."""
    _registry[name] = fn


def apply_lens(graph: Graph, name: str, **params) -> Graph:
    """Apply a named lens to a graph.

    Args:
        graph: The graph to transform.
        name: Registered lens name.
        **params: Optional parameters forwarded to the lens function.

    Returns:
        Transformed graph with lens-specific metadata.

    Raises:
        KeyError: If the lens is not registered.
    """
    fn = _registry.get(name)
    if fn is None:
        raise KeyError(f"Unknown lens: {name!r}. Available: {list(_registry)}")
    return fn(graph, **params)


def list_lenses() -> list[str]:
    """Return all registered lens names."""
    return list(_registry)
