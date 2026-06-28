"""Causal tracing for system dynamics models.

Provides tools to trace causes and effects through model structure:
- causes_tree: walk upstream dependencies recursively
- causes_strip: decompose a variable's value into contributing factors
- effects_tree: walk downstream to find all affected variables
- causal_trace: combined cause/effect analysis
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from dynafx.system.dsl import SysdModel, StockDef, AuxDef, FlowDef


@dataclass
class CausalNode:
    """A node in a causal tree."""
    name: str
    expr: str = ""
    value: float = 0.0
    polarity: int = 1  # +1 = positive, -1 = negative
    children: list[CausalNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expr": self.expr,
            "value": self.value,
            "polarity": self.polarity,
            "children": [c.to_dict() for c in self.children],
        }

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


@dataclass
class CausalStrip:
    """Decomposition of a variable's value into contributing factors."""
    variable: str
    total_value: float
    factors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "total_value": self.total_value,
            "factors": self.factors,
        }


def _extract_refs(expr: str, known_names: set[str]) -> set[str]:
    """Extract variable references from an expression string."""
    # Find all word tokens that match known variable names
    tokens = set(re.findall(r'\b([A-Za-z_]\w*)\b', expr))
    return tokens & known_names


def _get_dependencies(model: SysdModel) -> dict[str, tuple[str, set[str]]]:
    """Build dependency map: variable -> (expression, referenced_names).

    Structure:
    - Stock → depends on its flows (by name)
    - Flow → depends on expression references
    - Aux → depends on expression references
    """
    known_names: set[str] = set()
    deps: dict[str, tuple[str, set[str]]] = {}

    # Collect all known names first
    for s in model.stocks:
        known_names.add(s.name)
        for f in s.flows:
            known_names.add(f.name)
    for a in model.aux_vars:
        known_names.add(a.name)
    for t in model.tables:
        known_names.add(t.name)

    # Build dependencies for flows (before stocks, so stocks can reference flow names)
    for s in model.stocks:
        for f in s.flows:
            refs = _extract_refs(f.expr, known_names - {f.name})
            deps[f.name] = (f.expr, refs)

    # Build dependencies for stocks (depend on their flow names)
    for s in model.stocks:
        flow_names = {f.name for f in s.flows}
        expr = " + ".join(f"{f.direction}{f.name}" for f in s.flows)
        deps[s.name] = (expr, flow_names)

    # Build dependencies for aux variables
    for a in model.aux_vars:
        refs = _extract_refs(a.expr, known_names - {a.name})
        deps[a.name] = (a.expr, refs)

    return deps


def _get_reverse_deps(deps: dict[str, tuple[str, set[str]]]) -> dict[str, set[str]]:
    """Build reverse dependency map: variable -> set of variables that depend on it."""
    reverse: dict[str, set[str]] = {}
    for name, (_, refs) in deps.items():
        for ref in refs:
            reverse.setdefault(ref, set()).add(name)
    return reverse


def _build_causal_graph(model: SysdModel) -> tuple[dict[str, tuple[str, set[str]]], dict[str, set[str]]]:
    """Build both forward and reverse dependency graphs."""
    deps = _get_dependencies(model)
    reverse_deps = _get_reverse_deps(deps)
    return deps, reverse_deps


def causes_tree(
    model: SysdModel,
    variable: str,
    max_depth: int = 10,
) -> Optional[CausalNode]:
    """Walk upstream dependencies recursively to build a causes tree.

    Args:
        model: The SysdModel to analyze
        variable: The target variable to trace causes for
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        CausalNode tree showing upstream dependencies, or None if variable not found
    """
    deps, _ = _build_causal_graph(model)

    if variable not in deps:
        return None

    visited: set[str] = set()

    def _walk(name: str, depth: int) -> CausalNode:
        if name in visited or depth >= max_depth:
            return CausalNode(name=name, expr="(cycle or max depth)")

        visited.add(name)
        expr, refs = deps.get(name, ("", set()))
        node = CausalNode(name=name, expr=expr)

        for ref in sorted(refs):
            child = _walk(ref, depth + 1)
            node.children.append(child)

        visited.discard(name)
        return node

    return _walk(variable, 0)


def effects_tree(
    model: SysdModel,
    variable: str,
    max_depth: int = 10,
) -> Optional[CausalNode]:
    """Walk downstream to find all affected variables.

    Args:
        model: The SysdModel to analyze
        variable: The source variable to trace effects from
        max_depth: Maximum recursion depth

    Returns:
        CausalNode tree showing downstream effects, or None if variable not found
    """
    deps, reverse_deps = _build_causal_graph(model)

    if variable not in deps and variable not in reverse_deps:
        return None

    visited: set[str] = set()

    def _walk(name: str, depth: int) -> CausalNode:
        if name in visited or depth >= max_depth:
            return CausalNode(name=name, expr="(cycle or max depth)")

        visited.add(name)
        expr, _ = deps.get(name, ("", set()))
        targets = reverse_deps.get(name, set())
        node = CausalNode(name=name, expr=expr)

        for target in sorted(targets):
            child = _walk(target, depth + 1)
            node.children.append(child)

        visited.discard(name)
        return node

    return _walk(variable, 0)


def causes_strip(
    model: SysdModel,
    variable: str,
    state: dict[str, float],
) -> Optional[CausalStrip]:
    """Decompose a variable's value into contributing factors at a given state.

    For each upstream variable, shows its contribution to the target's value.

    Args:
        model: The SysdModel to analyze
        variable: The variable to decompose
        state: Current state dict mapping variable names to values

    Returns:
        CausalStrip with factor contributions, or None if variable not found
    """
    deps, _ = _build_causal_graph(model)

    if variable not in deps:
        return None

    expr, refs = deps[variable]
    total_value = state.get(variable, 0.0)
    strip = CausalStrip(variable=variable, total_value=total_value)

    for ref in sorted(refs):
        ref_value = state.get(ref, 0.0)
        # Estimate contribution: proportion of value from this reference
        # Simple heuristic: if ref_value > 0 and total_value > 0, contribution ~ ref_value/total_value
        # For more accuracy, we'd need to evaluate partial derivatives
        contribution = ref_value if abs(total_value) < 1e-10 else ref_value
        strip.factors.append({
            "name": ref,
            "value": ref_value,
            "contribution": contribution,
            "expr": deps.get(ref, ("", set()))[0],
        })

    return strip


def causal_trace(
    model: SysdModel,
    variable: str,
    state: dict[str, float],
    max_depth: int = 10,
) -> dict[str, Any]:
    """Combined causal analysis: what caused this value, what does it affect.

    Args:
        model: The SysdModel to analyze
        variable: The variable to trace
        state: Current state dict mapping variable names to values
        max_depth: Maximum recursion depth

    Returns:
        Dict with 'causes' tree, 'effects' tree, and 'strip' decomposition
    """
    causes = causes_tree(model, variable, max_depth)
    effects = effects_tree(model, variable, max_depth)
    strip = causes_strip(model, variable, state)

    return {
        "variable": variable,
        "causes": causes.to_dict() if causes else None,
        "effects": effects.to_dict() if effects else None,
        "strip": strip.to_dict() if strip else None,
    }
