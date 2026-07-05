"""Equation compilation and numerical simulation for system dynamics.

Builds symbolic equations from tagged graph using stock-flow template.
Simulates them forward in time with RK4 or Euler integration.
Loop polarity classification as post-pass (not separate template).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from dynafx.core.models import (
    Edge,
    Graph,
    Node,
    NodeType,
)

logger = logging.getLogger(__name__)


class LoopType:
    """Loop polarity classification."""
    REINFORCING = "reinforcing"
    BALANCING = "balancing"
    GOAL_SEEKING = "goal_seeking"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Equation:
    """Compiled equation for a stock node."""
    stock_id: UUID
    stock_name: str
    equation_type: str  # "stock_flow"
    inflow_ids: list[UUID]
    outflow_ids: list[UUID]
    inflow_expression: str
    outflow_expression: str
    full_expression: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _get_parameter_value(node: Node) -> Optional[float]:
    """Extract numeric value from node metadata.

    Reads a plain float from metadata["parameter"] or metadata["stock_value"].
    """
    param = node.metadata.get('parameter')
    if param is not None:
        if isinstance(param, (int, float)):
            return float(param)
    return None


def _get_node_role(node: Node) -> Optional[str]:
    """Extract role from node metadata."""
    return node.metadata.get('role')


def _find_incoming_edges(graph: Graph, node_id: UUID) -> list[Edge]:
    """Find all edges targeting this node."""
    return [e for e in graph.edges.values() if e.target_id == node_id]


def _find_outgoing_edges(graph: Graph, node_id: UUID) -> list[Edge]:
    """Find all edges originating from this node."""
    return [e for e in graph.edges.values() if e.source_id == node_id]


def _get_edge_polarity(edge: Edge) -> str:
    """Extract polarity from edge.polarity field or metadata fallback."""
    if edge.polarity == 1:
        return '+'
    if edge.polarity == -1:
        return '-'
    return edge.metadata.get('polarity', '+')


def _compile_stock_equation(
    graph: Graph,
    stock_node: Node,
) -> Optional[Equation]:
    """Compile equation for a single stock node.

    Stock-flow template: dx/dt = inflow - outflow

    Inflow: Sources that add to the stock (incoming edges with + polarity,
            or outgoing edges with - polarity pointing to the stock)
    Outflow: Sinks that deplete the stock (incoming edges with - polarity,
             or outgoing edges with + polarity from the stock)
    """
    stock_id = stock_node.id
    stock_name = stock_node.text

    # Find incoming edges (flows into stock)
    incoming = _find_incoming_edges(graph, stock_id)
    # Find outgoing edges (flows out of stock)
    outgoing = _find_outgoing_edges(graph, stock_id)

    # Categorize flows by polarity
    inflow_ids: list[UUID] = []
    outflow_ids: list[UUID] = []
    inflow_parts: list[str] = []
    outflow_parts: list[str] = []

    for edge in incoming:
        polarity = _get_edge_polarity(edge)
        source = graph.nodes.get(edge.source_id)
        if source is None:
            continue

        source_name = source.text
        param_value = _get_parameter_value(source)

        if polarity == '+':
            # Positive polarity incoming = source adds to stock (inflow)
            inflow_ids.append(edge.source_id)
            if param_value is not None:
                inflow_parts.append(f"{source_name}({param_value})")
            else:
                inflow_parts.append(source_name)
        else:
            # Negative polarity incoming = source depletes stock (outflow)
            outflow_ids.append(edge.source_id)
            if param_value is not None:
                outflow_parts.append(f"{source_name}({param_value})")
            else:
                outflow_parts.append(source_name)

    for edge in outgoing:
        polarity = _get_edge_polarity(edge)
        target = graph.nodes.get(edge.target_id)
        if target is None:
            continue

        target_name = target.text
        param_value = _get_parameter_value(target)

        if polarity == '-':
            # Negative polarity outgoing = stock depletes target (outflow)
            outflow_ids.append(edge.target_id)
            if param_value is not None:
                outflow_parts.append(f"{target_name}({param_value})")
            else:
                outflow_parts.append(target_name)
        else:
            # Positive polarity outgoing = stock adds to target (inflow)
            inflow_ids.append(edge.target_id)
            if param_value is not None:
                inflow_parts.append(f"{target_name}({param_value})")
            else:
                inflow_parts.append(target_name)

    # Build expressions
    inflow_expr = " + ".join(inflow_parts) if inflow_parts else "0"
    # Outflows are subtracted individually
    if outflow_parts:
        outflow_expr = " - ".join(outflow_parts)
        full_expr = f"d({stock_name})/dt = {inflow_expr} - {outflow_expr}"
    else:
        outflow_expr = "0"
        full_expr = f"d({stock_name})/dt = {inflow_expr}"

    return Equation(
        stock_id=stock_id,
        stock_name=stock_name,
        equation_type="stock_flow",
        inflow_ids=inflow_ids,
        outflow_ids=outflow_ids,
        inflow_expression=inflow_expr,
        outflow_expression=outflow_expr,
        full_expression=full_expr,
        metadata={
            "stock_value": _get_parameter_value(stock_node),
        },
    )


def _classify_loop_polarity(
    graph: Graph,
    cycle_nodes: list[UUID],
    cycle_edges: list[UUID],
) -> tuple[str, str]:
    """Classify loop polarity based on edge signs.

    Returns:
        (loop_type, gain_sign)
    """
    if not cycle_edges:
        return LoopType.UNKNOWN, "?"

    # Count positive and negative edges
    positive_count = 0
    negative_count = 0

    for edge_id in cycle_edges:
        edge = graph.edges.get(edge_id)
        if edge:
            polarity = _get_edge_polarity(edge)
            if polarity == '+':
                positive_count += 1
            else:
                negative_count += 1

    # Determine loop type
    total_edges = positive_count + negative_count
    if total_edges == 0:
        return LoopType.UNKNOWN, "?"

    # Calculate gain sign (product of all edge signs)
    # Even number of negative edges = positive gain, odd = negative gain
    if negative_count % 2 == 0:
        gain_sign = "+"
        loop_type = LoopType.REINFORCING
    else:
        gain_sign = "-"
        loop_type = LoopType.BALANCING

    return loop_type, gain_sign


def compile_equations(graph: Graph) -> list[Equation]:
    """Compile symbolic equations from tagged graph.

    Stock-flow template: dx/dt = inflow - outflow
    Loop polarity classification as post-pass.

    Returns:
        List of compiled equations
    """
    equations = []

    # Find all stock nodes
    stock_nodes = [
        node for node in graph.nodes.values()
        if node.type != NodeType.ENTITY and _get_node_role(node) == "stock"
    ]

    # Compile equation for each stock
    for stock_node in stock_nodes:
        equation = _compile_stock_equation(graph, stock_node)
        if equation is not None:
            equations.append(equation)
        else:
            # Template mismatch — create high uncertainty equation
            equations.append(Equation(
                stock_id=stock_node.id,
                stock_name=stock_node.text,
                equation_type="stock_flow",
                inflow_ids=[],
                outflow_ids=[],
                inflow_expression="?",
                outflow_expression="?",
                full_expression=f"d({stock_node.text})/dt = ? (template mismatch)",
                metadata={"template_mismatch": True},
            ))
            logger.warning(
                "Template mismatch for stock %s: insufficient flow information",
                stock_node.id,
            )

    return equations


def get_equation_summary(equations: list[Equation]) -> dict[str, Any]:
    """Get summary of compiled equations."""
    return {
        "total_equations": len(equations),
        "equations_with_explicit_values": sum(
            1 for eq in equations
            if eq.metadata.get("stock_value") is not None
        ),
        "equations_with_template_mismatch": sum(
            1 for eq in equations
            if eq.metadata.get("template_mismatch", False)
        ),
    }


# ─────────────────────────────────────────────────────────────
# Numerical integration — RK4 & Euler solvers
# ─────────────────────────────────────────────────────────────

_EXPR_RE = re.compile(r"^(\w[\w\s]*?)\(([^)]+)\)$")


def _parse_expression(expr: str) -> Callable[[float, dict[str, float]], float]:
    """Parse expression string into a callable.

    "production(1200)"  →  callable(t, params) returning 1200
    "150"               →  callable returning 150
    "?" or unknown      →  callable returning 0.0 (with warning)
    """
    expr = expr.strip()
    if not expr:
        logger.warning("Empty expression in _parse_expression, returning 0.0")
        return lambda _t, _params: 0.0
    m = _EXPR_RE.match(expr)
    if m:
        try:
            val = float(m.group(2))
            return lambda _t, _params: val
        except ValueError:
            pass
    try:
        val = float(expr)
        return lambda _t, _params: val
    except ValueError:
        logger.warning("Could not parse expression '%s', returning 0.0", expr)
        return lambda _t, _params: 0.0


def _build_ode_fn(
    equations: list[Equation],
) -> tuple[Callable, list[str]]:
    """Build ODE system function from compiled equations.

    Returns:
        (f(t, y, params) -> dy/dt, stock_names)
    """
    stock_names: list[str] = []
    inflow_fns: list[Callable] = []
    outflow_fns: list[Callable] = []

    for eq in equations:
        stock_names.append(eq.stock_name)
        inflow_fns.append(_parse_expression(eq.inflow_expression))
        outflow_fns.append(_parse_expression(eq.outflow_expression))

    def f(t: float, y: list[float], params: dict[str, float]) -> list[float]:
        return [
            infn(t, params) - outfn(t, params)
            for infn, outfn in zip(inflow_fns, outflow_fns, strict=False)
        ]

    return f, stock_names


def rk4_step(
    f: Callable[[float, list[float], dict[str, float]], list[float]],
    t: float,
    y: list[float],
    dt: float,
    params: dict[str, float] | None = None,
) -> list[float]:
    """Single 4th-order Runge-Kutta step."""
    if params is None:
        params = {}
    k1 = f(t, y, params)
    k2 = f(t + dt / 2, [yi + dti * dt / 2 for yi, dti in zip(y, k1, strict=False)], params)
    k3 = f(t + dt / 2, [yi + dti * dt / 2 for yi, dti in zip(y, k2, strict=False)], params)
    k4 = f(t + dt, [yi + dti * dt for yi, dti in zip(y, k3, strict=False)], params)
    return [
        yi + (k1i + 2 * k2i + 2 * k3i + k4i) * dt / 6
        for yi, k1i, k2i, k3i, k4i in zip(y, k1, k2, k3, k4, strict=False)
    ]


def euler_step(
    f: Callable[[float, list[float], dict[str, float]], list[float]],
    t: float,
    y: list[float],
    dt: float,
    params: dict[str, float] | None = None,
) -> list[float]:
    """Single forward Euler step."""
    if params is None:
        params = {}
    return [yi + dti * dt for yi, dti in zip(y, f(t, y, params), strict=False)]


def simulate_equations(
    equations: list[Equation],
    t_span: tuple[float, float] = (0.0, 100.0),
    dt: float = 1.0,
    method: str = "rk4",
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Numerically integrate the equation system forward in time.

    Args:
        equations: Compiled equations from compile_equations().
        t_span: (start, end) time interval.
        dt: Step size (can be negative for backward integration).
        method: "rk4" (default) or "euler".
        params: Optional parameter overrides passed to each callable.

    Returns:
        Dict with times, stock names, per-stock value histories,
        final state, method, and step count.
    """
    if dt == 0:
        raise ValueError("dt must be non-zero")
    if method not in ("rk4", "euler"):
        raise ValueError(f"Unknown method '{method}', expected 'rk4' or 'euler'")
    step_fn = rk4_step if method == "rk4" else euler_step
    f, stock_names = _build_ode_fn(equations)

    y0: list[float] = []
    for eq in equations:
        val = eq.metadata.get("stock_value")
        y0.append(float(val) if val is not None else 0.0)

    if params is None:
        params = {}

    t = t_span[0]
    t_end = t_span[1]
    direction = 1 if t_end >= t else -1
    times = [t]
    y_hist = [list(y0)]

    while abs(t - t_end) > 1e-12:
        remaining = abs(t_end - t)
        if remaining < abs(dt):
            actual_dt = direction * remaining
            y0 = step_fn(f, t, y0, actual_dt, params)
            t = t_end
        else:
            y0 = step_fn(f, t, y0, direction * dt, params)
            t += direction * dt
        times.append(t)
        y_hist.append(list(y0))

    return {
        "times": times,
        "stocks": stock_names,
        "values": {
            name: [row[i] for row in y_hist]
            for i, name in enumerate(stock_names)
        },
        "final_state": y_hist[-1],
        "method": method,
        "steps": len(times) - 1,
    }
