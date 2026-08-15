"""Numerical integration primitives (RK4 & Euler steps).

These low-level step functions are consumed by the DSL solver
(:mod:`dynafx.dynamics.dsl`). The legacy Graph-based equation
compilation API (``compile_equations``/``simulate_equations``) was
removed; build and simulate models via the DSL instead.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# ── Expression parser ─────────────────────────────────────────

_EXPR_RE = re.compile(r"^(\w[\w\s]*?)\(([^)]+)\)$")


def _parse_expression(expr: str) -> Callable[[float, dict[str, float]], float]:
    """Parse a constant expression string into a callable.

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


# ── Solver steps ──────────────────────────────────────────────


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
