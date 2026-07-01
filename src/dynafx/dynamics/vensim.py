"""Vensim .mdl model importer — converts Vensim to .sysd SysdModel.

Supports:
  - Stock equations: `Stock = INTEG( inflow - outflow, init )`
  - Aux/rate equations: `name = expr`
  - WITH LOOKUP: `WITH LOOKUP(name, pairs)`
  - SMOOTH, DELAY1, DELAY3, DELAYN, DELAY FIXED
  - MIN, MAX, IF
  - TIME → t
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from dynafx.dynamics.dsl import (
    AuxDef,
    StockDef,
    FlowDef,
    TableDef,
    SysdModel,
)


_VENSIM_FUNC_MAP: dict[str, str] = {
    "SMOOTH": "SMOOTH",
    "DELAY1": "SMOOTH",
    "DELAY3": "DELAY3",
    "DELAYN": "DELAYN",
    "DELAY FIXED": "DELAY_FIXED",
    "MIN": "MIN",
    "MAX": "MAX",
    "IF": "IF",
    "ABS": "ABS",
    "EXP": "exp",
    "LN": "log",
    "SQRT": "sqrt",
    "SIN": "sin",
    "COS": "cos",
}

_VENSIM_KEYWORD_PATTERNS: list[tuple[str, str]] = [
    (r"\bINITIAL\s+TIME\b", "INITIAL_TIME"),
    (r"\bFINISH\s+TIME\b", "FINISH_TIME"),
    (r"\bTIME\s+STEP\b", "TIME_STEP"),
]


@dataclass
class _RawEquation:
    name: str
    expr: str
    var_type: str
    init: Optional[float] = None


def _vensim_preprocess(source: str) -> str:
    """Preprocess Vensim source: join continuations, normalize whitespace."""
    lines = []
    for raw_line in source.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        # Join indented continuation lines or lines starting with ( or [
        is_continuation = indent > 0 or stripped.startswith("(") or stripped.startswith("[")
        has_equals = "=" in stripped.split("~")[0]
        if lines and is_continuation and not has_equals:
            lines[-1] += " " + stripped
        else:
            lines.append(stripped)
    result = "\n".join(lines)
    for pattern, replacement in _VENSIM_KEYWORD_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


def _clean_line(line: str) -> str:
    line = line.split("~")[0].strip() if "~" in line else line.strip()
    line = re.sub(r"\s*\|\s*$", "", line)
    return line.strip()


def _parse_vensim_lines(lines: list[str]) -> list[_RawEquation]:
    raw: list[_RawEquation] = []
    for raw_line in lines:
        line = _clean_line(raw_line)
        if not line or line.startswith("{") or line.startswith("\\\\"):
            continue

        m = re.match(r"(\w[\w_]*)\s*=", line)
        if not m:
            continue
        name = m.group(1)

        # Skip reserved Vensim keywords
        if name.upper() in (
            "INITIAL_TIME", "FINISH_TIME", "TIME_STEP",
            "SAVEPER", "SAVEUNIT", "START", "STOP",
        ):
            continue

        rest = line[m.end():].strip()

        # Stock equation
        integ_m = re.match(r"INTEG\s*\(\s*(.+?)\s*,\s*([^)]+)\s*\)", rest, re.IGNORECASE)
        if integ_m:
            flow_expr = integ_m.group(1).strip()
            init_str = integ_m.group(2).strip()
            try:
                init_val = float(init_str)
            except ValueError:
                init_val = 0.0
            raw.append(_RawEquation(name, flow_expr, "stock", init_val))
            continue

        # Check for WITH LOOKUP (table)
        if "WITH LOOKUP" in rest.upper() or "WITH_LOOKUP" in rest:
            raw.append(_RawEquation(name, rest, "table"))
            continue

        raw.append(_RawEquation(name, rest, "aux"))

    return raw


def _split_inflow_outflow(expr: str) -> tuple[list[str], list[str]]:
    """Split expression into inflow (+) and outflow (-) terms at top level.

    Handles parentheses, function calls, and nested expressions correctly.
    E.g. 'IF(cond, A, B) - C + D' → (['IF(cond, A, B)', 'D'], ['C'])
    """
    inflows: list[str] = []
    outflows: list[str] = []
    depth = 0
    current = ""
    sign = "+"

    for ch in expr:
        if ch in "([{":
            depth += 1
            current += ch
        elif ch in ")]}":
            depth -= 1
            current += ch
        elif depth == 0 and ch in "+-":
            term = current.strip()
            if term:
                if sign == "+":
                    inflows.append(term)
                else:
                    outflows.append(term)
            sign = ch
            current = ""
        else:
            current += ch

    term = current.strip()
    if term:
        if sign == "+":
            inflows.append(term)
        else:
            outflows.append(term)

    return inflows, outflows


def _vensim_to_sysd_expr(expr: str) -> str:
    result = expr
    for key, val in _VENSIM_FUNC_MAP.items():
        result = re.sub(rf"\b{key}\s*\(", f"{val}(", result, flags=re.IGNORECASE)
    result = re.sub(r"\bTIME\b", "t", result)
    return result


def _parse_with_lookup(expr: str, name: str) -> tuple[str, list[float], list[float]]:
    m = re.search(r"WITH\s+LOOKUP\s*\(\s*(\w[\w_]*)\s*,", expr, re.IGNORECASE)
    if not m:
        return name, [], []

    tbl_name = name
    x_vals: list[float] = []
    y_vals: list[float] = []

    range_end = expr.find(")]")
    data_part = expr[range_end + 2:] if range_end >= 0 else expr

    pairs_m = re.findall(r"\(([^)]+)\)", data_part)
    for p in pairs_m:
        try:
            nums = [float(x) for x in re.findall(r"[\d.]+(?:e[+-]?\d+)?", p)]
            if len(nums) >= 2:
                x_vals.append(nums[0])
                y_vals.append(nums[1])
        except ValueError:
            pass

    return tbl_name, x_vals, y_vals


def parse_mdl(source: str) -> SysdModel:
    """Parse Vensim .mdl source string into a SysdModel."""
    processed = _vensim_preprocess(source)
    lines = processed.split("\n")

    model_name = "Vensim Import"
    for line in lines:
        m = re.match(r"(.+?)\|", line)
        if m:
            candidate = m.group(1).strip()
            if candidate and not candidate.startswith("{"):
                model_name = candidate
                break

    raw_eqns = _parse_vensim_lines(lines)
    model = SysdModel(name=model_name, dt=1.0, t_span=(0.0, 100.0))

    for eq in raw_eqns:
        if eq.var_type == "stock":
            flow_expr = _vensim_to_sysd_expr(eq.expr)

            # Split into inflow and outflow terms at top level
            inflows, outflows = _split_inflow_outflow(flow_expr)
            stock = StockDef(name=eq.name, initial=eq.init or 0.0)

            if not inflows and not outflows:
                # No flows found — treat as net inflow
                stock.flows.append(FlowDef(
                    name=f"{eq.name}_net",
                    direction="+",
                    expr=flow_expr,
                ))
            else:
                for inf in inflows:
                    if inf and inf != "0":
                        stock.flows.append(FlowDef(
                            name=f"{eq.name}_in_{len(stock.flows)}",
                            direction="+",
                            expr=inf,
                        ))
                for out in outflows:
                    if out and out != "0":
                        stock.flows.append(FlowDef(
                            name=f"{eq.name}_out_{len(stock.flows)}",
                            direction="-",
                            expr=out,
                        ))
            model.stocks.append(stock)

        elif eq.var_type == "table":
            tbl_name, x_vals, y_vals = _parse_with_lookup(eq.expr, eq.name)
            if x_vals and y_vals:
                model.tables.append(TableDef(
                    name=tbl_name,
                    x=x_vals,
                    y=y_vals,
                ))

        elif eq.var_type == "aux":
            expr = _vensim_to_sysd_expr(eq.expr)
            model.aux_vars.append(AuxDef(name=eq.name, expr=expr))

    return model


def parse_mdl_file(path: str) -> SysdModel:
    with open(path, encoding="utf-8", errors="replace") as f:
        return parse_mdl(f.read())
