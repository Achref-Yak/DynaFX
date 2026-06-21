"""System dynamics DSL — parse .sysd files into simulation-ready models.

Syntax (Vensim-inspired):
    model "Name"
      dt 0.5
      from 0 to 100

      stock "Stock Name": initial_value
        + "Inflow Name": rate_expression
        - "Outflow Name": rate_expression

      table "Table Name"
        x: [0, 10, 20]
        y: [5, 15, 5]

Expressions support: +, -, *, /, parentheses, MIN(a,b), MAX(a,b),
IF(cond,a,b), SMOOTH(x,delay), and references to other stocks/flows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4

from cognitive_engine.core.decomposer import SystemDecomposer
from cognitive_engine.core.models import Parameter
from cognitive_engine.system.equations import rk4_step, euler_step


# ── AST Nodes ───────────────────────────────────────────────────

@dataclass
class FlowDef:
    name: str
    direction: str     # "+" inflow, "-" outflow
    expr: str          # raw source expression


@dataclass
class StockDef:
    name: str
    initial: float
    flows: list[FlowDef] = field(default_factory=list)


@dataclass
class TableDef:
    name: str
    x: list[float]
    y: list[float]


@dataclass
class SysdModel:
    name: str = ""
    dt: float = 1.0
    t_span: tuple[float, float] = (0.0, 100.0)
    stocks: list[StockDef] = field(default_factory=list)
    tables: list[TableDef] = field(default_factory=list)

    def to_decomposer(self) -> SystemDecomposer:
        d = SystemDecomposer(name=self.name)
        for s in self.stocks:
            nid = d.add_node(s.name, type="STOCK")
            if nid and s.initial != 0.0:
                node = d.graph.nodes.get(nid)
                if node:
                    node.metadata["parameter"] = Parameter(value=s.initial)
        for s in self.stocks:
            for f in s.flows:
                pol = 1 if f.direction == "+" else -1
                d.add_node(f.name, type="FLOW")
                d.add_edge(f.name, s.name, "CAUSES", polarity=pol)
        for t in self.tables:
            d.graph.metadata.setdefault("sysd_tables", {})[t.name] = {
                "x": t.x, "y": t.y,
            }
        d.graph.metadata["sysd_model"] = {
            "name": self.name,
            "dt": self.dt,
            "t_span": list(self.t_span),
        }
        return d

    def simulate(
        self,
        method: str = "rk4",
        t_span: Optional[tuple[float, float]] = None,
        dt: Optional[float] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Parse → compile → simulate in one call.

        Returns dict with times, stocks, values, final_state, and metadata.
        """
        t_span = t_span or self.t_span
        step = dt or self.dt
        step_fn = rk4_step if method == "rk4" else euler_step
        if params is None:
            params = {}

        # Build table lookup params
        for t in self.tables:
            params[t.name] = LookupTable(t.x, t.y)
        params["dt"] = step

        # Build ODE system
        f, stock_names, y0, aux_count = _build_system(self, params)

        t0, t_end = t_span
        direction = 1 if t_end >= t0 else -1
        y = list(y0)
        times = [t0]
        y_hist = [list(y)]

        while abs(t0 - t_end) > 1e-12:
            remaining = abs(t_end - t0)
            if remaining < abs(step):
                y = step_fn(f, t0, y, direction * remaining, params)
                t0 = t_end
            else:
                y = step_fn(f, t0, y, direction * step, params)
                t0 += direction * step
            times.append(t0)
            y_hist.append(list(y))

        # Strip aux state from output
        if aux_count:
            pure_stocks = len(stock_names) - aux_count
            y_hist = [row[:pure_stocks] for row in y_hist]
            stock_names = stock_names[:pure_stocks]

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


# ── Lookup table ────────────────────────────────────────────────

class LookupTable:
    """Linear-interpolated lookup table for time-varying parameters."""
    def __init__(self, x: list[float], y: list[float]):
        self.x = x
        self.y = y

    def __call__(self, t: float) -> float:
        if t <= self.x[0]:
            return self.y[0]
        if t >= self.x[-1]:
            return self.y[-1]
        for i in range(len(self.x) - 1):
            if self.x[i] <= t < self.x[i + 1]:
                frac = (t - self.x[i]) / (self.x[i + 1] - self.x[i])
                return self.y[i] + frac * (self.y[i + 1] - self.y[i])
        return self.y[-1]


# ── Expression AST and Parser ───────────────────────────────────

_TOKEN_RE = re.compile(r"""
    \s*(?:((?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)  # number (incl. sci notation)
         |([a-zA-Z_]\w*)           # identifier
         |(>=|<=|!=|==|[><+\-*/(),\[\]])  # operators
         |(\S)                     # unexpected char
    )""", re.VERBOSE)

_OP_CHARS = frozenset(["+", "-", "*", "/", "(", ")", ",", "[", "]", ">", "<", "=", ">=", "<=", "==", "!="])

Token = tuple[str, str]  # (type, value)


def _tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        if not m:
            pos += 1
            continue
        if m.group(1):
            tokens.append(("num", m.group(1)))
        elif m.group(2):
            tokens.append(("id", m.group(2)))
        elif m.group(3):
            ch = m.group(3)
            if ch in _OP_CHARS:
                tokens.append(("op", ch))
        pos = m.end()
    return tokens


class ExprNode: pass

@dataclass
class ExprLiteral(ExprNode):
    value: float

@dataclass
class ExprRef(ExprNode):
    name: str

@dataclass
class ExprBinOp(ExprNode):
    op: str
    left: ExprNode
    right: ExprNode

@dataclass
class ExprFuncCall(ExprNode):
    name: str
    args: list[ExprNode]


class ExprParser:
    def __init__(self, source: str):
        self.tokens = _tokenize(source)
        self.pos = 0

    def peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> Token:
        t = self.peek()
        if t is None:
            raise SyntaxError("Unexpected end of expression")
        if expected and t[1] != expected:
            raise SyntaxError(f"Expected '{expected}', got '{t[1]}'")
        self.pos += 1
        return t

    _COMP_OPS = frozenset([">", "<", ">=", "<=", "==", "!="])

    def parse(self) -> ExprNode:
        return self._comparison()

    def _comparison(self) -> ExprNode:
        left = self._expression()
        while self.peek() and self.peek()[1] in self._COMP_OPS:
            op = self.consume()[1]
            right = self._expression()
            left = ExprBinOp(op, left, right)
        return left

    def _expression(self) -> ExprNode:
        left = self._term()
        while self.peek() and self.peek()[1] in ("+", "-"):
            op = self.consume()[1]
            right = self._term()
            left = ExprBinOp(op, left, right)
        return left

    def _term(self) -> ExprNode:
        left = self._unary()
        while self.peek() and self.peek()[1] in ("*", "/"):
            op = self.consume()[1]
            right = self._unary()
            left = ExprBinOp(op, left, right)
        return left

    def _unary(self) -> ExprNode:
        if self.peek() and self.peek()[1] == "-":
            self.consume()
            return ExprBinOp("*", ExprLiteral(-1.0), self._unary())
        return self._primary()

    def _primary(self) -> ExprNode:
        t = self.peek()
        if t is None:
            raise SyntaxError("Expected expression, got end")
        if t[0] == "num":
            self.consume()
            return ExprLiteral(float(t[1]))
        if t[0] == "id":
            name = self.consume()[1]
            if self.peek() and self.peek()[1] == "(":
                self.consume()  # (
                args = []
                if self.peek() and self.peek()[1] != ")":
                    args.append(self._comparison())
                    while self.peek() and self.peek()[1] == ",":
                        self.consume()
                        args.append(self._comparison())
                self.consume(")")
                return ExprFuncCall(name, args)
            return ExprRef(name)
        if t[1] == "(":
            self.consume()
            node = self._comparison()
            self.consume(")")
            return node
        raise SyntaxError(f"Unexpected token '{t[1]}'")


# ── Expression compiler ─────────────────────────────────────────

def _compile_expr(node: ExprNode, stock_names: set[str]) -> str:
    """Compile expression AST to a Python expression string.

    Returns a string suitable for eval() with a namespace of
    stock/flow values + builtins.
    """
    if isinstance(node, ExprLiteral):
        return repr(node.value)
    if isinstance(node, ExprRef):
        if node.name == "dt":
            return "_p['dt']"
        if node.name == "t":
            return "t"
        return f"_s.get('{node.name}', 0.0)"
    if isinstance(node, ExprBinOp):
        left = _compile_expr(node.left, stock_names)
        right = _compile_expr(node.right, stock_names)
        return f"({left} {node.op} {right})"
    if isinstance(node, ExprFuncCall):
        args = ", ".join(
            _compile_expr(a, stock_names) for a in node.args
        )
        return f"{node.name}({args})"
    raise TypeError(f"Unknown node: {node}")


# ── System builder ──────────────────────────────────────────────

def _build_system(
    model: SysdModel,
    params: dict[str, Any],
) -> tuple[Callable, list[str], list[float], int]:
    """Build ODE system from SysdModel.

    Returns: (f(t, y, params), all_names, y0, aux_count)
    """
    stock_names = [s.name for s in model.stocks]
    name_set = set(stock_names)
    all_names = list(stock_names)
    y0 = [s.initial for s in model.stocks]

    # Collect flow expressions and detect SMOOTH calls
    smooth_vars: dict[str, str] = {}
    smooth_params: list[tuple[str, str, float]] = []  # (aux_name, input_expr, delay_time)

    flow_exprs: list[tuple[str, str]] = []  # (direction, expr_str)
    for s in model.stocks:
        for f in s.flows:
            flow_exprs.append((f.direction, f.expr))

    # First pass: find and replace SMOOTH calls in all flow expressions.
    # Track compiled expressions per stock as (inflow_list, outflow_list).
    stock_inflow: list[list[str]] = [[] for _ in model.stocks]
    stock_outflow: list[list[str]] = [[] for _ in model.stocks]

    for si, s in enumerate(model.stocks):
        for f in s.flows:
            node = ExprParser(f.expr).parse()
            modified = _replace_smooths(node, smooth_vars, smooth_params)
            compiled = _compile_expr(modified, name_set)
            if f.direction == "+":
                stock_inflow[si].append(compiled)
            else:
                stock_outflow[si].append(compiled)

    # Add SMOOTH aux variables
    smooth_names: list[str] = []
    for aux_name, input_expr_str, delay_time in smooth_params:
        input_node = ExprParser(input_expr_str).parse()
        input_compiled = _compile_expr(input_node, name_set)
        smooth_names.append(aux_name)
        all_names.append(aux_name)
        y0.append(0.0)
    smooth_idx = {n: i + len(stock_names) for i, n in enumerate(smooth_names)}

    # Build inflow/outflow strings for each stock (including from SMOOTH outputs)
    # Actually, SMOOTH variables are aux states with their own ODEs.
    # The stock equations reference them by name.

    inflow_strs: list[str] = []
    outflow_strs: list[str] = []
    for i in range(len(model.stocks)):
        inf = " + ".join(stock_inflow[i]) if stock_inflow[i] else "0.0"
        outf = " + ".join(stock_outflow[i]) if stock_outflow[i] else "0.0"
        inflow_strs.append(inf)
        outflow_strs.append(outf)

    aux_strs: list[str] = []
    for aux_name, input_expr_str, delay_time in smooth_params:
        input_node = ExprParser(input_expr_str).parse()
        input_compiled = _compile_expr(input_node, name_set)
        aux_strs.append(
            f"({input_compiled} - _s.get('{aux_name}', 0.0)) / {delay_time}"
        )

    _builtins = {
        "MIN": min, "MAX": max,
        "IF": lambda c, a, b: a if c else b,
    }

    def f(t: float, y: list[float], p: dict) -> list[float]:
        _s = dict(zip(all_names, y))
        _p = {**params, **p}
        _ns: dict = {**_builtins, "_s": _s, "_p": _p, "t": t}
        # Inject lookup tables into eval namespace by name
        for _k, _v in _p.items():
            if hasattr(_v, "__call__"):
                _ns[_k] = _v
        dydt: list[float] = []
        for i in range(len(model.stocks)):
            inflow_val = eval(inflow_strs[i], {"__builtins__": {}}, _ns)
            outflow_val = eval(outflow_strs[i], {"__builtins__": {}}, _ns)
            dydt.append(inflow_val - outflow_val)
        for j, expr in enumerate(aux_strs):
            val = eval(expr, {"__builtins__": {}}, _ns)
            dydt.append(val)
        return dydt

    return f, all_names, y0, len(smooth_names)


def _replace_smooths(
    node: ExprNode,
    smooth_vars: dict[str, str],
    smooth_params: list[tuple[str, str, float]],
) -> ExprNode:
    """Walk expression tree, replace SMOOTH calls with aux variable references."""
    if isinstance(node, ExprBinOp):
        return ExprBinOp(
            node.op,
            _replace_smooths(node.left, smooth_vars, smooth_params),
            _replace_smooths(node.right, smooth_vars, smooth_params),
        )
    if isinstance(node, ExprFuncCall):
        args = [_replace_smooths(a, smooth_vars, smooth_params) for a in node.args]
        if node.name == "SMOOTH" and len(args) >= 2:
            delay_node = args[1]
            if isinstance(delay_node, ExprLiteral):
                delay = delay_node.value
            else:
                delay = 1.0
            aux_name = f"_smooth_{len(smooth_params)}"
            input_expr = _serialize_expr(args[0])
            smooth_params.append((aux_name, input_expr, delay))
            smooth_vars[aux_name] = aux_name
            return ExprRef(aux_name)
        return ExprFuncCall(node.name, args)
    return node


def _serialize_expr(node: ExprNode) -> str:
    if isinstance(node, ExprLiteral):
        return repr(node.value)
    if isinstance(node, ExprRef):
        return node.name
    if isinstance(node, ExprBinOp):
        return f"({_serialize_expr(node.left)} {node.op} {_serialize_expr(node.right)})"
    if isinstance(node, ExprFuncCall):
        args = ", ".join(_serialize_expr(a) for a in node.args)
        return f"{node.name}({args})"
    return "0"


# ── Sysd Lexer / Structure Parser ───────────────────────────────

_COMMENT_RE = re.compile(r"//.*$")
_STRIP_RE = re.compile(r'^["\']|["\']$')

_TokenLine = tuple[int, str, str]  # (indent, keyword, args)


def _lex_sysd(source: str) -> list[_TokenLine]:
    lines: list[_TokenLine] = []
    for line in source.split("\n"):
        raw = _COMMENT_RE.sub("", line).rstrip()
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        content = raw.strip()
        m = re.match(r"(\w[\w.]*|[+\-])\s*(.*)", content)
        if not m:
            continue
        keyword = m.group(1)
        args = m.group(2).strip()
        lines.append((indent, keyword, args))
    return lines


def _build_tree(lines: list[_TokenLine]) -> SysdModel:
    """Convert indent-aware token lines into a SysdModel AST."""
    model = SysdModel()
    stack: list[tuple[int, StockDef | TableDef | None]] = [(-1, None)]

    for indent, keyword, args in lines:
        if keyword == "model":
            model.name = _STRIP_RE.sub("", args)
            continue
        if keyword == "dt":
            model.dt = float(args)
            continue
        if keyword == "from":
            parts = args.split()
            if parts:
                t0 = float(parts[0])
                if len(parts) >= 3 and parts[1] == "to":
                    t1 = float(parts[2])
                    model.t_span = (t0, t1)
                else:
                    model.t_span = (t0, model.t_span[1])
            continue
        if keyword == "stock":
            name, initial = _parse_name_value(args)
            name = name.replace(" ", "_")
            sd = StockDef(name=name, initial=initial)
            model.stocks.append(sd)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, sd))
            continue

        if keyword == "table":
            name = _STRIP_RE.sub("", args)
            td = TableDef(name=name, x=[], y=[])
            model.tables.append(td)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, td))
            continue

        if keyword in ("x", "y"):
            parent = stack[-1][1] if stack else None
            if isinstance(parent, TableDef):
                vals = _parse_list(args.lstrip(":"))
                if keyword == "x":
                    parent.x = vals
                else:
                    parent.y = vals
            continue

        if keyword in ("+", "-"):
            parent = stack[-1][1] if stack else None
            if isinstance(parent, StockDef):
                name, _ = _parse_name_value(args)
                expr = _split_expr(args)
                parent.flows.append(FlowDef(name=name, direction=keyword, expr=expr))
            continue

    return model


def _parse_name_value(args: str) -> tuple[str, float]:
    """Parse 'Warehouse Stock: 100' → ('Warehouse Stock', 100.0)."""
    if ":" in args:
        name, val = args.split(":", 1)
        name = _STRIP_RE.sub("", name.strip())
        try:
            return name, float(val.strip())
        except ValueError:
            return name, 0.0
    return _STRIP_RE.sub("", args.strip()), 0.0


def _split_expr(args: str) -> str:
    """Extract the expression part after 'name: expr'."""
    if ":" in args:
        _, expr = args.split(":", 1)
        return expr.strip()
    return args


def _parse_list(args: str) -> list[float]:
    """Parse '[1, 2, 3]' or '1, 2, 3' into [1.0, 2.0, 3.0]."""
    args = args.strip().strip("[]")
    parts = [p.strip() for p in args.split(",") if p.strip()]
    return [float(p) for p in parts]


# ── Public API ──────────────────────────────────────────────────

def parse_sysd(source: str) -> SysdModel:
    """Parse a .sysd source string into a SysdModel."""
    lines = _lex_sysd(source)
    return _build_tree(lines)


def parse_sysd_file(path: str) -> SysdModel:
    """Load and parse a .sysd file."""
    with open(path, encoding="utf-8") as f:
        return parse_sysd(f.read())
