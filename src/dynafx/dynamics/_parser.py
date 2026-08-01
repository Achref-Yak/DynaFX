"""Expression parser for the system dynamics DSL.

Provides tokenizer, AST nodes, recursive descent parser, expression compiler,
smooth/delay replacement, and expression serialization.

This is an internal module. Import via ``dynafx.dynamics.dsl``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# ── Expression tokenizer ─────────────────────────────────────────

_TOKEN_RE = re.compile(r"""
    \s*(?:((?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)  # number (incl. sci notation)
         |([a-zA-Z_]\w*)           # identifier
         |(\*\*|>=|<=|!=|==|[><+\-*/(),\[\]])  # operators (multi-char first)
         |(\S)                     # unexpected char
    )""", re.VERBOSE)

_OP_CHARS = frozenset(["+", "-", "*", "/", "**", "(", ")", ",", "[", "]", ">", "<", "=", ">=", "<=", "==", "!="])

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
        elif m.group(4):
            raise SyntaxError(f"Unexpected character '{m.group(4)}' at position {pos}")
        pos = m.end()
    return tokens


# ── Expression AST ───────────────────────────────────────────────

class ExprNode:
    pass


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


# ── Recursive descent parser ─────────────────────────────────────

class ExprParser:
    def __init__(self, source: str):
        self.tokens = _tokenize(source)
        self.pos = 0

    def peek(self) -> Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> Token:
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
        return self._power()

    def _power(self) -> ExprNode:
        left = self._primary()
        while self.peek() and self.peek()[1] == "**":
            self.consume()
            right = self._power()
            left = ExprBinOp("**", left, right)
        return left

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
                self.consume()
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

_COMPILED_CONSTANTS: dict[str, str] = {
    "PI": repr(math.pi),
}


def _compile_expr(node: ExprNode, stock_names: set[str], aux_names: set[str] = frozenset()) -> str:
    if isinstance(node, ExprLiteral):
        return repr(node.value)
    if isinstance(node, ExprRef):
        if node.name == "dt":
            return "_p['dt']"
        if node.name == "t":
            return "t"
        if node.name in _COMPILED_CONSTANTS:
            return _COMPILED_CONSTANTS[node.name]
        if node.name in aux_names:
            return f"_a['{node.name}']"
        if node.name in stock_names:
            return f"_s['{node.name}']"
        # Unknown name — may be a numeric param, lookup table, or ABM metric
        # Use safe fallback; compile-time warnings for truly undefined names
        return f"_s.get('{node.name}', 0.0)"
    if isinstance(node, ExprBinOp):
        left = _compile_expr(node.left, stock_names, aux_names)
        right = _compile_expr(node.right, stock_names, aux_names)
        return f"({left} {node.op} {right})"
    if isinstance(node, ExprFuncCall):
        if node.name == "IF":
            cond = _compile_expr(node.args[0], stock_names, aux_names)
            true_val = _compile_expr(node.args[1], stock_names, aux_names)
            false_val = _compile_expr(node.args[2], stock_names, aux_names)
            return f"({true_val}) if ({cond}) else ({false_val})"
        args = ", ".join(
            _compile_expr(a, stock_names, aux_names) for a in node.args
        )
        return f"{node.name}({args})"
    raise TypeError(f"Unknown node: {node}")


# ── Reference finder / topological sort ─────────────────────────

def _find_refs(node: ExprNode, names: set[str]) -> set[str]:
    """Collect all ExprRef names from an AST that are in `names`."""
    if isinstance(node, ExprRef):
        return {node.name} if node.name in names else set()
    if isinstance(node, ExprBinOp):
        return _find_refs(node.left, names) | _find_refs(node.right, names)
    if isinstance(node, ExprFuncCall):
        result: set[str] = set()
        for a in node.args:
            result |= _find_refs(a, names)
        return result
    return set()


def _topo_sort(names: list[str], expr_nodes: list[ExprNode], all_names: set[str]) -> list[int]:
    """Topological sort of indices based on cross-references in expression trees.

    Returns a list of indices in evaluation order. Falls back to definition
    order if a cycle is detected.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    adj: list[set[int]] = []
    for node in expr_nodes:
        refs = _find_refs(node, set(names))
        adj.append({name_to_idx[r] for r in refs if r in name_to_idx})

    in_degree = [len(deps) for deps in adj]
    queue = [i for i, d in enumerate(in_degree) if d == 0]
    rev_adj: list[set[int]] = [set() for _ in names]
    for j, deps in enumerate(adj):
        for i in deps:
            rev_adj[i].add(j)

    order: list[int] = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for j in rev_adj[i]:
            in_degree[j] -= 1
            if in_degree[j] == 0:
                queue.append(j)

    if len(order) == len(names):
        return order
    return list(range(len(names)))


# ── Smooth/delay replacement and expression serialization ────────

def _replace_smooths(
    node: ExprNode,
    smooth_params: list[tuple[str, str, str, float, float]],
) -> ExprNode:
    """Walk expression tree, replace SMOOTH/SMOOTHI/DELAY3/DELAYN/DELAY_FIXED calls with aux variable references."""
    if isinstance(node, ExprBinOp):
        return ExprBinOp(
            node.op,
            _replace_smooths(node.left, smooth_params),
            _replace_smooths(node.right, smooth_params),
        )
    if isinstance(node, ExprFuncCall):
        args = [_replace_smooths(a, smooth_params) for a in node.args]
        if node.name == "SMOOTH" and len(args) >= 2:
            delay_node = args[1]
            delay = delay_node.value if isinstance(delay_node, ExprLiteral) else delay_node
            aux_name = f"_smooth_{len(smooth_params)}"
            input_expr = _serialize_expr(args[0])
            smooth_params.append(("smooth", aux_name, input_expr, delay, 0.0))
            return ExprRef(aux_name)
        if node.name == "SMOOTHI" and len(args) >= 3:
            delay_node = args[1]
            delay = delay_node.value if isinstance(delay_node, ExprLiteral) else delay_node
            init_node = args[2]
            init_val = init_node.value if isinstance(init_node, ExprLiteral) else init_node
            aux_name = f"_smooth_{len(smooth_params)}"
            input_expr = _serialize_expr(args[0])
            smooth_params.append(("smooth", aux_name, input_expr, delay, init_val))
            return ExprRef(aux_name)
        if node.name == "DELAY3" and len(args) >= 2:
            delay_node = args[1]
            if isinstance(delay_node, ExprLiteral):
                total_delay = delay_node.value
                stage_delay = total_delay / 3.0
            else:
                stage_delay = ExprBinOp("/", delay_node, ExprLiteral(3.0))
            current_input = _serialize_expr(args[0])
            for _ in range(3):
                aux_name = f"_delay3_{len(smooth_params)}"
                smooth_params.append(("smooth", aux_name, current_input, stage_delay, 0.0))
                current_input = aux_name
            return ExprRef(current_input)
        if node.name == "DELAYN" and len(args) >= 3:
            delay_node = args[1]
            total_delay = delay_node.value if isinstance(delay_node, ExprLiteral) else delay_node
            n_node = args[2]
            n_stages = max(1, int(n_node.value)) if isinstance(n_node, ExprLiteral) else 3
            if isinstance(total_delay, (int, float)):
                stage_delay = total_delay / n_stages
            else:
                stage_delay = ExprBinOp("/", total_delay, ExprLiteral(n_stages))
            current_input = _serialize_expr(args[0])
            for _ in range(n_stages):
                aux_name = f"_delayn_{len(smooth_params)}"
                smooth_params.append(("smooth", aux_name, current_input, stage_delay, 0.0))
                current_input = aux_name
            return ExprRef(current_input)
        if node.name == "DELAY_FIXED" and len(args) >= 2:
            delay_node = args[1]
            delay = delay_node.value if isinstance(delay_node, ExprLiteral) else delay_node
            aux_name = f"_delay_fixed_{len(smooth_params)}"
            input_expr = _serialize_expr(args[0])
            smooth_params.append(("delay_fixed", aux_name, input_expr, delay, 0.0))
            return ExprRef(aux_name)
        if node.name == "CONVEY" and len(args) >= 2:
            delay_node = args[1]
            delay = delay_node.value if isinstance(delay_node, ExprLiteral) else delay_node
            aux_name = f"_convey_{len(smooth_params)}"
            input_expr = _serialize_expr(args[0])
            smooth_params.append(("convey", aux_name, input_expr, delay, 0.0))
            return ExprRef(aux_name)
        if node.name == "CONVEY_BATCH" and len(args) >= 3:
            delay_node = args[1]
            delay = delay_node.value if isinstance(delay_node, ExprLiteral) else delay_node
            batch_node = args[2]
            batch_size = batch_node.value if isinstance(batch_node, ExprLiteral) else batch_node
            acc_name = f"_cbatch_acc_{len(smooth_params)}"
            out_name = f"_cbatch_out_{len(smooth_params)}"
            input_expr = _serialize_expr(args[0])
            smooth_params.append(("convey_batch", acc_name, input_expr, delay, batch_size))
            smooth_params.append(("convey_batch_out", out_name, "0.0", 0.0, 0.0))
            return ExprRef(out_name)
        return ExprFuncCall(node.name, args)
    return node


def _serialize_expr(node: Any) -> str:
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


# ── User-defined function expansion (macros) ─────────────────────

def _expand_func_calls(node: ExprNode, func_map: dict[str, tuple[list[str], ExprNode]]) -> ExprNode:
    """Replace user-defined function calls with inlined body AST.

    ``func_map`` maps function name → (param_names, body_ast).
    Body AST is cloned and parameters are substituted with argument ASTs.
    Nested function calls are expanded recursively.
    """
    if isinstance(node, ExprFuncCall):
        if node.name in func_map:
            params, body = func_map[node.name]
            args = [_expand_func_calls(a, func_map) for a in node.args]
            if len(args) != len(params):
                raise SyntaxError(
                    f"Function '{node.name}' expects {len(params)} "
                    f"arguments, got {len(args)}"
                )
            sub_map = dict(zip(params, args, strict=False))
            result = _substitute_refs(body, sub_map)
            # Re-expand for nested user function calls in the body
            return _expand_func_calls(result, func_map)
        return ExprFuncCall(node.name, [_expand_func_calls(a, func_map) for a in node.args])
    if isinstance(node, ExprBinOp):
        return ExprBinOp(node.op, _expand_func_calls(node.left, func_map), _expand_func_calls(node.right, func_map))
    return node


def _substitute_refs(node: ExprNode, sub_map: dict[str, ExprNode]) -> ExprNode:
    """Clone an AST, replacing ExprRef nodes matching ``sub_map`` keys."""
    if isinstance(node, ExprRef):
        if node.name in sub_map:
            return _clone_node(sub_map[node.name])
        return ExprRef(node.name)
    if isinstance(node, ExprLiteral):
        return ExprLiteral(node.value)
    if isinstance(node, ExprBinOp):
        return ExprBinOp(node.op, _substitute_refs(node.left, sub_map), _substitute_refs(node.right, sub_map))
    if isinstance(node, ExprFuncCall):
        return ExprFuncCall(node.name, [_substitute_refs(a, sub_map) for a in node.args])
    return node


def _clone_node(node: ExprNode) -> ExprNode:
    """Deep clone an expression AST."""
    if isinstance(node, ExprLiteral):
        return ExprLiteral(node.value)
    if isinstance(node, ExprRef):
        return ExprRef(node.name)
    if isinstance(node, ExprBinOp):
        return ExprBinOp(node.op, _clone_node(node.left), _clone_node(node.right))
    if isinstance(node, ExprFuncCall):
        return ExprFuncCall(node.name, [_clone_node(a) for a in node.args])
    return node
