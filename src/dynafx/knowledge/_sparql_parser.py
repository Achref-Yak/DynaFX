"""SPARQL 1.1 query parser.

Extracted from the original sparql.py evaluator module.
Contains parser, AST types, and filter expression types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
)

# ── Operator / Function (used by new evaluator _eval_expr) ───────


@dataclass(frozen=True)
class Operator:
    op: str
    args: list[Any]


@dataclass(frozen=True)
class Function:
    name: str
    args: list[Any]


# ── Variable representation ──────────────────────────────────────


@dataclass(frozen=True)
class Variable:
    """A SPARQL variable (?x or $x)."""
    name: str

    def __repr__(self) -> str:
        return f"?{self.name}"


# ── SPARQL triple pattern (supports variables) ───────────────────


@dataclass(frozen=True)
class SPARQLTriplePattern:
    """Triple pattern where positions can be RDFNode, Variable, or None."""
    subject: Any | None = None
    predicate: Any | None = None
    object_: Any | None = None


# ── Tokenizer ────────────────────────────────────────────────────

_TOKEN_SPEC = [
    ("SELECT", r'(?i:select)'),
    ("ASK", r'(?i:ask)'),
    ("CONSTRUCT", r'(?i:construct)'),
    ("WHERE", r'(?i:where)'),
    ("DISTINCT", r'(?i:distinct)'),
    ("ORDER", r'(?i:order)'),
    ("BY", r'(?i:by)'),
    ("ASC", r'(?i:asc)'),
    ("DESC", r'(?i:desc)'),
    ("LIMIT", r'(?i:limit)'),
    ("OFFSET", r'(?i:offset)'),
    ("FILTER", r'(?i:filter)'),
    ("OPTIONAL", r'(?i:optional)'),
    ("UNION", r'(?i:union)'),
    ("REGEX", r'(?i:regex)'),
    ("BOUND", r'(?i:bound)'),
    ("PREFIX", r'(?i:prefix)'),
    ("BASE", r'(?i:base)'),
    ("AND", r'(?i:&&)'),
    ("OR", r'(?i:\|\|)'),
    ("NOT", r'(?i:not)'),
    ("TRUE", r'(?i:true)'),
    ("FALSE", r'(?i:false)'),
    ("IN", r'(?i:in)'),
    ("AS", r'(?i:as)'),
    ("A", r'(?i:a)'),
    ("VAR", r'[?$][a-zA-Z_][\w]*'),
    ("IRI", r'<[^>]*>'),
    ("PNAME_LN", r'(?:[a-zA-Z_][\w.-]*)?:[a-zA-Z_][\w.-]*'),
    ("PNAME_NS", r'(?:[a-zA-Z_][\w.-]*)?:'),
    ("STRING_LITERAL", r'"[^"\\]*(?:\\.[^"\\]*)*"'),
    ("DECIMAL", r'[+-]?\d+\.\d+'),
    ("INTEGER", r'[+-]?\d+'),
    ("LANGTAG", r'@[a-zA-Z]+(?:-[a-zA-Z0-9]+)*'),
    ("HAT_HAT", r'\^\^'),
    ("DOT", r'\.'),
    ("LBRACE", r'\{'),
    ("RBRACE", r'\}'),
    ("LPAREN", r'\('),
    ("RPAREN", r'\)'),
    ("SEMI", r';'),
    ("COMMA", r','),
    ("STAR", r'\*'),
    ("EQ", r'='),
    ("NE", r'!='),
    ("LE", r'<='),
    ("GE", r'>='),
    ("LT", r'<'),
    ("GT", r'>'),
    ("BANG", r'!'),
    ("PLUS", r'\+'),
    ("MINUS", r'-'),
    ("SLASH", r'/'),
    ("COMMENT", r'#[^\n]*'),
    ("WS", r'[ \t\r\n]+'),
    ("MISMATCH", r'.'),
]

_TOKEN_RE = re.compile(
    '|'.join(f'(?P<{name}>{pattern})' for name, pattern in _TOKEN_SPEC),
    re.IGNORECASE,
)


def tokenize(query: str) -> list[tuple[str, str, int]]:
    tokens: list[tuple[str, str, int]] = []
    for match in _TOKEN_RE.finditer(query):
        kind = match.lastgroup
        value = match.group()
        pos = match.start()
        if kind == "WS" or kind == "COMMENT":
            continue
        if kind == "STRING_LITERAL":
            value = value[1:-1]
            value = value.replace('\\"', '"').replace('\\\\', '\\')
            kind = "STRING"
        if kind == "IRI":
            value = value[1:-1]
        if kind == "MISMATCH":
            raise SyntaxError(f"Unexpected character {value!r} at position {pos}")
        tokens.append((kind, value, pos))
    tokens.append(("EOF", "", len(query)))
    return tokens


# ── SPARQL Algebra Nodes ─────────────────────────────────────────


class AlgebraNode:
    """Base class for SPARQL algebra nodes."""


@dataclass
class BGP(AlgebraNode):
    patterns: list[SPARQLTriplePattern]


@dataclass
class Filter(AlgebraNode):
    expr: Any
    inner: AlgebraNode


@dataclass
class Optional_(AlgebraNode):
    left: AlgebraNode
    right: AlgebraNode
    expr: Any | None = None


@dataclass
class Union(AlgebraNode):
    left: AlgebraNode
    right: AlgebraNode


@dataclass
class Project(AlgebraNode):
    vars: list[str]
    inner: AlgebraNode
    distinct: bool = False


@dataclass
class Ask(AlgebraNode):
    inner: AlgebraNode


@dataclass
class Construct(AlgebraNode):
    templates: list[SPARQLTriplePattern]
    inner: AlgebraNode


@dataclass
class OrderBy(AlgebraNode):
    conditions: list[tuple[str, str]]
    inner: AlgebraNode


@dataclass
class Slice(AlgebraNode):
    limit: int | None = None
    offset: int | None = None
    inner: AlgebraNode | None = None


# ── Filter Expression Nodes ──────────────────────────────────────


class FilterExpr:
    """Base class for filter expressions."""


@dataclass
class Comparison(FilterExpr):
    op: str
    left: FilterExpr
    right: FilterExpr


@dataclass
class And(FilterExpr):
    left: FilterExpr
    right: FilterExpr


@dataclass
class Or(FilterExpr):
    left: FilterExpr
    right: FilterExpr


@dataclass
class Not(FilterExpr):
    inner: FilterExpr


@dataclass
class VarRef(FilterExpr):
    name: str


@dataclass
class Constant(FilterExpr):
    value: RDFNode


@dataclass
class RegexFunc(FilterExpr):
    text: FilterExpr
    pattern: FilterExpr
    flags: FilterExpr | None = None


@dataclass
class BoundFunc(FilterExpr):
    name: str


# ── Parser ───────────────────────────────────────────────────────


class SPARQLParser:
    """Recursive descent SPARQL parser."""

    def __init__(self, query: str):
        self.tokens = tokenize(query)
        self.pos = 0
        self.prefixes: dict[str, str] = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }

    def parse(self) -> AlgebraNode:
        """Parse a full SPARQL query."""
        self._parse_prologue()
        tok = self.peek()
        if tok[0] == "SELECT":
            return self._parse_select()
        if tok[0] == "ASK":
            return self._parse_ask()
        if tok[0] == "CONSTRUCT":
            return self._parse_construct()
        raise SyntaxError(f"Expected query type (SELECT/ASK/CONSTRUCT) at position {tok[2]}")

    def peek(self) -> tuple[str, str, int]:
        return self.tokens[self.pos]

    def consume(self, expected_kind: str | None = None) -> tuple[str, str, int]:
        tok = self.tokens[self.pos]
        if expected_kind and tok[0] != expected_kind:
            raise SyntaxError(
                f"Expected {expected_kind}, got {tok[0]} ({tok[1]!r}) at position {tok[2]}"
            )
        self.pos += 1
        return tok

    def skip(self, kind: str) -> bool:
        if self.peek()[0] == kind:
            self.consume(kind)
            return True
        return False

    def _check(self, kind: str) -> bool:
        return self.peek()[0] == kind

    def _parse_prologue(self) -> None:
        while self._check("PREFIX") or self._check("BASE"):
            if self._check("PREFIX"):
                self.consume("PREFIX")
                ns = self._consume_pname_ns().rstrip(":")
                iri = self._consume_iri_ref()
                self.prefixes[ns] = iri
            elif self._check("BASE"):
                self.consume("BASE")
                self._consume_iri_ref()

    def _consume_pname_ns(self) -> str:
        tok = self.peek()
        if tok[0] in ("PNAME_NS", "PNAME_LN"):
            return self.consume()[1]
        raise SyntaxError(f"Expected prefix name at position {tok[2]}")

    def _consume_iri_ref(self) -> str:
        tok = self.consume("IRI")
        return tok[1]

    def _parse_select(self) -> AlgebraNode:
        self.consume("SELECT")
        distinct = self.skip("DISTINCT")
        vars_: list[str] = []
        if self._check("STAR"):
            self.consume("STAR")
        else:
            while self._check("VAR"):
                var_tok = self.consume("VAR")
                vars_.append(var_tok[1][1:])
        if self._check("WHERE"):
            self.consume("WHERE")
        inner = self._parse_group_graph_pattern()
        if not vars_:
            vars_ = _collect_vars(inner)
        order_conditions: list[tuple[str, str]] = []
        if self._check("ORDER"):
            self._parse_order_by(order_conditions)
        limit: int | None = None
        offset: int | None = None
        if self._check("LIMIT"):
            self.consume("LIMIT")
            limit = int(self.consume("INTEGER")[1])
        if self._check("OFFSET"):
            self.consume("OFFSET")
            offset = int(self.consume("INTEGER")[1])
        result: AlgebraNode = inner
        if order_conditions:
            result = OrderBy(order_conditions, result)
        if limit is not None or offset is not None:
            result = Slice(limit=limit, offset=offset, inner=result)
        return Project(vars_, result, distinct=distinct)

    def _parse_order_by(self, conditions: list[tuple[str, str]]) -> None:
        self.consume("ORDER")
        self.consume("BY")
        while not self._check("LIMIT") and not self._check("OFFSET") and not self._check("EOF"):
            if self._check("ASC"):
                self.consume("ASC")
                self.consume("LPAREN")
                var_tok = self.consume("VAR")
                conditions.append((var_tok[1][1:], "ASC"))
                self.consume("RPAREN")
            elif self._check("DESC"):
                self.consume("DESC")
                self.consume("LPAREN")
                var_tok = self.consume("VAR")
                conditions.append((var_tok[1][1:], "DESC"))
                self.consume("RPAREN")
            elif self._check("VAR"):
                var_tok = self.consume("VAR")
                conditions.append((var_tok[1][1:], "ASC"))

    def _parse_ask(self) -> AlgebraNode:
        self.consume("ASK")
        if self._check("WHERE"):
            self.consume("WHERE")
        inner = self._parse_group_graph_pattern()
        return Ask(inner)

    def _parse_construct(self) -> AlgebraNode:
        self.consume("CONSTRUCT")
        templates = self._parse_construct_templates()
        if self._check("WHERE"):
            self.consume("WHERE")
        inner = self._parse_group_graph_pattern()
        return Construct(templates, inner)

    def _parse_construct_templates(self) -> list[SPARQLTriplePattern]:
        self.consume("LBRACE")
        patterns: list[SPARQLTriplePattern] = []
        while not self._check("RBRACE") and not self._check("EOF"):
            pattern = self._parse_triple_pattern()
            if pattern is not None:
                patterns.append(pattern)
            self.skip("DOT")
        self.consume("RBRACE")
        return patterns

    def _parse_group_graph_pattern(self) -> AlgebraNode:
        self.consume("LBRACE")
        result = self._parse_group_graph_pattern_sub()
        self.consume("RBRACE")
        return result

    def _parse_group_graph_pattern_sub(self) -> AlgebraNode:
        result: AlgebraNode = BGP([])
        while not self._check("RBRACE") and not self._check("EOF"):
            if self._check("FILTER"):
                expr = self._parse_filter_constraint()
                result = Filter(expr, result)
                self.skip("DOT")
            elif self._check("OPTIONAL"):
                self.consume("OPTIONAL")
                right = self._parse_group_graph_pattern()
                result = Optional_(result, right)
                self.skip("DOT")
            elif self._check("UNION"):
                left = result
                self.consume("UNION")
                right = self._parse_group_graph_pattern()
                result = Union(left, right)
            elif self._check("LBRACE"):
                subgroup = self._parse_group_graph_pattern()
                result = _join_bgp(result, subgroup)
                self.skip("DOT")
            else:
                patterns = self._parse_triples_block()
                if patterns:
                    result = _join_bgp(result, BGP(patterns))
                self.skip("DOT")
        return result

    def _parse_triples_block(self) -> list[SPARQLTriplePattern]:
        patterns: list[SPARQLTriplePattern] = []
        while not self._check("RBRACE") and not self._check("FILTER") and not self._check("OPTIONAL") and not self._check("UNION") and not self._check("EOF"):
            pattern = self._parse_triple_pattern()
            if pattern is not None:
                patterns.append(pattern)
            if not self._check("DOT") and not self._check("RBRACE") and not self._check("FILTER") and not self._check("OPTIONAL") and not self._check("UNION") and not self._check("EOF"):
                break
            if self._check("DOT"):
                self.consume("DOT")
        return patterns

    def _parse_triple_pattern(self) -> SPARQLTriplePattern | None:
        if self._check("RBRACE") or self._check("FILTER") or self._check("OPTIONAL") or self._check("UNION") or self._check("EOF"):
            return None
        subject = self._parse_subject_or_var()
        predicate = self._parse_predicate_or_var()
        object_ = self._parse_object_or_var()
        return SPARQLTriplePattern(subject=subject, predicate=predicate, object_=object_)

    def _parse_subject_or_var(self):
        tok = self.peek()
        if tok[0] == "IRI":
            return NamedNode(self.consume()[1])
        if tok[0] == "PNAME_LN":
            return self._resolve_pname_ln()
        if tok[0] == "BLANK_NODE":
            return BlankNode(id=self.consume()[1])
        if tok[0] == "VAR":
            return Variable(self.consume("VAR")[1][1:])
        if tok[0] == "LPAREN":
            self.consume("LPAREN")
            while not self._check("RPAREN") and not self._check("EOF"):
                self._parse_object_or_var()
            self.consume("RPAREN")
            return None
        if tok[0] == "A":
            tok = self.consume("A")
            return None
        raise SyntaxError(f"Expected subject/variable at position {tok[2]}, got {tok[0]}")

    def _parse_predicate_or_var(self):
        if self._check("A"):
            self.consume("A")
            return Variable("a")
        tok = self.peek()
        if tok[0] == "IRI":
            return NamedNode(self.consume()[1])
        if tok[0] == "PNAME_LN":
            return self._resolve_pname_ln()
        if tok[0] == "VAR":
            return Variable(self.consume("VAR")[1][1:])
        raise SyntaxError(f"Expected predicate at position {tok[2]}, got {tok[0]}")

    def _parse_object_or_var(self):
        tok = self.peek()
        if tok[0] == "IRI":
            return NamedNode(self.consume()[1])
        if tok[0] == "PNAME_LN":
            return self._resolve_pname_ln()
        if tok[0] == "STRING":
            return self._parse_literal()
        if tok[0] == "INTEGER":
            return Literal(int(self.consume()[1]))
        if tok[0] == "DECIMAL":
            return Literal(float(self.consume()[1]))
        if tok[0] == "VAR":
            return Variable(self.consume("VAR")[1][1:])
        if tok[0] == "TRUE":
            self.consume()
            return Literal(True)
        if tok[0] == "FALSE":
            self.consume()
            return Literal(False)
        if tok[0] == "A":
            self.consume("A")
            return NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        raise SyntaxError(f"Expected object at position {tok[2]}, got {tok[0]}")

    def _parse_literal(self) -> Literal:
        value = self.consume("STRING")[1]
        if self.peek()[0] == "LANGTAG":
            lang = self.consume("LANGTAG")[1][1:]
            return Literal(value, lang_tag=lang)
        if self.peek()[0] == "HAT_HAT":
            self.consume("HAT_HAT")
            dtype = self._consume_datatype()
            return Literal(_coerce_literal_value(value, dtype), datatype=dtype)
        return Literal(value)

    def _consume_datatype(self) -> str:
        tok = self.peek()
        if tok[0] == "IRI":
            return self.consume()[1]
        if tok[0] == "PNAME_LN":
            return self._resolve_pname_ln().iri
        raise SyntaxError(f"Expected datatype IRI at position {tok[2]}")

    def _parse_filter_constraint(self) -> FilterExpr:
        self.consume("FILTER")
        if self._check("LPAREN"):
            self.consume("LPAREN")
            expr = self._parse_filter_expr()
            self.consume("RPAREN")
            return expr
        return self._parse_filter_expr()

    def _parse_filter_expr(self) -> FilterExpr:
        return self._parse_or_expr()

    def _parse_or_expr(self) -> FilterExpr:
        left = self._parse_and_expr()
        while self._check("OR"):
            self.consume("OR")
            right = self._parse_and_expr()
            left = Or(left, right)
        return left

    def _parse_and_expr(self) -> FilterExpr:
        left = self._parse_relational_expr()
        while self._check("AND"):
            self.consume("AND")
            right = self._parse_relational_expr()
            left = And(left, right)
        return left

    def _parse_relational_expr(self) -> FilterExpr:
        left = self._parse_unary_expr()
        if self._check("EQ") or self._check("NE") or self._check("LT") or self._check("GT") or self._check("LE") or self._check("GE"):
            op_map = {"EQ": "=", "NE": "!=", "LT": "<", "GT": ">", "LE": "<=", "GE": ">="}
            op_tok = self.consume()
            op = op_map.get(op_tok[0], op_tok[0])
            right = self._parse_unary_expr()
            return Comparison(op, left, right)
        return left

    def _parse_unary_expr(self) -> FilterExpr:
        if self._check("BANG"):
            self.consume("BANG")
            inner = self._parse_primary_expr()
            return Not(inner)
        return self._parse_primary_expr()

    def _parse_primary_expr(self) -> FilterExpr:
        tok = self.peek()
        if tok[0] == "LPAREN":
            self.consume("LPAREN")
            expr = self._parse_or_expr()
            self.consume("RPAREN")
            return expr
        if tok[0] in ("BOUND",):
            return self._parse_bound_func()
        if tok[0] in ("REGEX",):
            return self._parse_regex_func()
        if tok[0] == "VAR":
            return VarRef(self.consume("VAR")[1][1:])
        if tok[0] == "STRING":
            lit = self._parse_literal()
            return Constant(lit)
        if tok[0] == "INTEGER":
            return Constant(Literal(int(self.consume()[1])))
        if tok[0] == "DECIMAL":
            return Constant(Literal(float(self.consume()[1])))
        if tok[0] == "TRUE":
            self.consume()
            return Constant(Literal(True))
        if tok[0] == "FALSE":
            self.consume()
            return Constant(Literal(False))
        if tok[0] == "IRI":
            return Constant(NamedNode(self.consume()[1]))
        if tok[0] == "PNAME_LN":
            return Constant(self._resolve_pname_ln())
        raise SyntaxError(f"Expected filter expression at position {tok[2]}, got {tok[0]}")

    def _parse_bound_func(self) -> FilterExpr:
        self.consume("BOUND")
        self.consume("LPAREN")
        var_tok = self.consume("VAR")
        name = var_tok[1][1:]
        self.consume("RPAREN")
        return BoundFunc(name)

    def _parse_regex_func(self) -> RegexFunc:
        self.consume("REGEX")
        self.consume("LPAREN")
        text = self._parse_or_expr()
        self.consume("COMMA")
        pattern = self._parse_or_expr()
        flags: FilterExpr | None = None
        if self._check("COMMA"):
            self.consume("COMMA")
            flags = self._parse_or_expr()
        self.consume("RPAREN")
        return RegexFunc(text, pattern, flags)

    def _resolve_pname_ln(self) -> NamedNode:
        tok = self.consume("PNAME_LN")
        pname = tok[1]
        if ":" not in pname:
            raise SyntaxError(f"Invalid prefixed name: {pname}")
        ns, local = pname.split(":", 1)
        prefix_iri = self.prefixes.get(ns)
        if prefix_iri is None:
            raise SyntaxError(f"Undefined prefix: {ns}")
        return NamedNode(f"{prefix_iri}{local}")

    def _parse_blank_node(self, text: str) -> BlankNode:
        return BlankNode(id=text)


# ── Top-level parse function ─────────────────────────────────────


def parse_sparql(query: str) -> AlgebraNode:
    """Parse a SPARQL query string into an algebra tree."""
    parser = SPARQLParser(query)
    return parser.parse()


_parse = parse_sparql


# ── Helper: join two BGP nodes ───────────────────────────────────


def _join_bgp(left: AlgebraNode, right: AlgebraNode) -> AlgebraNode:
    if isinstance(left, BGP) and isinstance(right, BGP):
        return BGP(left.patterns + right.patterns)
    if isinstance(left, BGP) and not left.patterns:
        return right
    if isinstance(right, BGP) and not right.patterns:
        return left
    return right


# ── Helper: collect variable names from algebra ──────────────────


def _collect_vars(node: AlgebraNode) -> list[str]:
    vars_: set[str] = set()

    def walk(n: AlgebraNode) -> None:
        if isinstance(n, BGP):
            for p in n.patterns:
                for pos in (p.subject, p.predicate, p.object_):
                    if isinstance(pos, Variable):
                        vars_.add(pos.name)
        elif isinstance(n, Filter):
            walk(n.inner)
            _collect_filter_vars(n.expr, vars_)
        elif isinstance(n, (Optional_, Union)):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, (Project, OrderBy)) or (isinstance(n, Slice) and n.inner):
            walk(n.inner)

    walk(node)
    return sorted(vars_)


def _collect_filter_vars(expr: Any, vars_: set[str]) -> None:
    if isinstance(expr, VarRef):
        vars_.add(expr.name)
    elif isinstance(expr, (Comparison, And, Or)):
        _collect_filter_vars(expr.left, vars_)
        _collect_filter_vars(expr.right, vars_)
    elif isinstance(expr, Not):
        _collect_filter_vars(expr.inner, vars_)
    elif isinstance(expr, RegexFunc):
        _collect_filter_vars(expr.text, vars_)
        _collect_filter_vars(expr.pattern, vars_)
        if expr.flags:
            _collect_filter_vars(expr.flags, vars_)
    elif isinstance(expr, BoundFunc):
        pass


# ── Helper: coerce literal values ────────────────────────────────


def _coerce_literal_value(value: str, dtype: str) -> object:
    xsd_integer = "http://www.w3.org/2001/XMLSchema#integer"
    xsd_decimal = "http://www.w3.org/2001/XMLSchema#decimal"
    xsd_double = "http://www.w3.org/2001/XMLSchema#double"
    xsd_float = "http://www.w3.org/2001/XMLSchema#float"
    xsd_boolean = "http://www.w3.org/2001/XMLSchema#boolean"
    if dtype in (xsd_integer, "http://www.w3.org/2001/XMLSchema#int"):
        try:
            return int(value)
        except ValueError:
            return value
    if dtype in (xsd_decimal, xsd_double, xsd_float):
        try:
            return float(value)
        except ValueError:
            return value
    if dtype == xsd_boolean:
        return value.lower() in ("true", "1")
    return value
