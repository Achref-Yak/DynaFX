"""SPARQL 1.1 query parser and evaluator.

Supported:
    - SELECT (DISTINCT), ASK, CONSTRUCT
    - BGP (basic graph pattern)
    - FILTER (comparisons, &&, ||, !, REGEX, BOUND)
    - OPTIONAL, UNION
    - DISTINCT, ORDER BY (ASC/DESC), LIMIT, OFFSET
    - PREFIX declarations
    - Variables: ?x, $x
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from dynafx.core.models import Opinion
from dynafx.kb.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
)
from dynafx.kb.store import TripleStore
from dynafx.reason.fusion import cumulative_fusion


# ── Variable representation ──────────────────────────────────────


@dataclass(frozen=True)
class Variable:
    """A SPARQL variable (?x or $x)."""
    name: str  # without ?/$ prefix

    def __repr__(self) -> str:
        return f"?{self.name}"


# ── SPARQL triple pattern (supports variables) ───────────────────


@dataclass(frozen=True)
class SPARQLTriplePattern:
    """Triple pattern where positions can be RDFNode, Variable, or None."""
    subject: Optional[Any] = None
    predicate: Optional[Any] = None
    object_: Optional[Any] = None


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
    ("INTEGER", r'[+-]?\d+'),
    ("DECIMAL", r'[+-]?\d+\.\d*'),
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
    expr: Any  # FilterExpr
    inner: AlgebraNode


@dataclass
class Optional_(AlgebraNode):
    left: AlgebraNode
    right: AlgebraNode
    expr: Optional[Any] = None


@dataclass
class Union(AlgebraNode):
    left: AlgebraNode
    right: AlgebraNode


@dataclass
class Project(AlgebraNode):
    vars: list[str]  # variable names without ?
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
    conditions: list[Tuple[str, str]]  # (var_name, "ASC"|"DESC")
    inner: AlgebraNode


@dataclass
class Slice(AlgebraNode):
    limit: Optional[int] = None
    offset: Optional[int] = None
    inner: Optional[AlgebraNode] = None


# ── Filter Expression Nodes ──────────────────────────────────────


class FilterExpr:
    """Base class for filter expressions."""


@dataclass
class Comparison(FilterExpr):
    op: str  # =, !=, <, >, <=, >=
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
    flags: Optional[FilterExpr] = None


@dataclass
class BoundFunc(FilterExpr):
    name: str  # var name without ?


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

    # ── Lookahead helpers ─────────────────────────────────────

    def peek(self) -> tuple[str, str, int]:
        return self.tokens[self.pos]

    def consume(self, expected_kind: Optional[str] = None) -> tuple[str, str, int]:
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

    # ── Prologue ──────────────────────────────────────────────

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

    # ── SELECT ────────────────────────────────────────────────

    def _parse_select(self) -> AlgebraNode:
        self.consume("SELECT")
        distinct = self.skip("DISTINCT")
        vars_: list[str] = []
        if self._check("STAR"):
            self.consume("STAR")
            # Will fill in from WHERE clause
        else:
            while self._check("VAR"):
                var_tok = self.consume("VAR")
                vars_.append(var_tok[1][1:])  # strip ?/$
        # optional WHERE
        if self._check("WHERE"):
            self.consume("WHERE")
        inner = self._parse_group_graph_pattern()
        # Collect variables from BGP if SELECT *
        if not vars_:
            vars_ = _collect_vars(inner)
        # ORDER BY
        order_conditions: list[Tuple[str, str]] = []
        if self._check("ORDER"):
            self._parse_order_by(order_conditions)
        # LIMIT / OFFSET
        limit: Optional[int] = None
        offset: Optional[int] = None
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

    def _parse_order_by(self, conditions: list[Tuple[str, str]]) -> None:
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

    # ── ASK ───────────────────────────────────────────────────

    def _parse_ask(self) -> AlgebraNode:
        self.consume("ASK")
        if self._check("WHERE"):
            self.consume("WHERE")
        inner = self._parse_group_graph_pattern()
        return Ask(inner)

    # ── CONSTRUCT ─────────────────────────────────────────────

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

    # ── Group Graph Pattern ───────────────────────────────────

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
                # Sub-group (for nested patterns)
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

    # ── Triple Pattern ────────────────────────────────────────

    def _parse_triple_pattern(self) -> Optional[SPARQLTriplePattern]:
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
            # We don't have a BLANK_NODE token type in SPARQL lexer currently.
            # Use the Turtle-style parsing.
            return BlankNode(id=self.consume()[1])
        if tok[0] == "VAR":
            return Variable(self.consume("VAR")[1][1:])
        if tok[0] == "LPAREN":
            # Collection - skip for now
            self.consume("LPAREN")
            while not self._check("RPAREN") and not self._check("EOF"):
                self._parse_object_or_var()
            self.consume("RPAREN")
            return None
        if tok[0] == "A":
            # 'a' as subject is unusual but valid in some contexts
            tok = self.consume("A")
            return None
        raise SyntaxError(f"Expected subject/variable at position {tok[2]}, got {tok[0]}")

    def _parse_predicate_or_var(self):
        if self._check("A"):
            self.consume("A")
            return Variable("a")  # special marker for rdf:type
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

    # ── Filter expressions ────────────────────────────────────

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
        flags: Optional[FilterExpr] = None
        if self._check("COMMA"):
            self.consume("COMMA")
            flags = self._parse_or_expr()
        self.consume("RPAREN")
        return RegexFunc(text, pattern, flags)

    # ── Prefix resolution ─────────────────────────────────────

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

    # ── BLANK_NODE token support in SPARQL ────────────────────

    def _parse_blank_node(self, text: str) -> BlankNode:
        return BlankNode(id=text)


# ── Top-level parse function ─────────────────────────────────────


def parse_sparql(query: str) -> AlgebraNode:
    """Parse a SPARQL query string into an algebra tree."""
    parser = SPARQLParser(query)
    return parser.parse()


# ── Helper: join two BGP nodes ───────────────────────────────────


def _join_bgp(left: AlgebraNode, right: AlgebraNode) -> AlgebraNode:
    """Merge two algebra nodes, flattening adjacent BGPs."""
    if isinstance(left, BGP) and isinstance(right, BGP):
        return BGP(left.patterns + right.patterns)
    if isinstance(left, BGP) and not left.patterns:
        return right
    # Create a synthetic BGP join
    # In SPARQL algebra, sequential patterns are joined
    if isinstance(left, BGP) and isinstance(right, BGP):
        return BGP(left.patterns + right.patterns)
    return BGP(left.patterns + right.patterns) if isinstance(left, BGP) and isinstance(right, BGP) else right


# ── Helper: collect variable names from algebra ──────────────────


def _collect_vars(node: AlgebraNode) -> list[str]:
    """Collect all variable names referenced in an algebra node."""
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
        elif isinstance(n, Optional_):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, Union):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, Project):
            walk(n.inner)
        elif isinstance(n, OrderBy):
            walk(n.inner)
        elif isinstance(n, Slice) and n.inner:
            walk(n.inner)

    walk(node)
    return sorted(vars_)


def _collect_filter_vars(expr: Any, vars_: set[str]) -> None:
    """Collect variable names from a filter expression."""
    if isinstance(expr, VarRef):
        vars_.add(expr.name)
    elif isinstance(expr, Comparison):
        _collect_filter_vars(expr.left, vars_)
        _collect_filter_vars(expr.right, vars_)
    elif isinstance(expr, And):
        _collect_filter_vars(expr.left, vars_)
        _collect_filter_vars(expr.right, vars_)
    elif isinstance(expr, Or):
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
        pass  # var is bound by name, not a reference


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


# ── Query Result ─────────────────────────────────────────────────


@dataclass
class QueryResult:
    """Result of evaluating a SPARQL query."""
    vars: list[str]
    bindings: list[Dict[str, RDFNode]]
    opinions: list[Dict[str, Opinion]]
    cardinality: int


# ── Evaluator ────────────────────────────────────────────────────


Binding = Dict[str, RDFNode]
OpinionMap = Dict[str, Opinion]


def evaluate(algebra: AlgebraNode, store: TripleStore) -> QueryResult:
    """Evaluate a SPARQL algebra tree against a TripleStore."""
    if isinstance(algebra, Ask):
        bindings = list(_eval_node(algebra.inner, store, [({}, {})]))
        return QueryResult(
            vars=[],
            bindings=[],
            opinions=[],
            cardinality=1 if bindings else 0,
        )
    if isinstance(algebra, Construct):
        bindings_list = list(_eval_node(algebra.inner, store, [({}, {})]))
        construct_store = TripleStore()
        for binding, _opin in bindings_list:
            for tpl in algebra.templates:
                s = _resolve_tpl_position(tpl.subject, binding)
                p = _resolve_tpl_position(tpl.predicate, binding)
                o_ = _resolve_tpl_position(tpl.object_, binding)
                if s is not None and p is not None and o_ is not None:
                    construct_store.add(Triple(s, p, o_))
        result_triples = list(construct_store.triples(TriplePattern()))
        return QueryResult(
            vars=[],
            bindings=[{"_triple": t} for t in result_triples],
            opinions=[{}],
            cardinality=len(result_triples),
        )
    if isinstance(algebra, Project):
        bindings_list = list(_eval_node(algebra.inner, store, [({}, {})]))
        # Apply DISTINCT
        if algebra.distinct:
            bindings_list = _distinct_bindings(bindings_list, algebra.vars)
        # Project to selected vars
        result_bindings: list[Binding] = []
        result_opinions: list[OpinionMap] = []
        for binding, opin in bindings_list:
            projected = {v: binding[v] for v in algebra.vars if v in binding}
            projected_opinions = {v: opin.get(v, Opinion()) for v in algebra.vars if v in opin}
            result_bindings.append(projected)
            result_opinions.append(projected_opinions)
        vars_ = [v for v in algebra.vars if any(v in b for b in result_bindings)]
        return QueryResult(
            vars=vars_ if vars_ else algebra.vars,
            bindings=result_bindings,
            opinions=result_opinions,
            cardinality=len(result_bindings),
        )
    raise ValueError(f"Unsupported algebra node: {type(algebra).__name__}")


def _eval_node(
    node: AlgebraNode,
    store: TripleStore,
    initial: list[Tuple[Binding, OpinionMap]],
) -> Iterator[Tuple[Binding, OpinionMap]]:
    """Evaluate an algebra node, yielding (binding, opinions) tuples."""
    if isinstance(node, BGP):
        yield from _eval_bgp(node.patterns, store, initial)
    elif isinstance(node, Filter):
        for binding, opin in _eval_node(node.inner, store, initial):
            if _eval_filter(node.expr, binding):
                yield binding, opin
    elif isinstance(node, Optional_):
        # Left join
        left_results = list(_eval_node(node.left, store, initial))
        right_results = list(_eval_node(node.right, store, [({}, {})]))
        if not left_results:
            return
        for left_binding, left_opin in left_results:
            matched = False
            for right_binding, right_opin in right_results:
                merged = _merge_bindings(left_binding, right_binding)
                if merged is not None:
                    merged_opin = {**left_opin, **right_opin}
                    if node.expr is None or _eval_filter(node.expr, merged):
                        yield merged, merged_opin
                        matched = True
            if not matched:
                yield left_binding, left_opin
    elif isinstance(node, Union):
        yield from _eval_node(node.left, store, initial)
        yield from _eval_node(node.right, store, initial)
    elif isinstance(node, OrderBy):
        results = list(_eval_node(node.inner, store, initial))
        if not results:
            return
        for cond_var, direction in reversed(node.conditions):
            results.sort(
                key=lambda r: _binding_sort_key(r[0].get(cond_var)),
                reverse=(direction == "DESC"),
            )
        yield from results
    elif isinstance(node, Slice):
        results = list(_eval_node(node.inner, store, initial))
        start = node.offset or 0
        end = start + node.limit if node.limit is not None else None
        yield from results[start:end]
    else:
        raise ValueError(f"Unknown algebra node: {type(node).__name__}")


def _eval_bgp(
    patterns: list[SPARQLTriplePattern],
    store: TripleStore,
    initial: list[Tuple[Binding, OpinionMap]],
) -> Iterator[Tuple[Binding, OpinionMap]]:
    """Evaluate a BGP (basic graph pattern) using nested-loop join."""
    current: list[Tuple[Binding, OpinionMap]] = list(initial)
    for pattern in patterns:
        next_results: list[Tuple[Binding, OpinionMap]] = []
        for binding, opin in current:
            resolved = _resolve_pattern(pattern, binding)
            for triple in store.triples(resolved):
                new_bindings = _extract_bindings(pattern, triple)
                merged = _merge_bindings(binding, new_bindings)
                if merged is not None:
                    merged_opin = dict(opin)
                    # Track opinion from this triple
                    var_for_opinion = _find_var_for_pattern(pattern, triple)
                    for var in var_for_opinion:
                        if triple.opinion:
                            merged_opin[var] = triple.opinion
                    next_results.append((merged, merged_opin))
        current = next_results
        if not current:
            break
    yield from current


def _resolve_pattern(
    pattern: SPARQLTriplePattern,
    binding: Binding,
) -> TriplePattern:
    """Resolve a SPARQL pattern against a binding, substituting known vars."""
    s = pattern.subject
    p = pattern.predicate
    o = pattern.object_

    if isinstance(s, Variable) and s.name in binding:
        s = binding[s.name]
    elif isinstance(s, Variable):
        s = None
    if isinstance(p, Variable) and p.name == "a":
        # 'a' is a special variable meaning rdf:type
        p = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    elif isinstance(p, Variable) and p.name in binding:
        p = binding[p.name]
    elif isinstance(p, Variable):
        p = None
    if isinstance(o, Variable) and o.name in binding:
        o = binding[o.name]
    elif isinstance(o, Variable):
        o = None

    return TriplePattern(s, p, o)


def _extract_bindings(
    pattern: SPARQLTriplePattern,
    triple: Triple,
) -> Binding:
    """Extract variable bindings from a matching triple."""
    bindings: Binding = {}
    if isinstance(pattern.subject, Variable):
        bindings[pattern.subject.name] = triple.subject
    if isinstance(pattern.predicate, Variable):
        p_name = pattern.predicate.name
        if p_name != "a":
            bindings[p_name] = triple.predicate
    if isinstance(pattern.object_, Variable):
        bindings[pattern.object_.name] = triple.object_
    return bindings


def _find_var_for_pattern(pattern: SPARQLTriplePattern, triple: Triple) -> list[str]:
    """Find variable names that are bound by this pattern."""
    vars_: list[str] = []
    if isinstance(pattern.subject, Variable):
        vars_.append(pattern.subject.name)
    if isinstance(pattern.predicate, Variable) and pattern.predicate.name != "a":
        vars_.append(pattern.predicate.name)
    if isinstance(pattern.object_, Variable):
        vars_.append(pattern.object_.name)
    return vars_


def _merge_bindings(b1: Binding, b2: Binding) -> Optional[Binding]:
    """Merge two bindings. Returns None if they conflict."""
    merged = dict(b1)
    for k, v in b2.items():
        if k in merged:
            if not _rdf_equals(merged[k], v):
                return None
        else:
            merged[k] = v
    return merged


def _rdf_equals(a: Any, b: Any) -> bool:
    """Check RDF term equality."""
    if type(a) is not type(b):
        return False
    if isinstance(a, NamedNode):
        return a.iri == b.iri
    if isinstance(a, BlankNode):
        return a.id == b.id
    if isinstance(a, Literal):
        if a.lang_tag != b.lang_tag:
            return False
        if a.datatype != b.datatype:
            return False
        return a.value == b.value
    return a == b


def _distinct_bindings(
    bindings_list: list[Tuple[Binding, OpinionMap]],
    vars_: list[str],
) -> list[Tuple[Binding, OpinionMap]]:
    """Remove duplicate bindings for the given variables."""
    seen: set[tuple] = set()
    result: list[Tuple[Binding, OpinionMap]] = []
    for binding, opin in bindings_list:
        key = tuple(binding.get(v) for v in vars_)
        if key not in seen:
            seen.add(key)
            result.append((binding, opin))
    return result


def _resolve_tpl_position(pos: Any, binding: Binding) -> Optional[RDFNode]:
    """Resolve a template position (used in CONSTRUCT)."""
    if isinstance(pos, Variable) and pos.name in binding:
        return binding[pos.name]
    if isinstance(pos, RDFNode):
        return pos
    return None


# ── Filter evaluation ────────────────────────────────────────────


def _eval_filter(expr: Any, binding: Binding) -> bool:
    """Evaluate a filter expression against a binding."""
    try:
        if isinstance(expr, Comparison):
            left = _eval_filter_rvalue(expr.left, binding)
            right = _eval_filter_rvalue(expr.right, binding)
            if left is None or right is None:
                return False
            if expr.op == "=" or expr.op == "EQ":
                return _rdf_equals(left, right)
            if expr.op == "!=" or expr.op == "NE":
                return not _rdf_equals(left, right)
            if expr.op in ("<", ">", "<=", ">=", "LT", "GT", "LE", "GE"):
                return _compare_numeric(left, right, expr.op)
            return False
        if isinstance(expr, And):
            return _eval_filter(expr.left, binding) and _eval_filter(expr.right, binding)
        if isinstance(expr, Or):
            return _eval_filter(expr.left, binding) or _eval_filter(expr.right, binding)
        if isinstance(expr, Not):
            return not _eval_filter(expr.inner, binding)
        if isinstance(expr, VarRef):
            return expr.name in binding
        if isinstance(expr, Constant):
            return True  # constants are truthy
        if isinstance(expr, RegexFunc):
            text = _eval_filter_rvalue(expr.text, binding)
            pattern = _eval_filter_rvalue(expr.pattern, binding)
            if text is None or pattern is None:
                return False
            text_str = str(text.value) if isinstance(text, Literal) else str(text)
            pattern_str = str(pattern.value) if isinstance(pattern, Literal) else str(pattern)
            flags = 0
            if expr.flags:
                flag_val = _eval_filter_rvalue(expr.flags, binding)
                if flag_val is not None:
                    flag_str = str(flag_val.value) if isinstance(flag_val, Literal) else str(flag_val)
                    if "i" in flag_str:
                        flags |= re.IGNORECASE
            return bool(re.search(pattern_str, text_str, flags))
        if isinstance(expr, BoundFunc):
            return expr.name in binding
        return True
    except Exception:
        return False


def _eval_filter_rvalue(expr: Any, binding: Binding) -> Optional[Any]:
    """Evaluate a filter expression to an RDFNode or None."""
    if isinstance(expr, VarRef):
        return binding.get(expr.name)
    if isinstance(expr, Constant):
        return expr.value
    if isinstance(expr, Comparison):
        left = _eval_filter_rvalue(expr.left, binding)
        right = _eval_filter_rvalue(expr.right, binding)
        if left is None or right is None:
            return None
        if expr.op == "=" or expr.op == "EQ":
            return Literal(_rdf_equals(left, right))
        return Literal(_compare_numeric(left, right, expr.op))
    if isinstance(expr, And):
        return Literal(_eval_filter(expr.left, binding) and _eval_filter(expr.right, binding))
    if isinstance(expr, Or):
        return Literal(_eval_filter(expr.left, binding) or _eval_filter(expr.right, binding))
    if isinstance(expr, Not):
        return Literal(not _eval_filter(expr.inner, binding))
    if isinstance(expr, RegexFunc):
        return Literal(_eval_filter(expr, binding))
    if isinstance(expr, BoundFunc):
        return Literal(expr.name in binding)
    return None


def _compare_numeric(left: Any, right: Any, op: str) -> bool:
    """Compare two RDFNodes numerically."""
    try:
        lv = _to_numeric(left)
        rv = _to_numeric(right)
        if lv is None or rv is None:
            return False
        stripped_op = op.replace("LT", "<").replace("GT", ">").replace("LE", "<=").replace("GE", ">=")
        if stripped_op == "<":
            return lv < rv
        if stripped_op == ">":
            return lv > rv
        if stripped_op == "<=":
            return lv <= rv
        if stripped_op == ">=":
            return lv >= rv
        return False
    except (TypeError, ValueError):
        return False


def _to_numeric(node: Any) -> Optional[float]:
    """Convert an RDFNode to a numeric value for comparison."""
    if isinstance(node, Literal):
        val = node.value
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return None


def _binding_sort_key(node: Any) -> tuple:
    """Generate a sort key for a binding value."""
    if isinstance(node, Literal):
        val = node.value
        if isinstance(val, (int, float)):
            return (0, val)
        return (1, str(val))
    if isinstance(node, NamedNode):
        return (2, node.iri)
    return (3, str(node))
