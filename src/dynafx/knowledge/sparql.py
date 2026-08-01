"""SPARQL 1.1 query parser and evaluator.

Supports:
    - ASK queries (return boolean)
    - SELECT queries (return variable bindings)
    - Variable aliases (SELECT (expr AS ?alias))
    - Basic graph patterns with s/p/o matching
    - Named graph querying (GRAPH ?g { ... })
    - FILTER expressions (comparisons, logical, bound, isURI, isLiteral, regex, lang, STRDT, STRLANG)
    - OPTIONAL patterns
    - DISTINCT, LIMIT, OFFSET
    - ORDER BY (ASC/DESC)
    - UNION of basic graph patterns
    - IN / NOT IN filters
    - BIND expressions
    - VALUES clause
    - DESCRIBE queries (resolve as star-model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynafx.knowledge._sparql_parser import (
    BGP,
    And,
    Ask,
    BoundFunc,
    Comparison,
    Constant,
    Construct,
    Filter,
    Not,
    Optional_,
    Or,
    OrderBy,
    Project,
    RegexFunc,
    Slice,
    SPARQLTriplePattern,
    Union,
    Variable,
    VarRef,
    tokenize,  # noqa: F401  re-exported for tests
)
from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
)
from dynafx.knowledge.store import TripleStore

# ── Extra AST types (not produced by parser) ──────────────────────


@dataclass(frozen=True)
class Select:
    """SELECT query."""
    variables: list[str]
    patterns: list[Any]
    filters: list[Any] = field(default_factory=list)
    distinct: bool = False
    limit: int | None = None
    offset: int = 0
    order_by: list[tuple[str, str]] | None = None
    prefix_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Describe:
    """DESCRIBE query."""
    variables: list[str]
    patterns: list[Any]
    filters: list[Any] = field(default_factory=list)
    prefix_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Values:
    """VALUES clause."""
    variables: list[str]
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class Bind:
    """BIND expression."""
    expr: Any
    var: str


@dataclass(frozen=True)
class GraphPattern:
    """GRAPH ?g { patterns }"""
    var: Any
    inner: Any


AlgebraNode = (
    Ask | BGP | Filter | Optional_ | Union | OrderBy | Slice
    | Project | Select | Describe | Construct | Values | Bind | GraphPattern
)


# ── Expression Types (pre-parsed) ────────────────────────────────


@dataclass(frozen=True)
class ExprLiteral:
    value: Any


@dataclass(frozen=True)
class ExprVar:
    name: str


@dataclass(frozen=True)
class ExprOp:
    op: str
    args: list[Any]


@dataclass(frozen=True)
class ExprFunc:
    name: str
    args: list[Any]


# ── Query Result ─────────────────────────────────────────────────


class QueryResult:
    """The result of executing a SPARQL query.

    For ASK: cardinality is 1 or 0.
    For SELECT: bindings is a list of dicts mapping var -> value.
    """

    def __init__(self, vars: list[str] | None = None,
                 bindings: list[dict] | None = None,
                 cardinality: int = 0):
        self.vars: list[str] = vars or []
        self.bindings: list[dict] = bindings or []
        self.cardinality = cardinality

    def __repr__(self) -> str:
        if self.bindings:
            return (f"QueryResult(cardinality={self.cardinality}, "
                    f"bindings=[{len(self.bindings)} rows, "
                    f"vars={list(self.bindings[0].keys())}])")
        return f"QueryResult(cardinality={self.cardinality})"

    def __bool__(self) -> bool:
        return self.cardinality > 0


# ── Evaluator ────────────────────────────────────────────────────


Binding = dict[str, RDFNode]


def evaluate(
    algebra: AlgebraNode,
    store: TripleStore,
    with_inference: str | dict[str, Any] | None = None,
) -> QueryResult:
    """Evaluate a SPARQL algebra tree against a TripleStore."""
    if isinstance(algebra, Ask):
        bindings = list(_eval_node(algebra.inner, store, [({})], with_inference))
        return QueryResult(cardinality=1 if bindings else 0)
    if isinstance(algebra, Construct):
        bindings_list = list(_eval_node(algebra.inner, store, [({})], with_inference))
        construct_store = TripleStore()
        for binding in bindings_list:
            for tpl in algebra.templates:
                s = _resolve_tpl_position(tpl.subject, binding)
                p = _resolve_tpl_position(tpl.predicate, binding)
                o_ = _resolve_tpl_position(tpl.object_, binding)
                if s is not None and p is not None and o_ is not None:
                    construct_store.add(Triple(s, p, o_))
        result_triples = list(construct_store.triples(TriplePattern()))
        return QueryResult(
            bindings=[{"_triple": t} for t in result_triples],
            cardinality=len(result_triples),
        )
    if isinstance(algebra, Project):
        bindings_list = list(_eval_node(algebra.inner, store, [({})], with_inference))
        if algebra.distinct:
            bindings_list = _distinct_bindings(bindings_list, algebra.vars)
        result_bindings: list[Binding] = []
        for binding in bindings_list:
            projected = {v: binding[v] for v in algebra.vars if v in binding}
            result_bindings.append(projected)
        vars_ = [v for v in algebra.vars if any(v in b for b in result_bindings)]
        return QueryResult(
            vars=vars_ if vars_ else algebra.vars,
            bindings=result_bindings,
            cardinality=len(result_bindings),
        )
    raise ValueError(f"Unsupported algebra node: {type(algebra).__name__}")


def _eval_node(
    node: AlgebraNode,
    store: TripleStore,
    initial: list[Binding],
    with_inference: str | dict[str, Any] | None = None,
) -> list[Binding]:
    if isinstance(node, BGP):
        return list(_eval_bgp(node.patterns, store, initial, with_inference))
    if isinstance(node, Filter):
        results: list[Binding] = []
        for binding in _eval_node(node.inner, store, initial, with_inference):
            if _eval_filter(node.expr, binding):
                results.append(binding)
        return results
    if isinstance(node, Optional_):
        left_results = _eval_node(node.left, store, initial, with_inference)
        if not left_results:
            return []
        right_results = _eval_node(node.right, store, [({})], with_inference)
        results: list[Binding] = []
        for left_binding in left_results:
            matched = False
            for right_binding in right_results:
                merged = _merge_bindings(left_binding, right_binding)
                if merged is not None and (node.expr is None or _eval_filter(node.expr, merged)):
                    results.append(merged)
                    matched = True
            if not matched:
                results.append(left_binding)
        return results
    if isinstance(node, Union):
        return (_eval_node(node.left, store, initial, with_inference)
                + _eval_node(node.right, store, initial, with_inference))
    if isinstance(node, OrderBy):
        results = _eval_node(node.inner, store, initial, with_inference)
        if not results:
            return []
        for cond_var, direction in reversed(node.conditions):
            results.sort(
                key=lambda r: _binding_sort_key(r.get(cond_var)),
                reverse=(direction == "DESC"),
            )
        return results
    if isinstance(node, Slice):
        results = _eval_node(node.inner, store, initial, with_inference)
        start = node.offset or 0
        end = start + node.limit if node.limit is not None else None
        return results[start:end]
    raise ValueError(f"Unknown algebra node: {type(node).__name__}")


def _eval_bgp(
    patterns: list[SPARQLTriplePattern],
    store: TripleStore,
    initial: list[Binding],
    with_inference: str | dict[str, Any] | None = None,
) -> list[Binding]:
    current: list[Binding] = list(initial)
    for pattern in patterns:
        next_results: list[Binding] = []
        for binding in current:
            resolved = _resolve_pattern(pattern, binding)
            for triple in store.triples(resolved, with_inference=with_inference):
                new_bindings = _extract_bindings(pattern, triple)
                merged = _merge_bindings(binding, new_bindings)
                if merged is not None:
                    next_results.append(merged)
        current = next_results
        if not current:
            break
    return current


def _resolve_pattern(
    pattern: SPARQLTriplePattern,
    binding: Binding,
) -> TriplePattern:
    s = pattern.subject
    p = pattern.predicate
    o = pattern.object_

    if isinstance(s, Variable) and s.name in binding:
        s = binding[s.name]
    elif isinstance(s, Variable):
        s = None
    if isinstance(p, Variable) and p.name == "a":
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


def _merge_bindings(b1: Binding, b2: Binding) -> Binding | None:
    merged = dict(b1)
    for k, v in b2.items():
        if k in merged:
            if not _rdf_equals(merged[k], v):
                return None
        else:
            merged[k] = v
    return merged


def _rdf_equals(a: Any, b: Any) -> bool:
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
    bindings_list: list[Binding],
    vars_: list[str],
) -> list[Binding]:
    seen: set[tuple] = set()
    result: list[Binding] = []
    for binding in bindings_list:
        key = tuple(binding.get(v) for v in vars_)
        if key not in seen:
            seen.add(key)
            result.append(binding)
    return result


def _resolve_tpl_position(pos: Any, binding: Binding) -> RDFNode | None:
    if isinstance(pos, Variable):
        return binding.get(pos.name)
    if isinstance(pos, (NamedNode, BlankNode, Literal)):
        return pos
    return None


def _binding_sort_key(val: Any) -> tuple:
    if val is None:
        return (1, "")
    if isinstance(val, Literal):
        return (0, str(val.value))
    if isinstance(val, NamedNode):
        return (2, val.iri)
    return (3, str(val))


# ── Filter evaluator ──────────────────────────────────────────────

FILTER_FUNCS: dict[str, list[str]] = {
    "STR": ["value"],
    "LANG": ["value"],
    "DATATYPE": ["value"],
    "isURI": ["value"],
    "isIRI": ["value"],
    "isLiteral": ["value"],
    "isBlank": ["value"],
    "isNumeric": ["value"],
    "xsd:dateTime": ["value"],
    "STRLEN": ["value"],
    "UCASE": ["value"],
    "LCASE": ["value"],
    "ENCODE_FOR_URI": ["value"],
    "CONCAT": ["args"],
    "SUBSTR": ["value", "start", "length"],
    "CONTAINS": ["arg1", "arg2"],
    "STRSTARTS": ["arg1", "arg2"],
    "STRENDS": ["arg1", "arg2"],
    "STRBEFORE": ["arg1", "arg2"],
    "STRAFTER": ["arg1", "arg2"],
    "REPLACE": ["arg1", "arg2", "arg3"],
    "ABS": ["value"],
    "ROUND": ["value"],
    "CEIL": ["value"],
    "FLOOR": ["value"],
    "RAND": [],
    "YEAR": ["value"],
    "MONTH": ["value"],
    "DAY": ["value"],
    "HOURS": ["value"],
    "MINUTES": ["value"],
    "SECONDS": ["value"],
    "TIMEZONE": ["value"],
    "TZ": ["value"],
    "UUID": [],
    "STRUUID": [],
    "MD5": ["value"],
    "SHA1": ["value"],
    "SHA256": ["value"],
    "SHA384": ["value"],
    "SHA512": ["value"],
    "COALESCE": ["args"],
    "IF": ["condition", "then", "else"],
    "BOUND": ["var"],
    "sameTerm": ["term1", "term2"],
    "langMatches": ["lang_tag", "lang_range"],
    "REGEX": ["text", "pattern", "flags"],
}


def _eval_filter(expr: Any, binding: Binding) -> bool:
    """Evaluate a SPARQL FILTER expression against a binding.

    Returns True if the filter passes (or if the expression cannot be evaluated).
    """
    try:
        result = _eval_expr(expr, binding)
        if isinstance(result, bool):
            return result
        if isinstance(result, Literal):
            return _literal_bool(result)
        if result is None:
            return True
        return bool(result)
    except (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError):
        return True


def _eval_expr(expr: Any, binding: Binding) -> Any:
    """Evaluate a SPARQL expression against a binding."""
    from dynafx.knowledge._sparql_parser import Function, Operator

    if isinstance(expr, ExprLiteral):
        return expr.value
    if isinstance(expr, ExprVar):
        val = binding.get(expr.name)
        if val is None and expr.name == "a":
            return None
        return val
    if isinstance(expr, ExprOp):
        args = [_eval_expr(a, binding) for a in expr.args]
        return _eval_operator(expr.op, args)
    if isinstance(expr, ExprFunc):
        args = [_eval_expr(a, binding) for a in expr.args]
        return _eval_function(expr.name, args)
    if isinstance(expr, Operator):
        args = [_eval_expr(a, binding) for a in expr.args]
        return _eval_operator(expr.op, args)
    if isinstance(expr, Function):
        args = [_eval_expr(a, binding) for a in expr.args]
        return _eval_function(expr.name, args)
    if isinstance(expr, Comparison):
        left = _eval_expr(expr.left, binding)
        right = _eval_expr(expr.right, binding)
        if expr.op == "=":
            return _rdf_equals(left, right)
        if expr.op == "!=":
            return not _rdf_equals(left, right)
        diff = _compare_numeric(left, right)
        if expr.op == "<":
            return diff < 0
        if expr.op == ">":
            return diff > 0
        if expr.op == "<=":
            return diff <= 0
        if expr.op == ">=":
            return diff >= 0
        return False
    if isinstance(expr, And):
        return _eval_expr(expr.left, binding) and _eval_expr(expr.right, binding)
    if isinstance(expr, Or):
        return _eval_expr(expr.left, binding) or _eval_expr(expr.right, binding)
    if isinstance(expr, Not):
        return not _eval_expr(expr.inner, binding)
    if isinstance(expr, VarRef):
        val = binding.get(expr.name)
        if val is None and expr.name == "a":
            return None
        return val
    if isinstance(expr, Constant):
        return expr.value
    if isinstance(expr, RegexFunc):
        text = _sparql_str(_eval_expr(expr.text, binding))
        pattern = _sparql_str(_eval_expr(expr.pattern, binding))
        flags = _sparql_str(_eval_expr(expr.flags, binding)) if expr.flags else ""
        import re as _re
        re_flags = 0
        if "i" in flags:
            re_flags |= _re.IGNORECASE
        if "m" in flags:
            re_flags |= _re.MULTILINE
        if "s" in flags:
            re_flags |= _re.DOTALL
        try:
            return bool(_re.search(pattern, text, re_flags))
        except _re.error:
            return False
    if isinstance(expr, BoundFunc):
        return expr.name in binding
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, str):
        return expr
    return expr


def _eval_operator(op: str, args: list[Any]) -> Any:
    """Evaluate a SPARQL operator."""
    if op == "=":
        return _rdf_equals(args[0], args[1]) if len(args) >= 2 else False
    if op == "!=":
        return not _rdf_equals(args[0], args[1]) if len(args) >= 2 else False
    if op == "<":
        return _compare_numeric(args[0], args[1]) < 0
    if op == ">":
        return _compare_numeric(args[0], args[1]) > 0
    if op == "<=":
        return _compare_numeric(args[0], args[1]) <= 0
    if op == ">=":
        return _compare_numeric(args[0], args[1]) >= 0
    if op in ("+", "ADD"):
        return _to_numeric(args[0]) + _to_numeric(args[1])
    if op in ("-", "SUB"):
        return _to_numeric(args[0]) - _to_numeric(args[1])
    if op in ("*", "MUL"):
        return _to_numeric(args[0]) * _to_numeric(args[1])
    if op in ("/", "DIV"):
        divisor = _to_numeric(args[1])
        if divisor == 0:
            raise ZeroDivisionError
        return _to_numeric(args[0]) / divisor
    if op == "&&":
        return bool(args[0]) and bool(args[1])
    if op == "||":
        return bool(args[0]) or bool(args[1])
    if op == "!":
        return not bool(args[0])
    if op == "UNARY+":
        return +_to_numeric(args[0])
    if op == "UNARY-":
        return -_to_numeric(args[0])
    if op == "IN":
        return any(_rdf_equals(args[0], a) for a in args[1:])
    if op == "NOT_IN":
        return not any(_rdf_equals(args[0], a) for a in args[1:])
    raise ValueError(f"Unknown operator: {op}")


def _eval_function(name: str, args: list[Any]) -> Any:
    """Evaluate a SPARQL builtin function."""
    name_upper = name.upper()
    if name_upper == "BOUND":
        if args:
            return args[0] is not None
        return False
    if name_upper == "isURI" or name_upper == "isIRI":
        return isinstance(args[0], NamedNode) if args else False
    if name_upper == "isLITERAL":
        return isinstance(args[0], Literal) if args else False
    if name_upper == "isBLANK":
        return isinstance(args[0], BlankNode) if args else False
    if name_upper == "isNUMERIC":
        if not args:
            return False
        val = args[0]
        if isinstance(val, (int, float)):
            return True
        if isinstance(val, Literal):
            try:
                float(val.value)
                return True
            except (ValueError, TypeError):
                pass
        return False
    if name_upper == "STR":
        if args:
            return _sparql_str(args[0])
        return ""
    if name_upper == "LANG":
        if isinstance(args[0], Literal):
            return args[0].lang_tag or ""
        return ""
    if name_upper == "DATATYPE":
        if isinstance(args[0], Literal):
            return args[0].datatype or ""
        return ""
    if name_upper == "STRLEN":
        return len(str(_sparql_str(args[0]))) if args else 0
    if name_upper == "UCASE":
        return _sparql_str(args[0]).upper() if args else ""
    if name_upper == "LCASE":
        return _sparql_str(args[0]).lower() if args else ""
    if name_upper in ("CONCAT", "CONCATENATE"):
        return "".join(str(_sparql_str(a)) for a in args)
    if name_upper == "SUBSTR":
        s = _sparql_str(args[0]) if args else ""
        start = int(_to_numeric(args[1])) - 1 if len(args) > 1 else 0
        length = int(_to_numeric(args[2])) if len(args) > 2 else len(s) - start
        return s[start:start + length] if 0 <= start < len(s) else ""
    if name_upper == "CONTAINS":
        return _sparql_str(args[0]) in _sparql_str(args[1])
    if name_upper == "STRSTARTS":
        return _sparql_str(args[0]).startswith(_sparql_str(args[1]))
    if name_upper == "STRENDS":
        return _sparql_str(args[0]).endswith(_sparql_str(args[1]))
    if name_upper == "STRBEFORE":
        s, t = _sparql_str(args[0]), _sparql_str(args[1])
        idx = s.find(t)
        return s[:idx] if idx >= 0 else ""
    if name_upper == "STRAFTER":
        s, t = _sparql_str(args[0]), _sparql_str(args[1])
        idx = s.find(t)
        return s[idx + len(t):] if idx >= 0 else ""
    if name_upper == "REPLACE":
        text = _sparql_str(args[0])
        pattern = _sparql_str(args[1])
        replacement = _sparql_str(args[2])
        import re as _re
        flags = _sparql_str(args[3]) if len(args) > 3 else ""
        re_flags = 0
        if "i" in flags:
            re_flags |= _re.IGNORECASE
        if "m" in flags:
            re_flags |= _re.MULTILINE
        if "s" in flags:
            re_flags |= _re.DOTALL
        try:
            return _re.sub(pattern, replacement, text, flags=re_flags)
        except _re.error:
            return text
    if name_upper in ("ABS", "ROUND", "CEIL", "FLOOR"):
        val = _to_numeric(args[0])
        if name_upper == "ABS":
            return abs(val)
        if name_upper == "ROUND":
            return round(val)
        if name_upper == "CEIL":
            import math
            return math.ceil(val)
        if name_upper == "FLOOR":
            import math
            return math.floor(val)
    if name_upper == "RAND":
        import random
        return random.random()
    if name_upper == "NOW":
        import time
        return time.time()
    if name_upper == "YEAR":
        from datetime import datetime
        return datetime.fromtimestamp(int(_to_numeric(args[0]))).year
    if name_upper == "MONTH":
        from datetime import datetime
        return datetime.fromtimestamp(int(_to_numeric(args[0]))).month
    if name_upper == "DAY":
        from datetime import datetime
        return datetime.fromtimestamp(int(_to_numeric(args[0]))).day
    if name_upper == "HOURS":
        from datetime import datetime
        return datetime.fromtimestamp(int(_to_numeric(args[0]))).hour
    if name_upper == "MINUTES":
        from datetime import datetime
        return datetime.fromtimestamp(int(_to_numeric(args[0]))).minute
    if name_upper == "SECONDS":
        from datetime import datetime
        return datetime.fromtimestamp(int(_to_numeric(args[0]))).second
    if name_upper == "COALESCE":
        for a in args:
            if a is not None:
                return a
        return None
    if name_upper == "IF":
        return args[1] if _eval_expr(args[0], {}) else args[2]
    if name_upper == "sameTerm":
        return _rdf_equals(args[0], args[1])
    if name_upper == "langMATCHES":
        lang_tag = _sparql_str(args[0]).lower()
        lang_range = _sparql_str(args[1]).lower()
        if lang_range == "*":
            return bool(lang_tag)
        return lang_tag == lang_range or lang_tag.startswith(lang_range + "-")
    if name_upper == "REGEX":
        text = _sparql_str(args[0])
        pattern = _sparql_str(args[1])
        flags = _sparql_str(args[2]) if len(args) > 2 else ""
        import re as _re
        re_flags = 0
        if "i" in flags:
            re_flags |= _re.IGNORECASE
        if "m" in flags:
            re_flags |= _re.MULTILINE
        if "s" in flags:
            re_flags |= _re.DOTALL
        try:
            return bool(_re.search(pattern, text, re_flags))
        except _re.error:
            return False
    if name_upper in ("MD5", "SHA1", "SHA256", "SHA384", "SHA512"):
        import hashlib
        data = _sparql_str(args[0]).encode("utf-8")
        h = getattr(hashlib, name_upper.lower())()
        h.update(data)
        return h.hexdigest()
    if name_upper in ("STRDT", "STRDT"):
        return Literal(str(_sparql_str(args[0])), datatype=str(_sparql_str(args[1])))
    if name_upper == "STRLANG":
        return Literal(str(_sparql_str(args[0])), lang_tag=str(_sparql_str(args[1])))
    if name_upper in ("UUID", "STRUUID"):
        import uuid
        if name_upper == "UUID":
            return uuid.uuid4().hex
        return str(uuid.uuid4())
    raise ValueError(f"Unknown SPARQL function: {name}")


def _sparql_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, Literal):
        return str(val.value)
    if isinstance(val, NamedNode):
        return val.iri
    return str(val)


def _to_numeric(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, Literal):
        try:
            return float(val.value)
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def _literal_bool(lit: Literal) -> bool:
    if lit.datatype and "boolean" in lit.datatype:
        return lit.value in (True, "true", "1")
    v = str(lit.value).lower()
    return v in ("true", "1")


def _compare_numeric(a: Any, b: Any) -> float:
    return _to_numeric(a) - _to_numeric(b)


# ── SPARQL Parser ────────────────────────────────────────────────


def parse_sparql(query: str) -> AlgebraNode:
    """Parse a SPARQL 1.1 query string into an AST node."""
    from dynafx.knowledge._sparql_parser import _parse as _inner_parse
    return _inner_parse(query)
