"""Turtle / N-Triples parser and serializer.

Handles the common subset:
    - @prefix declarations
    - a (rdf:type)
    - String, integer, float, boolean literals
    - Language-tagged and typed literals
    - Blank nodes
    - ; and , grouping
    - Comments (# ...)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from dynafx.core.models import Opinion
from dynafx.kb.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
)
from dynafx.kb.store import TripleStore


# ── Tokenizer ────────────────────────────────────────────────────

_TOKEN_SPEC = [
    ("PREFIX", r'@prefix\b'),
    ("BASE", r'@base\b'),
    ("SPARQL_PREFIX", r'\bPREFIX\b'),
    ("SPARQL_BASE", r'\bBASE\b'),
    ("IRI", r'<[^>]*>'),
    ("BLANK_NODE", r'_:[\w.-]*'),
    ("PNAME_LN", r'(?:[a-zA-Z_][\w.-]*)?:[a-zA-Z_][\w.-]*'),
    ("PNAME_NS", r'(?:[a-zA-Z_][\w.-]*)?:'),
    ("A", r'\ba\b'),
    ("TRUE", r'\btrue\b'),
    ("FALSE", r'\bfalse\b'),
    ("DOUBLE", r'[+-]?(?:\d+\.\d*[eE][+-]?\d+|\.\d+[eE][+-]?\d+)'),
    ("DECIMAL", r'[+-]?\d+\.\d*(?:[eE][+-]?\d+)?'),
    ("INTEGER", r'[+-]?\d+'),
    ("LANGTAG", r'@[a-zA-Z]+(?:-[a-zA-Z0-9]+)*'),
    ("STRING_LITERAL_QUOTE", r'"[^"\\]*(?:\\.[^"\\]*)*"'),
    ("DOT", r'\.'),
    ("SEMI", r';'),
    ("COMMA", r','),
    ("LPAREN", r'\('),
    ("RPAREN", r'\)'),
    ("LBRACE", r'\{'),
    ("RBRACE", r'\}'),
    ("HAT_HAT", r'\^\^'),
    ("COMMENT", r'#[^\n]*'),
    ("WS", r'[ \t\r\n]+'),
    ("MISMATCH", r'.'),
]

_TOKEN_RE = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in _TOKEN_SPEC))


def tokenize(text: str) -> list[tuple[str, str, int]]:
    """Tokenize Turtle text. Returns list of (type, value, position)."""
    tokens: list[tuple[str, str, int]] = []
    for match in _TOKEN_RE.finditer(text):
        kind = match.lastgroup
        value = match.group()
        pos = match.start()
        if kind == "WS" or kind == "COMMENT":
            continue
        if kind == "STRING_LITERAL_QUOTE":
            # Unescape
            value = value[1:-1]
            value = value.replace('\\"', '"').replace('\\\\', '\\')
            if '\\' in value and value not in ('"', '\\'):
                value = value.replace('\\n', '\n').replace('\\t', '\t')
            kind = "STRING"
        if kind == "IRI":
            value = value[1:-1]
        if kind == "MISMATCH":
            raise SyntaxError(f"Unexpected character {value!r} at position {pos}")
        tokens.append((kind, value, pos))
    tokens.append(("EOF", "", len(text)))
    return tokens


# ── Parser ───────────────────────────────────────────────────────


class TurtleParser:
    """Recursive descent Turtle parser into a TripleStore."""

    def __init__(self, text: str, base_iri: str = "",
                 default_graph: str = "default"):
        self.tokens = tokenize(text)
        self.pos = 0
        self.store = TripleStore()
        self.prefixes: dict[str, str] = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }
        self.base_iri = base_iri
        self.default_graph = default_graph
        self.bnode_counter = 0

    def parse(self) -> TripleStore:
        """Parse the full Turtle document."""
        while self.peek()[0] != "EOF":
            tok = self.peek()
            if tok[0] in ("PREFIX", "SPARQL_PREFIX"):
                self._parse_directive()
            elif tok[0] in ("BASE", "SPARQL_BASE"):
                self._parse_directive()
            else:
                self._parse_triples()
        return self.store

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

    # ── Directives ────────────────────────────────────────────

    def _parse_directive(self) -> None:
        tok = self.consume()
        if tok[0] in ("PREFIX", "SPARQL_PREFIX"):
            ns = self._consume_pname_ns()
            iri = self._consume_iri_ref()
            self.prefixes[ns.rstrip(":")] = iri
            if tok[0] == "PREFIX":
                self.skip("DOT")
        elif tok[0] in ("BASE", "SPARQL_BASE"):
            iri = self._consume_iri_ref()
            self.base_iri = iri
            if tok[0] == "BASE":
                self.skip("DOT")

    def _consume_pname_ns(self) -> str:
        tok = self.peek()
        if tok[0] == "PNAME_NS":
            return self.consume()[1]
        if tok[0] == "PNAME_LN":
            return self.consume()[1]
        raise SyntaxError(f"Expected prefix name at position {tok[2]}")

    def _resolve_iri(self, iri: str) -> str:
        if self.base_iri and not iri.startswith("http"):
            from urllib.parse import urljoin
            iri = urljoin(self.base_iri, iri)
        return iri

    def _consume_iri_ref(self) -> str:
        tok = self.consume("IRI")
        return self._resolve_iri(tok[1])

    # ── Triples ───────────────────────────────────────────────

    def _parse_triples(self) -> None:
        subject = self._parse_subject()
        self._parse_predicate_object_list(subject)
        self.skip("DOT")

    def _parse_predicate_object_list(self, subject: RDFNode) -> None:
        while True:
            verb = self._parse_verb()
            self._parse_object_list(subject, verb)
            if not self.skip("SEMI"):
                break

    def _parse_object_list(self, subject: RDFNode, verb: NamedNode) -> None:
        while True:
            obj = self._parse_object()
            triple = Triple(subject, verb, obj)
            self.store.add(triple, graph=self.default_graph)
            if not self.skip("COMMA"):
                break

    # ── Subjects, Verbs, Objects ──────────────────────────────

    def _parse_subject(self) -> RDFNode:
        tok = self.peek()
        if tok[0] == "IRI":
            iri = self._resolve_iri(self.consume()[1])
            return NamedNode(iri)
        if tok[0] == "BLANK_NODE":
            id_ = self.consume()[1]
            return BlankNode(id=id_)
        if tok[0] == "PNAME_LN":
            return self._resolve_pname_ln()
        if tok[0] == "PNAME_NS":
            ns = self.consume()[1]
            return NamedNode(self.prefixes.get(ns.rstrip(":"), ns))
        if tok[0] == "LPAREN":
            return self._parse_blank_node_collection()
        if tok[0] == "STRING":
            return self._parse_literal()
        raise SyntaxError(f"Expected subject at position {tok[2]}, got {tok[0]}")

    def _parse_verb(self) -> NamedNode:
        if self.skip("A"):
            return NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        if self.peek()[0] == "IRI":
            return NamedNode(self._resolve_iri(self.consume("IRI")[1]))
        if self.peek()[0] == "PNAME_LN":
            return self._resolve_pname_ln()
        raise SyntaxError(f"Expected verb at position {self.peek()[2]}")

    def _parse_object(self) -> RDFNode:
        tok = self.peek()
        if tok[0] == "IRI":
            return NamedNode(self._resolve_iri(self.consume()[1]))
        if tok[0] == "BLANK_NODE":
            return BlankNode(id=self.consume()[1])
        if tok[0] == "PNAME_LN":
            return self._resolve_pname_ln()
        if tok[0] == "STRING":
            return self._parse_literal()
        if tok[0] == "INTEGER":
            return Literal(int(self.consume()[1]))
        if tok[0] == "DECIMAL":
            return Literal(float(self.consume()[1]))
        if tok[0] == "DOUBLE":
            return Literal(float(self.consume()[1]))
        if tok[0] == "TRUE":
            self.consume()
            return Literal(True)
        if tok[0] == "FALSE":
            self.consume()
            return Literal(False)
        if tok[0] == "LPAREN":
            return self._parse_blank_node_collection()
        raise SyntaxError(f"Expected object at position {tok[2]}, got {tok[0]}")

    # ── Literals ──────────────────────────────────────────────

    def _parse_literal(self) -> Literal:
        value = self.consume("STRING")[1]
        if self.peek()[0] == "LANGTAG":
            lang = self.consume("LANGTAG")[1][1:]
            return Literal(value, lang_tag=lang)
        if self.peek()[0] == "HAT_HAT":
            self.consume("HAT_HAT")
            dtype = self._consume_datatype()
            return Literal(self._coerce_literal(value, dtype), datatype=dtype)
        return Literal(value)

    def _consume_datatype(self) -> str:
        tok = self.peek()
        if tok[0] == "IRI":
            return self.consume()[1]
        if tok[0] == "PNAME_LN":
            return self._resolve_pname_ln().iri
        raise SyntaxError(f"Expected datatype IRI at position {tok[2]}")

    def _coerce_literal(self, value: str, dtype: str) -> object:
        xsd_integer = "http://www.w3.org/2001/XMLSchema#integer"
        xsd_int = "http://www.w3.org/2001/XMLSchema#int"
        xsd_decimal = "http://www.w3.org/2001/XMLSchema#decimal"
        xsd_double = "http://www.w3.org/2001/XMLSchema#double"
        xsd_float = "http://www.w3.org/2001/XMLSchema#float"
        xsd_boolean = "http://www.w3.org/2001/XMLSchema#boolean"
        if dtype in (xsd_integer, xsd_int):
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

    # ── Blank node collections ────────────────────────────────

    def _parse_blank_node_collection(self) -> BlankNode:
        self.consume("LPAREN")
        bnode = BlankNode(id=f"_:c{self.bnode_counter}")
        self.bnode_counter += 1
        # Parse objects in collection
        while self.peek()[0] != "RPAREN":
            obj = self._parse_object()
            triple = Triple(
                bnode,
                NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#first"),
                obj,
            )
            self.store.add(triple, graph=self.default_graph)
            # Chain: bnode rdf:rest next_bnode | rdf:nil
            next_bn = BlankNode(id=f"_:c{self.bnode_counter}")
            self.bnode_counter += 1
            rest = Triple(
                bnode,
                NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"),
                next_bn,
            )
            self.store.add(rest, graph=self.default_graph)
            bnode = next_bn
        self.consume("RPAREN")
        # Terminate the list
        nil = Triple(
            bnode,
            NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"),
            NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"),
        )
        self.store.add(nil, graph=self.default_graph)
        return BlankNode(id="_:c0")  # return the first node of the collection


# ── Top-level parse function ─────────────────────────────────────


def parse_turtle(text: str, base_iri: str = "",
                 default_graph: str = "default") -> TripleStore:
    """Parse Turtle text into a TripleStore.

    Args:
        text: Turtle-encoded RDF text.
        base_iri: Base IRI for resolving relative IRIs.
        default_graph: Named graph to store parsed triples in.

    Returns:
        TripleStore populated with parsed triples.
    """
    parser = TurtleParser(text, base_iri=base_iri, default_graph=default_graph)
    return parser.parse()


# ── Serializer ───────────────────────────────────────────────────


def serialize_turtle(
    triples: list[Triple],
    prefixes: Optional[Dict[str, str]] = None,
    comment_opinions: bool = True,
) -> str:
    """Serialize triples to pretty-printed Turtle.

    Args:
        triples: List of Triples to serialize.
        prefixes: Optional prefix map (e.g. {"ex": "http://example.org/"}).
        comment_opinions: Whether to append opinion comments.

    Returns:
        Turtle-encoded string.
    """
    if prefixes is None:
        prefixes = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }

    if not triples:
        return ""

    lines: list[str] = []
    for ns, iri in sorted(prefixes.items()):
        lines.append(f"@prefix {ns}: <{iri}> .")
    if prefixes:
        lines.append("")

    # Group by subject
    by_subject: dict[RDFNode, list[Triple]] = {}
    for t in triples:
        by_subject.setdefault(t.subject, []).append(t)

    for subject, subject_triples in by_subject.items():
        by_predicate: dict[NamedNode, list[Triple]] = {}
        for t in subject_triples:
            by_predicate.setdefault(t.predicate, []).append(t)

        subj_str = _n3_with_prefix(subject, prefixes)
        pred_items = list(by_predicate.items())
        for i, (predicate, pred_triples) in enumerate(pred_items):
            pred_str = _n3_with_prefix(predicate, prefixes)
            obj_parts: list[str] = []
            for t in pred_triples:
                obj_str = _n3_with_prefix(t.object_, prefixes)
                if comment_opinions and t.opinion:
                    obj_str += f"  # b={t.opinion.belief:.2f} d={t.opinion.disbelief:.2f} u={t.opinion.uncertainty:.2f}"
                obj_parts.append(obj_str)
            objects_str = ", ".join(obj_parts)
            is_last = (i == len(pred_items) - 1)
            suffix = " ." if is_last else " ;"
            if i == 0:
                lines.append(f"{subj_str}  {pred_str} {objects_str}{suffix}")
            else:
                indent = max(len(subj_str), 8)
                lines.append(f"{' ' * indent}  {pred_str} {objects_str}{suffix}")

    return "\n".join(lines) + ("\n" if lines else "")


def _n3_with_prefix(node: RDFNode, prefixes: Dict[str, str]) -> str:
    """Serialize an RDFNode using prefix abbreviations where possible."""
    if isinstance(node, NamedNode):
        iri = node.iri
        # Check if IRI starts with any prefix
        for ns, prefix_iri in sorted(prefixes.items(), key=lambda x: -len(x[1])):
            if iri.startswith(prefix_iri):
                local = iri[len(prefix_iri):]
                if local and (local[0].isalpha() or local[0] == '_'):
                    return f"{ns}:{local}"
        return node.n3()
    if isinstance(node, BlankNode):
        if node.id.startswith("_:"):
            return node.id
        return node.n3()
    if isinstance(node, Literal):
        return node.n3()
    return str(node)


# ── N-Triples (simple line-based format) ─────────────────────────


def parse_ntriples(text: str, graph: str = "default") -> TripleStore:
    """Parse N-Triples text into a TripleStore.

    N-Triples is line-based: one triple per line.
    """
    store = TripleStore()
    for line_num, line in enumerate(text.strip().split("\n"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(" ."):
            line = line[:-2]
        # Very simple N-Triples parser — just handle <s> <p> <o> .
        parts = _split_ntriples_line(line)
        if len(parts) < 3:
            continue
        try:
            s = _parse_ntriples_node(parts[0])
            p = _parse_ntriples_node(parts[1])
            o = _parse_ntriples_node(parts[2])
            if s is not None and p is not None and o is not None:
                store.add(Triple(s, p, o), graph=graph)
        except Exception:
            pass
    return store


def _split_ntriples_line(line: str) -> list[str]:
    """Split N-Triples line into tokens, respecting quoted strings."""
    parts: list[str] = []
    current = ""
    in_quotes = False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
            current += ch
        elif ch in (' ', '\t') and not in_quotes:
            if current:
                parts.append(current)
                current = ""
        else:
            current += ch
    if current:
        parts.append(current)
    return parts


def _parse_ntriples_node(text: str) -> Optional[RDFNode]:
    text = text.strip()
    if text.startswith("<") and text.endswith(">"):
        return NamedNode(text[1:-1])
    if text.startswith("_:"):
        return BlankNode(id=text)
    if text.startswith('"'):
        # Literal — handle lang tag and datatype
        if text.rstrip().endswith(">"):
            # Has datatype: "value"^^<type> or "value"^^type
            dtype_start = text.rfind("^^")
            if dtype_start > 0:
                value_part = text[:dtype_start].strip().strip('"')
                dtype_part = text[dtype_start+2:].strip().strip('<>')
                return Literal(value_part, datatype=dtype_part)
        # Check lang tag
        if '"@' in text:
            idx = text.rfind('"@')
            value_part = text[1:idx]
            lang_part = text[idx+2:]
            return Literal(value_part, lang_tag=lang_part)
        value_part = text.strip('"')
        return Literal(value_part)
    return None


def serialize_ntriples(triples: list[Triple]) -> str:
    """Serialize triples to N-Triples format (one triple per line)."""
    if not triples:
        return ""
    lines: list[str] = []
    for t in triples:
        s = _n3_node(t.subject)
        p = _n3_node(t.predicate)
        o = _n3_node(t.object_)
        line = f"{s} {p} {o} ."
        if t.opinion:
            line += f"  # b={t.opinion.belief:.2f} d={t.opinion.disbelief:.2f} u={t.opinion.uncertainty:.2f}"
        lines.append(line)
    return "\n".join(lines) + "\n" if lines else ""


def _n3_node(node: RDFNode) -> str:
    if isinstance(node, NamedNode):
        return f"<{node.iri}>"
    if isinstance(node, BlankNode):
        return node.id
    if isinstance(node, Literal):
        val = str(node.value).replace('\\', '\\\\').replace('"', '\\"')
        result = f'"{val}"'
        if node.lang_tag:
            result += f"@{node.lang_tag}"
        elif node.datatype:
            result += f"^^<{node.datatype}>"
        return result
    return str(node)
