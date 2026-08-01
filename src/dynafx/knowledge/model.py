"""RDF data model — nodes, triples, and patterns.

Provides the core RDF data model types:
    - NamedNode, BlankNode, Literal (RDF dataset nodes)
    - Triple (subject-predicate-object statement)
    - TriplePattern (query/inference pattern with wildcards via None)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

# ── Node types ───────────────────────────────────────────────────


class RDFNode:
    """Abstract base for all RDF nodes (NamedNode, BlankNode, Literal)."""
    def n3(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class NamedNode(RDFNode):
    """An RDF node identified by an IRI.

    Example:
        NamedNode("http://example.org/Person")
    """
    iri: str

    def n3(self) -> str:
        return f"<{self.iri}>"


@dataclass(frozen=True)
class BlankNode(RDFNode):
    """An RDF blank node with an internal identifier.

    If no id is provided, a unique id is auto-generated.
    """
    id: str = field(default_factory=lambda: f"_:b{uuid4().hex[:8]}")

    def n3(self) -> str:
        return self.id


@dataclass(frozen=True)
class Literal(RDFNode):
    """An RDF literal value with optional datatype and language tag.

    Example:
        Literal("hello", lang_tag="en")
        Literal(42, datatype=xsd("integer"))
    """
    value: Any
    datatype: Optional[str] = None
    lang_tag: Optional[str] = None

    def n3(self) -> str:
        if self.lang_tag:
            return f'{_quote_literal(self.value, force_quotes=True)}@{self.lang_tag}'
        if self.datatype:
            dtype = _wrap_iri(self.datatype)
            return f'{_quote_literal(self.value, force_quotes=True)}^^{dtype}'
        return _quote_literal(self.value)


def _wrap_iri(iri: str) -> str:
    if iri.startswith("<") and iri.endswith(">"):
        return iri
    return f"<{iri}>"


def _quote_literal(value: Any, force_quotes: bool = False) -> str:
    if force_quotes or isinstance(value, str):
        escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    if value is True:
        return '"true"^^<http://www.w3.org/2001/XMLSchema#boolean>'
    if value is False:
        return '"false"^^<http://www.w3.org/2001/XMLSchema#boolean>'
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


# ── XSD type shortcuts ──────────────────────────────────────────

_XSD = "http://www.w3.org/2001/XMLSchema"


def xsd(typename: str) -> NamedNode:
    """Create an XSD type NamedNode.

    Example:
        xsd("integer") → NamedNode("http://www.w3.org/2001/XMLSchema#integer")
    """
    return NamedNode(f"{_XSD}#{typename}")


# Commonly used XSD types
XSD_STRING = xsd("string")
XSD_INTEGER = xsd("integer")
XSD_DOUBLE = xsd("double")
XSD_BOOLEAN = xsd("boolean")
XSD_DATE_TIME = xsd("dateTime")


# ── Triple ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Triple:
    """An RDF triple.

    Equality and hashing are based on (subject, predicate, object_) only.

    Example:
        Triple(
            subject=NamedNode("http://example.org/jane"),
            predicate=NamedNode("http://example.org/worksAt"),
            object_=NamedNode("http://example.org/acme"),
        )
    """
    subject: NamedNode | BlankNode
    predicate: NamedNode
    object_: NamedNode | BlankNode | Literal

    @property
    def spo(self) -> tuple:
        """Return the (subject, predicate, object) identity tuple."""
        return (self.subject, self.predicate, self.object_)

    def __hash__(self) -> int:
        return hash(self.spo)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triple):
            return NotImplemented
        return self.spo == other.spo


# ── TriplePattern (wildcards) ────────────────────────────────────


@dataclass(frozen=True)
class TriplePattern:
    """A triple pattern with None as wildcard for query matching.

    Example:
        TriplePattern(
            subject=NamedNode("http://example.org/jane"),
            predicate=None,
            object_=None,
        )
    """
    subject: Optional[NamedNode | BlankNode | Literal] = None
    predicate: Optional[NamedNode | BlankNode | Literal] = None
    object_: Optional[NamedNode | BlankNode | Literal] = None


# ── RDF namespace constants ──────────────────────────────────────

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"

RDF_TYPE = NamedNode(f"{RDF}type")
RDF_PROPERTY = NamedNode(f"{RDF}Property")
RDFS_DOMAIN = NamedNode(f"{RDFS}domain")
RDFS_RANGE = NamedNode(f"{RDFS}range")
RDFS_SUBCLASS_OF = NamedNode(f"{RDFS}subClassOf")
RDFS_SUBPROPERTY_OF = NamedNode(f"{RDFS}subPropertyOf")
OWL_SAME_AS = NamedNode(f"{OWL}sameAs")
OWL_INVERSE_OF = NamedNode(f"{OWL}inverseOf")
OWL_FUNCTIONAL_PROPERTY = NamedNode(f"{OWL}FunctionalProperty")
OWL_TRANSITIVE_PROPERTY = NamedNode(f"{OWL}TransitiveProperty")
OWL_SYMMETRIC_PROPERTY = NamedNode(f"{OWL}SymmetricProperty")
