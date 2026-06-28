"""Knowledge base package — RDF/OWL/SPARQL with SL confidence grading.

Submodules:
    model      — RDF data model (NamedNode, BlankNode, Literal, Triple)
    store      — TripleStore with named graphs and opinion dedup
    turtle     — Turtle/N-Triples parser and serializer
    sparql     — SPARQL 1.1 query parser and evaluator
    inference  — Forward-chaining rule engine (RDFS, OWL RL)
    confidence — Graph fusion and query grading

Integration:
    - `reason/` provides SL Opinion, cumulative_fusion, EvidenceMatrix
    - `tbox/` provides OWL2-style type hierarchy consumed by store/inference
"""

from dynafx.kb.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
    xsd,
)
from dynafx.kb.store import TripleStore
from dynafx.kb.turtle import parse_turtle, serialize_turtle
from dynafx.kb.sparql import QueryResult, evaluate as _evaluate, parse_sparql as _parse_sparql

from dynafx.kb.inference import (
    InferencePattern,
    Rule,
    RuleEngine,
    Var,
    owl_rl_rules,
    propagate_opinion,
    rdfs_rules,
)
from dynafx.kb.confidence import (
    FusionResult,
    QueryGrade,
    argumentative_filter,
    fuse_graphs,
    grade_query,
)
from dynafx.reason.fusion import (
    consensus_compromise,
    cumulative_fusion,
)
from dynafx.core.models import (
    FusionSituation,
    Opinion,
)

# Canonical names
sparql_evaluate = _evaluate

import warnings as _warnings

def __getattr__(name: str):
    if name == "evaluate":
        _warnings.warn(
            "dynafx.kb.evaluate is deprecated, use dynafx.kb.sparql_evaluate",
            DeprecationWarning, stacklevel=2,
        )
        return _evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# parse_sparql is fine — specific enough
from dynafx.kb.sparql import parse_sparql

__all__ = [
    # Model
    "BlankNode",
    "Literal",
    "NamedNode",
    "RDFNode",
    "Triple",
    "TriplePattern",
    "xsd",
    # Store
    "TripleStore",
    # Turtle
    "parse_turtle",
    "serialize_turtle",
    # SPARQL
    "QueryResult",
    "sparql_evaluate",
    "parse_sparql",
    # Inference
    "InferencePattern",
    "Rule",
    "RuleEngine",
    "Var",
    "owl_rl_rules",
    "propagate_opinion",
    "rdfs_rules",
    # Confidence
    "FusionResult",
    "QueryGrade",
    "argumentative_filter",
    "fuse_graphs",
    "grade_query",
    # SL integration
    "Opinion",
    "FusionSituation",
    "cumulative_fusion",
    "consensus_compromise",
]
