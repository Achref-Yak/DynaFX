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

from cognitive_engine.kb.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
    xsd,
)
from cognitive_engine.kb.store import TripleStore
from cognitive_engine.kb.turtle import parse_turtle, serialize_turtle
from cognitive_engine.kb.sparql import QueryResult, evaluate, parse_sparql
from cognitive_engine.kb.inference import (
    InferencePattern,
    Rule,
    RuleEngine,
    Var,
    owl_rl_rules,
    propagate_opinion,
    rdfs_rules,
)
from cognitive_engine.kb.confidence import (
    FusionResult,
    QueryGrade,
    argumentative_filter,
    fuse_graphs,
    grade_query,
)
from cognitive_engine.reason.fusion import (
    consensus_compromise,
    cumulative_fusion,
)
from cognitive_engine.core.models import (
    FusionSituation,
    Opinion,
)

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
    "evaluate",
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
