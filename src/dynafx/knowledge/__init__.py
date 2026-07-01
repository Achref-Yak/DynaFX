"""Knowledge — RDF/OWL/SPARQL triple store with SL confidence grading.

Submodules:
    model      — RDF data model (NamedNode, BlankNode, Literal, Triple)
    store      — TripleStore with named graphs and opinion dedup
    turtle     — Turtle/N-Triples parser and serializer
    sparql     — SPARQL 1.1 query parser and evaluator
    inference  — Forward-chaining rule engine (RDFS, OWL RL)
    confidence — Graph fusion and query grading
    hierarchy  — OWL2-style type hierarchy
    loader     — TBox loader

Integration:
    - `epistemics/` provides SL Opinion, cumulative_fusion, EvidenceMatrix
"""

from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
    xsd,
)
from dynafx.knowledge.store import TripleStore
from dynafx.knowledge.turtle import parse_turtle, serialize_turtle
from dynafx.knowledge.sparql import QueryResult, evaluate as _evaluate, parse_sparql as _parse_sparql
from dynafx.knowledge.inference import (
    InferencePattern,
    Rule,
    RuleEngine,
    Var,
    owl_rl_rules,
    propagate_opinion,
    rdfs_rules,
)
from dynafx.knowledge.confidence import (
    FusionResult,
    QueryGrade,
    argumentative_filter,
    fuse_graphs,
    grade_query,
)
from dynafx.knowledge.loader import TBox, load_tbox, GENERAL_TBOX
from dynafx.knowledge.hierarchy import TypeNode, TypeHierarchy, MDM_TYPE_HIERARCHY
from dynafx.knowledge.production import (
    Action,
    ActionResult,
    AggregationCondition,
    AndCondition,
    BridgeAction,
    ComparisonCondition,
    Condition,
    ConditionResult,
    LogAction,
    NotCondition,
    OrCondition,
    ProductionRule,
    ProductionRuleEngine,
    RetractAction,
    SimulateAction,
    SparqlCondition,
    TripleAction,
    TripleCondition,
)
from dynafx.knowledge.transactions import (
    Transaction,
    TransactionQuery,
    TransactionStore,
)
from dynafx.knowledge.execution import (
    ExecutionRecord,
    ExecutionStore,
)
from dynafx.epistemics.fusion import (
    consensus_compromise,
    cumulative_fusion,
)
from dynafx.core.models import (
    FusionSituation,
    Opinion,
)

sparql_evaluate = _evaluate

import warnings as _warnings

def __getattr__(name: str):
    if name == "evaluate":
        _warnings.warn(
            "dynafx.knowledge.evaluate is deprecated, use dynafx.knowledge.sparql_evaluate",
            DeprecationWarning, stacklevel=2,
        )
        return _evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

parse_sparql = _parse_sparql

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
    # TBox
    "TBox",
    "load_tbox",
    "GENERAL_TBOX",
    "TypeNode",
    "TypeHierarchy",
    "MDM_TYPE_HIERARCHY",
    # Production rules
    "Action",
    "ActionResult",
    "AggregationCondition",
    "AndCondition",
    "BridgeAction",
    "ComparisonCondition",
    "Condition",
    "ConditionResult",
    "LogAction",
    "NotCondition",
    "OrCondition",
    "ProductionRule",
    "ProductionRuleEngine",
    "RetractAction",
    "SimulateAction",
    "SparqlCondition",
    "TripleAction",
    "TripleCondition",
    # Transactions
    "Transaction",
    "TransactionQuery",
    "TransactionStore",
    # Execution
    "ExecutionRecord",
    "ExecutionStore",
    # SL integration
    "Opinion",
    "FusionSituation",
    "cumulative_fusion",
    "consensus_compromise",
]
