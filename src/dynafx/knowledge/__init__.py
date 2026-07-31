"""Knowledge — RDF/OWL/SPARQL triple store.

Submodules:
    model      — RDF data model (NamedNode, BlankNode, Literal, Triple)
    store      — TripleStore with named graphs
    turtle     — Turtle/N-Triples parser and serializer
    sparql     — SPARQL 1.1 query parser and evaluator
    inference  — Forward-chaining rule engine (RDFS, OWL RL)
    hierarchy  — OWL2-style type hierarchy
    loader     — TBox loader
"""

from dynafx.knowledge.execution import (
    ExecutionRecord,
    ExecutionStore,
)
from dynafx.knowledge.hierarchy import MDM_TYPE_HIERARCHY, TypeHierarchy, TypeNode
from dynafx.knowledge.inference import (
    InferencePattern,
    Rule,
    RuleEngine,
    Var,
    owl_rl_rules,
    rdfs_rules,
)
from dynafx.knowledge.ingest_csv import (
    ColumnMapping,
    IngestReport,
    MappingDef,
    ingest_csv,
    load_all_mappings,
)
from dynafx.knowledge.loader import GENERAL_TBOX, TBox, load_tbox
from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
    xsd,
)
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
from dynafx.knowledge.sparql import QueryResult
from dynafx.knowledge.sparql import evaluate as _evaluate
from dynafx.knowledge.sparql import parse_sparql as _parse_sparql
from dynafx.knowledge.store import TripleStore
from dynafx.knowledge.transactions import (
    Transaction,
    TransactionQuery,
    TransactionStore,
)
from dynafx.knowledge.turtle import parse_turtle, serialize_turtle

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
    "rdfs_rules",
    # CSV ingestion
    "ColumnMapping",
    "IngestReport",
    "ingest_csv",
    "MappingDef",
    "load_all_mappings",
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
]
