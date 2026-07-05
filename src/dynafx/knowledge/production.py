"""Production Rule Engine — event-driven business rules over a TripleStore.

Enables IF-THEN rules with complex conditions (triple patterns, SPARQL,
comparisons, aggregation) and diverse actions (KB mutations, simulation,
scenario generation, optimization, logging).

Usage::

    from dynafx.knowledge.production import (
        ProductionRuleEngine, ProductionRule,
        TripleCondition, SparqlCondition, ComparisonCondition,
        TripleAction, SimulateAction, LogAction,
    )

    engine = ProductionRuleEngine(store)
    engine.add_rule(ProductionRule(
        name="detect-container-delay",
        body=[TripleCondition(InferencePattern(None, sc_status, sc_Delayed))],
        head=[LogAction("Container delay detected")],
    ))
    engine.start()
"""

from __future__ import annotations

import logging
import operator
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from dynafx.core.models import Opinion
from dynafx.knowledge.inference import InferencePattern
from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
)
from dynafx.knowledge.sparql import evaluate as _sparql_evaluate
from dynafx.knowledge.sparql import parse_sparql as _parse_sparql
from dynafx.knowledge.store import TripleStore

logger = logging.getLogger(__name__)


# ── Condition types ────────────────────────────────────────────────


@dataclass(frozen=True)
class ConditionResult:
    """Result of evaluating a condition."""
    matched: bool
    bindings: dict[str, RDFNode] = field(default_factory=dict)


class Condition:
    """A boolean condition evaluated against KB state.

    Subclasses implement evaluate(store, current_bindings) -> ConditionResult.
    """

    def evaluate(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ConditionResult:
        raise NotImplementedError


@dataclass(frozen=True)
class TripleCondition(Condition):
    """Match a triple pattern against the store.

    Variables in the pattern (from InferencePattern) are bound from the
    current bindings if already known, or newly extracted from matching triples.
    """

    pattern: InferencePattern

    def evaluate(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ConditionResult:
        resolved = self._resolve(self.pattern, bindings)
        for triple in store.triples(resolved):
            new_bindings = self._extract(self.pattern, triple, bindings)
            if new_bindings is not None:
                return ConditionResult(matched=True, bindings=new_bindings)
        return ConditionResult(matched=False, bindings=bindings)

    @staticmethod
    def _resolve(pat: InferencePattern, bindings: dict[str, RDFNode]) -> TriplePattern:
        s, p, o = pat.subject, pat.predicate, pat.object_
        if isinstance(s, str) and s.startswith("?"):
            s = bindings.get(s[1:])
        if isinstance(p, str) and p.startswith("?"):
            p = bindings.get(p[1:])
        if isinstance(o, str) and o.startswith("?"):
            o = bindings.get(o[1:])
        return TriplePattern(s, p, o)

    @staticmethod
    def _extract(
        pat: InferencePattern,
        triple: Triple,
        current: dict[str, RDFNode],
    ) -> Optional[dict[str, RDFNode]]:
        new = dict(current)
        for var_name, pat_val in [
            ("subject", pat.subject), ("predicate", pat.predicate),
            ("object_", pat.object_),
        ]:
            triple_val = getattr(triple, var_name)
            if isinstance(pat_val, str) and pat_val.startswith("?"):
                name = pat_val[1:]
                if name in new:
                    if not _rdf_equal(new[name], triple_val):
                        return None
                else:
                    new[name] = triple_val
        return new


@dataclass(frozen=True)
class SparqlCondition(Condition):
    """Match if a SPARQL query returns at least min_results results.

    Bindings are set from the first result row's variables.
    """

    query: str
    min_results: int = 1

    def evaluate(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ConditionResult:
        try:
            ast = _parse_sparql(self.query)
            qr = _sparql_evaluate(ast, store)
            count = len(qr.bindings) if qr.bindings else (1 if qr.cardinality > 0 else 0)
            if count >= self.min_results:
                new_bindings = dict(bindings)
                if qr.bindings:
                    for k, v in qr.bindings[0].items():
                        new_bindings[k] = v
                return ConditionResult(matched=True, bindings=new_bindings)
        except Exception:
            logger.warning("SparqlCondition '%s' failed", self.query[:80])
        return ConditionResult(matched=False, bindings=bindings)


@dataclass(frozen=True)
class ComparisonCondition(Condition):
    """Evaluate a numeric comparison: left op right.

    left/right can be:
        - float literal (e.g. 0.5)
        - string starting with ? for a bound variable name
        - string starting with $ for a value extracted via SPARQL
    op is one of: ">", ">=", "<", "<=", "==", "!="
    """

    left: Any
    op: str
    right: Any

    _OPS = {
        ">": operator.gt, ">=": operator.ge,
        "<": operator.lt, "<=": operator.le,
        "==": operator.eq, "!=": operator.ne,
    }

    def evaluate(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ConditionResult:
        lv = self._resolve_value(self.left, store, bindings)
        rv = self._resolve_value(self.right, store, bindings)
        op_fn = self._OPS.get(self.op)
        if op_fn is None:
            return ConditionResult(matched=False, bindings=bindings)
        try:
            matched = op_fn(float(lv), float(rv))
        except (ValueError, TypeError):
            matched = False
        return ConditionResult(matched=matched, bindings=bindings)

    @staticmethod
    def _resolve_value(val: Any, store: TripleStore, bindings: dict[str, RDFNode]) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            if val.startswith("?"):
                node = bindings.get(val[1:])
                if isinstance(node, Literal):
                    try:
                        return float(node.value)
                    except (ValueError, TypeError):
                        return 0.0
                return 0.0
        return 0.0


@dataclass(frozen=True)
class AggregationCondition(Condition):
    """Match if a SPARQL aggregate result meets a threshold.

    query: SPARQL SELECT with aggregate (COUNT, SUM, AVG, MIN, MAX).
    threshold: numeric threshold.
    op: comparison operator (default ">=").
    """

    query: str
    threshold: float
    op: str = ">="

    def evaluate(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ConditionResult:
        try:
            ast = _parse_sparql(self.query)
            qr = _sparql_evaluate(ast, store)
            val = float(len(qr.bindings)) if qr.bindings else 0.0
            if qr.bindings and qr.bindings[0]:
                first_key = next(iter(qr.bindings[0]))
                first_val = qr.bindings[0][first_key]
                if isinstance(first_val, Literal):
                    try:
                        val = float(first_val.value)
                    except (ValueError, TypeError):
                        val = float(len(qr.bindings))
            op_fn = ComparisonCondition._OPS.get(self.op, operator.ge)
            matched = op_fn(val, self.threshold)
            new_bindings = dict(bindings)
            new_bindings["_agg_result"] = Literal(val)
            return ConditionResult(matched=matched, bindings=new_bindings)
        except Exception:
            return ConditionResult(matched=False, bindings=bindings)


@dataclass(frozen=True)
class AndCondition(Condition):
    """AND of sub-conditions — all must match. Merges bindings."""

    conditions: list[Condition]

    def evaluate(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ConditionResult:
        merged = dict(bindings)
        for cond in self.conditions:
            result = cond.evaluate(store, merged)
            if not result.matched:
                return ConditionResult(matched=False, bindings=bindings)
            merged.update(result.bindings)
        return ConditionResult(matched=True, bindings=merged)


@dataclass(frozen=True)
class OrCondition(Condition):
    """OR of sub-conditions — first match wins. Uses its bindings."""

    conditions: list[Condition]

    def evaluate(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ConditionResult:
        for cond in self.conditions:
            result = cond.evaluate(store, bindings)
            if result.matched:
                return result
        return ConditionResult(matched=False, bindings=bindings)


@dataclass(frozen=True)
class NotCondition(Condition):
    """Negation — matches if inner condition does NOT match."""

    condition: Condition

    def evaluate(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ConditionResult:
        result = self.condition.evaluate(store, bindings)
        return ConditionResult(matched=not result.matched, bindings=bindings)


# ── Action types ───────────────────────────────────────────────────


@dataclass
class ActionResult:
    """Result of executing an action."""
    action_type: str
    action_id: str = ""
    success: bool = True
    message: str = ""
    output: dict[str, Any] = field(default_factory=dict)


class Action:
    """An action executed when a ProductionRule's body matches.

    Subclasses implement execute(store, bindings) -> ActionResult.
    """

    def execute(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ActionResult:
        raise NotImplementedError


@dataclass
class TripleAction(Action):
    """Add a triple to the store.

    s/p/o can be concrete RDFNodes or strings starting with ? for bound variables.
    """

    subject: Any
    predicate: Any
    object_: Any
    graph: str = "default"
    belief: float = 1.0
    disbelief: float = 0.0
    uncertainty: float = 0.0

    def execute(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ActionResult:
        s = self._resolve(self.subject, bindings)
        p = self._resolve(self.predicate, bindings)
        o = self._resolve(self.object_, bindings)
        if s is None or p is None or o is None:
            return ActionResult("triple_add", success=False, message="Unresolved node")
        opin = Opinion(self.belief, self.disbelief, self.uncertainty)
        triple = Triple(s, p, o, opinion=opin)
        store.add(triple, graph=self.graph)
        return ActionResult(
            "triple_add", action_id=str(uuid.uuid4()),
            output={"s": str(s), "p": str(p), "o": str(o), "graph": self.graph},
        )

    @staticmethod
    def _resolve(val: Any, bindings: dict[str, RDFNode]) -> Optional[RDFNode]:
        if isinstance(val, (NamedNode, BlankNode, Literal)):
            return val
        if isinstance(val, str) and val.startswith("?"):
            return bindings.get(val[1:])
        return None


@dataclass
class RetractAction(Action):
    """Remove triples matching a pattern from the store."""

    pattern: TriplePattern
    graph: Optional[str] = None

    def execute(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ActionResult:
        count = store.remove(self.pattern, graph=self.graph)
        return ActionResult(
            "retract", action_id=str(uuid.uuid4()),
            output={"removed": count},
        )


@dataclass
class LogAction(Action):
    """Log a message. Supports ?var substitution from bindings."""

    message: str
    level: str = "info"

    def execute(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ActionResult:
        msg = self._interpolate(self.message, bindings)
        level_fn = getattr(logger, self.level, logger.info)
        level_fn("[ProductionRule] %s", msg)
        return ActionResult("log", action_id=str(uuid.uuid4()), message=msg)

    @staticmethod
    def _interpolate(template: str, bindings: dict[str, RDFNode]) -> str:
        result = template
        for k, v in bindings.items():
            placeholder = f"?{k}"
            if placeholder in result:
                val = v.iri if hasattr(v, "iri") else (v.value if hasattr(v, "value") else str(v))
                result = result.replace(placeholder, str(val))
        return result


@dataclass
class BridgeAction(Action):
    """Run a KBSimBridge params_from_kb + simulate round-trip."""

    bridge: Any = None
    model: Any = None
    claim_map: list = field(default_factory=list)
    params_override: dict = field(default_factory=dict)

    def execute(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ActionResult:
        if self.bridge is None or self.model is None:
            return ActionResult("bridge", success=False, message="No bridge/model configured")
        try:
            params = self.bridge.params_from_kb(self.claim_map)
            params.update(self.params_override)
            result = self.bridge.run_with_kb(self.model, params=params)
            end_cash = result.values.get("Cash_Reserves", [0])[-1] if hasattr(result, "values") else 0
            return ActionResult(
                "bridge", action_id=str(uuid.uuid4()),
                output={
                    "end_cash": end_cash,
                    "steps": len(getattr(result, "times", [])),
                    "result": result,
                },
            )
        except Exception as e:
            return ActionResult("bridge", success=False, message=str(e))


@dataclass
class SimulateAction(Action):
    """Run a SysdModel simulation with given params."""

    model: Any = None
    params: dict = field(default_factory=dict)
    method: str = "euler"

    def execute(
        self,
        store: TripleStore,
        bindings: dict[str, RDFNode],
    ) -> ActionResult:
        if self.model is None:
            return ActionResult("simulate", success=False, message="No model configured")
        try:
            result = self.model.simulate(params=dict(self.params), method=self.method)
            end_cash = result.values.get("Cash_Reserves", [0])[-1] if hasattr(result, "values") else 0
            return ActionResult(
                "simulate", action_id=str(uuid.uuid4()),
                output={"end_cash": end_cash, "steps": len(getattr(result, "times", []))},
            )
        except Exception as e:
            return ActionResult("simulate", success=False, message=str(e))


# ── ProductionRule ─────────────────────────────────────────────────


@dataclass
class ProductionRule:
    """A production rule: IF body (conditions) THEN head (actions).

    Attributes:
        name: Unique rule name.
        description: Human-readable description.
        enabled: If False, the rule is skipped during evaluation.
        event: Trigger type — "on_change" (default), "on_timer", "on_sim_complete".
        body: List of conditions (AND semantics — all must match).
        head: List of actions executed in order if body matches.
        priority: Lower value = evaluated first.
        fire_once: If True, only fires once per unique trigger triple.
        max_fires: Maximum number of times to fire (0 = unlimited).
    """

    name: str
    description: str = ""
    enabled: bool = True
    event: str = "on_change"
    body: list[Condition] = field(default_factory=list)
    head: list[Action] = field(default_factory=list)
    priority: int = 0
    fire_once: bool = True
    max_fires: int = 0


# ── ProductionRuleEngine ────────────────────────────────────────────


class ProductionRuleEngine:
    """Event-driven forward-chaining production rule engine.

    Evaluates rules against a TripleStore. When a rule's body conditions
    all match, its head actions are executed.

    The engine subscribes to TripleStore.on_add() so rules fire
    automatically when new triples are added.
    """

    def __init__(self, store: TripleStore):
        self.store = store
        self.rules: list[ProductionRule] = []
        self._fired_count: dict[str, int] = {}
        self._fired_signatures: dict[str, set] = {}
        self._add_listener: Optional[Callable] = None
        self._remove_listener: Optional[Callable] = None
        self._started: bool = False
        self._in_evaluate: int = 0  # re-entrant depth counter

    def add_rule(self, rule: ProductionRule) -> None:
        """Register a production rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def remove_rule(self, name: str) -> None:
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != name]
        self._fired_count.pop(name, None)
        self._fired_signatures.pop(name, None)

    def get_rule(self, name: str) -> Optional[ProductionRule]:
        for r in self.rules:
            if r.name == name:
                return r
        return None

    def start(self) -> None:
        """Subscribe to TripleStore events and start evaluating."""
        if self._started:
            return
        self._add_listener = self._on_triple_added
        self.store.on_add(self._add_listener)
        self._started = True
        self.evaluate()

    def stop(self) -> None:
        """Unsubscribe from TripleStore events."""
        self._started = False

    def evaluate(
        self,
        trigger_triple: Optional[Triple] = None,
        trigger_graph: Optional[str] = None,
    ) -> list[ActionResult]:
        """Evaluate all enabled rules and return fired actions.

        Re-entrant calls (from actions that add triples to the store during
        evaluation) are counted via ``_in_evaluate``. When depth exceeds
        ``MAX_EVAL_DEPTH``, the call is a no-op to prevent infinite loops.
        """
        MAX_EVAL_DEPTH = 10
        if self._in_evaluate >= MAX_EVAL_DEPTH:
            return []
        self._in_evaluate += 1
        results: list[ActionResult] = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            if rule.max_fires > 0 and self._fired_count.get(rule.name, 0) >= rule.max_fires:
                continue
            if rule.fire_once and trigger_triple is not None:
                sig = self._make_signature(rule, trigger_triple, trigger_graph)
                if sig in self._fired_signatures.get(rule.name, set()):
                    continue

            merged_bindings: dict[str, RDFNode] = {}
            all_matched = True
            for cond in rule.body:
                result = cond.evaluate(self.store, merged_bindings)
                if not result.matched:
                    all_matched = False
                    break
                merged_bindings.update(result.bindings)

            if not all_matched:
                continue

            if rule.fire_once and trigger_triple is not None:
                sig = self._make_signature(rule, trigger_triple, trigger_graph)
                self._fired_signatures.setdefault(rule.name, set()).add(sig)

            self._fired_count[rule.name] = self._fired_count.get(rule.name, 0) + 1

            for action in rule.head:
                try:
                    action_result = action.execute(self.store, merged_bindings)
                    results.append(action_result)
                except Exception as e:
                    results.append(ActionResult(
                        type(action).__name__, success=False, message=str(e),
                    ))

        self._in_evaluate -= 1
        return results

    def reset(self) -> None:
        """Reset all fired-counts and signatures (re-enables fire_once rules)."""
        self._fired_count.clear()
        self._fired_signatures.clear()

    # ── Internal ──────────────────────────────────────────────

    def _on_triple_added(self, triple: Triple, graph: str) -> None:
        """Called when a triple is added to the store."""
        self.evaluate(trigger_triple=triple, trigger_graph=graph)

    @staticmethod
    def _make_signature(
        rule: ProductionRule,
        triple: Triple,
        graph: Optional[str],
    ) -> str:
        """Create a unique signature for fire_once dedup."""
        return f"{rule.name}:{triple.spo}@{graph or 'default'}"


# ── Helpers ────────────────────────────────────────────────────────


def _rdf_equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, NamedNode):
        return a.iri == b.iri
    if isinstance(a, BlankNode):
        return a.id == b.id
    if isinstance(a, Literal):
        return (a.value == b.value and a.datatype == b.datatype and a.lang_tag == b.lang_tag)
    return a == b
