"""Inference engine with RDFS and OWL RL rule sets.

Forward-chaining rule engine with opinion propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from dynafx.core.models import Opinion
from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
)
from dynafx.knowledge.store import TripleStore

# ── Namespace constants ──────────────────────────────────────────

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"

RDF_TYPE = NamedNode(f"{RDF_NS}type")
RDFS_DOMAIN = NamedNode(f"{RDFS_NS}domain")
RDFS_RANGE = NamedNode(f"{RDFS_NS}range")
RDFS_SUBCLASS_OF = NamedNode(f"{RDFS_NS}subClassOf")
RDFS_SUBPROPERTY_OF = NamedNode(f"{RDFS_NS}subPropertyOf")
OWL_INVERSE_OF = NamedNode(f"{OWL_NS}inverseOf")
OWL_SYMMETRIC_PROPERTY = NamedNode(f"{OWL_NS}SymmetricProperty")
OWL_TRANSITIVE_PROPERTY = NamedNode(f"{OWL_NS}TransitiveProperty")
OWL_FUNCTIONAL_PROPERTY = NamedNode(f"{OWL_NS}FunctionalProperty")
OWL_SAME_AS = NamedNode(f"{OWL_NS}sameAs")


# ── Variable representation ──────────────────────────────────────


@dataclass(frozen=True)
class Var:
    """A rule variable (?x, ?p, ?c, etc.)."""
    name: str

    def __repr__(self) -> str:
        return f"?{self.name}"


# ── Inference pattern (supports variables) ───────────────────────


@dataclass(frozen=True)
class InferencePattern:
    """Triple pattern where positions can be RDFNode, Var, or None."""
    subject: Optional[Any] = None
    predicate: Optional[Any] = None
    object_: Optional[Any] = None


# ── Rule ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Rule:
    """An inference rule with body and head patterns.

    Attributes:
        name: Human-readable rule name.
        head: List of patterns to infer (conclusion).
        body: List of patterns to match (premise).
        confidence_fn: How to combine body opinions:
            "min" - minimum belief, maximum uncertainty
            "product" - product-based combination
            "average" - weighted mean of (b, d, u)
    """
    name: str
    head: list[InferencePattern]
    body: list[InferencePattern]
    confidence_fn: str = "min"


# ── RDFS Rule Set ────────────────────────────────────────────────


def rdfs_rules() -> list[Rule]:
    """Return the 7 RDFS inference rules.

    1.  (?x ?p ?o) ∧ (?p rdfs:domain ?c) → (?x rdf:type ?c)
    2.  (?x ?p ?o) ∧ (?p rdfs:range ?c) → (?o rdf:type ?c)
    3.  (?x rdfs:subClassOf ?y) ∧ (?y rdfs:subClassOf ?z) → (?x rdfs:subClassOf ?z)
    4.  (?x rdf:type ?c) ∧ (?c rdfs:subClassOf ?d) → (?x rdf:type ?d)
    5.  (?p rdfs:subPropertyOf ?q) ∧ (?q rdfs:subPropertyOf ?r) → (?p rdfs:subPropertyOf ?r)
    6.  (?x ?p ?o) ∧ (?p rdfs:subPropertyOf ?q) → (?x ?q ?o)
    7.  (?x rdfs:subClassOf ?y) → (?x rdf:type ?y)
    """
    x = Var("x")
    p = Var("p")
    o = Var("o")
    c = Var("c")
    d = Var("d")
    q = Var("q")
    r = Var("r")
    y = Var("y")
    z = Var("z")

    return [
        Rule(
            name="rdfs-domain",
            head=[InferencePattern(x, RDF_TYPE, c)],
            body=[InferencePattern(x, p, o), InferencePattern(p, RDFS_DOMAIN, c)],
        ),
        Rule(
            name="rdfs-range",
            head=[InferencePattern(o, RDF_TYPE, c)],
            body=[InferencePattern(x, p, o), InferencePattern(p, RDFS_RANGE, c)],
        ),
        Rule(
            name="rdfs-subclass-trans",
            head=[InferencePattern(x, RDFS_SUBCLASS_OF, z)],
            body=[
                InferencePattern(x, RDFS_SUBCLASS_OF, y),
                InferencePattern(y, RDFS_SUBCLASS_OF, z),
            ],
        ),
        Rule(
            name="rdfs-subclass-usage",
            head=[InferencePattern(x, RDF_TYPE, d)],
            body=[
                InferencePattern(x, RDF_TYPE, c),
                InferencePattern(c, RDFS_SUBCLASS_OF, d),
            ],
        ),
        Rule(
            name="rdfs-subproperty-trans",
            head=[InferencePattern(p, RDFS_SUBPROPERTY_OF, r)],
            body=[
                InferencePattern(p, RDFS_SUBPROPERTY_OF, q),
                InferencePattern(q, RDFS_SUBPROPERTY_OF, r),
            ],
        ),
        Rule(
            name="rdfs-subproperty-usage",
            head=[InferencePattern(x, q, o)],
            body=[
                InferencePattern(x, p, o),
                InferencePattern(p, RDFS_SUBPROPERTY_OF, q),
            ],
        ),
        Rule(
            name="rdfs-class-instance",
            head=[InferencePattern(x, RDF_TYPE, y)],
            body=[InferencePattern(x, RDFS_SUBCLASS_OF, y)],
        ),
    ]


# ── OWL RL Rule Set ──────────────────────────────────────────────


def owl_rl_rules() -> list[Rule]:
    """Return 4 OWL RL inference rules.

    1.  (?p owl:inverseOf ?q) ∧ (?x ?p ?y) → (?y ?q ?x)
    2.  (?p a owl:SymmetricProperty) ∧ (?x ?p ?y) → (?y ?p ?x)
    3.  (?p a owl:TransitiveProperty) ∧ (?x ?p ?y) ∧ (?y ?p ?z) → (?x ?p ?z)
    4.  (?p a owl:FunctionalProperty) ∧ (?x ?p ?y) ∧ (?x ?p ?z) → (?y owl:sameAs ?z)
    """
    p = Var("p")
    q = Var("q")
    x = Var("x")
    y = Var("y")
    z = Var("z")

    return [
        Rule(
            name="owl-inverse",
            head=[InferencePattern(y, q, x)],
            body=[
                InferencePattern(p, OWL_INVERSE_OF, q),
                InferencePattern(x, p, y),
            ],
        ),
        Rule(
            name="owl-symmetric",
            head=[InferencePattern(y, p, x)],
            body=[
                InferencePattern(p, RDF_TYPE, OWL_SYMMETRIC_PROPERTY),
                InferencePattern(x, p, y),
            ],
        ),
        Rule(
            name="owl-transitive",
            head=[InferencePattern(x, p, z)],
            body=[
                InferencePattern(p, RDF_TYPE, OWL_TRANSITIVE_PROPERTY),
                InferencePattern(x, p, y),
                InferencePattern(y, p, z),
            ],
        ),
        Rule(
            name="owl-functional",
            head=[InferencePattern(y, OWL_SAME_AS, z)],
            body=[
                InferencePattern(p, RDF_TYPE, OWL_FUNCTIONAL_PROPERTY),
                InferencePattern(x, p, y),
                InferencePattern(x, p, z),
            ],
        ),
    ]


# ── Rule Engine ──────────────────────────────────────────────────


class RuleEngine:
    """Forward-chaining rule engine.

    Applies rules iteratively until fixpoint or max_iterations.
    """

    def __init__(self, rules: Optional[list[Rule]] = None,
                 max_iterations: int = 10):
        self.rules: list[Rule] = list(rules) if rules else []
        self.max_iterations = max_iterations

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def clear_rules(self) -> None:
        self.rules.clear()

    def apply(self, store: TripleStore) -> int:
        """Forward-chain all rules against the store.

        Returns the total number of new triples inferred.
        """
        total_new = 0
        for iteration in range(self.max_iterations):
            inferred: list[Triple] = []
            for rule in self.rules:
                bindings = self._eval_body(rule.body, store)
                for binding, body_opinions in bindings:
                    for head_pat in rule.head:
                        t = self._instantiate(head_pat, binding, rule, body_opinions)
                        if t is not None:
                            inferred.append(t)
            added = 0
            for t in inferred:
                # Check if triple already exists (by identity)
                pat = TriplePattern(t.subject, t.predicate, t.object_)
                if pat not in store:
                    store.add(t)
                    added += 1
            total_new += added
            if added == 0:
                break
        return total_new

    # ── Body evaluation ───────────────────────────────────────

    def _eval_body(
        self,
        body: list[InferencePattern],
        store: TripleStore,
    ) -> list[tuple[dict[str, RDFNode], list[Optional[Opinion]]]]:
        """Evaluate body patterns against the store.

        Returns list of (binding, opinions) tuples where binding maps
        variable names to RDFNodes and opinions is a parallel list of
        opinions from each body pattern match.
        """
        result: list[tuple[dict[str, RDFNode], list[Optional[Opinion]]]] = [({}, [])]
        for pat in body:
            next_result: list[tuple[dict[str, RDFNode], list[Optional[Opinion]]]] = []
            for binding, opin_list in result:
                resolved = self._resolve_pat(pat, binding)
                for triple in store.triples(resolved):
                    new_binding = self._extract_bindings(pat, triple, binding)
                    if new_binding is not None:
                        new_opinions = list(opin_list) + [triple.opinion]
                        next_result.append((new_binding, new_opinions))
            result = next_result
            if not result:
                break
        return result

    @staticmethod
    def _resolve_pat(
        pat: InferencePattern,
        binding: dict[str, RDFNode],
    ) -> TriplePattern:
        """Resolve an inference pattern by substituting known bindings."""
        s = pat.subject
        p = pat.predicate
        o = pat.object_

        if isinstance(s, Var) and s.name in binding:
            s = binding[s.name]
        elif isinstance(s, Var):
            s = None
        if isinstance(p, Var) and p.name in binding:
            p = binding[p.name]
        elif isinstance(p, Var):
            p = None
        if isinstance(o, Var) and o.name in binding:
            o = binding[o.name]
        elif isinstance(o, Var):
            o = None

        return TriplePattern(s, p, o)

    @staticmethod
    def _extract_bindings(
        pat: InferencePattern,
        triple: Triple,
        current_binding: dict[str, RDFNode],
    ) -> Optional[dict[str, RDFNode]]:
        """Extract new bindings from a matching triple, checking consistency."""
        new_binding = dict(current_binding)
        for pos_name, pos_val in [("subject", pat.subject),
                                   ("predicate", pat.predicate),
                                   ("object_", pat.object_)]:
            triple_val = getattr(triple, pos_name)
            if isinstance(pos_val, Var):
                name = pos_val.name
                if name in new_binding:
                    if not _rdf_equal(new_binding[name], triple_val):
                        return None
                else:
                    new_binding[name] = triple_val
        return new_binding

    # ── Head instantiation ────────────────────────────────────

    @staticmethod
    def _instantiate(
        pat: InferencePattern,
        binding: dict[str, RDFNode],
        rule: Rule,
        body_opinions: list[Optional[Opinion]],
    ) -> Optional[Triple]:
        """Instantiate a head pattern with variable bindings."""
        s = pat.subject
        p = pat.predicate
        o = pat.object_

        if isinstance(s, Var):
            s = binding.get(s.name)
        if isinstance(p, Var):
            p = binding.get(p.name)
        if isinstance(o, Var):
            o = binding.get(o.name)

        if s is None or p is None or o is None:
            return None
        if not isinstance(s, (NamedNode, BlankNode)):
            return None
        if isinstance(o, Literal):
            pass  # literals are allowed as objects
        elif not isinstance(o, (NamedNode, BlankNode)):
            return None
        if not isinstance(p, NamedNode):
            return None

        opin = propagate_opinion(rule, body_opinions)
        return Triple(s, p, o, opinion=opin)

    # ── Queries ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.rules)


# ── Opinion propagation ──────────────────────────────────────────


def propagate_opinion(
    rule: Rule,
    body_opinions: list[Optional[Opinion]],
) -> Optional[Opinion]:
    """Compute the opinion for an inferred triple from body opinions."""
    valid = [o for o in body_opinions if o is not None]
    if not valid:
        return Opinion(0.5, 0.3, 0.2)

    if rule.confidence_fn == "min":
        return _propagate_min(valid)
    if rule.confidence_fn == "product":
        return _propagate_product(valid)
    if rule.confidence_fn == "average":
        return _propagate_average(valid)
    return _propagate_min(valid)


def _propagate_min(opinions: list[Opinion]) -> Opinion:
    b = min(o.belief for o in opinions)
    d = max(o.disbelief for o in opinions)  # worst-case disbelief
    u = max(o.uncertainty for o in opinions)
    return Opinion(b, d, u)


def _propagate_product(opinions: list[Opinion]) -> Opinion:
    prod_bu = 1.0
    prod_u = 1.0
    for o in opinions:
        prod_bu *= (o.belief + o.uncertainty)
        prod_u *= o.uncertainty
    b = prod_bu - prod_u
    u = prod_u
    d = 1.0 - b - u
    return Opinion(max(0.0, min(1.0, b)),
                   max(0.0, min(1.0, d)),
                   max(0.0, min(1.0, u)))


def _propagate_average(opinions: list[Opinion]) -> Opinion:
    n = len(opinions)
    b = sum(o.belief for o in opinions) / n
    d = sum(o.disbelief for o in opinions) / n
    u = sum(o.uncertainty for o in opinions) / n
    return Opinion(b, d, u)


def _rdf_equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, NamedNode):
        return a.iri == b.iri
    if isinstance(a, BlankNode):
        return a.id == b.id
    if isinstance(a, Literal):
        return (a.value == b.value and
                a.datatype == b.datatype and
                a.lang_tag == b.lang_tag)
    return a == b
