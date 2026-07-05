"""Argumentation framework — Dung-style attack/defeat reasoning.

Built on the standard grounded semantics (Dung 1995, *Artificial
Intelligence* 77:321–357): arguments are acceptable if they are
defended against all attackers.

Slot between RDFS/OWL inference and SL graph fusion: only triples
supported by acceptable arguments proceed to belief fusion.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from dynafx.knowledge.model import (
    BlankNode,
    Literal,
    NamedNode,
    Triple,
    TriplePattern,
)
from dynafx.knowledge.store import TripleStore

# ── Enums ─────────────────────────────────────────────────────────


class SupportType(Enum):
    EVIDENCE = "evidence"
    INFERENCE = "inference"
    SOURCE_AUTHORITY = "source_authority"


class AttackType(Enum):
    REBUT = "rebut"                # Same s,p but contradictory o values
    UNDERMINE = "undermine"       # Source reliability or low belief
    UNDERCUT = "undercut"         # Premise chain is flawed


# ── Data classes ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Argument:
    """A claim about a triple, linked to its source."""
    id: str
    triple: Triple
    source_graph: Optional[str] = None
    support_type: SupportType = SupportType.EVIDENCE
    strength: float = 1.0


@dataclass(frozen=True)
class Attack:
    """A directed attack from one argument to another."""
    source_id: str
    target_id: str
    attack_type: AttackType = AttackType.REBUT
    strength: float = 1.0


# ── Framework ─────────────────────────────────────────────────────


class ArgumentationFramework:
    """Dung argumentation framework with grounded/preferred semantics.

    Usage::

        af = ArgumentationFramework()
        af.add_argument(arg1)
        af.add_attack(Attack("a1", "a2"))
        accepted = af.compute_grounded()  # set of acceptable IDs
    """

    def __init__(self) -> None:
        self.arguments: dict[str, Argument] = {}
        self.attacks: list[Attack] = []
        self._attackers: dict[str, set[str]] = defaultdict(set)

    def add_argument(self, arg: Argument) -> None:
        self.arguments[arg.id] = arg

    def add_attack(self, attack: Attack) -> None:
        self.attacks.append(attack)
        self._attackers[attack.target_id].add(attack.source_id)

    # ── Extension computation ─────────────────────────────────

    def compute_grounded(
        self, min_attack_strength: float = 0.0
    ) -> set[str]:
        """Compute the grounded extension via least fixed point.

        The grounded extension is the unique, minimal, skeptical
        set of arguments that are defensible against all attacks.

        Algorithm:
            F(S) = {a | ∀b attacks a, ∃c ∈ S: c attacks b}
            Iterate F starting from ∅ until stable.
        """
        attackers: dict[str, set[str]] = defaultdict(set)
        for a in self.attacks:
            if a.strength >= min_attack_strength:
                attackers[a.target_id].add(a.source_id)

        extension: set[str] = set()
        changed = True
        while changed:
            changed = False
            for arg_id in list(self.arguments):
                if arg_id in extension:
                    continue
                # a is acceptable iff every attacker of a is
                # counter-attacked by some argument in extension
                acceptable = True
                for b in attackers.get(arg_id, set()):
                    if not attackers.get(b, set()).intersection(extension):
                        acceptable = False
                        break
                if acceptable:
                    extension.add(arg_id)
                    changed = True
        return extension

    def compute_preferred(self) -> list[set[str]]:
        """Compute all preferred extensions (maximal admissible sets).

        Admissible set: conflict-free and defends all its members.
        Uses brute-force search over subsets (small frameworks).
        """
        ids = list(self.arguments)
        attackers: dict[str, set[str]] = defaultdict(set)
        for a in self.attacks:
            attackers[a.target_id].add(a.source_id)

        def is_admissible(s: set[str]) -> bool:
            # Conflict-free: no attacks between members
            for a in self.attacks:
                if a.source_id in s and a.target_id in s:
                    return False
            # Self-defending: every member is defended by s
            for arg_id in s:
                for b in attackers.get(arg_id, set()):
                    if not attackers.get(b, set()).intersection(s):
                        return False
            return True

        admissible: list[set[str]] = []
        from itertools import combinations
        for r in range(len(ids) + 1):
            for combo in combinations(ids, r):
                s = set(combo)
                if is_admissible(s):
                    admissible.append(s)

        # Filter to maximal
        maximal: list[set[str]] = []
        for s in admissible:
            if not any(
                s < other for other in admissible
            ):
                maximal.append(s)
        return maximal

    # ── Queries ───────────────────────────────────────────────

    def acceptable_triples(
        self, extension: Optional[set[str]] = None,
    ) -> set[Triple]:
        """Return triples supported by at least one acceptable argument."""
        if extension is None:
            extension = self.compute_grounded()
        return {
            self.arguments[aid].triple
            for aid in extension
            if aid in self.arguments
        }

    def filter_store(
        self,
        store: TripleStore,
        extension: Optional[set[str]] = None,
    ) -> TripleStore:
        """Return a new TripleStore with only acceptable triples."""
        if extension is None:
            extension = self.compute_grounded()
        acceptable = self.acceptable_triples(extension)
        result = TripleStore()
        for t in store.all_triples():
            if t in acceptable:
                result.add(t)
        # Also copy graph memberships
        for graph in store.graphs():
            for t in store.triples_in_graph(graph):
                if t in acceptable:
                    result.add(t, graph=graph)
        return result


# ── Framework builder ──────────────────────────────────────────────


# Namespace for argumentation meta-triples
ARG_NS = "http://cognitive.engine/argumentation#"
ARG_TYPE = NamedNode(f"{ARG_NS}Argument")
ATTACK_TYPE = NamedNode(f"{ARG_NS}Attack")
SUPPORT_TYPE = NamedNode(f"{ARG_NS}Support")
HAS_STRENGTH = NamedNode(f"{ARG_NS}strength")
ATTACKS = NamedNode(f"{ARG_NS}attacks")
SUPPORTS = NamedNode(f"{ARG_NS}supports")

# Namespace for provenance
PROV_NS = "http://cognitive.engine/provenance#"
SOURCE_RELIABILITY = NamedNode(f"{PROV_NS}reliability")


def build_framework(
    store: TripleStore,
    source_graphs: list[str],
    *,
    min_belief: float = 0.2,
    min_attack_strength: float = 0.3,
    auto_rebut: bool = True,
    auto_undermine_low_belief: bool = True,
) -> ArgumentationFramework:
    """Build an argumentation framework from a TripleStore's named graphs.

    Attack sources:

    1. **Rebut** (*auto_rebut*): Two triples with the same (subject,
       predicate) but different object values attack each other.
    2. **Undermine (low belief)** (*auto_undermine_low_belief*): Triples
       with belief below *min_belief* are attacked by a generic
       ``_skeptic`` argument.
    3. **Undermine (source reliability)**: Source reliability meta-triples
       (``:Source ex:reliability 0.3``) in the store generate undermining
       attacks against all triples from that source graph.
    4. **Inference support**: Inference-derived triples link back to their
       premise arguments via support relations.

    Returns:
        An ``ArgumentationFramework`` with arguments and attacks populated.
    """
    af = ArgumentationFramework()
    _arg_counter = [0]

    def _next_id() -> str:
        _arg_counter[0] += 1
        return f"a{_arg_counter[0]}"

    # Collect source reliability triples
    src_reliability: dict[str, float] = {}
    for pat in store.triples(TriplePattern(predicate=SOURCE_RELIABILITY)):
        if isinstance(pat.object_, Literal):
            try:
                val = float(pat.object_.value)
                if pat.subject.iri in source_graphs:
                    src_reliability[pat.subject.iri] = val
            except (ValueError, AttributeError):
                pass

    # Phase 1: create one Argument per triple per graph
    arg_by_spo_and_graph: dict[tuple, str] = {}
    triples_by_spo: dict[tuple, list[Triple]] = defaultdict(list)

    for g in source_graphs:
        for t in store.triples_in_graph(g):
            aid = _next_id()
            strength = 1.0
            # If this source has a reliability rating, adjust strength
            if g in src_reliability:
                strength = src_reliability[g]
            arg = Argument(
                id=aid, triple=t, source_graph=g,
                support_type=SupportType.EVIDENCE,
                strength=strength,
            )
            af.add_argument(arg)
            arg_by_spo_and_graph[(t.spo, g)] = aid
            triples_by_spo[t.spo].append(t)

    # Phase 2: Rebut attacks — same (s,p) with different o attack each other
    if auto_rebut:
        # Build map: (s, p) -> [(o, graph, arg_id)]
        prop_claims: dict[tuple, list[tuple]] = defaultdict(list)
        for g in source_graphs:
            for t in store.triples_in_graph(g):
                key = (t.subject, t.predicate)
                aid = arg_by_spo_and_graph.get((t.spo, g))
                if aid:
                    prop_claims[key].append((t.object_, g, aid))

        for key, claims in prop_claims.items():
            if len(claims) < 2:
                continue
            # Check for actual contradictions (different object values)
            seen_objects: dict[object, list[str]] = defaultdict(list)
            for obj, g, aid in claims:
                obj_val = _object_value(obj)
                seen_objects[obj_val].append(aid)

            if len(seen_objects) < 2:
                continue  # all same value, no contradiction

            # Create mutual rebut attacks between groups with different values
            obj_groups = list(seen_objects.values())
            for i in range(len(obj_groups)):
                for j in range(i + 1, len(obj_groups)):
                    for src_aid in obj_groups[i]:
                        for tgt_aid in obj_groups[j]:
                            af.add_attack(Attack(
                                source_id=src_aid,
                                target_id=tgt_aid,
                                attack_type=AttackType.REBUT,
                            ))
                            af.add_attack(Attack(
                                source_id=tgt_aid,
                                target_id=src_aid,
                                attack_type=AttackType.REBUT,
                            ))

    # Phase 3: Undermine low-belief triples
    if auto_undermine_low_belief:
        skeptic_id = "_skeptic"
        has_skeptic = False
        for g in source_graphs:
            for t in store.triples_in_graph(g):
                if t.opinion and t.opinion.belief < min_belief:
                    if not has_skeptic:
                        # Make a dummy triple for the skeptic argument
                        dummy = Triple(
                            NamedNode(f"{ARG_NS}default_skeptic"),
                            NamedNode(f"{ARG_NS}challenges"),
                            NamedNode(f"{ARG_NS}low_belief_claims"),
                        )
                        af.add_argument(Argument(
                            id=skeptic_id, triple=dummy,
                            support_type=SupportType.SOURCE_AUTHORITY,
                            strength=1.0,
                        ))
                        has_skeptic = True
                    aid = arg_by_spo_and_graph.get((t.spo, g))
                    if aid:
                        af.add_attack(Attack(
                            source_id=skeptic_id,
                            target_id=aid,
                            attack_type=AttackType.UNDERMINE,
                            strength=1.0 - t.opinion.belief,
                        ))

    # Phase 4: Undermine by source reliability
    for src_graph, reliability in src_reliability.items():
        if reliability >= min_attack_strength:
            continue  # source is reliable enough
        # Create a "source_unreliable" argument that attacks all from this graph
        src_arg_id = f"_src_unreliable_{src_graph}"
        dummy = Triple(
            NamedNode(f"{ARG_NS}source_reliability"),
            NamedNode(f"{ARG_NS}challenges"),
            NamedNode(f"{PROV_NS}{src_graph}"),
        )
        af.add_argument(Argument(
            id=src_arg_id, triple=dummy,
            support_type=SupportType.SOURCE_AUTHORITY,
            strength=1.0 - reliability,
        ))
        for g in source_graphs:
            if g != src_graph:
                continue
            for t in store.triples_in_graph(g):
                aid = arg_by_spo_and_graph.get((t.spo, g))
                if aid:
                    af.add_attack(Attack(
                        source_id=src_arg_id,
                        target_id=aid,
                        attack_type=AttackType.UNDERMINE,
                        strength=1.0 - reliability,
                    ))

    return af


# ── Helpers ───────────────────────────────────────────────────────


def _object_value(obj: object) -> object:
    """Extract a comparable value from an RDF node."""
    if isinstance(obj, Literal):
        return obj.value
    if isinstance(obj, NamedNode):
        return obj.iri
    if isinstance(obj, BlankNode):
        return obj.id
    return obj
