"""Relate operator — first-principles relationship mapper.

Three universal axioms (domain-agnostic):
  1. Epistemic Opposition — subjective logic opinion_conflict + numeric opposition
  2. Type Compatibility  — loaded TBox valid_edges triples
  3. Lexical Signals    — causal/temporal trigger words, shared entities

After axiom-based edge creation, a concept-driven step consults the
ConceptRegistry to create edges for same-concept and same-parent-concept
pairs based on their temporal semantics and provenance weights.

Finally, Dung's Argumentation is applied as post-processing with
provenance-aware weighting to filter edges by acceptability.

Cosine similarity is intentionally excluded from edge creation — it measures
topical overlap, not logical support (e.g., "40%" and "60%" share words but
are logically opposed). Cosine is preserved for node matching in other
operators (compare, analogy, abduce, induce, attention, align).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional
from uuid import UUID

from dynafx.core.concept import ConceptDef, ConceptRegistry, ConflictType, TemporalSemantics
from dynafx.core.math import cumulative_fusion, dung_semantics, opinion_conflict
from dynafx.core.models import EDGE_BFO_CONSTRAINTS, Edge, EdgeType, Node, Opinion
from dynafx.core.state import State
from dynafx.tbox.loader import load_tbox, TBox

logger = logging.getLogger(__name__)

# ── Universal lexical signal patterns (English function words only) ──

_TEMPORAL_WORDS: frozenset[str] = frozenset({
    "before", "after", "during", "while", "subsequently",
    "then", "later", "earlier", "prior", "following",
    "simultaneously", "concurrently", "afterwards",
})

_CAUSAL_WORDS: frozenset[str] = frozenset({
    "because", "causes", "caused", "leading", "leads to",
    "results in", "resulted in", "due to", "therefore",
    "hence", "consequently", "thus", "accordingly",
})

_DEONTIC_WORDS: frozenset[str] = frozenset({
    "must", "shall", "required", "obligated", "duty",
    "demand", "require", "mandatory", "forbidden", "prohibited",
})

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(%|percent|dollars?|\$|units?|days?|hours?|months?|years?|km|m|gb|mb)?",
    re.IGNORECASE,
)

_SHARED_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# Regex patterns for extracting asserted values from identity-slot nodes
_NAME_VALUE_RE = re.compile(r"(?:my|his|her|its|the)\s+name\s+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.IGNORECASE)
_EMAIL_VALUE_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_VALUE_RE = re.compile(r"\+?\d[\d\s()-]{7,}")
_ID_VALUE_RE = re.compile(r"(?:id|number|code)[:\s]+(\S+)", re.IGNORECASE)


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(w.lower().strip(".,!?;:\"'()[]{}") for w in text.split())


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER_RE.findall(text)]


def _extract_numbers_with_units(text: str) -> list[tuple[float, str]]:
    """Extract (value, unit) pairs from text. Unit is normalized to lowercase."""
    results = []
    for match in _UNIT_RE.finditer(text):
        value = float(match.group(1))
        unit = (match.group(2) or "").lower()
        if unit in ("$", "dollar", "dollars"):
            unit = "currency"
        elif unit in ("%", "percent"):
            unit = "percentage"
        elif unit in ("unit", "units"):
            unit = "count"
        results.append((value, unit))
    return results


def _has_shared_proper_nouns(text_a: str, text_b: str) -> bool:
    entities_a = set(_SHARED_ENTITY_RE.findall(text_a))
    entities_b = set(_SHARED_ENTITY_RE.findall(text_b))
    shared = entities_a & entities_b
    shared -= {"The", "A", "An", "This", "That", "It", "I", "He", "She", "They"}
    return len(shared) > 0


def _extract_asserted_value(text: str, concept_name: str) -> Optional[str]:
    """Extract the asserted value from a node's text for identity-slot concepts.

    For PERSON_NAME: extracts the name (e.g., "Alice" from "My name Alice")
    For EMAIL: extracts the email address
    For PHONE: extracts the phone number
    For ID codes: extracts the code after "id:" or "number:"
    """
    if concept_name in ("PERSON_NAME", "NAME"):
        m = _NAME_VALUE_RE.search(text)
        if m:
            return m.group(1).strip()
        # Fallback: extract capitalized words after "name"
        words = text.split()
        for i, w in enumerate(words):
            if w.lower() == "name" and i + 1 < len(words):
                return " ".join(words[i + 1:])
        return None

    if concept_name in ("EMAIL", "EMAIL_ADDRESS"):
        m = _EMAIL_VALUE_RE.search(text)
        return m.group(0) if m else None

    if concept_name in ("PHONE", "PHONE_NUMBER"):
        m = _PHONE_VALUE_RE.search(text)
        return m.group(0) if m else None

    if concept_name in ("ID_CODE", "IDENTIFIER"):
        m = _ID_VALUE_RE.search(text)
        return m.group(1) if m else None

    return None


# ── Priority ordering for edge types ──────────────────────────────

_EDGE_PRIORITY: dict[EdgeType, int] = {
    # World-model edges (highest priority)
    EdgeType.CAUSES:        100,
    EdgeType.ENABLES:        95,
    EdgeType.PART_OF:        90,
    EdgeType.DEPENDS:        85,
    EdgeType.FLOWS_TO:       80,
    EdgeType.TRANSFORMS:     78,
    EdgeType.PRODUCES:       76,
    EdgeType.CONSUMES:       74,
    EdgeType.USES:           72,
    EdgeType.HAS_ATTRIBUTE:  70,
    EdgeType.HAS_GOAL:       68,
    EdgeType.INTENDS:        66,
    EdgeType.KNOWS:          64,
    EdgeType.COMMUNICATED:   62,
    EdgeType.PREFERS:        60,
    # Structural edges
    EdgeType.CITES:          50,
    EdgeType.ASSOCIATED_WITH: 45,
    EdgeType.SIMILAR:        40,
    EdgeType.LOCATED_AT:     38,
    EdgeType.EMPLOYED_BY:    36,
    EdgeType.CONTACT_OF:     34,
    # Argumentation edges (lowest priority)
    EdgeType.CONTRADICTS:    30,
    EdgeType.REBUTS:         28,
    EdgeType.SUPPORTS:       26,
    EdgeType.INFERS:         24,
    EdgeType.ATTACKS:        22,
    EdgeType.JUSTIFIES:      20,
    EdgeType.EVIDENCE:       18,
    EdgeType.QUALIFIES:      16,
    EdgeType.DIRECT:         14,
    EdgeType.CIRCUMSTANTIAL: 12,
    EdgeType.HEARSAY:        10,
    EdgeType.SUPPORT:        8,
    EdgeType.TEMPORAL:       6,
}

# ── Universal type compatibility (derived from NodeType semantics) ──

_TYPE_PATTERNS: dict[tuple[str, str], list[tuple[str, float]]] = {
    # World-model patterns
    ("AGENT", "PROCESS"):      [("ENABLES", 0.8), ("INTENDS", 0.7)],
    ("AGENT", "GOAL"):         [("HAS_GOAL", 0.9)],
    ("AGENT", "ACTION"):       [("INTENDS", 0.8)],
    ("AGENT", "BELIEF"):       [("KNOWS", 0.7)],
    ("AGENT", "KNOWLEDGE"):    [("KNOWS", 0.8)],
    ("AGENT", "RESOURCE"):     [("USES", 0.7)],
    ("PROCESS", "STATE"):      [("CAUSES", 0.8), ("TRANSFORMS", 0.7)],
    ("PROCESS", "RESOURCE"):   [("PRODUCES", 0.7), ("CONSUMES", 0.6)],
    ("STATE", "PROPERTY"):     [("HAS_ATTRIBUTE", 0.8)],
    ("STATE", "STATE"):        [("TRANSFORMS", 0.7), ("CAUSES", 0.6)],
    ("DECISION", "ACTION"):    [("CAUSES", 0.8), ("INTENDS", 0.7)],
    ("ACTION", "STATE"):       [("CAUSES", 0.7)],
    ("ACTION", "RESOURCE"):    [("PRODUCES", 0.6), ("CONSUMES", 0.6)],
    ("GOAL", "ACTION"):        [("ENABLES", 0.7)],
    ("CONSTRAINT", "ACTION"):  [("ENABLES", 0.6)],
    ("CONSTRAINT", "PROCESS"): [("ENABLES", 0.6)],
    ("OBSERVATION", "STATE"):  [("SUPPORTS", 0.6)],
    # Argumentation patterns (preserved)
    ("COUNTERCLAIM", "CLAIM"):    [("ATTACKS", 0.8), ("CONTRADICTS", 0.7)],
    ("CLAIM", "COUNTERCLAIM"):    [("REBUTS", 0.6)],
    ("EVIDENCE", "CLAIM"):        [("SUPPORTS", 0.85), ("EVIDENCE", 0.8)],
    ("EVIDENCE", "COUNTERCLAIM"): [("ATTACKS", 0.7)],
    ("RULE", "EVENT"):            [("QUALIFIES", 0.7)],
    ("EVENT", "RULE"):            [("CONTRADICTS", 0.75), ("SUPPORTS", 0.5)],
    ("AXIOM", "CLAIM"):           [("INFERS", 0.9)],
    ("CLAIM", "JUSTIFICATION"):   [("JUSTIFIES", 0.8)],
    ("HYPOTHESIS", "EVIDENCE"):   [("SUPPORTS", 0.7)],
    ("EVIDENCE", "HYPOTHESIS"):   [("SUPPORTS", 0.6)],
    ("CONDITION", "ACTION"):      [("ENABLES", 0.7)],
    ("OBSERVATION", "HYPOTHESIS"): [("SUPPORTS", 0.6)],
    ("CONCEPT", "ENTITY"):        [("PART_OF", 0.7)],
}


# Edge types where the relation is symmetric: if A→B holds, B→A also holds.
_SYMMETRIC_EDGES: frozenset[EdgeType] = frozenset({
    EdgeType.CONTRADICTS, EdgeType.REBUTS, EdgeType.SUPPORTS,
    EdgeType.SIMILAR, EdgeType.ASSOCIATED_WITH,
})


class RelateOperator:
    """First-principles relationship mapper.

    Evaluates every pair of nodes against 3 universal axioms, then
    applies concept-driven edge creation and Dung's Argumentation
    as post-processing.
    """
    name = "relate"

    # Axiom configuration defaults
    SIMILARITY_THRESHOLD: float = 0.6
    NUMERIC_RATIO_THRESHOLD: float = 0.5
    OPINION_CONFLICT_THRESHOLD: float = 0.6
    MAX_EDGES_PER_NODE: int = 5

    def __call__(
        self,
        state: State,
        similarity_threshold: Optional[float] = None,
        max_edges_per_node: Optional[int] = None,
        **kwargs,
    ) -> State:
        nodes = list(state.graph.nodes.items())
        if len(nodes) < 2:
            state.record(self.name, "Fewer than 2 nodes — no edges to create")
            return state

        sim_th = similarity_threshold if similarity_threshold is not None else self.SIMILARITY_THRESHOLD
        max_per = max_edges_per_node if max_edges_per_node is not None else self.MAX_EDGES_PER_NODE

        # Pre-load TBox for Axiom 2
        tbox = load_tbox()

        # Snapshot existing edges for extraction-layer preservation
        existing_edges = dict(state.graph.edges)

        # Each pair maps to a LIST of edges (one per axiom that fires).
        # This prevents cross-axiom suppression: e.g. CAUSES + CONTRADICTS
        # can coexist for the same pair.
        scored_candidates: dict[tuple[UUID, UUID], list[tuple[EdgeType, float, Opinion]]] = defaultdict(list)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                nid_a, node_a = nodes[i]
                nid_b, node_b = nodes[j]

                edges_a_to_b = self._evaluate_pair(
                    node_a, node_b, tbox, sim_th,
                )
                for edge_type, weight, opinion in edges_a_to_b:
                    scored_candidates[(nid_a, nid_b)].append((edge_type, weight, opinion))
                    # Symmetric edges: also create the reverse direction
                    if edge_type in _SYMMETRIC_EDGES:
                        scored_candidates[(nid_b, nid_a)].append((edge_type, weight, opinion))

        # Concept-driven edge creation: creates bidirectional REBUTS/SUPPORTS
        # for symmetric conflicts, and may remove NO_CONFLICT pairs.
        self._apply_concept_axiom(scored_candidates, nodes, state.concepts)

        # Flatten to individual edges for cap and creation
        flat: list[tuple[UUID, UUID, EdgeType, float, Opinion]] = []
        for (src, tgt), edges in scored_candidates.items():
            for etype, weight, opinion in edges:
                flat.append((src, tgt, etype, weight, opinion))

        capped = self._cap_candidates(flat, max_per)

        # Create edges, preserving extraction-layer edges
        existing_keys: set[tuple[UUID, UUID, EdgeType]] = {
            (e.source_id, e.target_id, e.type) for e in state.graph.edges.values()
        }
        created = 0
        for src_id, tgt_id, etype, weight, opinion in capped:
            key = (src_id, tgt_id, etype)
            if key in existing_keys:
                # Edge already exists — merge opinions via cumulative fusion
                for edge in state.graph.edges.values():
                    if (edge.source_id, edge.target_id, edge.type) == key:
                        if edge.opinion and opinion:
                            edge.opinion = cumulative_fusion(
                                (edge.opinion.belief, edge.opinion.disbelief,
                                 edge.opinion.uncertainty, edge.opinion.prior),
                                (opinion.belief, opinion.disbelief,
                                 opinion.uncertainty, opinion.prior),
                            )
                        edge.weight = max(edge.weight, weight)
                        break
                continue
            existing_keys.add(key)
            edge = Edge(
                source_id=src_id,
                target_id=tgt_id,
                type=etype,
                weight=weight,
                opinion=opinion,
            )
            state.graph.edges[edge.id] = edge
            created += 1

        # Dung's Argumentation post-processing with provenance-aware weighting
        self._apply_dung_postprocessing(state)

        state.record(
            self.name,
            f"Evaluated {len(nodes)} nodes ({len(nodes)*(len(nodes)-1)//2} pairs), "
            f"created {created} edges",
        )
        return state

    # ── Private helpers ────────────────────────────────────────────

    def _evaluate_pair(
        self,
        a: Node,
        b: Node,
        tbox: TBox,
        sim_th: float,
    ) -> list[tuple[EdgeType, float, Opinion]]:
        """Score all axioms for a pair, return ALL edges that fire (one per axiom).

        Returns a list — multiple axioms can contribute independent edges
        to the same pair (e.g. CAUSES from lexical + CONTRADICTS from numeric).
        """
        scores: dict[EdgeType, float] = defaultdict(float)

        # Axiom 3: Lexical signals (highest specificity)
        self._apply_lexical_axiom(a.text, b.text, scores)

        # Axiom 1: Epistemic opposition
        self._apply_epistemic_axiom(a, b, scores, sim_th)

        # Axiom 2: Type compatibility (TBox + universal patterns)
        self._apply_type_axiom(a, b, tbox, scores)

        # BFO compatibility filter: reject edge types whose domain/range
        # don't match the source/target BFO categories.
        for etype in list(scores):
            if not self._check_bfo_compatibility(a, b, etype):
                del scores[etype]

        if not scores:
            return []

        # Return one edge per score, with SL-invariant opinions.
        # Multiple axioms contributing to the same edge type are folded
        # via max (already done in the scores dict).
        result: list[tuple[EdgeType, float, Opinion]] = []
        for etype, score in scores.items():
            belief = score * 0.5
            uncertainty = 1.0 - belief  # SL invariant: b + d + u = 1.0, d = 0
            opinion = Opinion(
                belief=belief,
                disbelief=0.0,
                uncertainty=uncertainty,
                prior=0.5,
            )
            result.append((etype, score, opinion))
        return result

    @staticmethod
    def _check_bfo_compatibility(a: Node, b: Node, etype: EdgeType) -> bool:
        constraints = EDGE_BFO_CONSTRAINTS.get(etype)
        if constraints is None:
            return True
        # If either node lacks a BFO category (unassigned), skip the check
        if a.bfo_category is None or b.bfo_category is None:
            return True
        src_allowed, tgt_allowed = constraints
        return a.bfo_category in src_allowed and b.bfo_category in tgt_allowed

    @staticmethod
    def _apply_lexical_axiom(text_a: str, text_b: str, scores: dict[EdgeType, float]) -> None:
        tokens_a = _tokenize(text_a)
        tokens_b = _tokenize(text_b)

        # Causal signal
        if _CAUSAL_WORDS & tokens_a or _CAUSAL_WORDS & tokens_b:
            scores[EdgeType.CAUSES] = max(scores[EdgeType.CAUSES], 0.7)

        # Temporal signal
        if _TEMPORAL_WORDS & tokens_a or _TEMPORAL_WORDS & tokens_b:
            scores[EdgeType.TEMPORAL] = max(scores[EdgeType.TEMPORAL], 0.65)

        # Shared proper nouns → CITES
        if _has_shared_proper_nouns(text_a, text_b):
            scores[EdgeType.CITES] = max(scores[EdgeType.CITES], 0.6)

    @staticmethod
    def _apply_epistemic_axiom(a: Node, b: Node, scores: dict[EdgeType, float], sim_th: float) -> None:
        # Numeric opposition: same-unit values with strongly diverging values
        nums_a = _extract_numbers_with_units(a.text)
        nums_b = _extract_numbers_with_units(b.text)
        if nums_a and nums_b:
            units_a = {unit for _, unit in nums_a}
            units_b = {unit for _, unit in nums_b}
            shared_units = units_a & units_b
            if shared_units:
                vals_a = [v for v, u in nums_a if u in shared_units]
                vals_b = [v for v, u in nums_b if u in shared_units]
                min_val = min(min(vals_a), min(vals_b))
                max_val = max(max(vals_a), max(vals_b))
                if max_val > 0:
                    ratio = min_val / max_val
                    if ratio < 0.3:
                        score = 0.5 + (1.0 - ratio) * 0.4
                        scores[EdgeType.CONTRADICTS] = max(scores[EdgeType.CONTRADICTS], score)

    @staticmethod
    def _apply_type_axiom(a: Node, b: Node, tbox: TBox, scores: dict[EdgeType, float]) -> None:
        # TBox valid_edges triples
        for src_type, etype_str, tgt_type in tbox.valid_edges:
            edge_type = EdgeType[etype_str.upper()]
            weight = tbox.edge_types.get(etype_str.upper(), 0.5)
            if a.type.name == src_type and b.type.name == tgt_type:
                scores[edge_type] = max(scores[edge_type], weight)
            elif a.type.name == tgt_type and b.type.name == src_type:
                scores[edge_type] = max(scores[edge_type], weight)

        # Universal type pattern table
        key_forward = (a.type.name, b.type.name)
        key_reverse = (b.type.name, a.type.name)
        for key, patterns in _TYPE_PATTERNS.items():
            if key == key_forward or key == key_reverse:
                for etype_str, weight in patterns:
                    etype = EdgeType[etype_str.upper()]
                    scores[etype] = max(scores.get(etype, 0.0), weight)

    @staticmethod
    def _apply_concept_axiom(
        candidates: dict[tuple[UUID, UUID], list[tuple[EdgeType, float, Opinion]]],
        nodes: list[tuple[UUID, Node]],
        registry: ConceptRegistry,
    ) -> None:
        """Concept-driven edge creation.

        For every pair of nodes that share a concept or parent concept,
        consult the ConceptRegistry to determine the appropriate edge type
        based on temporal semantics and provenance weight.

        Symmetric conflicts (REBUTS, SUPPORTS) are created in both directions.
        NO_CONFLICT removes any prior CONTRADICTS/REBUTS for the pair.
        """
        # Track which pairs were resolved as NO_CONFLICT so we can clean up
        no_conflict_pairs: set[tuple[UUID, UUID]] = set()

        # ── Pass 1: Same-concept pairs ──
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                nid_a, node_a = nodes[i]
                nid_b, node_b = nodes[j]
                concept_a = node_a.metadata.get("concept")
                concept_b = node_b.metadata.get("concept")

                if concept_a and concept_a == concept_b:
                    conflict_type = registry.resolve_conflict(concept_a)
                    try:
                        cd = registry.get(concept_a)
                    except KeyError:
                        continue

                    if conflict_type == ConflictType.HISTORICAL_CORRECTION:
                        val_a = _extract_asserted_value(node_a.text, concept_a)
                        val_b = _extract_asserted_value(node_b.text, concept_b)

                        if val_a and val_b and val_a != val_b:
                            w = cd.provenance_weight
                            belief = w * 0.5
                            opinion = Opinion(
                                belief=belief, disbelief=0.0,
                                uncertainty=1.0 - belief, prior=0.5,
                            )
                            # Bidirectional REBUTS
                            candidates[(nid_a, nid_b)].append((EdgeType.REBUTS, w, opinion))
                            candidates[(nid_b, nid_a)].append((EdgeType.REBUTS, w, opinion))
                        elif val_a and val_b and val_a == val_b:
                            w = cd.provenance_weight
                            belief = w * 0.4
                            opinion = Opinion(
                                belief=belief, disbelief=0.0,
                                uncertainty=1.0 - belief, prior=0.5,
                            )
                            # Bidirectional SUPPORTS
                            candidates[(nid_a, nid_b)].append((EdgeType.SUPPORTS, w * 0.8, opinion))
                            candidates[(nid_b, nid_a)].append((EdgeType.SUPPORTS, w * 0.8, opinion))
                        else:
                            # Values not extractable — assume conflict for identity concepts
                            w = cd.provenance_weight
                            belief = w * 0.5
                            opinion = Opinion(
                                belief=belief, disbelief=0.0,
                                uncertainty=1.0 - belief, prior=0.5,
                            )
                            candidates[(nid_a, nid_b)].append((EdgeType.REBUTS, w, opinion))
                            candidates[(nid_b, nid_a)].append((EdgeType.REBUTS, w, opinion))

                    elif conflict_type == ConflictType.COMPETING_CLAIMS:
                        opinion = Opinion(belief=0.35, disbelief=0.35, uncertainty=0.3, prior=0.5)
                        candidates[(nid_a, nid_b)].append((EdgeType.CONTRADICTS, 0.7, opinion))
                        candidates[(nid_b, nid_a)].append((EdgeType.CONTRADICTS, 0.7, opinion))

                    elif conflict_type == ConflictType.NO_CONFLICT:
                        no_conflict_pairs.add((nid_a, nid_b))
                        no_conflict_pairs.add((nid_b, nid_a))

        # ── Pass 2: Different concepts, same parent ──
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                nid_a, node_a = nodes[i]
                nid_b, node_b = nodes[j]
                ca = node_a.metadata.get("concept")
                cb = node_b.metadata.get("concept")
                if not ca or not cb or ca == cb:
                    continue
                try:
                    cda = registry.get(ca)
                    cdb = registry.get(cb)
                except KeyError:
                    continue
                if cda.parent is None or cdb.parent is None or cda.parent != cdb.parent:
                    continue
                parent_sem = registry.resolve_temporal_semantics(cda.parent)
                if parent_sem == TemporalSemantics.SUPERSEEDE_WITH_HISTORY:
                    w = max(cda.provenance_weight, cdb.provenance_weight)
                    belief = w * 0.5
                    opinion = Opinion(
                        belief=belief, disbelief=0.0,
                        uncertainty=1.0 - belief, prior=0.5,
                    )
                    candidates[(nid_a, nid_b)].append((EdgeType.REBUTS, w, opinion))
                    candidates[(nid_b, nid_a)].append((EdgeType.REBUTS, w, opinion))

        # ── Pass 3: Remove NO_CONFLICT pairs from candidates ──
        for pair in no_conflict_pairs:
            if pair in candidates:
                candidates[pair] = [
                    (et, w, op) for et, w, op in candidates[pair]
                    if et not in (EdgeType.CONTRADICTS, EdgeType.REBUTS)
                ]

    @staticmethod
    def _apply_dung_postprocessing(state: State) -> None:
        """Apply Dung's Argumentation as post-processing evaluator.

        After axioms 1-3 and concept-driven edge creation, Dung's
        acceptability semantics down-weights edges from non-acceptable
        arguments by x0.3 (soft demotion, not deletion — preserves
        traceability and audit trail).

        Provenance-aware: uses ConceptDef.provenance_weight to weight
        attacks — a high-provenance node is harder to defeat.
        """
        if not state.graph.edges:
            return

        # Build argumentation framework: attacks/supports relations
        arg_edges = [
            e for e in state.graph.edges.values()
            if e.type in (EdgeType.ATTACKS, EdgeType.CONTRADICTS, EdgeType.SUPPORTS,
                          EdgeType.REBUTS, EdgeType.INFERS, EdgeType.EVIDENCE)
        ]

        if not arg_edges:
            return

        # Build beliefs dict from node opinions
        beliefs: dict[UUID, float] = {}
        for nid, node in state.graph.nodes.items():
            if node.opinion:
                beliefs[nid] = node.opinion.belief
            else:
                beliefs[nid] = 0.5  # Default belief

        # Build attack graph: for each node, list of nodes that attack it
        attack_graph: dict[UUID, list[UUID]] = defaultdict(list)
        for edge in arg_edges:
            if edge.type in (EdgeType.ATTACKS, EdgeType.CONTRADICTS, EdgeType.REBUTS):
                attack_graph[edge.target_id].append(edge.source_id)

        # Build provenance weights from concept registry
        weights: dict[UUID, float] = {}
        for nid, node in state.graph.nodes.items():
            concept_name = node.metadata.get("concept")
            if concept_name:
                try:
                    cd = state.concepts.get(concept_name)
                    weights[nid] = cd.provenance_weight
                except KeyError:
                    weights[nid] = 1.0
            else:
                weights[nid] = 1.0

        # Get acceptable arguments with provenance weighting
        acceptable = dung_semantics(beliefs, attack_graph, weights=weights)

        # Down-weight edges from non-acceptable arguments
        for edge in arg_edges:
            if edge.source_id not in acceptable:
                edge.weight *= 0.3  # Reduce weight for non-acceptable sources

    @staticmethod
    def _cap_candidates(
        flat: list[tuple[UUID, UUID, EdgeType, float, Opinion]],
        max_per_node: int,
    ) -> list[tuple[UUID, UUID, EdgeType, float, Opinion]]:
        """Cap edges per node, preserving highest-priority edges."""
        if max_per_node <= 0:
            return []

        from collections import Counter
        edge_counts: Counter[UUID] = Counter()
        # Sort by priority desc, then weight desc
        sorted_edges = sorted(
            flat,
            key=lambda item: (_EDGE_PRIORITY.get(item[2], 0), item[3]),
            reverse=True,
        )
        result: list[tuple[UUID, UUID, EdgeType, float, Opinion]] = []
        for src, tgt, etype, weight, opinion in sorted_edges:
            if edge_counts[src] < max_per_node and edge_counts[tgt] < max_per_node:
                result.append((src, tgt, etype, weight, opinion))
                edge_counts[src] += 1
                edge_counts[tgt] += 1
        return result
