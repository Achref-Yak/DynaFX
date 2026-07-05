"""TripleStore — in-memory RDF triple store with SPO/POS/OSP indices and named graphs.

Supports:
    - O(1) pattern matching via 3-index prefix strategy
    - Named graphs per information source
    - Dedup: same (s,p,o) with higher-belief opinion replaces lower
    - Graph-level isolation, copy, and removal
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Optional

from dynafx.knowledge.model import (
    NamedNode,
    Triple,
    TriplePattern,
)

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
_RDF_TYPE = NamedNode(f"{RDF_NS}type")
_RDFS_SUBCLASS_OF = NamedNode(f"{RDFS_NS}subClassOf")
_RDFS_SUBPROPERTY_OF = NamedNode(f"{RDFS_NS}subPropertyOf")
_RDFS_DOMAIN = NamedNode(f"{RDFS_NS}domain")
_RDFS_RANGE = NamedNode(f"{RDFS_NS}range")


def _parse_inference_config(with_inference: Any) -> dict:
    """Normalise the with_inference parameter into a config dict.

    Accepts:
        - None → {}
        - "rdfs" → {"mode": "rdfs"}
        - dict  → passed through (e.g. {"mode": "rdfs", "min_belief": 0.5})
    """
    if with_inference is None:
        return {}
    if isinstance(with_inference, str):
        return {"mode": with_inference}
    if isinstance(with_inference, dict):
        return with_inference
    return {}


class TripleStore:
    """In-memory triple store with three-index prefix strategy.

    Indices:
        _spo: subject → predicate → object → set of triple_keys
        _pos: predicate → object → subject → set of triple_keys
        _osp: object → subject → predicate → set of triple_keys

    Each triple_key is the identity tuple (subject, predicate, object).
    """

    def __init__(self) -> None:
        # Triple storage by identity key
        self._triples: dict[tuple, Triple] = {}

        # Three nested indices for O(1) prefix matching
        self._spo: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self._pos: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self._osp: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

        # Named graphs: graph_name -> set of triple_keys
        self._graphs: dict[str, set[tuple]] = defaultdict(set)

        # RDFS closure caches (invalidated on write)
        self._rdfs_type_closure: Optional[dict[tuple, set[tuple]]] = None
        self._rdfs_subproperty_map: Optional[dict[NamedNode, set[NamedNode]]] = None

    # ── Event callbacks ───────────────────────────────────────────

    _on_add_callbacks: Optional[list[Callable[[Triple, str], None]]] = None
    _on_remove_callbacks: Optional[list[Callable[[TriplePattern, Optional[str]], None]]] = None
    _suppress_callbacks: int = 0  # > 0 means callbacks are skipped

    @contextmanager
    def suppress_callbacks(self) -> Iterator[None]:
        """Context manager to temporarily suppress on_add/on_remove callbacks."""
        self._suppress_callbacks += 1
        try:
            yield
        finally:
            self._suppress_callbacks -= 1

    def on_add(self, fn: Callable[[Triple, str], None]) -> None:
        """Register a callback fired when a triple is added.
        Callback receives (triple, graph_name).
        """
        if self._on_add_callbacks is None:
            self._on_add_callbacks = []
        self._on_add_callbacks.append(fn)

    def on_remove(self, fn: Callable[[TriplePattern, Optional[str]], None]) -> None:
        """Register a callback fired when triples are removed.
        Callback receives (pattern, graph_name_or_None).
        """
        if self._on_remove_callbacks is None:
            self._on_remove_callbacks = []
        self._on_remove_callbacks.append(fn)

    # ── Core mutation ───────────────────────────────────────────

    def add(self, triple: Triple, graph: str = "default") -> None:
        """Add a triple to the store, optionally in a named graph.

        If a triple with the same (s, p, o) already exists, the one with
        higher belief (or higher belief + certainty) is retained.
        """
        key = triple.spo

        # Check if key exists — keep max-belief opinion
        existing = self._triples.get(key)
        if existing is not None and existing.opinion and triple.opinion:
            if _opinion_strength(existing.opinion) >= _opinion_strength(triple.opinion):
                # Existing is stronger — just ensure it's in this graph
                self._graphs[graph].add(key)
                return

        # Index in all 3 permutations
        s, p, o = key
        self._spo[s][p][o].add(key)
        self._pos[p][o][s].add(key)
        self._osp[o][s][p].add(key)

        # Store triple
        self._triples[key] = triple
        self._graphs[graph].add(key)

        # Invalidate RDFS closures
        self._rdfs_type_closure = None
        self._rdfs_subproperty_map = None

        # Fire callbacks (unless suppressed)
        if self._suppress_callbacks == 0 and self._on_add_callbacks is not None:
            for fn in self._on_add_callbacks:
                fn(triple, graph)

    def remove(self, pattern: TriplePattern, graph: Optional[str] = None) -> int:
        """Remove triples matching a pattern, optionally in a graph.

        Returns the number of triples removed.
        """
        keys_to_remove = self._resolve_pattern(pattern, graph)
        count = len(keys_to_remove)

        for key in keys_to_remove:
            s, p, o = key
            self._spo[s][p][o].discard(key)
            if not self._spo[s][p][o]:
                del self._spo[s][p][o]
                if not self._spo[s][p]:
                    del self._spo[s][p]
                    if not self._spo[s]:
                        del self._spo[s]

            self._pos[p][o][s].discard(key)
            if not self._pos[p][o][s]:
                del self._pos[p][o][s]
                if not self._pos[p][o]:
                    del self._pos[p][o]
                    if not self._pos[p]:
                        del self._pos[p]

            self._osp[o][s][p].discard(key)
            if not self._osp[o][s][p]:
                del self._osp[o][s][p]
                if not self._osp[o][s]:
                    del self._osp[o][s]
                    if not self._osp[o]:
                        del self._osp[o]

            del self._triples[key]

            # Remove from all graphs
            for g in list(self._graphs.keys()):
                self._graphs[g].discard(key)

        # Invalidate RDFS closures
        self._rdfs_type_closure = None
        self._rdfs_subproperty_map = None

        # Fire callbacks
        if self._on_remove_callbacks is not None and count > 0:
            for fn in self._on_remove_callbacks:
                fn(pattern, graph)

        return count

    # ── Query ───────────────────────────────────────────────────

    def triples(
        self,
        pattern: TriplePattern,
        graph: Optional[str] = None,
        with_inference: Optional[str | dict] = None,
    ) -> Iterator[Triple]:
        """Iterate over triples matching a pattern, optionally in a graph.

        Args:
            pattern: The triple pattern to match.
            graph: Optional named graph to restrict to.
            with_inference: Inference config. Can be:
                - ``"rdfs"``: RDFS inference (subClassOf, subPropertyOf, domain, range).
                - ``dict``: e.g. ``{"mode": "rdfs", "min_belief": 0.5}``
                  ``min_belief`` filters results by opinion belief threshold.
                  ``min_confidence`` filters by ``belief + (1 - uncertainty)``.
        """
        cfg = _parse_inference_config(with_inference)
        keys = self._resolve_pattern(pattern, graph)

        if cfg.get("mode") == "rdfs":
            inferred = self._resolve_inferred(pattern, graph)
            seen: set = set(keys)
            for key in inferred:
                if key not in seen:
                    seen.add(key)
                    s, p, o = key
                    yield Triple(s, p, o)

        for key in keys:
            triple = self._triples[key]
            if cfg.get("min_belief") is not None or cfg.get("min_confidence") is not None:
                op = triple.opinion
                if op is not None:
                    if cfg.get("min_belief") is not None and op.belief < cfg["min_belief"]:
                        continue
                    if cfg.get("min_confidence") is not None:
                        conf = op.belief + (1.0 - op.uncertainty)
                        if conf < cfg["min_confidence"]:
                            continue
            yield triple

    def all_triples(self) -> Iterator[Triple]:
        """Iterate over every triple in the store (across all graphs)."""
        for t in self._triples.values():
            yield t

    def __contains__(self, pattern: TriplePattern) -> bool:
        """Check if any triple matches the pattern."""
        try:
            next(self.triples(pattern))
            return True
        except StopIteration:
            return False

    def __len__(self) -> int:
        return len(self._triples)

    # ── Named graph operations ─────────────────────────────────

    def graphs(self) -> list[str]:
        """Return the list of named graph names."""
        return list(self._graphs.keys())

    def triples_in_graph(self, graph: str) -> Iterator[Triple]:
        """Iterate over all triples in a named graph."""
        for key in self._graphs.get(graph, set()):
            yield self._triples[key]

    def remove_graph(self, graph: str) -> int:
        """Remove all triples in a named graph.

        Triples that exist in other graphs are preserved.
        Returns the number of triples removed from this graph.
        """
        keys = set(self._graphs.get(graph, set()))
        count = len(keys)
        self._graphs[graph] = set()
        if graph in self._graphs:
            del self._graphs[graph]

        # Remove triples that have no remaining graph references
        for key in keys:
            in_other_graph = any(
                key in gset for gname, gset in self._graphs.items()
            )
            if not in_other_graph:
                self._remove_key(key)

        # Invalidate RDFS closures
        self._rdfs_type_closure = None
        self._rdfs_subproperty_map = None

        return count

    def copy_graph(self, src: str, dst: str) -> None:
        """Copy all triples from src graph to dst graph."""
        if src not in self._graphs:
            return
        for key in self._graphs[src]:
            self._graphs[dst].add(key)

    # ── RDFS inference helpers ─────────────────────────────────

    def _build_rdfs_type_closure(self) -> dict[tuple, set[tuple]]:
        """Compute inferred rdf:type triples from RDFS rules.

        Applies:
            - rdfs-subclass-usage:  (?x rdf:type ?c) ∧ (?c rdfs:subClassOf ?d)
                                    → (?x rdf:type ?d)
            - rdfs-domain:          (?x ?p ?o) ∧ (?p rdfs:domain ?c)
                                    → (?x rdf:type ?c)
            - rdfs-range:           (?x ?p ?o) ∧ (?p rdfs:range ?c)
                                    → (?o rdf:type ?c)

        Returns dict: type_iri → set of (subject_iri, type_iri) inferred keys.
        """
        # 1. Build subClassOf transitive closure
        parent_map: dict[NamedNode, set[NamedNode]] = defaultdict(set)
        all_classes: set[NamedNode] = set()
        for key in self._triples:
            s, p, o = key
            if p == _RDFS_SUBCLASS_OF:
                parent_map[s].add(o)
                all_classes.add(s)
                all_classes.add(o)

        ancestors: dict[NamedNode, set[NamedNode]] = {}
        for cls in all_classes:
            visited: set[NamedNode] = set()
            queue = list(parent_map.get(cls, set()))
            while queue:
                parent = queue.pop()
                if parent not in visited:
                    visited.add(parent)
                    queue.extend(parent_map.get(parent, set()))
            ancestors[cls] = visited

        # 2. Domain and range maps
        domain_map: dict[NamedNode, set[NamedNode]] = defaultdict(set)
        range_map: dict[NamedNode, set[NamedNode]] = defaultdict(set)
        for key in self._triples:
            s, p, o = key
            if p == _RDFS_DOMAIN:
                domain_map[s].add(o)
            elif p == _RDFS_RANGE:
                range_map[s].add(o)

        # 3. Build inferred type triples
        inferred: dict[NamedNode, set[tuple[NamedNode, NamedNode]]] = defaultdict(set)
        for key in self._triples:
            s, p, o = key
            if p == _RDF_TYPE:
                types_to_add = {o} | ancestors.get(o, set())
                for t in types_to_add:
                    inferred[t].add((s, t))
            # rdfs-domain: (?x ?p ?o) ∧ (?p rdfs:domain ?c) → (?x rdf:type ?c)
            if p in domain_map:
                for dt in domain_map[p]:
                    inferred[dt].add((s, dt))
                    for anc in ancestors.get(dt, set()):
                        inferred[anc].add((s, anc))
            # rdfs-range: (?x ?p ?o) ∧ (?p rdfs:range ?c) → (?o rdf:type ?c)
            if p in range_map and isinstance(o, NamedNode):
                for rt in range_map[p]:
                    inferred[rt].add((o, rt))
                    for anc in ancestors.get(rt, set()):
                        inferred[anc].add((o, anc))

        self._rdfs_type_closure = dict(inferred)
        return self._rdfs_type_closure

    def _build_rdfs_subproperty_map(self) -> dict[NamedNode, set[NamedNode]]:
        """Build a map: super-property → set of sub-properties (transitive closure).

        If ``ex:subProp rdfs:subPropertyOf ex:prop``, then
        ``_rdfs_subproperty_map[ex:prop]`` includes ``ex:subProp``.

        The transitive closure means that if
        ``ex:verySpecific rdfs:subPropertyOf ex:subProp``, then
        ``ex:verySpecific`` is also included under ``ex:prop``.
        """
        if self._rdfs_subproperty_map is not None:
            return self._rdfs_subproperty_map

        # Direct adjacency: sub → {supers}, super → {direct subs}
        sub_to_supers: dict[NamedNode, set[NamedNode]] = defaultdict(set)
        super_to_subs: dict[NamedNode, set[NamedNode]] = defaultdict(set)
        all_props: set[NamedNode] = set()
        for key in self._triples:
            s, p, o = key
            if p == _RDFS_SUBPROPERTY_OF:
                sub_to_supers[s].add(o)
                super_to_subs[o].add(s)
                all_props.add(s)
                all_props.add(o)

        # Transitive closure via BFS from each super-property
        result: dict[NamedNode, set[NamedNode]] = {}
        for prop in all_props:
            visited: set[NamedNode] = set()
            queue = list(super_to_subs.get(prop, set()))
            while queue:
                child = queue.pop()
                if child not in visited:
                    visited.add(child)
                    queue.extend(super_to_subs.get(child, set()))
            result[prop] = visited

        self._rdfs_subproperty_map = result
        return self._rdfs_subproperty_map

    def _resolve_inferred(
        self,
        pattern: TriplePattern,
        graph: Optional[str] = None,
    ) -> set[tuple]:
        """Resolve pattern against the RDFS closure.

        Handles:
          - rdf:type patterns → type expansion (subClassOf, domain, range)
          - Other bound predicates → subPropertyOf expansion
        """
        s, p, o = pattern.subject, pattern.predicate, pattern.object_

        if p is None:
            return set()

        if p == _RDF_TYPE:
            # ── Type inference from type closure ──
            if self._rdfs_type_closure is None:
                self._build_rdfs_type_closure()

            all_keys: set[tuple] = set()
            if o is not None:
                for type_nn, inferred_entries in self._rdfs_type_closure.items():
                    if type_nn == o:
                        for (subj, _) in inferred_entries:
                            key = (subj, _RDF_TYPE, type_nn)
                            if graph is None or key in self._graphs.get(graph, set()):
                                all_keys.add(key)
            else:
                for type_nn, inferred_entries in self._rdfs_type_closure.items():
                    for (subj, _) in inferred_entries:
                        key = (subj, _RDF_TYPE, type_nn)
                        if graph is None or key in self._graphs.get(graph, set()):
                            all_keys.add(key)
            return all_keys

        # ── Phase 2: subPropertyOf expansion ──
        # If pattern has predicate P, also match triples whose predicate
        # is a sub-property of P (Q rdfs:subPropertyOf P → match Q triples)
        subprop_map = self._build_rdfs_subproperty_map()
        sub_props = subprop_map.get(p, set())
        if not sub_props:
            return set()

        extra: set[tuple] = set()
        for sub_p in sub_props:
            sub_pattern = TriplePattern(s, sub_p, o)
            extra |= self._resolve_pattern(sub_pattern, graph)
        return extra

    # ── Internal helpers ───────────────────────────────────────

    def _resolve_pattern(
        self,
        pattern: TriplePattern,
        graph: Optional[str] = None,
    ) -> set[tuple]:
        """Resolve a TriplePattern to a set of matching triple keys."""
        keys: set[tuple] = set()
        s, p, o = pattern.subject, pattern.predicate, pattern.object_

        if s is not None and p is not None and o is not None:
            keys = set(self._spo.get(s, {}).get(p, {}).get(o, set()))
        elif s is not None and p is not None:
            for o_sub in self._spo.get(s, {}).get(p, {}).values():
                keys.update(o_sub)
        elif p is not None and o is not None:
            for s_sub in self._pos.get(p, {}).get(o, {}).values():
                keys.update(s_sub)
        elif s is not None and o is not None:
            for p_sub in self._osp.get(o, {}).get(s, {}).values():
                keys.update(p_sub)
        elif s is not None:
            for p_sub in self._spo.get(s, {}).values():
                for o_sub in p_sub.values():
                    keys.update(o_sub)
        elif p is not None:
            for o_sub in self._pos.get(p, {}).values():
                for s_sub in o_sub.values():
                    keys.update(s_sub)
        elif o is not None:
            for s_sub in self._osp.get(o, {}).values():
                for p_sub in s_sub.values():
                    keys.update(p_sub)
        else:
            # Full scan — return all keys
            keys = set(self._triples.keys())

        if graph is not None:
            if graph in self._graphs:
                keys &= self._graphs[graph]
            else:
                return set()

        return keys

    def _remove_key(self, key: tuple) -> None:
        """Remove a triple key from all indices (no graph check)."""
        s, p, o = key

        self._spo[s][p][o].discard(key)
        if not self._spo[s][p][o]:
            del self._spo[s][p][o]
            if not self._spo[s][p]:
                del self._spo[s][p]
                if not self._spo[s]:
                    del self._spo[s]

        self._pos[p][o][s].discard(key)
        if not self._pos[p][o][s]:
            del self._pos[p][o][s]
            if not self._pos[p][o]:
                del self._pos[p][o]
                if not self._pos[p]:
                    del self._pos[p]

        self._osp[o][s][p].discard(key)
        if not self._osp[o][s][p]:
            del self._osp[o][s][p]
            if not self._osp[o][s]:
                del self._osp[o][s]
                if not self._osp[o]:
                    del self._osp[o]

        self._triples.pop(key, None)


def _opinion_strength(op) -> float:
    """Compute a scalar strength for an opinion (higher = more certain)."""
    return op.belief + (1.0 - op.uncertainty)  # belief + certainty
