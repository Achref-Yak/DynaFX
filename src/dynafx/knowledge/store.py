"""TripleStore — in-memory RDF triple store with SPO/POS/OSP indices and named graphs.

Supports:
    - O(1) pattern matching via 3-index prefix strategy
    - Named graphs per information source
    - Dedup: same (s,p,o) is idempotent (last write wins)
    - Graph-level isolation, copy, and removal
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

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


class TripleStore:
    """In-memory triple store with three-index prefix strategy.

    Indices:
        _spo: subject -> predicate -> object -> set of triple_keys
        _pos: predicate -> object -> subject -> set of triple_keys
        _osp: object -> subject -> predicate -> set of triple_keys

    Each triple_key is the identity tuple (subject, predicate, object).
    """

    def __init__(self) -> None:
        self._triples: dict[tuple, Triple] = {}

        self._spo: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self._pos: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self._osp: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

        self._graphs: dict[str, set[tuple]] = defaultdict(set)

        self._rdfs_type_closure: dict[tuple, set[tuple]] | None = None
        self._rdfs_subproperty_map: dict[NamedNode, set[NamedNode]] | None = None

    # ── Event callbacks ───────────────────────────────────────────

    _on_add_callbacks: list[Callable[[Triple, str], None]] | None = None
    _on_remove_callbacks: list[Callable[[TriplePattern, str | None], None]] | None = None
    _suppress_callbacks: int = 0

    @contextmanager
    def suppress_callbacks(self) -> Iterator[None]:
        self._suppress_callbacks += 1
        try:
            yield
        finally:
            self._suppress_callbacks -= 1

    def on_add(self, fn: Callable[[Triple, str], None]) -> None:
        if self._on_add_callbacks is None:
            self._on_add_callbacks = []
        self._on_add_callbacks.append(fn)

    def on_remove(self, fn: Callable[[TriplePattern, str | None], None]) -> None:
        if self._on_remove_callbacks is None:
            self._on_remove_callbacks = []
        self._on_remove_callbacks.append(fn)

    # ── Core mutation ───────────────────────────────────────────

    def add(self, triple: Triple, graph: str = "default") -> None:
        """Add a triple to the store, optionally in a named graph."""
        key = triple.spo

        # If key already exists, skip (idempotent add)
        if key in self._triples:
            self._graphs[graph].add(key)
            return

        s, p, o = key
        self._spo[s][p][o].add(key)
        self._pos[p][o][s].add(key)
        self._osp[o][s][p].add(key)

        self._triples[key] = triple
        self._graphs[graph].add(key)

        self._rdfs_type_closure = None
        self._rdfs_subproperty_map = None

        if self._suppress_callbacks == 0 and self._on_add_callbacks is not None:
            for fn in self._on_add_callbacks:
                fn(triple, graph)

    def remove(self, pattern: TriplePattern, graph: str | None = None) -> int:
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

            for g in list(self._graphs.keys()):
                self._graphs[g].discard(key)

        self._rdfs_type_closure = None
        self._rdfs_subproperty_map = None

        if self._on_remove_callbacks is not None and count > 0:
            for fn in self._on_remove_callbacks:
                fn(pattern, graph)

        return count

    # ── Query ───────────────────────────────────────────────────

    def triples(
        self,
        pattern: TriplePattern,
        graph: str | None = None,
        with_inference: str | dict | None = None,
    ) -> Iterator[Triple]:
        """Iterate over triples matching a pattern, optionally in a graph.

        Args:
            pattern: The triple pattern to match.
            graph: Optional named graph to restrict to.
            with_inference: ``"rdfs"`` or dict to enable RDFS inference.
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
            yield self._triples[key]

    def all_triples(self) -> Iterator[Triple]:
        """Iterate over every triple in the store (across all graphs)."""
        yield from self._triples.values()

    def __contains__(self, pattern: TriplePattern) -> bool:
        try:
            next(self.triples(pattern))
            return True
        except StopIteration:
            return False

    def __len__(self) -> int:
        return len(self._triples)

    # ── Named graph operations ─────────────────────────────────

    def graphs(self) -> list[str]:
        return list(self._graphs.keys())

    def triples_in_graph(self, graph: str) -> Iterator[Triple]:
        for key in self._graphs.get(graph, set()):
            yield self._triples[key]

    def remove_graph(self, graph: str) -> int:
        keys = set(self._graphs.get(graph, set()))
        count = len(keys)
        self._graphs[graph] = set()
        if graph in self._graphs:
            del self._graphs[graph]

        for key in keys:
            in_other_graph = any(
                key in gset for gname, gset in self._graphs.items()
            )
            if not in_other_graph:
                self._remove_key(key)

        self._rdfs_type_closure = None
        self._rdfs_subproperty_map = None

        return count

    def copy_graph(self, src: str, dst: str) -> None:
        if src not in self._graphs:
            return
        for key in self._graphs[src]:
            self._graphs[dst].add(key)

    # ── RDFS inference helpers ─────────────────────────────────

    def _build_rdfs_type_closure(self) -> dict[tuple, set[tuple]]:
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

        domain_map: dict[NamedNode, set[NamedNode]] = defaultdict(set)
        range_map: dict[NamedNode, set[NamedNode]] = defaultdict(set)
        for key in self._triples:
            s, p, o = key
            if p == _RDFS_DOMAIN:
                domain_map[s].add(o)
            elif p == _RDFS_RANGE:
                range_map[s].add(o)

        inferred: dict[NamedNode, set[tuple[NamedNode, NamedNode]]] = defaultdict(set)
        for key in self._triples:
            s, p, o = key
            if p == _RDF_TYPE:
                types_to_add = {o} | ancestors.get(o, set())
                for t in types_to_add:
                    inferred[t].add((s, t))
            if p in domain_map:
                for dt in domain_map[p]:
                    inferred[dt].add((s, dt))
                    for anc in ancestors.get(dt, set()):
                        inferred[anc].add((s, anc))
            if p in range_map and isinstance(o, NamedNode):
                for rt in range_map[p]:
                    inferred[rt].add((o, rt))
                    for anc in ancestors.get(rt, set()):
                        inferred[anc].add((o, anc))

        self._rdfs_type_closure = dict(inferred)
        return self._rdfs_type_closure

    def _build_rdfs_subproperty_map(self) -> dict[NamedNode, set[NamedNode]]:
        if self._rdfs_subproperty_map is not None:
            return self._rdfs_subproperty_map

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
        graph: str | None = None,
    ) -> set[tuple]:
        s, p, o = pattern.subject, pattern.predicate, pattern.object_

        if p is None:
            return set()

        if p == _RDF_TYPE:
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
        graph: str | None = None,
    ) -> set[tuple]:
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
            keys = set(self._triples.keys())

        if graph is not None:
            if graph in self._graphs:
                keys &= self._graphs[graph]
            else:
                return set()

        return keys

    def _remove_key(self, key: tuple) -> None:
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


def _parse_inference_config(with_inference: Any) -> dict:
    if with_inference is None:
        return {}
    if isinstance(with_inference, str):
        return {"mode": with_inference}
    if isinstance(with_inference, dict):
        return with_inference
    return {}
