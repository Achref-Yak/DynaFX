"""TripleStore — in-memory RDF triple store with SPO/POS/OSP indices and named graphs.

Supports:
    - O(1) pattern matching via 3-index prefix strategy
    - Named graphs per information source
    - Dedup: same (s,p,o) with higher-belief opinion replaces lower
    - Graph-level isolation, copy, and removal
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterator, List, Optional, Set, Tuple

from cognitive_engine.kb.model import (
    BlankNode,
    Literal,
    NamedNode,
    RDFNode,
    Triple,
    TriplePattern,
)


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

        return count

    # ── Query ───────────────────────────────────────────────────

    def triples(
        self,
        pattern: TriplePattern,
        graph: Optional[str] = None,
    ) -> Iterator[Triple]:
        """Iterate over triples matching a pattern, optionally in a graph."""
        keys = self._resolve_pattern(pattern, graph)
        for key in keys:
            yield self._triples[key]

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

    def graphs(self) -> List[str]:
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

        return count

    def copy_graph(self, src: str, dst: str) -> None:
        """Copy all triples from src graph to dst graph."""
        if src not in self._graphs:
            return
        for key in self._graphs[src]:
            self._graphs[dst].add(key)

    # ── Internal helpers ───────────────────────────────────────

    def _resolve_pattern(
        self,
        pattern: TriplePattern,
        graph: Optional[str] = None,
    ) -> Set[tuple]:
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
