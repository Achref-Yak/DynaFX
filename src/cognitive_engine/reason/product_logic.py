from __future__ import annotations

from enum import IntEnum
from typing import Tuple

from cognitive_engine.core.models import Graph, Violation, Severity
from cognitive_engine.domain import domain as _domain


class _CategoryNameLookup:
    """Lazily resolves category names from the active domain config."""

    def get(self, key: int, default: str = "?") -> str:
        return _domain.active().category_levels.get(key, default)


CATEGORY_NAMES: _CategoryNameLookup = _CategoryNameLookup()


class Category(IntEnum):
    NECESSITY = 1
    FACT = 2
    BELIEF = 3
    CONCEPT = 4


def valuate(category: int, truth: bool) -> int:
    return category * (1 if truth else 0)


def negation_cat(cat: int) -> int:
    return cat


def conjunction_cat(a: int, b: int) -> int:
    return max(a, b)


def disjunction_cat(a: int, b: int) -> int:
    return min(a, b)


def implication_valid(src_cat: int, tgt_cat: int) -> bool:
    return src_cat <= tgt_cat


def equivalence_valid(a: int, b: int) -> bool:
    return a == b


def validate_categories(graph: Graph) -> list[Violation]:
    violations: list[Violation] = []
    for edge in graph.edges.values():
        source = graph.nodes.get(edge.source_id)
        target = graph.nodes.get(edge.target_id)
        if source is None or target is None:
            violations.append(
                Violation(
                    type="MISSING_NODE",
                    severity=Severity.ERROR,
                    description=f"Edge {edge.id.hex[:8]} references non-existent node",
                    edge_id=edge.id,
                )
            )
            continue
        src_cat = getattr(source, "category", 2)
        tgt_cat = getattr(target, "category", 2)

        if src_cat > tgt_cat:
            violations.append(
                Violation(
                    type="CATEGORY_ERROR",
                    severity=Severity.ERROR,
                    description=f"Cannot imply category {tgt_cat} "
                    f"({CATEGORY_NAMES.get(tgt_cat, '?')}) from category {src_cat} "
                    f"({CATEGORY_NAMES.get(src_cat, '?')}): "
                    f"'{source.text[:50]}' → '{target.text[:50]}'",
                    edge_id=edge.id,
                )
            )

        if edge.type.name in ("CONTRADICTS", "ATTACKS"):
            if src_cat == tgt_cat == 4:
                violations.append(
                    Violation(
                        type="CATEGORY_ERROR",
                        severity=Severity.WARNING,
                        description=f"CONTRADICTS/ATTACKS between two Concept-level "
                        f"nodes ({source.text[:30]} vs {target.text[:30]}) — "
                        f"likely a category mistake",
                        edge_id=edge.id,
                    )
                )
    return violations
