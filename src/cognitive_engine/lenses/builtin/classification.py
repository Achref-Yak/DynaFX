"""Classification lens — label nodes by belief strength vs domain thresholds.

Each node gets a 'classification' metadata key:
    - "strong"        if b >= threshold_high
    - "moderate"      if b >= threshold_moderate and b < threshold_high
    - "weak"          if b < threshold_moderate and b > d
    - "conflicted"    if conflict detected (b and d both above conflict_threshold)
    - "inconclusive"  otherwise (high uncertainty, u > 0.5)

Produces summary statistics and by-type breakdown in metadata.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from cognitive_engine.core.models import Graph, Opinion
from cognitive_engine.domain import domain as _domain


def classification_lens(graph: Graph, **params) -> Graph:
    cfg = _domain.active()
    threshold_high = params.get("threshold_high", 0.75)
    threshold_moderate = params.get("threshold_moderate", 0.5)
    conflict_threshold = params.get("conflict_threshold", cfg.conflict_threshold)

    result = deepcopy(graph)

    # Classify each node
    distribution: dict[str, int] = defaultdict(int)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    needs_attention: list[dict] = []

    for node in result.nodes.values():
        b, d, u, _ = node.opinion
        label = _classify(b, d, u, threshold_high, threshold_moderate, conflict_threshold)
        node.metadata["classification"] = label

        distribution[label] += 1
        by_type[node.type.name][label] += 1

        if label in ("conflicted", "inconclusive", "weak"):
            needs_attention.append({
                "id": node.id.hex,
                "text": node.text[:60],
                "type": node.type.name,
                "belief": round(b, 4),
                "classification": label,
                "reason": _classification_reason(label, b, d, u),
            })

    # Convert defaultdicts to regular dicts for JSON serialization
    result.metadata["classification_summary"] = {
        "distribution": dict(distribution),
        "total_nodes": len(result.nodes),
        "needs_attention": needs_attention,
        "by_type": {
            t: dict(labels) for t, labels in sorted(by_type.items())
        },
    }
    result.metadata["lens"] = "classification"
    return result


def _classify(
    b: float, d: float, u: float,
    high: float, moderate: float, conflict: float,
) -> str:
    if b >= high:
        return "strong"
    if b >= moderate and b > d:
        return "moderate"
    if b > conflict and d > conflict:
        return "conflicted"
    if u > 0.5:
        return "inconclusive"
    if b > d:
        return "weak"
    return "inconclusive"


def _classification_reason(label: str, b: float, d: float, u: float) -> str:
    """Generate a human-readable reason for the classification."""
    if label == "conflicted":
        return f"High belief ({b:.2f}) AND high disbelief ({d:.2f}) — evidence conflicts"
    if label == "inconclusive":
        return f"High uncertainty ({u:.2f}) — insufficient evidence"
    if label == "weak":
        return f"Low belief ({b:.2f}) with minimal disbelief ({d:.2f})"
    return ""
