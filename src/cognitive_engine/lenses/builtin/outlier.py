"""Outlier lens — flag nodes with unusually low or unsupported opinions.

Compares each node's belief against the mean belief of its cohort (nodes of
the same type). A node is flagged as an outlier if:
    |b - mean_b(cohort)| > threshold

Produces a ranked list of outliers with explanations.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from statistics import mean, stdev

from cognitive_engine.core.models import Graph


def outlier_lens(graph: Graph, **params) -> Graph:
    threshold = params.get("outlier_threshold", 0.3)

    result = deepcopy(graph)

    # Build cohorts by node type
    cohorts: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for node in result.nodes.values():
        cohorts[node.type.name].append((node.id.hex, node.opinion[0]))

    # Compute cohort statistics
    cohort_stats: dict[str, dict] = {}
    for type_name, members in cohorts.items():
        beliefs = [b for _, b in members]
        if len(beliefs) > 1:
            cohort_stats[type_name] = {
                "mean": round(mean(beliefs), 4),
                "stdev": round(stdev(beliefs), 4) if len(beliefs) > 2 else 0.0,
                "min": round(min(beliefs), 4),
                "max": round(max(beliefs), 4),
                "count": len(beliefs),
            }
        else:
            cohort_stats[type_name] = {
                "mean": round(beliefs[0], 4),
                "stdev": 0.0,
                "min": round(beliefs[0], 4),
                "max": round(beliefs[0], 4),
                "count": 1,
            }

    # Flag outliers and build ranking
    outlier_ranking: list[dict] = []

    for node in result.nodes.values():
        stats = cohort_stats.get(node.type.name)
        mean_b = stats["mean"] if stats else None

        if mean_b is None or stats["count"] <= 1:
            node.metadata["outlier"] = False
            node.metadata["outlier_deviation"] = 0.0
            node.metadata["outlier_cohort_mean"] = None
        else:
            b = node.opinion[0]
            deviation = abs(b - mean_b)
            is_outlier = deviation > threshold
            node.metadata["outlier"] = is_outlier
            node.metadata["outlier_deviation"] = round(deviation, 4)
            node.metadata["outlier_cohort_mean"] = round(mean_b, 4)

            if is_outlier:
                direction = "below" if b < mean_b else "above"
                outlier_ranking.append({
                    "id": node.id.hex,
                    "text": node.text[:80],
                    "type": node.type.name,
                    "belief": round(b, 4),
                    "cohort_mean": round(mean_b, 4),
                    "cohort_stdev": stats["stdev"],
                    "deviation": round(deviation, 4),
                    "direction": direction,
                    "explanation": (
                        f"Belief ({b:.2f}) is {deviation:.2f} {direction} "
                        f"{node.type.name} cohort mean ({mean_b:.2f})"
                    ),
                })

    # Sort by deviation (highest first)
    outlier_ranking.sort(key=lambda x: x["deviation"], reverse=True)

    result.metadata["outlier_ranking"] = outlier_ranking
    result.metadata["outlier_count"] = len(outlier_ranking)
    result.metadata["outlier_threshold"] = threshold
    result.metadata["cohort_stats"] = cohort_stats
    result.metadata["lens"] = "outlier"
    return result
