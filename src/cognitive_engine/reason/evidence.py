from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from cognitive_engine.core.config import Priors
from cognitive_engine.core.models import EvidenceCounts, Graph, Opinion
from cognitive_engine.domain import domain as _domain

logger = logging.getLogger(__name__)


def opinion_from_counts(counts: EvidenceCounts) -> tuple[float, float, float, float]:
    from cognitive_engine.core.math import opinion_from_counts as _from_counts
    return _from_counts(
        counts.positive, counts.negative,
        counts.uncertainty_pseudocount, 0.5,
    )


def mean_opinion(opinions: list[Opinion]) -> tuple[float, float, float, float]:
    from cognitive_engine.core.math import mean_opinion as _mean
    return _mean(list(opinions))


def mean_opinion_pair(pairs: list[tuple[Opinion, Opinion]]) -> tuple[Opinion, Opinion]:
    first = [p[0] for p in pairs]
    second = [p[1] for p in pairs]
    return (mean_opinion(first), mean_opinion(second))


def _collect_node_counts(
    graphs: list[Graph],
) -> dict[str, EvidenceCounts]:
    w = _domain.active().uncertainty_pseudocount
    counts: dict[str, EvidenceCounts] = defaultdict(lambda: EvidenceCounts(uncertainty_pseudocount=w))
    for graph in graphs:
        for node in graph.nodes.values():
            key = node.type.name
            b, d, _, _ = node.opinion
            if b > d + _domain.active().opinion_positive_threshold:
                counts[key].positive += 1
            else:
                counts[key].negative += 1
    return dict(counts)


def _collect_edge_counts(
    graphs: list[Graph],
) -> dict[str, list[tuple[Opinion, Opinion]]]:
    warrants: dict[str, list[tuple[Opinion, Opinion]]] = defaultdict(list)
    for graph in graphs:
        for edge in graph.edges.values():
            key = edge.type.name
            if edge.warrant is not None:
                warrants[key].append(edge.warrant)
    return dict(warrants)


@dataclass
class CorpusResult:
    graph_count: int = 0
    node_counts: Dict[str, EvidenceCounts] = field(default_factory=dict)
    edge_warrants: Dict[str, list[tuple[Opinion, Opinion]]] = field(default_factory=dict)

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: str | Path,
        max_files: Optional[int] = None,
        config_path: Optional[str] = None,
    ) -> CorpusResult:
        path = Path(corpus_dir)
        if not path.is_dir():
            raise NotADirectoryError(f"Corpus directory not found: {corpus_dir}")

        txt_files = sorted(path.rglob("*.txt"))
        if max_files is not None:
            txt_files = txt_files[:max_files]

        if not txt_files:
            logger.warning("No .txt files found in %s", corpus_dir)
            return cls()

        graphs: list[Graph] = []
        for f in txt_files:
            logger.warning("  Skipped %s: corpus extraction removed in this build", f.name)

        node_counts = _collect_node_counts(graphs)
        edge_warrants = _collect_edge_counts(graphs)

        return cls(
            graph_count=len(graphs),
            node_counts=node_counts,
            edge_warrants=edge_warrants,
        )

    def to_priors(self) -> Priors:
        cfg = _domain.active()
        node_type_map = cfg.source_type_map

        source_type_map: dict[str, str] = {}
        learned_opinions: dict[str, Opinion] = {}
        for node_type_name, template_name in node_type_map.items():
            counts = self.node_counts.get(node_type_name)
            if counts is not None and (counts.positive > 0 or counts.negative > 0):
                learned_opinions[template_name] = opinion_from_counts(counts)
            source_type_map[node_type_name] = template_name

        learned_opinions["total_ignorance"] = cfg.total_ignorance
        for template in list(cfg.default_opinions):
            if template not in learned_opinions:
                learned_opinions[template] = cfg.default_opinions[template]

        learned_warrants: dict[str, tuple[Opinion, Opinion]] = {}
        for edge_type_name, pairs in self.edge_warrants.items():
            if pairs:
                learned_warrants[edge_type_name] = mean_opinion_pair(pairs)

        base = Priors()
        warrants = learned_warrants if learned_warrants else base.edge_warrants
        if "SUPPORTS" not in warrants:
            warrants["SUPPORTS"] = base.edge_warrants["SUPPORTS"]
        for et_name in base.edge_warrants:
            if et_name not in warrants:
                warrants[et_name] = base.edge_warrants[et_name]

        return Priors(
            default_opinions=learned_opinions,
            source_type_map=source_type_map,
            edge_warrants=warrants,
            default_warrant=base.default_warrant,
        )

    def to_dict(self) -> dict:
        return {
            "graph_count": self.graph_count,
            "node_counts": {
                k: {"positive": v.positive, "negative": v.negative,
                    "uncertainty_pseudocount": v.uncertainty_pseudocount}
                for k, v in self.node_counts.items()
            },
            "edge_warrants": {
                k: [[list(a), list(b)] for a, b in v]
                for k, v in self.edge_warrants.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Saved corpus result to %s", p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> CorpusResult:
        p = Path(path)
        data = json.loads(p.read_text())
        node_counts = {
            k: EvidenceCounts(**v)
            for k, v in data.get("node_counts", {}).items()
        }
        edge_warrants = {
            k: [(tuple(a), tuple(b)) for a, b in v]
            for k, v in data.get("edge_warrants", {}).items()
        }
        return cls(
            graph_count=data.get("graph_count", 0),
            node_counts=node_counts,
            edge_warrants=edge_warrants,
        )
