import json
from pathlib import Path

import pytest

from dynafx.core.models import EvidenceCounts, Graph, Node, Edge, NodeType, EdgeType
from dynafx.epistemics.evidence import (
    CorpusResult,
    mean_opinion,
    mean_opinion_pair,
    opinion_from_counts,
)


class TestOpinionFromCounts:
    def test_all_positive(self):
        result = opinion_from_counts(EvidenceCounts(positive=10, negative=0))
        assert result[0] == pytest.approx(10 / 12)  # b = 10/(10+0+2)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(2 / 12)

    def test_all_negative(self):
        result = opinion_from_counts(EvidenceCounts(positive=0, negative=10))
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(10 / 12)
        assert result[2] == pytest.approx(2 / 12)

    def test_mixed(self):
        result = opinion_from_counts(EvidenceCounts(positive=7, negative=3))
        assert result[0] == pytest.approx(7 / 12)
        assert result[1] == pytest.approx(3 / 12)
        assert result[2] == pytest.approx(2 / 12)

    def test_zero_observations(self):
        result = opinion_from_counts(EvidenceCounts(positive=0, negative=0))
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_custom_pseudocount(self):
        result = opinion_from_counts(EvidenceCounts(positive=5, negative=5, uncertainty_pseudocount=0.5))
        assert result[2] == pytest.approx(0.5 / 10.5)

    def test_invariant(self):
        for pos, neg in [(0, 0), (5, 0), (3, 7), (100, 200)]:
            result = opinion_from_counts(EvidenceCounts(positive=pos, negative=neg))
            assert sum(result[:3]) == pytest.approx(1.0)


class TestMeanOpinion:
    def test_empty_list(self):
        result = mean_opinion([])
        assert result == (0.0, 0.0, 1.0, 0.5)

    def test_single(self):
        result = mean_opinion([(0.8, 0.1, 0.1, 0.5)])
        assert result == pytest.approx((0.8, 0.1, 0.1, 0.5))

    def test_two_opinions(self):
        result = mean_opinion([
            (0.8, 0.1, 0.1, 0.5),
            (0.4, 0.3, 0.3, 0.5),
        ])
        assert result[0] == pytest.approx(0.6)
        assert result[1] == pytest.approx(0.2)
        assert result[2] == pytest.approx(0.2)

    def test_normalizes(self):
        result = mean_opinion([
            (0.9, 0.2, 0.1, 0.5),
            (0.1, 0.3, 0.2, 0.5),
        ])
        assert sum(result[:3]) == pytest.approx(1.0, abs=1e-3)


class TestMeanOpinionPair:
    def test_basic(self):
        pairs = [
            ((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5)),
            ((0.6, 0.2, 0.2, 0.5), (0.3, 0.5, 0.2, 0.5)),
        ]
        first, second = mean_opinion_pair(pairs)
        assert first[0] == pytest.approx(0.7)
        assert second[0] == pytest.approx(0.25)


class TestCorpusResult:
    def test_empty_corpus_loads(self, tmp_path: Path):
        result = CorpusResult()
        assert result.graph_count == 0
        assert result.node_counts == {}

    def test_save_load_roundtrip(self, tmp_path: Path):
        original = CorpusResult(
            graph_count=2,
            node_counts={
                "CLAIM": EvidenceCounts(positive=10, negative=3),
                "EVIDENCE": EvidenceCounts(positive=5, negative=1),
            },
            edge_warrants={
                "SUPPORTS": [
                    ((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5)),
                ],
            },
        )
        path = tmp_path / "corpus.json"
        original.save(path)
        assert path.exists()

        loaded = CorpusResult.load(path)
        assert loaded.graph_count == 2
        assert loaded.node_counts["CLAIM"].positive == 10
        assert loaded.node_counts["CLAIM"].negative == 3
        assert loaded.edge_warrants["SUPPORTS"][0] == ((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5))

    def test_to_priors_with_counts(self):
        result = CorpusResult(
            graph_count=2,
            node_counts={
                "CLAIM": EvidenceCounts(positive=20, negative=5),
                "EVIDENCE": EvidenceCounts(positive=15, negative=3),
            },
            edge_warrants={},
        )
        priors = result.to_priors()
        assert "consensus_principle" in priors.default_opinions
        assert "empirical_pattern" in priors.default_opinions
        cp = priors.default_opinions["consensus_principle"]
        assert cp[0] > cp[1]  # more positive than negative
        ep = priors.default_opinions["empirical_pattern"]
        assert ep[0] > ep[1]

    def test_to_priors_preserves_source_type_map(self):
        result = CorpusResult(graph_count=1, node_counts={}, edge_warrants={})
        priors = result.to_priors()
        assert priors.source_type_map["CLAIM"] == "consensus_principle"
        assert priors.source_type_map["EVIDENCE"] == "empirical_pattern"

    def test_to_priors_empty_falls_back_to_defaults(self):
        result = CorpusResult(graph_count=0, node_counts={}, edge_warrants={})
        priors = result.to_priors()
        assert priors.default_opinions["consensus_principle"] == (0.7, 0.1, 0.2, 0.5)

    def test_learn_from_corpus_directory(self, tmp_path: Path):
        d = tmp_path / "corpus"
        d.mkdir()
        (d / "doc1.txt").write_text("This is a claim. The evidence supports it.")
        (d / "doc2.txt").write_text("A counterargument exists. However, the claim stands.")

        result = CorpusResult.from_corpus(d, max_files=2)
        # Corpus extraction removed — from_corpus returns empty result
        assert result.graph_count == 0

    def test_non_existent_corpus_raises(self):
        with pytest.raises(NotADirectoryError):
            CorpusResult.from_corpus("/nonexistent/corpus")

    def test_save_produces_valid_json(self, tmp_path: Path):
        result = CorpusResult(
            graph_count=1,
            node_counts={"CLAIM": EvidenceCounts(positive=3, negative=1)},
            edge_warrants={},
        )
        path = tmp_path / "out.json"
        result.save(path)
        data = json.loads(path.read_text())
        assert data["graph_count"] == 1
        assert data["node_counts"]["CLAIM"]["positive"] == 3

    def test_to_dict_structure(self):
        result = CorpusResult(
            graph_count=1,
            node_counts={"TEST": EvidenceCounts(positive=2, negative=1)},
            edge_warrants={"SUPPORTS": [((0.8, 0.1, 0.1, 0.5), (0.2, 0.7, 0.1, 0.5))]},
        )
        d = result.to_dict()
        assert d["graph_count"] == 1
        assert d["node_counts"]["TEST"]["positive"] == 2
        assert len(d["edge_warrants"]["SUPPORTS"]) == 1
