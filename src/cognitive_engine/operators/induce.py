"""Ι (Induce) operator — Generalize from observations to rules.

Deterministic induction via pattern extraction and generalization.
No LLM needed — rules are extracted from structured observations.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from cognitive_engine.core.models import Graph, Node, NodeType
from cognitive_engine.core.state import State


@dataclass
class InductionRule:
    """A generalized rule extracted from observations."""
    pattern: str
    rule_text: str
    observations: list[str]
    coverage: float
    specificity: float
    node_ids: list[UUID] = field(default_factory=list)


class InductionOperator:
    """Ι: Generalize from observations to rules.

    Core mechanism:
        1. Collect nodes matching a pattern (by type, embedding, or text)
        2. Find common properties (shared tokens, entities, structure)
        3. Generalize: create a rule from the common pattern
        4. Score by coverage (% matched) and specificity

    Example:
        Observations: ["Swan 1 is white", "Swan 2 is white", "Swan 3 is white"]
        → Rule: "Swans tend to be white"
    """
    name = "induce"

    def __call__(
        self,
        state: State,
        pattern: str = None,
        node_type: str = None,
        min_instances: int = 3,
        similarity_threshold: float = 0.6,
        max_rules: int = 5,
        auto_detect: bool = True,
        **kwargs,
    ) -> State:
        graph = state.graph
        if not graph.nodes:
            state.metadata["induction_result"] = {
                "rules": [],
                "total_observations": 0,
            }
            return state

        if not pattern and auto_detect:
            pattern = self._auto_detect_pattern(graph)
            if pattern:
                similarity_threshold = 0.3

        observations = self._collect_observations(
            graph, pattern, node_type, similarity_threshold
        )

        if len(observations) < min_instances:
            state.metadata["induction_result"] = {
                "rules": [],
                "total_observations": len(observations),
                "min_instances": min_instances,
                "detected_pattern": pattern,
            }
            state.record(
                self.name,
                f"Induction: {len(observations)} observations < {min_instances} minimum",
            )
            return state

        rules = self._generalize(observations, graph)
        rules = self._score_rules(rules, observations)
        rules.sort(key=lambda r: r.specificity * r.coverage, reverse=True)
        rules = rules[:max_rules]

        state.metadata["induction_result"] = {
            "pattern": pattern or node_type or "all",
            "rules": [
                {
                    "pattern": r.pattern,
                    "rule_text": r.rule_text,
                    "observations": r.observations,
                    "coverage": r.coverage,
                    "specificity": r.specificity,
                    "node_ids": [str(nid) for nid in r.node_ids],
                }
                for r in rules
            ],
            "total_observations": len(observations),
            "detected_pattern": pattern,
        }

        rule_texts = [f"'{r.rule_text}' (coverage={r.coverage:.1%}, specificity={r.specificity:.2f})" for r in rules[:5]]
        state.record(
            self.name,
            f"Induced {len(rules)} general rules from {len(observations)} specific observations. "
            f"Pattern detected: '{state.metadata.get('induction_result', {}).get('detected_pattern', 'auto-detect')}'. "
            f"Rules discovered: {'; '.join(rule_texts)}. "
            f"Coverage measures what fraction of observations each rule explains; specificity measures how precise the pattern is. "
            f"Induction generalizes observed instances into reusable reasoning patterns for future cycles.",
        )
        return state

    def _auto_detect_pattern(self, graph: Graph) -> str:
        """Auto-detect common themes across all nodes."""
        all_tokens = []
        for node in graph.nodes.values():
            tokens = set(re.findall(r'\b\w+\b', node.text.lower()))
            tokens = {t for t in tokens if len(t) > 3}
            all_tokens.append(tokens)

        if not all_tokens:
            return None

        counter = Counter()
        for tokens in all_tokens:
            for t in tokens:
                counter[t] += 1

        common = [t for t, count in counter.items() if count >= 2]
        if common:
            return " ".join(common[:5])

        return None

    def _collect_observations(
        self,
        graph: Graph,
        pattern: str,
        node_type: str,
        threshold: float,
    ) -> list[Node]:
        """Collect nodes that match the pattern or type."""
        observations = []

        for node in graph.nodes.values():
            # Filter by node type
            if node_type:
                if node.type.name.upper() != node_type.upper():
                    continue

            # Filter by pattern (text similarity)
            if pattern:
                from cognitive_engine.core.embeddings import EmbeddingModel
                model = EmbeddingModel.get_instance()

                emb = node.embedding
                if emb is None:
                    emb = model.encode(node.text)
                    node.embedding = emb

                pattern_emb = model.encode(pattern)
                sim = EmbeddingModel.cosine_similarity(pattern_emb, emb)
                if sim < threshold:
                    continue

            observations.append(node)

        return observations

    def _generalize(
        self,
        observations: list[Node],
        graph: Graph,
    ) -> list[InductionRule]:
        """Extract generalizations from observations."""
        rules = []

        # Method 1: Common keyword extraction
        rule = self._extract_common_keywords(observations)
        if rule:
            rules.append(rule)

        # Method 2: Sentence structure generalization
        rule = self._extract_structure(observations)
        if rule:
            rules.append(rule)

        # Method 3: Entity-based generalization
        rule = self._extract_entities(observations)
        if rule:
            rules.append(rule)

        return rules

    def _extract_common_keywords(self, observations: list[Node]) -> Optional[InductionRule]:
        """Extract common keywords from observations."""
        if not observations:
            return None

        # Tokenize all observations
        all_tokens = []
        for obs in observations:
            tokens = set(re.findall(r'\b\w+\b', obs.text.lower()))
            tokens = {t for t in tokens if len(t) > 3}  # skip short words
            all_tokens.append(tokens)

        # Find common tokens (appear in >50% of observations)
        if not all_tokens:
            return None

        counter = Counter()
        for tokens in all_tokens:
            for t in tokens:
                counter[t] += 1

        threshold = len(observations) * 0.5
        common = [t for t, count in counter.items() if count >= threshold]

        if not common:
            return None

        # Build rule text
        rule_text = "Tend to involve: " + ", ".join(sorted(common)[:5])

        return InductionRule(
            pattern=",".join(sorted(common)[:3]),
            rule_text=rule_text,
            observations=[obs.text[:100] for obs in observations[:5]],
            coverage=0.0,  # will be scored later
            specificity=0.0,  # will be scored later
            node_ids=[obs.id for obs in observations],
        )

    def _extract_structure(self, observations: list[Node]) -> Optional[InductionRule]:
        """Extract sentence structure pattern."""
        if not observations:
            return None

        # Simple structure: subject-verb pattern
        patterns = []
        for obs in observations:
            # Find first verb-like pattern
            words = obs.text.split()
            if len(words) >= 3:
                patterns.append(f"{words[0]} ... {words[-1]}")

        if not patterns:
            return None

        # Most common structure
        counter = Counter(patterns)
        most_common = counter.most_common(1)[0][0]

        return InductionRule(
            pattern=most_common,
            rule_text=f"Common structure: {most_common}",
            observations=[obs.text[:100] for obs in observations[:5]],
            coverage=0.0,
            specificity=0.0,
            node_ids=[obs.id for obs in observations],
        )

    def _extract_entities(self, observations: list[Node]) -> Optional[InductionRule]:
        """Extract entity-based pattern."""
        if not observations:
            return None

        # Simple: capitalized words as entities
        entities = []
        for obs in observations:
            words = obs.text.split()
            for w in words:
                if w[0].isupper() and len(w) > 2:
                    entities.append(w)

        if not entities:
            return None

        counter = Counter(entities)
        common_entities = [e for e, c in counter.items() if c >= 2]

        if not common_entities:
            return None

        return InductionRule(
            pattern=",".join(common_entities[:3]),
            rule_text=f"Involves: {', '.join(common_entities[:3])}",
            observations=[obs.text[:100] for obs in observations[:5]],
            coverage=0.0,
            specificity=0.0,
            node_ids=[obs.id for obs in observations],
        )

    def _score_rules(
        self,
        rules: list[InductionRule],
        observations: list[Node],
    ) -> list[InductionRule]:
        """Score rules by coverage and specificity."""
        total = len(observations)
        if total == 0:
            return rules

        for rule in rules:
            # Coverage: what % of observations match
            rule.coverage = len(rule.node_ids) / total

            # Specificity: how many distinct tokens in pattern vs total
            pattern_tokens = set(rule.pattern.split(","))
            rule.specificity = min(1.0, len(pattern_tokens) / 5.0)

        return rules
