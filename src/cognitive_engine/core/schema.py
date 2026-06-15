from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cognitive_engine.core.models import EdgeType, NodeType


@dataclass
class Schema:
    """Domain-specific configuration for operators.

    Defines:
      - Node type mappings (text pattern -> NodeType)
      - Edge type mappings (source_type, target_type, label -> EdgeType)
      - Merge strategy (how to combine graphs)
      - Conflict resolution (how to handle contradictions)
    """
    name: str
    node_types: dict[str, NodeType] = field(default_factory=dict)
    edge_types: dict[tuple, EdgeType] = field(default_factory=dict)
    type_rules: list[Callable] = field(default_factory=list)
    merge_strategy: str = "average"  # "average", "weighted", "dempster_shafer", "keep_both"
    conflict_resolution: str = "higher_confidence"  # "keep_both", "higher_confidence", "schema_rules"
    dedup_threshold: float = 0.8  # similarity threshold for node dedup
    metadata: dict = field(default_factory=dict)

    def get_node_type(self, label: str) -> Optional[NodeType]:
        """Look up NodeType by label."""
        return self.node_types.get(label)

    def get_edge_type(self, source: NodeType, target: NodeType, relation: str) -> Optional[EdgeType]:
        """Look up EdgeType by (source, target, relation)."""
        return self.edge_types.get((source, target, relation))

    def classify_node(self, text: str, **kwargs) -> Optional[NodeType]:
        """Apply type rules to classify a node."""
        for rule in self.type_rules:
            result = rule(text, **kwargs)
            if result is not None:
                return result
        return None


def merge_schemas(*schemas: Schema, name: str = "merged") -> Schema:
    """Merge multiple schemas into one (later schemas override earlier)."""
    node_types = {}
    edge_types = {}
    type_rules = []
    merge_strategy = "average"
    conflict_resolution = "higher_confidence"
    dedup_threshold = 0.8
    metadata = {}

    for s in schemas:
        node_types.update(s.node_types)
        edge_types.update(s.edge_types)
        type_rules.extend(s.type_rules)
        merge_strategy = s.merge_strategy
        conflict_resolution = s.conflict_resolution
        dedup_threshold = s.dedup_threshold
        metadata.update(s.metadata)

    return Schema(
        name=name,
        node_types=node_types,
        edge_types=edge_types,
        type_rules=type_rules,
        merge_strategy=merge_strategy,
        conflict_resolution=conflict_resolution,
        dedup_threshold=dedup_threshold,
        metadata=metadata,
    )
