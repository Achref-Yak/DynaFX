"""Lens framework — Graph → Graph analytical transformers.

A lens is a callable that takes a Graph and returns a transformed Graph
with additional metadata (classifications, funnel chains, outlier flags, etc.).

Usage:
    from cognitive_engine.lenses import apply_lens, list_lenses

    graph = apply_lens(graph, "classification")
    graph = apply_lens(graph, "funnel")
    print(list_lenses())
"""

from cognitive_engine.lenses.registry import register_lens, apply_lens, list_lenses
from cognitive_engine.lenses.builtin.classification import classification_lens
from cognitive_engine.lenses.builtin.funnel import funnel_lens
from cognitive_engine.lenses.builtin.decision_tree import decision_tree_lens
from cognitive_engine.lenses.builtin.outlier import outlier_lens
from cognitive_engine.lenses.builtin.aggregation import aggregation_lens

register_lens("classification", classification_lens)
register_lens("funnel", funnel_lens)
register_lens("decision-tree", decision_tree_lens)
register_lens("outlier", outlier_lens)
register_lens("aggregation", aggregation_lens)

__all__ = ["register_lens", "apply_lens", "list_lenses"]
