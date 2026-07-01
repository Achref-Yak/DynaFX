"""Feedback loop detection for system dynamics models.

Detects and classifies feedback loops in model structure:
- Reinforcing (positive) loops: even number of negative edges
- Balancing (negative) loops: odd number of negative edges
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from dynafx.dynamics.dsl import SysdModel
from dynafx.dynamics.causal import _get_dependencies, _get_reverse_deps


@dataclass
class FeedbackLoop:
    """A detected feedback loop in the model."""
    name: str
    nodes: list[str] = field(default_factory=list)
    polarity: str = "reinforcing"  # "reinforcing" or "balancing"
    negative_edges: int = 0
    edge_polarities: dict[tuple[str, str], int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": self.nodes,
            "polarity": self.polarity,
            "negative_edges": self.negative_edges,
            "edge_polarities": {f"{k[0]}->{k[1]}": v for k, v in self.edge_polarities.items()},
        }


@dataclass
class LoopAnalysis:
    """Complete feedback loop analysis of a model."""
    loops: list[FeedbackLoop] = field(default_factory=list)
    variable_loops: dict[str, list[str]] = field(default_factory=dict)  # variable -> loop names

    def to_dict(self) -> dict:
        return {
            "loops": [l.to_dict() for l in self.loops],
            "variable_loops": self.variable_loops,
            "num_reinforcing": sum(1 for l in self.loops if l.polarity == "reinforcing"),
            "num_balancing": sum(1 for l in self.loops if l.polarity == "balancing"),
        }


def _get_loop_polarity(
    nodes: list[str],
    edge_polarities: dict[tuple[str, str], int],
) -> tuple[str, int]:
    """Determine loop polarity from edge polarities.

    Returns (polarity_name, negative_edge_count).
    Reinforcing: even number of negative edges (positive feedback)
    Balancing: odd number of negative edges (negative feedback)
    """
    neg_count = 0
    for i in range(len(nodes)):
        src = nodes[i]
        dst = nodes[(i + 1) % len(nodes)]
        pol = edge_polarities.get((src, dst), 1)
        if pol < 0:
            neg_count += 1

    polarity = "reinforcing" if neg_count % 2 == 0 else "balancing"
    return polarity, neg_count


def _find_simple_cycles(
    deps: dict[str, tuple[str, set[str]]],
) -> list[list[str]]:
    """Find all simple cycles in the dependency graph using DFS."""
    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        _, refs = deps.get(node, ("", set()))
        for ref in refs:
            if ref not in visited:
                dfs(ref)
            elif ref in rec_stack:
                # Found a cycle
                cycle_start = path.index(ref)
                cycle = path[cycle_start:] + [ref]
                # Normalize: start with smallest element
                min_idx = cycle[:-1].index(min(cycle[:-1]))
                normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
                if normalized not in cycles:
                    cycles.append(normalized)

        path.pop()
        rec_stack.discard(node)

    for node in sorted(deps.keys()):
        if node not in visited:
            dfs(node)

    return cycles


def _determine_edge_polarities(
    model: SysdModel,
    deps: dict[str, tuple[str, set[str]]],
) -> dict[tuple[str, str], int]:
    """Determine polarity of each edge in the dependency graph.

    Polarity is +1 (positive) or -1 (negative).
    For stocks, outflow edges are negative (increasing outflow decreases stock).
    For expressions, we check if the reference appears with a negative sign.
    """
    edge_pols: dict[tuple[str, str], int] = {}

    # Stock flow edges
    for s in model.stocks:
        for f in s.flows:
            pol = 1 if f.direction == "+" else -1
            edge_pols[(f.name, s.name)] = pol

    # Expression references — check for negation
    import re
    for name, (expr, refs) in deps.items():
        for ref in refs:
            if (name, ref) not in edge_pols:
                # Check if ref appears negated in expression
                # Simple heuristic: look for -ref or ref preceded by minus
                neg_pattern = rf'-\s*{re.escape(ref)}\b'
                pos_pattern = rf'\b{re.escape(ref)}\b'
                has_neg = bool(re.search(neg_pattern, expr))
                has_pos = bool(re.search(pos_pattern, expr))
                if has_neg and not has_pos:
                    edge_pols[(name, ref)] = -1
                elif has_pos:
                    edge_pols[(name, ref)] = 1
                else:
                    edge_pols[(name, ref)] = 1  # default positive

    return edge_pols


def detect_feedback_loops(
    model: SysdModel,
    max_loop_length: int = 20,
) -> LoopAnalysis:
    """Detect and classify all feedback loops in the model.

    Args:
        model: The SysdModel to analyze
        max_loop_length: Maximum cycle length to consider

    Returns:
        LoopAnalysis with detected loops and variable-loop mappings
    """
    deps = _get_dependencies(model)
    edge_pols = _determine_edge_polarities(model, deps)

    # Find all simple cycles
    raw_cycles = _find_simple_cycles(deps)

    # Filter and classify
    analysis = LoopAnalysis()
    seen_loops: set[tuple] = set()

    for cycle in raw_cycles:
        if len(cycle) > max_loop_length + 1:
            continue

        # Normalize cycle for dedup
        nodes = cycle[:-1]  # Remove the repeated start node
        normalized = tuple(sorted(nodes))
        if normalized in seen_loops:
            continue
        seen_loops.add(normalized)

        # Determine polarity
        polarity, neg_count = _get_loop_polarity(nodes, edge_pols)

        # Build edge polarities for this loop
        loop_edge_pols = {}
        for i in range(len(nodes)):
            src = nodes[i]
            dst = nodes[(i + 1) % len(nodes)]
            loop_edge_pols[(src, dst)] = edge_pols.get((src, dst), 1)

        loop = FeedbackLoop(
            name=f"Loop_{len(analysis.loops) + 1}",
            nodes=nodes,
            polarity=polarity,
            negative_edges=neg_count,
            edge_polarities=loop_edge_pols,
        )
        analysis.loops.append(loop)

        # Map variables to loops
        for node in nodes:
            analysis.variable_loops.setdefault(node, []).append(loop.name)

    return analysis


def loops_for_variable(
    analysis: LoopAnalysis,
    variable: str,
) -> list[FeedbackLoop]:
    """Get all loops that a given variable participates in."""
    loop_names = analysis.variable_loops.get(variable, [])
    return [l for l in analysis.loops if l.name in loop_names]
