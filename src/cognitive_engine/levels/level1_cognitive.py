"""Level 1: Cognitive Architecture (ACT-R / SOAR inspired).

Implements activation-based memory retrieval, production rule firing,
goal management, and chunk learning.

Core formulas:
    A_i = B_i + Σ_j W_j * S_ij                     (activation)
    P(retrieve i) = e^{A_i / τ} / Σ_k e^{A_k / τ}  (retrieval softmax)
    P(r_j) = e^{U_j / τ} / Σ_k e^{U_k / τ}        (rule firing)
    S_{t+1} = δ(S_t, a_t)                          (state transition)
    r_new = compress(trace)                          (chunk learning)

Usage:
    from cognitive_engine.levels.level1_cognitive import CognitiveLevel
    level = CognitiveLevel()
    level.add_chunk("red_light", content="The light was red")
    level.add_production_rule(
        condition=lambda ctx: ctx.get("red_light"),
        action="should_stop",
        utility=0.9,
    )
    result = level.step()
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from cognitive_engine.core.models import Graph, NodeType
from cognitive_engine.levels.base import BaseLevel, LevelOutput, ReasoningContext

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A memory chunk with activation.

    Attributes:
        id: Unique identifier.
        content: The chunk's content (text or structured data).
        base_level_activation: Base-level activation B_i.
        creation_time: When the chunk was created.
        access_count: Number of times accessed.
        last_access: Last access timestamp.
        metadata: Additional chunk metadata.
    """
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    content: Any = None
    base_level_activation: float = 0.0
    creation_time: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ProductionRule:
    """A production rule: IF condition THEN action.

    Attributes:
        id: Unique identifier.
        condition: Function that takes context dict and returns bool.
        action: Name of the action/conclusion.
        utility: Rule utility U_j for firing probability.
        strength: Rule strength [0, 1].
    """
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    condition: Callable[[dict], bool] = field(default_factory=lambda: lambda ctx: True)
    action: str = ""
    utility: float = 0.5
    strength: float = 1.0


@dataclass
class ProductionResult:
    """Result of firing a production rule.

    Attributes:
        rule: The rule that fired.
        action: The action name.
        context_snapshot: Context at time of firing.
    """
    rule: ProductionRule
    action: str
    context_snapshot: dict = field(default_factory=dict)


class CognitiveLevel(BaseLevel):
    """Level 1: Cognitive Architecture.

    Models activation-based memory, production rule firing, and
    chunk learning inspired by ACT-R/SOAR cognitive architectures.
    """

    @property
    def name(self) -> str:
        return "Cognitive Architecture"

    @property
    def level_number(self) -> int:
        return 1

    def __init__(
        self,
        temperature: float = 1.0,
        decay_rate: float = 0.05,
        firing_threshold: float = 0.0,
    ) -> None:
        self.temperature = temperature
        self.decay_rate = decay_rate
        self.firing_threshold = firing_threshold

        self.chunks: dict[str, Chunk] = {}
        self.production_rules: list[ProductionRule] = []
        self.goals: list[str] = []
        self.context: dict[str, Any] = {}
        self._trace: list[ProductionResult] = []

    def add_chunk(
        self, content: Any, chunk_id: Optional[str] = None,
        activation: float = 0.0,
    ) -> Chunk:
        """Add a new chunk to memory."""
        chunk = Chunk(
            id=chunk_id or uuid4().hex[:12],
            content=content,
            base_level_activation=activation,
        )
        self.chunks[chunk.id] = chunk
        return chunk

    def add_production_rule(
        self,
        condition: Callable[[dict], bool],
        action: str,
        utility: float = 0.5,
        strength: float = 1.0,
    ) -> ProductionRule:
        """Add a production rule."""
        rule = ProductionRule(
            condition=condition,
            action=action,
            utility=utility,
            strength=strength,
        )
        self.production_rules.append(rule)
        return rule

    def push_goal(self, goal: str) -> None:
        """Push a new goal onto the goal stack."""
        self.goals.append(goal)

    def pop_goal(self) -> Optional[str]:
        """Pop the current goal from the stack."""
        return self.goals.pop() if self.goals else None

    def compute_activation(self, chunk_id: str) -> float:
        """Compute activation for a chunk: A_i = B_i + Σ W_j S_ij.

        For now, simplified to base-level activation only.
        Full spreading activation requires chunk-chunk associations.
        """
        chunk = self.chunks.get(chunk_id)
        if chunk is None:
            return -10.0  # Very low activation for unknown chunks

        # Base-level activation with power-law decay
        t = time.time() - chunk.creation_time
        if t > 0 and chunk.access_count > 0:
            B = chunk.base_level_activation + math.log(
                chunk.access_count / (t ** self.decay_rate)
            )
        else:
            B = chunk.base_level_activation

        return B

    def retrieve(self, goal: str) -> Optional[Chunk]:
        """Retrieve the most active chunk relevant to the goal.

        Uses softmax: P(retrieve i) = e^{A_i/τ} / Σ_k e^{A_k/τ}
        """
        if not self.chunks:
            return None

        # Compute activations
        activations = {}
        for chunk_id in self.chunks:
            activations[chunk_id] = self.compute_activation(chunk_id)

        # Softmax selection
        chunk_ids = list(activations.keys())
        act_values = [activations[cid] for cid in chunk_ids]

        if not act_values:
            return None

        # Temperature-scaled softmax
        max_act = max(act_values)
        exp_values = [
            math.exp((a - max_act) / self.temperature)
            for a in act_values
        ]
        total = sum(exp_values)

        if total == 0:
            return None

        probs = [e / total for e in exp_values]

        # Select by highest probability (deterministic for now)
        best_idx = probs.index(max(probs))
        best_chunk = self.chunks[chunk_ids[best_idx]]

        # Update access stats
        best_chunk.access_count += 1
        best_chunk.last_access = time.time()

        return best_chunk

    def fire_rules(self) -> list[ProductionResult]:
        """Fire the highest-utility matching production rule.

        P(r_j) = e^{U_j/τ} / Σ_k e^{U_k/τ}
        """
        matching = [
            rule for rule in self.production_rules
            if rule.condition(self.context) and rule.utility >= self.firing_threshold
        ]

        if not matching:
            return []

        # Softmax over utilities
        utilities = [rule.utility for rule in matching]
        max_u = max(utilities)
        exp_values = [math.exp((u - max_u) / self.temperature) for u in utilities]
        total = sum(exp_values)

        if total == 0:
            return []

        probs = [e / total for e in exp_values]

        # Fire rules in probability order (multiple can fire)
        results = []
        fired = set()
        for idx in sorted(range(len(matching)), key=lambda i: -probs[i]):
            rule = matching[idx]
            if rule.action not in fired:
                result = ProductionResult(
                    rule=rule,
                    action=rule.action,
                    context_snapshot=dict(self.context),
                )
                results.append(result)
                self._trace.append(result)
                fired.add(rule.action)
                self.context[rule.action] = True

        return results

    def learn_chunk(self, trace: Optional[list[ProductionResult]] = None) -> Optional[Chunk]:
        """Learn a new chunk by compressing a trace of production firings.

        r_new = compress(trace of steps)
        """
        if trace is None:
            trace = self._trace

        if not trace:
            return None

        # Compress trace into a single chunk
        actions = [r.action for r in trace]
        content = {
            "actions": actions,
            "length": len(actions),
            "source_context": trace[0].context_snapshot if trace else {},
        }

        chunk = self.add_chunk(content, activation=1.0)
        logger.debug("Learned chunk %s from trace of %d steps", chunk.id, len(trace))

        # Clear trace after learning
        self._trace.clear()

        return chunk

    def step(self) -> dict[str, Any]:
        """Run one cognitive cycle: retrieve → fire → learn.

        Returns the updated context.
        """
        # Retrieve based on current goal
        current_goal = self.goals[-1] if self.goals else None
        if current_goal:
            retrieved = self.retrieve(current_goal)
            if retrieved:
                self.context["retrieved"] = retrieved.content

        # Fire production rules
        results = self.fire_rules()

        # Learn from trace if enough steps
        if len(self._trace) >= 3:
            self.learn_chunk()

        return dict(self.context)

    def compute(
        self, graph: Graph, context: ReasoningContext,
    ) -> LevelOutput:
        """Run cognitive processing on the graph.

        1. Map graph nodes to chunks
        2. Set up production rules from node types
        3. Run cognitive cycles
        4. Return beliefs based on chunk activations
        """
        if not graph.nodes:
            return LevelOutput(beliefs={}, metadata={})

        # Apply coefficient overrides
        if context.coefficients:
            self.temperature = context.coefficients.level1_temperature
            self.decay_rate = context.coefficients.level1_decay_rate
            self.firing_threshold = context.coefficients.level1_firing_threshold

        # Clear state
        self.chunks.clear()
        self.production_rules.clear()
        self.context.clear()
        self._trace.clear()

        # Map graph nodes to chunks
        node_to_chunk = {}
        for node_id, node in graph.nodes.items():
            chunk = self.add_chunk(
                content={"node_id": str(node_id), "text": node.text, "type": node.type.name},
                activation=0.0,
            )
            node_to_chunk[node_id] = chunk

        # Set up production rules based on edge structure
        for edge in graph.edges:
            if edge.source_id in node_to_chunk and edge.target_id in node_to_chunk:
                source_chunk = node_to_chunk[edge.source_id]
                target_chunk = node_to_chunk[edge.target_id]

                def make_condition(sc):
                    return lambda ctx: ctx.get("retrieved") == sc.content

                self.add_production_rule(
                    condition=make_condition(source_chunk),
                    action=f"activate_{target_chunk.id}",
                    utility=0.7,
                )

        # Run cognitive cycles
        for _ in range(10):  # max cycles
            self.push_goal("reason")
            self.step()
            self.pop_goal()

        # Map chunk activations to beliefs
        beliefs = {}
        for node_id, chunk in node_to_chunk.items():
            activation = self.compute_activation(chunk.id)
            # Convert activation to [0, 1] via sigmoid
            beliefs[node_id] = 1.0 / (1.0 + math.exp(-activation))

        return LevelOutput(
            beliefs=beliefs,
            metadata={
                "num_chunks": len(self.chunks),
                "num_rules": len(self.production_rules),
                "num_cycles": 10,
                "context_size": len(self.context),
            },
        )

    def clear(self) -> None:
        """Reset all cognitive state."""
        self.chunks.clear()
        self.production_rules.clear()
        self.goals.clear()
        self.context.clear()
        self._trace.clear()
