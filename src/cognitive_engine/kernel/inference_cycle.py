"""InferenceCycle — the 9-step iterative reasoning loop.

Replaces BrainCycle (same math, honest name). Runs until
state delta norm falls below ε or cycle limit is reached.

Steps:
    1. Extract — Ξ operator (idempotent, guarded)
    2. Structural pass — schema, graph, constraint, temporal
    3. Policy evaluation — OperatorSelector picks next operators
    4. Evidential pass — propagate + selected evidential operators
    5. Conflict pass — debate, verify, constraint
    6. State delta computation
    7. Memory consolidation — STM → LTM if overflow
    8. Convergence check
    9. Tick — increment cycle, emit state diff
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from cognitive_engine.core.math import (
    convergence_norm, graph_distance, hidden_state_distance,
)
from cognitive_engine.core.state import State
from cognitive_engine.core.trace import StateDelta
from cognitive_engine.kernel.assertion_gate import AssertionGate
from cognitive_engine.kernel.self_reflect import SelfReflectOperator, SelfReflectionConfig
from cognitive_engine.memory.store import MemoryStore
from cognitive_engine.memory.feedback import FeedbackStore
from cognitive_engine.agents.manager import ManagerAgent
from cognitive_engine.agents.blackboard import BlackboardAgent
from cognitive_engine.policy.engine import PolicyEngine, PolicySelection
from cognitive_engine.tbox.loader import TBox

logger = logging.getLogger(__name__)


@dataclass
class InferenceCycleConfig:
    """Configuration for the InferenceCycle."""
    epsilon: float = 1e-4
    max_cycles: int = 20
    stm_capacity: int = 128
    policy_name: str = "default"
    domain: str = "general"
    convergence_window: int = 3
    self_reflect_frequency: int = 0  # 0 = disabled


@dataclass
class CycleReport:
    """Report for a single inference cycle."""
    cycle: int
    norm: float
    converged: bool
    operator_log: list[str] = field(default_factory=list)
    policy_selection: Optional[PolicySelection] = None
    state_snapshot: Optional[dict] = None
    duration: float = 0.0


@dataclass
class InferenceResult:
    """Final result of the inference loop."""
    state: State
    cycles: list[CycleReport] = field(default_factory=list)
    converged: bool = False
    total_duration: float = 0.0
    final_norm: float = 0.0
    total_cycles: int = 0


class InferenceCycle:
    """9-step iterative reasoning loop (replaces BrainCycle).

    The InferenceCycle runs three passes per cycle:
        - Pass 1 (Structural): What exists, how is it connected?
        - Pass 2 (Evidential): How strongly is each claim supported?
        - Pass 3 (Conflict): What contradicts, and how to resolve?

    All formal semantics (SL, category theory, master equation)
    are delegated to cognitive_engine.core.math.
    """

    def __init__(
        self,
        operators: dict[str, Any],
        config: Optional[InferenceCycleConfig] = None,
        memory_store: Optional[MemoryStore] = None,
        assertion_gate: Optional[AssertionGate] = None,
        tbox: Optional[TBox] = None,
        policy: Optional[PolicyEngine] = None,
        feedback_store: Optional[FeedbackStore] = None,
        self_reflect: Optional[SelfReflectOperator] = None,
        manager: Optional[ManagerAgent] = None,
        blackboard: Optional[BlackboardAgent] = None,
    ):
        self.operators = operators
        self.config = config or InferenceCycleConfig()
        self.memory = memory_store
        self.gate = assertion_gate or AssertionGate()
        self.tbox = tbox
        self.policy = policy or PolicyEngine()
        self.feedback = feedback_store
        # Auto-create self_reflect if frequency is configured
        if self_reflect is not None:
            self.self_reflect = self_reflect
        elif self.config.self_reflect_frequency > 0:
            self.self_reflect = SelfReflectOperator(
                SelfReflectionConfig(frequency=self.config.self_reflect_frequency)
            )
        else:
            self.self_reflect = None
        self.manager = manager
        self.blackboard = blackboard

    def run(self, initial_state: State) -> InferenceResult:
        """Run the 9-step InferenceCycle to convergence.

        Args:
            initial_state: The initial reasoning state.

        Returns:
            InferenceResult with final state and cycle history.
        """
        state = initial_state
        cycles: list[CycleReport] = []
        start_time = time.time()
        prev_state_snapshot: Optional[dict] = None
        stall_count = 0

        for cycle in range(1, self.config.max_cycles + 1):
            cycle_start = time.time()
            operator_log: list[str] = []
            state.metadata["cycle"] = cycle

            # ── Step 1: Extract (idempotent) ──────────────────────
            if not state.metadata.get("extracted") and "extract" in self.operators:
                state = self.operators["extract"](state)
                state.metadata["extracted"] = True
                operator_log.append("extract")
                state.record("extract", "Text → Graph extraction")

            # ── Step 2: Structural pass ───────────────────────────
            structural_ops = ["schema", "graph", "constraint", "temporal"]
            for op_name in structural_ops:
                if op_name in self.operators:
                    try:
                        state = self.operators[op_name](state)
                        operator_log.append(op_name)
                        state.record(op_name, f"Structural pass: {op_name}")
                    except Exception:
                        logger.warning("Structural operator %s failed, skipping", op_name)

            # ── Step 3: Policy evaluation ─────────────────────────
            selection = self.policy.select(state, cycle=cycle, domain=self.config.domain)
            operator_log.append(f"policy={selection.policy_name}:{selection.operators}")

            # ── Step 4: Evidential pass ───────────────────────────
            evidential_ops = ["propagate", "abduce", "induce", "analogy",
                              "reason", "align", "attention", "merge", "simulate"]
            for op_name in evidential_ops:
                if op_name in self.operators and op_name not in operator_log:
                    if op_name in selection.operators:
                        try:
                            state = self.operators[op_name](state)
                            operator_log.append(op_name)
                            state.record(op_name, f"Evidential pass: {op_name}")
                        except Exception:
                            logger.warning("Evidential operator %s failed, skipping", op_name)

            # Always run propagate (mandatory evidential operator)
            if "propagate" in self.operators and "propagate" not in operator_log:
                try:
                    state = self.operators["propagate"](state)
                    operator_log.append("propagate")
                    state.record("propagate", "Mandatory SL propagation")
                except Exception:
                    logger.warning("propagate failed, continuing")

            # ── Step 5: Conflict pass ─────────────────────────────
            conflict_ops = ["debate", "verify", "constraint", "compress"]
            for op_name in conflict_ops:
                if op_name in self.operators and op_name not in operator_log:
                    if op_name in selection.operators:
                        try:
                            state = self.operators[op_name](state)
                            operator_log.append(op_name)
                            state.record(op_name, f"Conflict pass: {op_name}")
                        except Exception:
                            logger.warning("Conflict operator %s failed, skipping", op_name)

            # ── Step 6: State delta computation ───────────────────
            curr_snapshot = self._state_snapshot(state)
            if prev_state_snapshot is not None:
                d_norm = self._compute_norm(prev_state_snapshot, curr_snapshot)
            else:
                d_norm = 1.0
            state.metadata["last_norm"] = d_norm

            # ── Step 7: Memory consolidation ──────────────────────
            if self.memory:
                try:
                    self.memory.store(state.graph)
                except Exception:
                    logger.warning("Memory store failed, continuing")

            # ── Step 7b: Feedback fusion ──────────────────────────
            if self.feedback:
                try:
                    fused = self.feedback.fuse_all()
                    for nid, opinion in fused.items():
                        if str(nid) in state.graph.nodes:
                            node = state.graph.nodes[str(nid)]
                            if node.opinion:
                                from cognitive_engine.core.math import cumulative_fusion
                                node.opinion = cumulative_fusion(node.opinion, opinion)
                except Exception:
                    logger.warning("Feedback fusion failed, continuing")

            # ── Step 7c: Self-reflection ──────────────────────────
            if self.self_reflect and self.self_reflect.should_reflect(cycle):
                try:
                    reflection = self.self_reflect.reflect(state, cycle)
                    state.metadata["self_reflection"] = {
                        "cycle": cycle,
                        "tier_counts": reflection.tier_counts,
                        "low_belief_ratio": reflection.low_belief_ratio,
                        "high_conflict_nodes": reflection.high_conflict_nodes,
                        "recommendations": reflection.recommendations,
                    }
                    if reflection.recommendations:
                        logger.info("Self-reflection at cycle %d: %s",
                                    cycle, "; ".join(reflection.recommendations[:3]))
                except Exception:
                    logger.warning("Self-reflection failed, continuing")

            # ── Step 7d: Agent health check + blackboard ──────────
            if self.manager:
                try:
                    health = self.manager.health_check(state.graph)
                    state.metadata["health"] = {
                        "healthy": health.healthy,
                        "convergence_rate": health.convergence_rate,
                        "evidence_density": health.evidence_density,
                        "conflict_ratio": health.conflict_ratio,
                        "belief_variance": health.belief_variance,
                        "world_model_coverage": health.world_model_coverage,
                        "bottlenecks": health.bottlenecks,
                    }
                except Exception:
                    logger.warning("Manager health check failed, continuing")
            if self.blackboard:
                try:
                    self.blackboard.publish(f"cycle_{cycle}", {
                        "norm": d_norm,
                        "operators": operator_log,
                        "converged": d_norm < self.config.epsilon,
                    }, publisher="inference_cycle")
                except Exception:
                    logger.warning("Blackboard publish failed, continuing")

            # ── Step 8: Convergence check ─────────────────────────
            converged = d_norm < self.config.epsilon
            if converged:
                logger.info("Converged at cycle %d (‖Δs‖=%.6f)", cycle, d_norm)

            # Stall detection
            stall_count = stall_count + 1 if d_norm < self.config.epsilon or prev_state_snapshot == curr_snapshot else 0
            state.metadata["convergence_stalled"] = stall_count >= self.config.convergence_window

            # ── Step 9: Tick ──────────────────────────────────────
            cycle_duration = time.time() - cycle_start
            report = CycleReport(
                cycle=cycle,
                norm=d_norm,
                converged=converged,
                operator_log=operator_log,
                policy_selection=selection,
                state_snapshot=curr_snapshot,
                duration=cycle_duration,
            )
            cycles.append(report)
            prev_state_snapshot = curr_snapshot

            # Record state delta
            state.record(
                f"cycle_{cycle}",
                f"C{cycle}: d={d_norm:.4f}, ops={operator_log}",
                effect_type="cycle",
                norm=d_norm,
                converged=converged,
                operators=operator_log,
            )

            if converged:
                break

            # Update policy from domain config
            if "constraint" in operator_log and self.config.policy_name == "default":
                pass  # Policy engine handles this via rules

        total_duration = time.time() - start_time
        last_norm = cycles[-1].norm if cycles else 1.0

        return InferenceResult(
            state=state,
            cycles=cycles,
            converged=cycles[-1].converged if cycles else False,
            total_duration=total_duration,
            final_norm=last_norm,
            total_cycles=len(cycles),
        )

    def _state_snapshot(self, state: State) -> dict:
        """Create a serializable state snapshot for comparison."""
        graph = state.graph
        beliefs = {}
        for nid, node in graph.nodes.items():
            if node.opinion:
                b, d, u, a = node.opinion
                beliefs[str(nid)] = b + a * u
            else:
                beliefs[str(nid)] = 0.5
        edges = []
        for eid, edge in graph.edges.items():
            edges.append((str(edge.source_id), str(edge.target_id), edge.type.name))
        return {
            "node_ids": {str(nid) for nid in graph.nodes},
            "edges": set(edges),
            "beliefs": beliefs,
            "operator": state.metadata.get("last_operator", ""),
        }

    def _compute_norm(self, prev: dict, curr: dict) -> float:
        """Compute ‖Δs‖ convergence norm."""
        g_dist = graph_distance(
            prev.get("node_ids", set()),
            curr.get("node_ids", set()),
            list(prev.get("edges", set())),
            list(curr.get("edges", set())),
            prev.get("beliefs", {}),
            curr.get("beliefs", {}),
        )
        op_change = 0.0 if prev.get("operator") == curr.get("operator") else 1.0
        return convergence_norm(
            graph_distance=g_dist,
            attention_distance=g_dist,
            hidden_distance=1.0,
            operator_change=op_change,
        )
