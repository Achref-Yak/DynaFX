"""π (Plan) operator — Multi-step operator sequence generation.

Generates a plan (ordered list of operators) using the PolicyEngine,
then stores it in state metadata for sequential execution by the
InferenceCycle.
"""

from __future__ import annotations

from dynafx.core.state import State
from dynafx.policy.engine import PolicyEngine
from dynafx.policy.schema import OperatorPolicy


class PlanOperator:
    """π: Generate multi-step plan using policy-based selection.

    At each step, the PolicyEngine evaluates all available operator
    policies against the current state metrics and selects the
    highest-scoring one. Repeated for ``plan_horizon`` steps.

    The plan is stored in ``state.metadata["plan"]`` as a list of
    operator names. The InferenceCycle consumes one step per cycle
    via ``state.metadata["plan_index"]``.
    """
    name = "plan"

    def __init__(self, horizon: int = 3):
        self._engine = PolicyEngine()
        self._horizon = horizon

    def __call__(self, state: State, **kwargs) -> State:
        horizon = kwargs.get("plan_horizon") or state.metadata.get(
            "plan_horizon", self._horizon
        )
        plan = []

        for _ in range(horizon):
            selection = self._engine.select(state, cycle=state.metadata.get("cycle", 0))
            op = selection.operator if selection else "propagate"
            if op == "done":
                break
            if op == "plan":
                continue
            plan.append(op)

        if not plan:
            plan = ["propagate"]

        state.metadata["plan"] = plan
        state.metadata["plan_index"] = 0
        state.metadata["plan_horizon"] = len(plan)

        state.record(
            self.name,
            f"Devised a multi-step reasoning plan: {' → '.join(plan)} ({len(plan)} steps). "
            f"The PolicyEngine selected this sequence by evaluating operator rules against current state metrics. "
            f"Step 1 ({plan[0] if plan else 'none'}) addresses the most salient gap or opportunity in the current belief structure.",
        )
        return state
