from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cognitive_engine.core.operator import Operator
from cognitive_engine.core.state import State


@dataclass
class Pipeline:
    """Composition of operators.

    G_out = (O_n ∘ ... ∘ O_1)(S)

    Operators are applied in order. Each operator receives the State
    produced by the previous operator.

    Example:
        pipeline = Pipeline(
            name="paper_consensus",
            operators=[
                ExtractOperator(),
                PropagateOperator(),
                CompressOperator(),
            ]
        )
        result = pipeline.run(initial_state)
    """
    name: str
    operators: list[Operator] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def run(self, state: State) -> State:
        """Compose: O_n ∘ ... ∘ O_1"""
        for op in self.operators:
            state = op(state)
            state.record(
                operator_name=op.name,
                description=f"Applied {op.name}",
            )
        return state

    def add(self, operator: Operator) -> Pipeline:
        """Fluent API for building pipelines (returns self)."""
        self.operators.append(operator)
        return self

    def insert(self, index: int, operator: Operator) -> Pipeline:
        """Insert operator at position."""
        self.operators.insert(index, operator)
        return self

    def remove(self, name: str) -> Pipeline:
        """Remove first operator with given name."""
        self.operators = [op for op in self.operators if op.name != name]
        return self

    def before(self, existing: str, new_op: Operator) -> Pipeline:
        """Insert new_op before the operator named 'existing'."""
        for i, op in enumerate(self.operators):
            if op.name == existing:
                self.operators.insert(i, new_op)
                return self
        raise ValueError(f"Operator '{existing}' not found in pipeline")

    def after(self, existing: str, new_op: Operator) -> Pipeline:
        """Insert new_op after the operator named 'existing'."""
        for i, op in enumerate(self.operators):
            if op.name == existing:
                self.operators.insert(i + 1, new_op)
                return self
        raise ValueError(f"Operator '{existing}' not found in pipeline")

    def replace(self, name: str, new_op: Operator) -> Pipeline:
        """Replace operator named 'name' with new_op."""
        for i, op in enumerate(self.operators):
            if op.name == name:
                self.operators[i] = new_op
                return self
        raise ValueError(f"Operator '{name}' not found in pipeline")

    def copy(self, name: str | None = None) -> Pipeline:
        """Create a copy of this pipeline."""
        return Pipeline(
            name=name or self.name,
            operators=list(self.operators),
            metadata=dict(self.metadata),
        )

    def __len__(self) -> int:
        return len(self.operators)

    def __repr__(self) -> str:
        op_names = " → ".join(op.name for op in self.operators)
        return f"Pipeline({self.name}: {op_names})"
